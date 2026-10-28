#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""校验 C++ 的韩语 G2P 与上游的 Python 实现是否逐条一致。

上游 `G2P/Korean/KoreanG2P.py` 依赖 g2pk2，而 g2pk2 依赖 mecab-ko
形态分析（mecab 是 C++ 库 + 100 MB 词典）。本脚本把 g2pk2 的 mecab
换成 stub —— 形态标注那一步因此不做任何标注，这正是"C++ 侧不依赖
形态分析"那条降级路径的参考输出（见 docs/upstream-audit.md §18）。

用法::

    NLTK_DATA=/tmp/nltk_data \\
    python3 tools/g2p/verify_korean_g2p.py --binary /tmp/dump_korean

退出码非 0 表示有不一致。
"""

import argparse
import os
import subprocess
import sys
import types
import importlib.util

REF = "/workspace/genie-tts-ref/src/genie_tts"
ORACLE = "/workspace/oracle"


class _StubMecab:
    """没有 mecab 时的替身：整句当成一个 token。

    g2pk2 的 annotate() 里 `string.replace(" ", "") != "".join(tokens)`
    会判定失败，于是原样返回 —— 也就是"不做任何 /J /E /P /B 标注"。
    """

    def pos(self, text):
        return [(text, "NN")]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_upstream():
    """加载上游的 KoreanG2P（绕开 genie_tts.__init__ 的 onnxruntime 依赖）。"""
    pkg = types.ModuleType("genie_tts")
    pkg.__path__ = [REF]
    sys.modules["genie_tts"] = pkg
    core = types.ModuleType("genie_tts.Core")
    core.__path__ = [REF + "/Core"]
    sys.modules["genie_tts.Core"] = core
    g2p_pkg = types.ModuleType("genie_tts.G2P")
    g2p_pkg.__path__ = [REF + "/G2P"]
    sys.modules["genie_tts.G2P"] = g2p_pkg
    _load("genie_tts.G2P.SymbolsV2", REF + "/G2P/SymbolsV2.py")
    korean = types.ModuleType("genie_tts.G2P.Korean")
    korean.__path__ = [REF + "/G2P/Korean"]
    sys.modules["genie_tts.G2P.Korean"] = korean

    # g2pk2 的 G2p 需要 mecab —— 换成 stub。
    import g2pk2.g2pk2 as g2p2_module

    g2p2_module.G2p.check_mecab = lambda self: None
    g2p2_module.G2p.get_mecab = lambda self: _StubMecab()

    return _load("genie_tts.G2P.Korean.KoreanG2P", REF + "/G2P/Korean/KoreanG2P.py")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="/tmp/dump_korean", help="C++ dump 工具路径")
    parser.add_argument("--corpus", default="tools/g2p/korean_corpus.txt", help="语料文件")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", help="写出黄金语料（JSON）")
    args = parser.parse_args()

    os.environ.setdefault("NLTK_DATA", "/tmp/nltk_data")

    with open(args.corpus, encoding="utf-8") as handle:
        corpus = [line.rstrip("\n") for line in handle if line.strip()]

    upstream = build_upstream()

    result = subprocess.run(
        [args.binary],
        input="\n".join(corpus) + "\n",
        capture_output=True,
        text=True,
        check=True,
    )
    lines = result.stdout.splitlines()
    if len(lines) != len(corpus):
        print(f"dump 输出行数 {len(lines)} != 语料行数 {len(corpus)}", file=sys.stderr)
        return 1

    golden = []
    failures = 0
    for text, line in zip(corpus, lines):
        expected = upstream.KoreanG2P.g2p(text)
        actual = [p for p in line.split("|") if p]
        golden.append({"text": text, "phonemes": expected})
        if expected != actual:
            failures += 1
            print(f"✗ {text!r}")
            print(f"  期望: {expected}")
            print(f"  实际: {actual}")
        elif args.verbose:
            print(f"✓ {text!r} -> {actual}")

    if args.output:
        import json

        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump({"cases": golden}, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"wrote {args.output} ({len(golden)} cases)")

    print(f"{len(corpus) - failures}/{len(corpus)} 条一致")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
