#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""校验 C++ 的日语 G2P 与上游的 Python 实现（pyopenjtalk）是否逐条一致。

上游 `G2P/Japanese/JapaneseG2P.py` 直接用 pyopenjtalk 做全上下文标注，
本项目的 C++ 侧内置了同源的 OpenJTalk（r9y9/open_jtalk 的 1.11 分支）——
两者对同一段文本应该给出**完全相同**的音素序列。

用法::

    python3 tools/g2p/verify_japanese_g2p.py --binary /tmp/dump_japanese

注意：pyopenjtalk 的 wheel 是针对 numpy 1.x 编译的 —— 需要 numpy<2，
否则 import 时报 "numpy.dtype size changed"。

退出码非 0 表示有不一致。
"""

import argparse
import importlib.util
import os
import subprocess
import sys
import types

REF = "/workspace/genie-tts-ref/src/genie_tts"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_upstream():
    """加载上游的 JapaneseG2P（绕开 genie_tts.__init__ 的 onnxruntime 依赖）。"""
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
    japanese = types.ModuleType("genie_tts.G2P.Japanese")
    japanese.__path__ = [REF + "/G2P/Japanese"]
    sys.modules["genie_tts.G2P.Japanese"] = japanese

    return _load("genie_tts.G2P.Japanese.JapaneseG2P", REF + "/G2P/Japanese/JapaneseG2P.py")


def _segments(upstream, text):
    """上游的切段结果（日语片段 ✓）。"""
    normalized = upstream.JapaneseG2P._text_normalize(text)
    return upstream._JAPANESE_MARKS_RE.split(normalized)


def _labels(upstream, text):
    """每段的 OpenJTalk 全上下文标签 ✓（词典相关的那部分 ✓）。"""
    import pyopenjtalk

    out = []
    for segment in _segments(upstream, text):
        if segment:
            out.append(pyopenjtalk.make_label(pyopenjtalk.run_frontend(segment)))
        else:
            out.append([])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="/tmp/dump_japanese", help="C++ dump 工具路径")
    parser.add_argument("--corpus", default="tools/g2p/japanese_corpus.txt", help="语料文件")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", help="写出黄金语料（JSON）")
    args = parser.parse_args()

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
        expected = upstream.JapaneseG2P.g2p(text)
        actual = [p for p in line.split("|") if p]
        golden.append(
            {
                "text": text,
                "normalized": upstream.JapaneseG2P._text_normalize(text),
                "segments": _segments(upstream, text),
                "labels": _labels(upstream, text),
                "phonemes": expected,
            }
        )
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
