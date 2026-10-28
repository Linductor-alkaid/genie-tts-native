#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""校验 C++ 的英文 G2P 与上游的 Python 实现是否逐条一致。

上游的英文链路（`G2P/English/`）依赖 nltk（分词 + 词性标注）、
CMU 发音词典、以及 g2p_en 的 GRU 权重。本脚本把上游的 Python 当
**oracle**，与 C++ 的输出逐音素比对：

    python3 tools/g2p/verify_english_g2p.py --binary /tmp/dump_english

需要的资源（沙箱里可用 pypi 镜像拿）：

  - `nltk` + `nltk_data`（分词器与 averaged perceptron 词性标注）
  - CMU 发音词典（pypi 包 `cmudict`）
  - g2p_en 的 `checkpoint20.npz`（OOV 兜底）
  - 上游的 `English_G2P_DIR`（本脚本用 --upstream-dir 指定）

用法::

    python3 tools/g2p/verify_english_g2p.py --binary /tmp/dump_english \\
        --corpus tools/g2p/english_corpus.txt

退出码非 0 表示有不一致。
"""

import argparse
import os
import subprocess
import sys
import types
import importlib.util

REF = "/workspace/genie-tts-ref/src/genie_tts"
DATA = "/tmp/fake_genie"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_upstream(upstream_dir):
    """加载上游的 EnglishG2P（绕开 genie_tts.__init__ 的 onnxruntime 依赖）。"""
    pkg = types.ModuleType("genie_tts")
    pkg.__path__ = [REF]
    sys.modules["genie_tts"] = pkg
    core = types.ModuleType("genie_tts.Core")
    core.__path__ = [REF + "/Core"]
    sys.modules["genie_tts.Core"] = core
    resources = types.ModuleType("genie_tts.Core.Resources")
    resources.English_G2P_DIR = upstream_dir
    resources.Chinese_G2P_DIR = DATA + "/G2P/ChineseG2P"
    sys.modules["genie_tts.Core.Resources"] = resources
    g2p_pkg = types.ModuleType("genie_tts.G2P")
    g2p_pkg.__path__ = [REF + "/G2P"]
    sys.modules["genie_tts.G2P"] = g2p_pkg
    _load("genie_tts.G2P.SymbolsV2", REF + "/G2P/SymbolsV2.py")
    english = types.ModuleType("genie_tts.G2P.English")
    english.__path__ = [REF + "/G2P/English"]
    sys.modules["genie_tts.G2P.English"] = english
    _load("genie_tts.G2P.English.Normalization", REF + "/G2P/English/Normalization.py")
    _load("genie_tts.G2P.English.WordSegment", REF + "/G2P/English/WordSegment.py")
    return _load("genie_tts.G2P.English.EnglishG2P", REF + "/G2P/English/EnglishG2P.py")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="/tmp/dump_english", help="C++ dump 工具路径")
    parser.add_argument("--corpus", default="tools/g2p/english_corpus.txt", help="语料文件")
    parser.add_argument("--upstream-dir", default=DATA + "/G2P/EnglishG2P")
    parser.add_argument("--cmu-dict", default="third_party/english/cmudict.bin")
    parser.add_argument("--pos-tagger", default="third_party/english/pos_tagger.bin")
    parser.add_argument("--g2p-model", default="third_party/english/english_g2p.bin")
    parser.add_argument("--word-segment-dir", default="")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", help="写出黄金语料（JSON）")
    args = parser.parse_args()

    os.environ.setdefault("NLTK_DATA", "/tmp/nltk_data")

    with open(args.corpus, encoding="utf-8") as handle:
        corpus = [line.rstrip("\n") for line in handle if line.strip()]

    upstream = build_upstream(args.upstream_dir)
    normalization = sys.modules["genie_tts.G2P.English.Normalization"]

    def run_dump(word_segment_dir):
        env = dict(os.environ)
        env["GENIE_CMU_DICT"] = args.cmu_dict
        env["GENIE_POS_TAGGER"] = args.pos_tagger
        env["GENIE_ENGLISH_G2P_MODEL"] = args.g2p_model
        if word_segment_dir:
            env["GENIE_WORD_SEGMENT_DIR"] = word_segment_dir
        else:
            env.pop("GENIE_WORD_SEGMENT_DIR", None)
        out = subprocess.run(
            [args.binary],
            input="\n".join(corpus) + "\n",
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        return out.stdout.splitlines()

    lines = run_dump(args.word_segment_dir)
    # ngram 分词词典不随仓库发布（12 MB）。没有它的时候，
    # 依赖分词的 OOV 词会走 GRU 兜底 —— 结果与上游不同。
    # 这类用例在黄金语料里标出来，单测跳过音素比对（规范化仍然比对），
    # 完整比对由本脚本在有词典时完成。
    lines_without_segmenter = run_dump("") if args.word_segment_dir else lines
    if len(lines) != len(corpus):
        print(f"dump 输出行数 {len(lines)} != 语料行数 {len(corpus)}", file=sys.stderr)
        return 1

    golden = []
    failures = 0
    for index, (text, line) in enumerate(zip(corpus, lines)):
        normalized, _, rest = line.partition("\t")
        text_normalized, _, phonemes = rest.partition("\t")
        expected = upstream.g2p(text)
        actual = [p for p in phonemes.split("|") if p]
        without_segmenter = lines_without_segmenter[index].partition("\t")[2].partition("\t")[2]
        needs_segmenter = without_segmenter != phonemes
        # 规范化也要逐条一致（上游 normalize() 只做规范化，
        # text_normalize 才是"规范化 + REP_MAP + 连续标点折叠"）。
        expected_normalized = normalization.normalize(text)
        if expected_normalized != normalized:
            failures += 1
            print(f"✗ 规范化不一致 {text!r}")
            print(f"  期望: {expected_normalized!r}")
            print(f"  实际: {normalized!r}")
        expected_text_normalized = upstream.text_normalize(text)
        if expected_text_normalized != text_normalized:
            failures += 1
            print(f"✗ text_normalize 不一致 {text!r}")
            print(f"  期望: {expected_text_normalized!r}")
            print(f"  实际: {text_normalized!r}")
        golden.append(
            {
                "text": text,
                "normalized": normalized,
                "text_normalized": text_normalized,
                "phonemes": expected,
                "needs_word_segment": needs_segmenter,
            }
        )
        if expected != actual:
            failures += 1
            print(f"✗ {text!r}")
            print(f"  期望: {expected}")
            print(f"  实际: {actual}")
            if normalized:
                print(f"  规范化: {normalized!r}")
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
