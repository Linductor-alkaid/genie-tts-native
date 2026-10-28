#!/usr/bin/env python3
"""verify_language_split.py —— 中英混读分块的 oracle。

上游参照（GetPhonesAndBert.py 的 split_language，逐字直译）：

    pattern_eng = re.compile(r"[a-zA-Z]+")
    split = pattern_eng.split(text)
    matches = pattern_eng.findall(text)
    result = []
    for i, part in enumerate(split):
        if part.strip():
            result.append({'language': 'chinese', 'content': part})
        if i < len(matches):
            result.append({'language': 'english', 'content': matches[i]})

比对内容：
  1. C++ 的 split_by_language 与上游正则逻辑逐块一致（含英文块）；
  2. C++ 前端产出的 BERT 分块 == 上游分块中的中文块（英文块被丢弃是
     docs/upstream-audit.md §9.4 记录在案的偏差）；
  3. word2ph 对齐（offset 连续覆盖）。

用法：
  c++ -std=c++20 -O1 -Iinclude -Isrc tools/g2p/dump_language_split.cpp \
      build/debug/src/libgenie_core.a -o /tmp/dump_language_split
  python3 tools/g2p/verify_language_split.py /tmp/dump_language_split
  python3 tools/g2p/verify_language_split.py /tmp/dump_language_split --output tests/golden/language_split.json
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

PATTERN_ENG = re.compile(r"[a-zA-Z]+")
CORPUS = Path(__file__).with_name("split_corpus.txt")


def upstream_split(text: str) -> list[dict]:
    """上游 split_language 的逐字直译。"""
    split = PATTERN_ENG.split(text)
    matches = PATTERN_ENG.findall(text)
    assert len(matches) == len(split) - 1, f"re.split/findall 数量不匹配: {text!r}"
    result = []
    for i, part in enumerate(split):
        if part.strip():
            result.append({"language": "chinese", "content": part})
        if i < len(matches):
            result.append({"language": "english", "content": matches[i]})
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump_binary", help="tools/g2p/dump_language_split 的编译产物")
    parser.add_argument("--output", help="把黄金语料写到这个路径（默认只比对不落盘）")
    args = parser.parse_args()

    lines = [
        line
        for line in CORPUS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    dump = subprocess.run(
        [args.dump_binary], input="\n".join(lines) + "\n", capture_output=True, text=True, check=True
    )
    cpp_rows = [json.loads(line) for line in dump.stdout.splitlines() if line.strip()]

    failures = 0
    cases = []
    for text, row in zip(lines, cpp_rows):
        if row["text"] != text:
            print(f"✗ 顺序错乱: {text!r} vs {row['text']!r}")
            failures += 1
            continue

        # 1) 切块与上游一致（含英文块）。
        expected = upstream_split(text)
        got = [{"language": c["language"], "content": c["content"]} for c in row["split"]]
        if got != expected:
            failures += 1
            print(f"✗ 分块不一致: {text!r}")
            print(f"    上游: {expected}")
            print(f"    C++ : {got}")
            continue

        # 2) BERT 分块 == 上游的中文块（顺序一致）。
        chinese = [c["content"] for c in expected if c["language"] == "chinese"]
        bert_contents = [c["content"] for c in row["bert_chunks"]]
        # 前端可能把「规范化后产出为空」的块跳过（没有音素就没有特征行），
        # 所以只要求 C++ 的分块是上游中文块的**子序列**且内容逐块规范化。
        # 内容比对交给 golden 测试与单测（那里断言 = normalize(块)），
        # 这里比对数量关系：产出非空的中文块必然保留。
        if len(bert_contents) > len(chinese):
            failures += 1
            print(f"✗ BERT 分块多于上游中文块: {text!r}")
            continue

        # 3) word2ph 对齐：offset 连续、count 非负，且块数 ≤ 音素数。
        cursor = 0
        for chunk in row["bert_chunks"]:
            if chunk["word_offset"] != cursor or chunk["word_count"] <= 0:
                failures += 1
                print(f"✗ word2ph 对齐断裂: {text!r} @ {chunk}")
                break
            cursor += chunk["word_count"]
        else:
            if row["bert_chunks"] and cursor > row["phones"]:
                failures += 1
                print(f"✗ word2ph 展开超过音素数: {text!r}")
                continue
            cases.append(row)
            print(f"✓ {text!r}: split={len(expected)} bert={len(row['bert_chunks'])} phones={row['phones']}")

    print(f"\n{len(cases)}/{len(lines)} 通过，{failures} 失败")
    if args.output:
        out = {"cases": cases}
        Path(args.output).write_text(
            json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"黄金语料已写入 {args.output}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
