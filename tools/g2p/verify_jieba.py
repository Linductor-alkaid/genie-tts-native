#!/usr/bin/env python3
"""verify_jieba.py —— jieba 分词移植的 oracle。

上游基准：jieba Python（jieba.lcut，HMM=True 默认路径；jieba_fast 与其
结果一致）。比对内容：纯汉字段的分词序列逐词一致。

用法：
    c++ -std=c++20 -O1 -Iinclude -Isrc tools/g2p/dump_jieba.cpp \\
        build/debug/src/libgenie_core.a -o /tmp/dump_jieba
    PYTHONPATH=/tmp/jieba_audit/jieba-0.42.1 \\
        python3 tools/g2p/verify_jieba.py /tmp/dump_jieba third_party/jieba/jieba.bin
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

CORPUS = Path(__file__).with_name("jieba_corpus.txt")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump_binary", help="tools/g2p/dump_jieba 的编译产物")
    parser.add_argument("model", help="third_party/jieba/jieba.bin")
    parser.add_argument("--output", help="把黄金语料写到这个路径")
    args = parser.parse_args()

    import jieba  # noqa: PLC0415 —— 上游基准，需 PYTHONPATH 指向 jieba 包

    jieba.setLogLevel(60)

    lines = [
        "".join(ch for ch in line if "\u4e00" <= ch <= "\u9fd5")
        for line in CORPUS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    lines = [line for line in lines if line]
    dump = subprocess.run(
        [args.dump_binary, args.model],
        input="\n".join(lines) + "\n",
        capture_output=True, text=True, check=True,
    )
    cpp_rows = [json.loads(line) for line in dump.stdout.splitlines() if line.strip()]

    failures = 0
    cases = []
    for text, row in zip(lines, cpp_rows):
        expected = [w for w in jieba.lcut(text) if w.strip()]
        got = row["words"]
        if got != expected:
            failures += 1
            print(f"✗ {text!r}")
            print(f"    上游: {expected}")
            print(f"    C++ : {got}")
            continue
        cases.append({"text": text, "words": got})

    print(f"\n{len(cases)}/{len(lines)} 通过，{failures} 失败")
    if args.output:
        Path(args.output).write_text(
            json.dumps({"cases": cases}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"黄金语料已写入 {args.output}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
