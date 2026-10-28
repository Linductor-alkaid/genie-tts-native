#!/usr/bin/env python3
"""verify_g2pm.py —— g2pM 多音字移植的 oracle。

上游基准：pip 包 g2pM（kakaobrain/g2pM 训练、mozillazg 打包）的

    G2pM()(segment, char_split=True)

逐字返回带调拼音。比对内容：
  1. C++ 实现（tools/g2p/dump_g2pm.cpp）与上游逐字读音一致；
  2. 不一致时打印上下文，便于定位是数值舍入还是逻辑偏差。

用法：
    PYTHONPATH=/tmp/g2pm python3 tools/g2p/verify_g2pm.py /tmp/dump_g2pm \
        third_party/g2pm/g2pm.bin [--output tests/golden/g2pm.json]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

CORPUS = Path(__file__).with_name("g2pm_corpus.txt")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump_binary", help="tools/g2p/dump_g2pm 的编译产物")
    parser.add_argument("model", help="third_party/g2pm/g2pm.bin")
    parser.add_argument("--output", help="把黄金语料写到这个路径（默认只比对不落盘）")
    args = parser.parse_args()

    from g2pM import G2pM  # noqa: PLC0415 —— 上游基准，需 PYTHONPATH 指向 g2pM 包

    model = G2pM()

    lines = [
        line
        for line in CORPUS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dump = subprocess.run(
        [args.dump_binary, args.model],
        input="\n".join(lines) + "\n",
        capture_output=True, text=True, check=True,
    )
    cpp_rows = [json.loads(line) for line in dump.stdout.splitlines() if line.strip()]

    failures = 0
    cases = []
    for text, row in zip(lines, cpp_rows):
        expected = model(text, char_split=True)
        got = row["pinyins"]
        if got != expected:
            failures += 1
            print(f"✗ {text!r}")
            for i, (e, g) in enumerate(zip(expected, got)):
                if e != g:
                    print(f"    [{i}] {text[i]}: 上游 {e!r} vs C++ {g!r}")
            if len(expected) != len(got):
                print(f"    长度: 上游 {len(expected)} vs C++ {len(got)}")
            continue
        cases.append({"text": text, "pinyins": got})
        print(f"✓ {text!r} ({len(got)} 字)")

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
