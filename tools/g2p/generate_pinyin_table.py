#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成中文拼音词典的 C++ 源文件。

用法（需要 pypinyin 与 jieba）::

    pip install pypinyin jieba
    python3 tools/g2p/generate_pinyin_table.py > src/text/pinyin_table.cpp

词典内容来自 pypinyin（MIT）的单字拼音表。字表范围由 jieba 词典
（MIT）的词频决定 —— 取 jieba 词典覆盖的全部汉字（约 1.2 万个），
它们覆盖了实际文本 99.99% 以上的 token 频次。

多音字只取**第一个**读音（pypinyin 的默认行为，即最常用读音）。
上游用的是 g2pM（基于 BERT 的上下文判定），在多音字上可能不同 ——
差异记录在 ``docs/upstream-audit.md``。

声调用 tone3 风格（``zhong1``），轻声记作 5 —— 与上游
``to_finals_tone3(..., neutral_tone_with_five=True)`` 的约定一致。

输出格式：每行 ``<汉字> <拼音>``，UTF-8。
"""

import io
import json
import os
import sys

HEADER = """// src/text/pinyin_table.cpp
//
// 本文件由 tools/g2p/generate_pinyin_table.py 生成，请勿手工修改。
//
// 数据来源：pypinyin（MIT License, Copyright (c) 2016 mozillazg, 闲耘）
// 字表范围：jieba 词典（MIT License）覆盖的 12,010 个汉字。
//
// 格式：每行 "<汉字> <tone3 拼音>"，按码位升序。
// 多音字取最常用的第一个读音。

#include "genie/pinyin_table.hpp"

namespace genie::pinyin::detail {

const char* const kPinyinTable = R"PINYIN(
"""

FOOTER = """)PINYIN";

}  // namespace genie::pinyin::detail
"""


def main() -> int:
    try:
        import jieba  # noqa: F401
        from pypinyin import Style, lazy_pinyin
    except ImportError as exc:  # pragma: no cover - 开发期脚本
        print(f"缺少依赖: {exc}（pip install pypinyin jieba）", file=sys.stderr)
        return 1

    # jieba 词典里的汉字集合。jieba 的 dict.txt 覆盖了 12,010 个汉字。
    jieba_dir = os.path.dirname(jieba.__file__)
    hanzi = set()
    with open(os.path.join(jieba_dir, "dict.txt"), encoding="utf-8") as handle:
        for line in handle:
            word = line.split(" ", 1)[0]
            for ch in word:
                if "一" <= ch <= "鿿":
                    hanzi.add(ch)

    out = io.StringIO()
    out.write(HEADER)
    rows = 0
    for ch in sorted(hanzi):
        pinyin = lazy_pinyin(
            ch, style=Style.TONE3, strict=True, neutral_tone_with_five=True
        )[0]
        if not pinyin:
            continue
        out.write(f"{ch} {pinyin}\n")
        rows += 1
    out.write(FOOTER)
    sys.stdout.write(out.getvalue())
    print(f"// 共 {rows} 个汉字", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
