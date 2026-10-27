#!/usr/bin/env python3
"""tools/g2p/generate_opencpop_table.py — 生成 opencpop 词表的 C++ 源文件。

用法：
    python3 tools/g2p/generate_opencpop_table.py <opencpop-strict.txt> [输出.cpp]

词表来源：上游 Genie-TTS 的 GenieData/G2P/ChineseG2P/opencpop-strict.txt
（HuggingFace High-Logic/Genie 仓库，随上游资源分发；opencpop 词表本身的
授权见上游 NOTICE）。格式：每行 "<拼音键>\\t<声母> <韵母>"。

生成物：src/text/opencpop_table.cpp —— 内嵌原始词表文本，运行期惰性解析。
"""

import sys


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    source = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else "src/text/opencpop_table.cpp"

    with open(source, encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]
    lines = [line for line in lines if line.strip()]
    # 键序稳定：按拼音键排序，diff 友好。
    lines.sort(key=lambda line: line.split("\t")[0])
    entries = len(lines)

    body = "\n".join(lines)
    # 原始串里不能出现定界符本身。
    assert "OPENCPOP" not in body

    with open(output, "w", encoding="utf-8") as f:
        f.write(f"""// src/text/opencpop_table.cpp
//
// 本文件由 tools/g2p/generate_opencpop_table.py 生成，请勿手工修改。
//
// 数据来源：上游 Genie-TTS 随资源分发的 opencpop-strict.txt
// （拼音键 → "声母 韵母"，{entries} 条，按键升序）。
// 上游用它把 to_initials/to_finals_tone3 的切分结果转成音素符号对
// （ChineseG2P._pinyin_to_opencpop_phones），是符号转换的唯一权威
// —— 翘舌 ir、ü 的 v、二声 EE 等特殊表示全部来自这张表。

#include "text/opencpop_table.hpp"

namespace genie::pinyin::detail {{

const char* const kOpencpopTable = R"OPENCPOP(
{body}
)OPENCPOP";

}}  // namespace genie::pinyin::detail
""")
    print(f"wrote {output} ({entries} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
