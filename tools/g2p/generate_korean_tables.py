#!/usr/bin/env python3
"""生成韩语 G2P 的静态表。

数据来源（都是 pip 上能拿到的公开包）：

  - `jamo` 的 U+11xx.json / U+31xx.json —— 组合字母（U+11xx）与兼容字母
    （U+3131–U+314E）之间的映射。jamo 靠 Unicode 名字匹配，本仓库把它
    固化成一张表。
  - `g2pk2` 的 idioms.txt —— 惯用词的读音替换表（上游按顺序逐条做正则替换）。
  - `g2pk2` 的 table.csv —— "받침 + 起首辅音" 的规则表（402 条）。

用法：
    python3 tools/g2p/generate_korean_tables.py --out src/text/korean/tables.hpp
"""

import argparse
import csv
import json
import pathlib
import re
import sys


def find_package_dir(name: str) -> pathlib.Path:
    """在 site-packages 里找包目录。"""
    try:
        module = __import__(name)
    except ImportError:
        raise SystemExit(f"找不到 {name} —— 先 pip install {name}")
    return pathlib.Path(module.__file__).parent


def jamo_pairs(jamo_dir: pathlib.Path):
    """产出 (兼容字母, 组合字母) 的配对。

    jamo 的做法是按 Unicode 名字匹配：把组合字母名字里的 "CHOSEONG/JUNGSEONG/
    TAIL" 换成 "LETTER"，再找同名的兼容字母。
    """
    modern = json.loads((jamo_dir / "data" / "U+11xx.json").read_text(encoding="utf-8"))
    compat = json.loads((jamo_dir / "data" / "U+31xx.json").read_text(encoding="utf-8"))
    by_name = {name: char for char, name in compat.items()}

    out = []
    for char, name in modern.items():
        renamed = re.sub("(?<=HANGUL )(\\w+)", "LETTER", name)
        if renamed in by_name:
            out.append((by_name[renamed], char))
    out.sort(key=lambda pair: ord(pair[1]))
    return out


def read_idioms(g2pk2_dir: pathlib.Path):
    """读 idioms.txt：每行 "str1===str2"，'#' 之后是注释。"""
    out = []
    for line in (g2pk2_dir / "idioms.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if "===" not in line:
            continue
        left, right = line.split("===")
        out.append((left.strip(), right.strip()))
    return out


def read_regular_table(g2pk2_dir: pathlib.Path):
    """读 table.csv —— "받침 + 起首辅音" 的规则表。

    表头是起首辅音（每格带一个可选空格），每行第一格是 받침。
    单元格里形如 "ᆨ\\1ᄂ(12)"：括号前是替换串，括号里是规则号（只用于
    verbose 输出，本仓库不需要）。
    """
    path = g2pk2_dir / "table.csv"
    # 表头带 BOM。
    text = path.read_text(encoding="utf-8-sig")
    rows = list(csv.reader(text.splitlines()))
    onsets = rows[0]
    table = []
    for row in rows[1:]:
        if not row:
            continue
        coda = row[0]
        for i, onset in enumerate(onsets):
            if i == 0 or i >= len(row):
                continue
            cell = row[i]
            if not cell:
                continue
            if "(" in cell:
                replacement = cell.split("(")[0]
            else:
                replacement = cell
            table.append((coda, onset, replacement))
    return table


HEADER = """// src/text/korean/tables.hpp
//
// 韩语 G2P 的静态表（**生成文件**，不要手改）。
//
// 由 tools/g2p/generate_korean_tables.py 生成，数据来自：
//   - jamo（组合字母 ↔ 兼容字母）
//   - g2pk2（idioms.txt、table.csv）
//
// 数据来源与授权见 docs/upstream-audit.md §18。

#ifndef GENIE_SRC_TEXT_KOREAN_TABLES_HPP
#define GENIE_SRC_TEXT_KOREAN_TABLES_HPP

#include <array>
#include <cstdint>
#include <string_view>

namespace genie::text::korean::tables {

/// 组合字母（U+1100 起）→ 兼容字母（U+3131 起）。
///
/// 注意是**多对一**：组合字母分"起首/中声/收音"，兼容字母不区分 ——
/// ᄀ（起首ㄱ）和 ᆨ（收音ㄱ）都映射到 ㄱ ✓（与 jamo 包一致）。
struct JamoPair {
    std::uint32_t modern;
    std::uint32_t compat;
};

/// {jamo_count} 条映射（按组合字母码位升序，二分查找）。
inline constexpr std::array<JamoPair, {jamo_count}> kJamoPairs = {{
"""

FOOTER = """}};

/// 惯用语替换：左 → 右（上游按顺序逐条做**正则**替换）。
struct Idiom {
    std::string_view from;
    std::string_view to;
};

/// {idiom_count} 条惯用语。
inline constexpr std::array<Idiom, {idiom_count}> kIdioms = {{
"""

MIDDLE = """}};

/// "받침 + 起首辅音" 的规则：coda + onset → replacement。
///
/// replacement 里的 "\\1" 指代**中间那个可选空格**（上游正则的分组 1）——
/// 也就是说这条规则把 받침 换掉、但保留中间的空格。
struct RegularRule {
    std::string_view coda;
    std::string_view onset;
    std::string_view replacement;
};

/// {regular_count} 条规则（顺序与上游 table.csv 一致）。
inline constexpr std::array<RegularRule, {regular_count}> kRegularRules = {{
"""

TAIL = """}};

}  // namespace genie::text::korean::tables

#endif  // GENIE_SRC_TEXT_KOREAN_TABLES_HPP
"""


def escape(text: str) -> str:
    """把字符串写成 C++ 字面量（转义反斜杠与双引号）。"""
    return text.replace("\\", "\\\\").replace('"', '\\"')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=pathlib.Path, help="输出头文件")
    parser.add_argument("--jamo-dir", type=pathlib.Path, help="jamo 包目录")
    parser.add_argument("--g2pk2-dir", type=pathlib.Path, help="g2pk2 包目录")
    args = parser.parse_args()

    jamo_dir = args.jamo_dir or find_package_dir("jamo")
    g2pk2_dir = args.g2pk2_dir or find_package_dir("g2pk2")

    pairs = jamo_pairs(jamo_dir)
    idioms = read_idioms(g2pk2_dir)
    rules = read_regular_table(g2pk2_dir)

    out = args.out.open("w", encoding="utf-8")
    out.write(HEADER.replace("{jamo_count}", str(len(pairs))))
    for compat, modern in pairs:
        out.write(f"    {{0x{ord(modern):04X},0x{ord(compat):04X}}},\n")
    out.write(FOOTER.replace("{idiom_count}", str(len(idioms))))
    for left, right in idioms:
        out.write(f'    {{"{escape(left)}","{escape(right)}"}},\n')
    out.write(MIDDLE.replace("{regular_count}", str(len(rules))))
    for coda, onset, replacement in rules:
        out.write(f'    {{"{escape(coda)}","{escape(onset)}","{escape(replacement)}"}},\n')
    out.write(TAIL)
    out.close()
    print(
        f"wrote {args.out}: {len(pairs)} jamo pairs, {len(idioms)} idioms, "
        f"{len(rules)} regular rules"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
