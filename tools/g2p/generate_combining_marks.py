#!/usr/bin/env python3
"""生成 src/text/english/combining_marks.hpp。

上游英文规范化第一步是 `unicodedata.normalize("NFD", text)` 再丢掉
所有 Mn（组合记号）类字符。C++ 侧没有 Unicode 类别表，这里用 Python
的 unicodedata 当基准，把 Mn 的码位区间导出来。

用法：
    python3 tools/g2p/generate_combining_marks.py > src/text/english/combining_marks.hpp
"""

import sys
import unicodedata

MAX_CODEPOINT = 0x110000

HEADER = """// src/text/english/combining_marks.hpp
//
// Unicode Mn（组合记号）类字符的码位区间表。
//
// 上游英文规范化第一步是 `unicodedata.normalize("NFD", text)`，然后丢掉
// 所有 Mn 类字符 —— 也就是把 "é" 拆成 "e" + 组合记号，再把组合记号丢掉。
// 等价于：预组合字符 → 基字符。
//
// 表由 tools/g2p/generate_combining_marks.py 生成（Python 的 unicodedata
// 是基准）。区间是闭区间 [first, last]。

#ifndef GENIE_SRC_TEXT_ENGLISH_COMBINING_MARKS_HPP
#define GENIE_SRC_TEXT_ENGLISH_COMBINING_MARKS_HPP

#include <array>
#include <cstdint>

namespace genie::text::english::marks {{

/// 一段连续的 Mn 码位（闭区间）。
struct Range {{
    std::uint32_t first;
    std::uint32_t last;
}};

/// {count} 段 Mn 区间。
inline constexpr std::array<Range, {count}> kCombiningMarkRanges = {{{{
"""

FOOTER = """}}};

/// 码点是不是 Mn（组合记号）。
[[nodiscard]] inline bool is_combining_mark(std::uint32_t codepoint) noexcept {{
    for (const auto& range : kCombiningMarkRanges) {{
        if (codepoint < range.first) {{
            return false;  // 区间有序 → 后面的都不可能
        }}
        if (codepoint <= range.last) {{
            return true;
        }}
    }}
    return false;
}}

}}  // namespace genie::text::english::marks

#endif  // GENIE_SRC_TEXT_ENGLISH_COMBINING_MARKS_HPP
"""


def ranges():
    """产出 Mn 类码位的闭区间。"""
    out = []
    start = None
    prev = None
    for cp in range(MAX_CODEPOINT):
        if unicodedata.category(chr(cp)) == "Mn":
            if start is None:
                start = cp
            prev = cp
        elif start is not None:
            out.append((start, prev))
            start = None
    if start is not None:
        out.append((start, prev))
    return out


def main() -> int:
    entries = ranges()
    out = sys.stdout
    out.write(HEADER.format(count=len(entries)))
    for i in range(0, len(entries), 4):
        chunk = entries[i : i + 4]
        out.write(
            "    "
            + ", ".join(f"{{0x{first:04X},0x{last:04X}}}" for first, last in chunk)
            + ",\n"
        )
    # FOOTER 里是 C++ 代码（大括号是成对的），直接把占位用的双大括号还原。
    out.write(FOOTER.replace("{{", "{").replace("}}", "}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
