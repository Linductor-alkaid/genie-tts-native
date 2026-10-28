#!/usr/bin/env python3
"""生成 src/text/english/latin_decomposition.hpp。

上游英文规范化第一步是 `unicodedata.normalize("NFD", text)` 再丢掉
所有 Mn（组合记号）类字符。C++ 侧没有内置的 Unicode 分解表，这里用
Python 的 unicodedata 当基准，生成「预组合拉丁字母 → 基字符」的表。

用法：
    python3 tools/g2p/generate_latin_decomposition.py > src/text/english/latin_decomposition.hpp
"""

import sys
import unicodedata

RANGES = list(range(0xC0, 0x180)) + list(range(0x1E00, 0x1F00)) + list(range(0x2000, 0x2100))

HEADER = """// src/text/english/latin_decomposition.hpp
//
// 预组合拉丁字母的分解表（上游 Normalization.normalize 的第一步：
// `unicodedata.normalize("NFD", text)` 后丢掉所有 Mn 类字符）。
//
// 上游用的是 Python 的完整 Unicode NFD —— 那要带一整张分解表。
// 英文 TTS 里实际会碰到的是带重音的拉丁字母，所以这里只收
// Latin-1 Supplement / Latin Extended-A / Latin Extended Additional
// 里「基字符 + 组合记号」形态的 {count} 个预组合字符：
// 遇到就换成基字符，等价于 NFD 之后再丢组合记号。
//
// 表由 tools/g2p/generate_latin_decomposition.py 生成（用 Python 的
// unicodedata 当基准）。不在表里的字符保持原样 —— 差异见
// docs/upstream-audit.md §17。

#ifndef GENIE_SRC_TEXT_ENGLISH_LATIN_DECOMPOSITION_HPP
#define GENIE_SRC_TEXT_ENGLISH_LATIN_DECOMPOSITION_HPP

#include <array>
#include <cstdint>
#include <string_view>

namespace genie::text::english::latin {{

/// 一条分解规则：预组合码位 → 分解后的基字符（UTF-8 字节）。
struct Rule {{
    std::uint32_t composed;
    std::string_view base;
}};

/// {count} 条规则，按码位升序。
inline constexpr std::array<Rule, {count}> kRules = {{{{
"""

FOOTER = """}};

}  // namespace genie::text::english::latin

#endif  // GENIE_SRC_TEXT_ENGLISH_LATIN_DECOMPOSITION_HPP
"""


def rules():
    """产出 (预组合码位, NFD 去掉组合记号之后的基字符)。

    上游是 `unicodedata.normalize("NFD", text)` 之后丢掉所有 Mn 类字符 ——
    所以 "é" → "e"，而 "Æ" → "AE"（NFD 本身就是两个字母，没有组合记号）。
    """
    for cp in RANGES:
        ch = chr(cp)
        decomposed = unicodedata.normalize("NFD", ch)
        if len(decomposed) <= 1:
            continue
        base = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
        if base and base != ch:
            yield cp, base


def main() -> int:
    entries = list(rules())
    out = sys.stdout
    out.write(HEADER.format(count=len(entries)))
    for i in range(0, len(entries), 5):
        chunk = entries[i : i + 5]
        out.write(
            "    "
            + ", ".join(f'{{0x{cp:04X},"{base}"}}' for cp, base in chunk)
            + ",\n"
        )
    out.write(FOOTER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
