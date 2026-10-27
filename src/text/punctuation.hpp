// src/text/punctuation.hpp
//
// 标点相关的共用小工具（内部头，不对外暴露）。
//
// 为什么单独放一个头：折叠连续标点这件事**必须按码点做**。
// 按字节做会把多字节汉字吃掉 —— "四"（E5 9B 9B）后两个字节相同，
// 按字节折叠就变成 "E5 9B"，输出直接是乱码。
//
// 这个坑很隐蔽：Linux 上不一定复现，MSVC 上可能表现不同，
// 而症状是"某个字莫名其妙少了一半"。

#ifndef GENIE_TEXT_PUNCTUATION_HPP
#define GENIE_TEXT_PUNCTUATION_HPP

#include <cstdint>
#include <string>
#include <string_view>

#include "genie/utf8.hpp"

namespace genie::text::detail {

/// 上游的六个标点符号。
inline constexpr std::uint32_t kEllipsis = 0x2026;  // …

/// 折叠**连续相同**的标点（上游 `pattern_consecutive`）。
///
/// 上游的正则是 `([!?…,.-])\1+` —— 只折叠这六种标点，
/// 而且按**字符**匹配。
[[nodiscard]] inline std::string collapse_punctuation(std::string_view text) {
    const auto is_punctuation = [](std::uint32_t codepoint) {
        return codepoint == '!' || codepoint == '?' || codepoint == ',' || codepoint == '.' ||
               codepoint == '-' || codepoint == kEllipsis;
    };

    std::string result;
    result.reserve(text.size());
    std::size_t offset = 0;
    utf8::Char ch;
    std::uint32_t previous = 0;
    bool has_previous = false;
    while (utf8::decode_next(text, offset, ch)) {
        if (has_previous && ch.codepoint == previous && is_punctuation(ch.codepoint)) {
            continue;
        }
        result += utf8::encode(ch.codepoint);
        previous = ch.codepoint;
        has_previous = true;
    }
    return result;
}

}  // namespace genie::text::detail

#endif  // GENIE_TEXT_PUNCTUATION_HPP
