// include/genie/utf8.hpp
//
// 最小 UTF-8 工具。文本前端需要按"字符"而不是按"字节"处理文本，
// 而把一个完整的 UTF-8 库引进来对端侧部署来说太重 —— 这里只实现
// 本项目真正用到的部分：
//
//   - 逐字符解码（含非法字节的容错）
//   - 码点 → UTF-8 编码
//   - 全角 ASCII → 半角
//   - 常用文字区段判定
//
// 解码策略：遇到非法序列时**不抛异常**，而是把该字节当作一个
// U+FFFD 之外的"未知字符"跳过。文本前端要处理的是用户输入，
// 一个坏字节不该让整次合成失败。

#ifndef GENIE_UTF8_HPP
#define GENIE_UTF8_HPP

#include <cstdint>
#include <string>
#include <string_view>

namespace genie::utf8 {

/// 一个解码出来的字符。
struct Char {
    /// Unicode 码位。非法字节用 kReplacementCodepoint。
    std::uint32_t codepoint = 0;
    /// 在原串中的字节偏移。
    std::size_t offset = 0;
    /// 字节长度。
    std::size_t length = 0;
};

/// 非法序列的替换码位（U+FFFD REPLACEMENT CHARACTER）。
inline constexpr std::uint32_t kReplacementCodepoint = 0xFFFD;

/// 解码下一个字符。到达末尾返回 false。
[[nodiscard]] bool decode_next(std::string_view text, std::size_t& offset, Char& out) noexcept;

/// 码点 → UTF-8。非法码位返回替换字符的编码。
[[nodiscard]] std::string encode(std::uint32_t codepoint);

/// 码点 → UTF-8 的**字节长度**（不解码、不分配）。
[[nodiscard]] std::size_t encoded_length(std::uint32_t codepoint) noexcept;

/// 是否是汉字（CJK 统一表意文字基本区 + 扩展 A + 兼容表意文字）。
[[nodiscard]] bool is_han(std::uint32_t codepoint) noexcept;

/// 是否是日语假名（平假名 + 片假名）。
[[nodiscard]] bool is_kana(std::uint32_t codepoint) noexcept;

/// 是否是韩文谚文。
[[nodiscard]] bool is_hangul(std::uint32_t codepoint) noexcept;

/// 是否是 ASCII 字母。
[[nodiscard]] bool is_ascii_letter(std::uint32_t codepoint) noexcept;

/// 是否是 ASCII 数字。
[[nodiscard]] bool is_ascii_digit(std::uint32_t codepoint) noexcept;

/// 是否是 ASCII 空白。
[[nodiscard]] bool is_ascii_space(std::uint32_t codepoint) noexcept;

/// 全角 ASCII（U+FF01..U+FF5E）→ 半角。其他码位原样返回。
[[nodiscard]] std::uint32_t to_halfwidth(std::uint32_t codepoint) noexcept;

/// 全角空格（U+3000）→ 半角空格。
[[nodiscard]] std::uint32_t ideographic_space_to_halfwidth(std::uint32_t codepoint) noexcept;

}  // namespace genie::utf8

#endif  // GENIE_UTF8_HPP
