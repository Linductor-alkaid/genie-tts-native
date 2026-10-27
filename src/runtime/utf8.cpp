// src/runtime/utf8.cpp

#include "genie/utf8.hpp"

namespace genie::utf8 {

bool decode_next(std::string_view text, std::size_t& offset, Char& out) noexcept {
    if (offset >= text.size()) {
        return false;
    }
    const auto lead = static_cast<unsigned char>(text[offset]);
    out.offset = offset;

    std::size_t width = 0;
    std::uint32_t codepoint = 0;
    if (lead < 0x80U) {
        width = 1;
        codepoint = lead;
    } else if ((lead & 0xE0U) == 0xC0U) {
        width = 2;
        codepoint = lead & 0x1FU;
    } else if ((lead & 0xF0U) == 0xE0U) {
        width = 3;
        codepoint = lead & 0x0FU;
    } else if ((lead & 0xF8U) == 0xF0U) {
        width = 4;
        codepoint = lead & 0x07U;
    } else {
        // 续字节或非法首字节。
        out.codepoint = kReplacementCodepoint;
        out.length = 1;
        ++offset;
        return true;
    }

    if (offset + width > text.size()) {
        out.codepoint = kReplacementCodepoint;
        out.length = 1;
        ++offset;
        return true;
    }
    for (std::size_t i = 1; i < width; ++i) {
        const auto continuation = static_cast<unsigned char>(text[offset + i]);
        if ((continuation & 0xC0U) != 0x80U) {
            out.codepoint = kReplacementCodepoint;
            out.length = 1;
            ++offset;
            return true;
        }
        codepoint = (codepoint << 6U) | (continuation & 0x3FU);
    }

    // 过长编码、代理区、越界码位一律当作替换字符。
    const bool overlong = (width == 2 && codepoint < 0x80U) || (width == 3 && codepoint < 0x800U) ||
                          (width == 4 && codepoint < 0x10000U);
    const bool surrogate = codepoint >= 0xD800U && codepoint <= 0xDFFFU;
    if (overlong || surrogate || codepoint > 0x10FFFFU) {
        out.codepoint = kReplacementCodepoint;
        out.length = 1;
        ++offset;
        return true;
    }

    out.codepoint = codepoint;
    out.length = width;
    offset += width;
    return true;
}

std::size_t encoded_length(std::uint32_t codepoint) noexcept {
    // 与 encode() 保持一致：非法码位按替换字符（U+FFFD，3 字节）算。
    if (codepoint > 0x10FFFFU || (codepoint >= 0xD800U && codepoint <= 0xDFFFU)) {
        return 3;
    }
    if (codepoint <= 0x7F) {
        return 1;
    }
    if (codepoint <= 0x7FF) {
        return 2;
    }
    if (codepoint <= 0xFFFF) {
        return 3;
    }
    return 4;
}

std::string encode(std::uint32_t codepoint) {
    std::string out;
    if (codepoint > 0x10FFFFU || (codepoint >= 0xD800U && codepoint <= 0xDFFFU)) {
        codepoint = kReplacementCodepoint;
    }
    if (codepoint < 0x80U) {
        out.push_back(static_cast<char>(codepoint));
    } else if (codepoint < 0x800U) {
        out.push_back(static_cast<char>(0xC0U | (codepoint >> 6U)));
        out.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else if (codepoint < 0x10000U) {
        out.push_back(static_cast<char>(0xE0U | (codepoint >> 12U)));
        out.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else {
        out.push_back(static_cast<char>(0xF0U | (codepoint >> 18U)));
        out.push_back(static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    }
    return out;
}

bool is_han(std::uint32_t codepoint) noexcept {
    return (codepoint >= 0x4E00U && codepoint <= 0x9FA5U) ||
           (codepoint >= 0x3400U && codepoint <= 0x4DBFU) ||
           (codepoint >= 0xF900U && codepoint <= 0xFAFFU);
}

bool is_kana(std::uint32_t codepoint) noexcept {
    return (codepoint >= 0x3040U && codepoint <= 0x309FU) ||  // 平假名
           (codepoint >= 0x30A0U && codepoint <= 0x30FFU) ||  // 片假名
           (codepoint >= 0x31F0U && codepoint <= 0x31FFU);    // 扩展
}

bool is_hangul(std::uint32_t codepoint) noexcept {
    return (codepoint >= 0x1100U && codepoint <= 0x11FFU) ||  // 字母
           (codepoint >= 0x3130U && codepoint <= 0x318FU) ||  // 兼容字母
           (codepoint >= 0xAC00U && codepoint <= 0xD7AFU);    // 音节
}

bool is_ascii_letter(std::uint32_t codepoint) noexcept {
    return (codepoint >= 'a' && codepoint <= 'z') || (codepoint >= 'A' && codepoint <= 'Z');
}

bool is_ascii_digit(std::uint32_t codepoint) noexcept {
    return codepoint >= '0' && codepoint <= '9';
}

bool is_ascii_space(std::uint32_t codepoint) noexcept {
    return codepoint == ' ' || codepoint == '\t' || codepoint == '\n' || codepoint == '\r' ||
           codepoint == '\f' || codepoint == '\v';
}

std::uint32_t to_halfwidth(std::uint32_t codepoint) noexcept {
    if (codepoint >= 0xFF01U && codepoint <= 0xFF5EU) {
        return codepoint - 0xFEE0U;
    }
    return codepoint;
}

std::uint32_t ideographic_space_to_halfwidth(std::uint32_t codepoint) noexcept {
    return codepoint == 0x3000U ? ' ' : codepoint;
}

}  // namespace genie::utf8
