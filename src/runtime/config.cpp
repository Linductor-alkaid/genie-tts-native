// src/runtime/config.cpp

#include "genie/config.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <string>
#include <vector>

namespace genie {
namespace {

std::string to_lower(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    for (const char ch : text) {
        out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }
    return out;
}

// Unicode 码点区间判定（UTF-8 解码后按码点判断）。
//   假名：U+3040..U+30FF（平/片假名）
//   谚文：U+AC00..U+D7AF（谚文音节）以及 U+1100..U+11FF（谚文字母）
//   汉字：U+4E00..U+9FFF（基本区）、U+3400..U+4DBF（扩展 A）
bool is_japanese_codepoint(std::uint32_t cp) {
    return (cp >= 0x3040 && cp <= 0x30FF) || (cp >= 0x31F0 && cp <= 0x31FF);
}

bool is_korean_codepoint(std::uint32_t cp) {
    return (cp >= 0xAC00 && cp <= 0xD7AF) || (cp >= 0x1100 && cp <= 0x11FF) ||
           (cp >= 0x3130 && cp <= 0x318F);
}

bool is_han_codepoint(std::uint32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
           (cp >= 0xF900 && cp <= 0xFAFF);
}

/// 极简 UTF-8 迭代器：只关心码点，不做完整 Unicode 规范化。
template <typename Fn>
void for_each_codepoint(std::string_view text, Fn&& fn) {
    std::size_t i = 0;
    const std::size_t n = text.size();
    while (i < n) {
        const auto byte = static_cast<unsigned char>(text[i]);
        std::uint32_t cp = 0;
        std::size_t len = 1;
        if (byte < 0x80) {
            cp = byte;
            len = 1;
        } else if ((byte & 0xE0) == 0xC0) {
            cp = byte & 0x1FU;
            len = 2;
        } else if ((byte & 0xF0) == 0xE0) {
            cp = byte & 0x0FU;
            len = 3;
        } else if ((byte & 0xF8) == 0xF0) {
            cp = byte & 0x07U;
            len = 4;
        } else {
            // 非法起始字节，跳过。
            ++i;
            continue;
        }
        if (i + len > n) {
            break;
        }
        for (std::size_t k = 1; k < len; ++k) {
            const auto cont = static_cast<unsigned char>(text[i + k]);
            if ((cont & 0xC0) != 0x80) {
                cp = 0xFFFD;  // 截断序列
                break;
            }
            cp = (cp << 6) | (cont & 0x3FU);
        }
        fn(cp);
        i += len;
    }
}

bool contains_ascii_letter(std::string_view text) {
    return std::any_of(text.begin(), text.end(), [](const char ch) {
        return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z');
    });
}

}  // namespace

std::optional<Language> language_from_string(std::string_view text) noexcept {
    const std::string key = to_lower(text);
    if (key == "auto" || key == "zh" || key == "cn" || key == "chinese") {
        return key == "auto" ? Language::Auto : Language::Chinese;
    }
    if (key == "en" || key == "english") {
        return Language::English;
    }
    if (key == "ja" || key == "jp" || key == "japanese") {
        return Language::Japanese;
    }
    if (key == "ko" || key == "kr" || key == "korean") {
        return Language::Korean;
    }
    if (key == "hybrid-chinese-english" || key == "hybrid" || key == "zh-en") {
        return Language::HybridChineseEnglish;
    }
    return std::nullopt;
}

std::string language_to_string(Language language) {
    switch (language) {
        case Language::Auto:
            return "auto";
        case Language::Chinese:
            return "chinese";
        case Language::English:
            return "english";
        case Language::Japanese:
            return "japanese";
        case Language::Korean:
            return "korean";
        case Language::HybridChineseEnglish:
            return "hybrid-chinese-english";
    }
    return "unknown";
}

Language detect_language(std::string_view text, Language requested) {
    // 上游规则：显式 chinese 且含 ASCII 字母 → 混读模式。
    // 中文 G2P 的 pattern_filter 会删掉所有英文字母，混读文本若走纯中文路径
    // 会得到空音素序列并产生噪声。
    if (requested == Language::Chinese && contains_ascii_letter(text)) {
        return Language::HybridChineseEnglish;
    }
    if (requested != Language::Auto) {
        return requested;
    }

    bool has_kana = false;
    bool has_hangul = false;
    bool has_han = false;
    bool has_ascii = false;

    for_each_codepoint(text, [&](const std::uint32_t cp) {
        if (is_japanese_codepoint(cp)) {
            has_kana = true;
        } else if (is_korean_codepoint(cp)) {
            has_hangul = true;
        } else if (is_han_codepoint(cp)) {
            has_han = true;
        } else if (cp < 0x80) {
            if ((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z')) {
                has_ascii = true;
            }
        }
    });

    if (has_kana) {
        return Language::Japanese;
    }
    if (has_hangul) {
        return Language::Korean;
    }
    if (has_han) {
        // 纯中文 + 英文混读 → 混读模式。
        return has_ascii ? Language::HybridChineseEnglish : Language::Chinese;
    }
    return Language::English;
}

}  // namespace genie
