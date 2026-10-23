// src/runtime/text.cpp
//
// 文本规范化。
//
// 规则来源是上游的中文 G2P 栈：
//   - `G2P/Chinese/ChineseG2P.py`  的 PUNCTUATION_REPLACEMENTS / pattern_filter
//   - `G2P/Chinese/Normalization/`  的 F2H_* 与 num.py
//
// 这里实现的是其中**可独立复现**的部分。繁简转换、日期/时间/电话/量词
// 这些依赖上游 65KB 词表的规则没有实现，差异记录在 docs/upstream-audit.md。

#include "genie/text.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include "genie/utf8.hpp"

namespace genie::text {
namespace {

// "…" 的码位。不能用字符字面量 —— 它在 UTF-8 里是 3 字节。
constexpr std::uint32_t kEllipsis = 0x2026;

// 上游 ChineseG2P.PUNCTUATION_REPLACEMENTS
struct PunctuationReplacement {
    std::string_view from;
    std::string_view to;
};

constexpr std::array kPunctuationReplacements = {
    PunctuationReplacement{"\xEF\xBC\x9A", ","},  // ：
    PunctuationReplacement{"\xEF\xBC\x9B", ","},  // ；
    PunctuationReplacement{"\xEF\xBC\x8C", ","},  // ，
    PunctuationReplacement{"\xE3\x80\x82", "."},  // 。
    PunctuationReplacement{"\xEF\xBC\x81", "!"},  // ！
    PunctuationReplacement{"\xEF\xBC\x9F", "?"},  // ？
    PunctuationReplacement{"\n", "."},
    PunctuationReplacement{"\xC2\xB7", ","},      // ·
    PunctuationReplacement{"\xE3\x80\x81", ","},  // 、
    PunctuationReplacement{"$", "."},
    PunctuationReplacement{"/", ","},
    PunctuationReplacement{"\xE2\x80\x94", "-"},             // —
    PunctuationReplacement{"~", "\xE2\x80\xA6"},             // ~ → …
    PunctuationReplacement{"\xEF\xBD\x9E", "\xE2\x80\xA6"},  // ～ → …
};

// 上游 SPECIAL_REPLACEMENTS
constexpr PunctuationReplacement kSpecialReplacement{"...", "\xE2\x80\xA6"};

/// 数字读法用到的字。
constexpr std::array<std::string_view, 10> kDigits = {
    "\xE9\x9B\xB6",  // 零
    "\xE4\xB8\x80",  // 一
    "\xE4\xBA\x8C",  // 二
    "\xE4\xB8\x89",  // 三
    "\xE5\x9B\x9B",  // 四
    "\xE4\xBA\x94",  // 五
    "\xE5\x85\xAD",  // 六
    "\xE4\xB8\x83",  // 七
    "\xE5\x85\xAB",  // 八
    "\xE4\xB9\x9D",  // 九
};

/// 上游 num.py 的 UNITS：位 → 量词。索引 0 不用。
constexpr std::array<std::string_view, 9> kUnits = {
    "", "\xE5\x8D\x81", "\xE7\x99\xBE", "\xE5\x8D\x83", "\xE4\xB8\x87", "", "", "", "\xE4\xBA\xBF",
};

/// 数字 → 中文（上游 num.py 的 _get_value）。
///
/// @param value 纯数字串，可以带前导零。
/// @param use_zero 是否为被去掉的零补一个"零"字。
[[nodiscard]] std::vector<std::string> get_value(const std::string& value, bool use_zero) {
    const std::size_t begin = value.find_first_not_of('0');
    const std::string stripped = begin == std::string::npos ? std::string() : value.substr(begin);
    if (stripped.empty()) {
        return {};
    }
    if (stripped.size() == 1) {
        const auto digit = static_cast<std::size_t>(stripped[0] - '0');
        if (use_zero && stripped.size() < value.size()) {
            return {std::string(kDigits[0]), std::string(kDigits[digit])};
        }
        return {std::string(kDigits[digit])};
    }
    // 找最大的、严格小于当前位数的单位。
    std::size_t largest_unit = 0;
    for (const std::size_t power : {8U, 4U, 3U, 2U, 1U}) {
        if (power < stripped.size()) {
            largest_unit = power;
            break;
        }
    }
    const std::string first = value.substr(0, value.size() - largest_unit);
    const std::string second = value.substr(value.size() - largest_unit);
    auto result = get_value(first, use_zero);
    result.emplace_back(kUnits[largest_unit]);
    auto tail = get_value(second, use_zero);
    result.insert(result.end(), tail.begin(), tail.end());
    return result;
}

/// 整数读法（上游 num.py 的 verbalize_cardinal）。
[[nodiscard]] std::string verbalize_cardinal(const std::string& value) {
    if (value.empty()) {
        return {};
    }
    const std::size_t begin = value.find_first_not_of('0');
    const std::string stripped = begin == std::string::npos ? std::string() : value.substr(begin);
    if (stripped.empty()) {
        return std::string(kDigits[0]);
    }
    auto symbols = get_value(value, /*use_zero=*/true);
    // "一十X" 缩写成 "十X"。
    if (symbols.size() >= 2 && symbols[0] == kDigits[1] && symbols[1] == kUnits[1]) {
        symbols.erase(symbols.begin());
    }
    std::string result;
    for (const auto& symbol : symbols) {
        result += symbol;
    }
    return result;
}

/// 逐位读法（上游 num.py 的 verbalize_digit）。
///
/// alt_one 把"一"读成"幺"（用于报号码），上游对应 verbalize_digit 的 alt_one。
[[nodiscard]] std::string verbalize_digit(const std::string& value, bool alt_one) {
    static constexpr std::string_view kYao = "\xE5\xB9\xB2";  // 幺
    std::string result;
    for (const char ch : value) {
        const auto digit = kDigits[static_cast<std::size_t>(ch - '0')];
        if (alt_one && digit == kDigits[1]) {
            result += kYao;
        } else {
            result += digit;
        }
    }
    return result;
}

/// 数字 → 中文（上游 num.py 的 num2str）。
[[nodiscard]] std::string num_to_chinese(const std::string& value) {
    const auto dot = value.find('.');
    const std::string integer = dot == std::string::npos ? value : value.substr(0, dot);
    std::string decimal = dot == std::string::npos ? std::string() : value.substr(dot + 1);

    std::string result = verbalize_cardinal(integer);

    // 上游的规则：小数末尾的 0 保留一位。
    if (!decimal.empty() && decimal.back() == '0') {
        decimal = decimal.substr(0, decimal.find_last_not_of('0') + 1);
        decimal += "0";
    } else {
        decimal = decimal.substr(0, decimal.find_last_not_of('0') + 1);
    }

    if (!decimal.empty()) {
        if (result.empty()) {
            result = kDigits[0];
        }
        result += "\xE7\x82\xB9";  // 点
        result += verbalize_digit(decimal, /*alt_one=*/false);
    }
    return result;
}

/// 把文本里的阿拉伯数字段替换成中文读法。
///
/// 只处理"整段连续数字"（可带小数点），不处理日期、单位等 —— 那些上游
/// 有专门的规则，这里没有实现。
void replace_numbers(std::string& text) {
    std::string result;
    result.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        if (!utf8::is_ascii_digit(static_cast<unsigned char>(text[i]))) {
            result.push_back(text[i]);
            ++i;
            continue;
        }
        // 收集一段数字（可含一个小数点）。
        const std::size_t start = i;
        while (i < text.size() && utf8::is_ascii_digit(static_cast<unsigned char>(text[i]))) {
            ++i;
        }
        std::string number = text.substr(start, i - start);
        if (i < text.size() && text[i] == '.' && i + 1 < text.size() &&
            utf8::is_ascii_digit(static_cast<unsigned char>(text[i + 1]))) {
            ++i;
            const std::size_t frac_start = i;
            while (i < text.size() && utf8::is_ascii_digit(static_cast<unsigned char>(text[i]))) {
                ++i;
            }
            number += "." + text.substr(frac_start, i - frac_start);
        }
        result += num_to_chinese(number);
    }
    text = std::move(result);
}

/// 全角 ASCII → 半角（上游 constants.py 的 F2H_* 规则）。
[[nodiscard]] std::string to_halfwidth(std::string_view text) {
    std::string result;
    result.reserve(text.size());
    std::size_t offset = 0;
    utf8::Char ch;
    while (utf8::decode_next(text, offset, ch)) {
        const auto converted =
            utf8::ideographic_space_to_halfwidth(utf8::to_halfwidth(ch.codepoint));
        result += utf8::encode(converted);
    }
    return result;
}

/// 标点映射（上游 ChineseG2P 的两轮替换）。
[[nodiscard]] std::string map_punctuation(std::string_view text) {
    std::string result;
    result.reserve(text.size());
    std::size_t pos = 0;
    while (pos < text.size()) {
        bool replaced = false;
        for (const auto& rule : kPunctuationReplacements) {
            if (text.compare(pos, rule.from.size(), rule.from) == 0) {
                result += rule.to;
                pos += rule.from.size();
                replaced = true;
                break;
            }
        }
        if (!replaced) {
            result.push_back(text[pos]);
            ++pos;
        }
    }
    // 多字符替换（"..." → "…"）放在最后。
    {
        std::string::size_type found = 0;
        while ((found = result.find(kSpecialReplacement.from, found)) != std::string::npos) {
            result.replace(found, kSpecialReplacement.from.size(), kSpecialReplacement.to);
            found += kSpecialReplacement.to.size();
        }
    }
    return result;
}

/// 连续标点折叠成一个（上游 pattern_consecutive）。
[[nodiscard]] std::string collapse_punctuation(std::string_view text) {
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
        result.append(text.substr(ch.offset, ch.length));
        previous = ch.codepoint;
        has_previous = true;
    }
    return result;
}

/// 只保留目标语言需要的字符（上游 pattern_filter）。
[[nodiscard]] std::string filter_characters(std::string_view text, Language language) {
    const auto is_common_punctuation = [](std::uint32_t codepoint) {
        return codepoint == '!' || codepoint == '?' || codepoint == kEllipsis || codepoint == ',' ||
               codepoint == '.' || codepoint == '-';
    };

    std::string result;
    result.reserve(text.size());
    std::size_t offset = 0;
    utf8::Char ch;
    while (utf8::decode_next(text, offset, ch)) {
        bool keep = false;
        switch (language) {
            case Language::Chinese:
            case Language::HybridChineseEnglish:
                keep = utf8::is_han(ch.codepoint) || is_common_punctuation(ch.codepoint);
                break;
            case Language::English:
                keep = utf8::is_ascii_letter(ch.codepoint) || utf8::is_ascii_digit(ch.codepoint) ||
                       ch.codepoint == ' ' || ch.codepoint == '\'' ||
                       is_common_punctuation(ch.codepoint);
                break;
            case Language::Japanese:
                keep = utf8::is_kana(ch.codepoint) || utf8::is_han(ch.codepoint) ||
                       is_common_punctuation(ch.codepoint);
                break;
            case Language::Korean:
                keep = utf8::is_hangul(ch.codepoint) || ch.codepoint == ' ' ||
                       is_common_punctuation(ch.codepoint);
                break;
            case Language::Auto:
                keep = true;
                break;
        }
        if (keep) {
            result.append(text.substr(ch.offset, ch.length));
        }
    }
    return result;
}

/// 折叠连续空白并去掉首尾空白。
[[nodiscard]] std::string collapse_whitespace(std::string_view text) {
    std::string result;
    result.reserve(text.size());
    bool previous_space = false;
    for (const char ch : text) {
        const bool is_space = ch == ' ' || ch == '\t';
        if (is_space) {
            if (previous_space) {
                continue;
            }
            previous_space = true;
        } else {
            previous_space = false;
        }
        result.push_back(ch);
    }
    const auto first = result.find_first_not_of(' ');
    if (first == std::string::npos) {
        return {};
    }
    const auto last = result.find_last_not_of(' ');
    return result.substr(first, last - first + 1);
}

}  // namespace

std::string normalize(std::string_view text, Language language) {
    std::string result = to_halfwidth(text);
    result = map_punctuation(result);

    switch (language) {
        case Language::Chinese:
        case Language::HybridChineseEnglish:
            replace_numbers(result);
            result = filter_characters(result, language);
            result = collapse_punctuation(result);
            break;
        case Language::English:
            result = filter_characters(result, language);
            result = collapse_whitespace(result);
            break;
        case Language::Japanese:
        case Language::Korean:
            result = filter_characters(result, language);
            result = collapse_whitespace(result);
            break;
        case Language::Auto:
            result = collapse_whitespace(result);
            break;
    }
    return result;
}

}  // namespace genie::text
