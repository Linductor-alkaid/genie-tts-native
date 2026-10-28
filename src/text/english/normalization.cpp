// src/text/english/normalization.cpp —— 英文文本规范化的 C++ 移植。
//
// 移植自上游 `G2P/English/Normalization.py` 的 `normalize()`，
// 外加 `EnglishG2P.py` 里包在它外面的两步：
//
//   1. `REP_MAP`（中文标点 → 英文标点）
//   2. `replace_consecutive_punctuation`（折叠连续标点 —— 注意它**也**
//      折叠连续空白，因为正则的第一个分组里含 `\s`）
//
// 上游是 20 多条正则依次替换，每条都**全文扫描**。本实现不用
// `std::regex`（ECMAScript 没有反向先行断言，且 MSVC 实现差异多），
// 改成手写扫描器，**顺序与替换语义逐条对齐** —— 顺序会互相影响：
// 比如 `_currency_suffix_re` 先把 "$3 million" 变成
// "three million dollars"，后面的 `_currency_re` 就再也匹配不到它了。
//
// 每条规则对应的上游正则写在函数注释里。

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include "genie/utf8.hpp"
#include "text/english/combining_marks.hpp"
#include "text/english/english.hpp"
#include "text/english/latin_decomposition.hpp"

namespace genie::text::english {
namespace {

// 货币符号是**多字节** UTF-8 —— 不能写成字符字面量（那是 multi-char
// constant，编译器只警告不报错，运行期值也是错的）。
constexpr std::string_view kPound = "\xC2\xA3";     // £
constexpr std::string_view kEuro = "\xE2\x82\xAC";  // €
constexpr std::string_view kDollar = "$";

/// 货币符号的字节长度（0 = 不是货币符号）。
[[nodiscard]] std::size_t currency_symbol_length(std::string_view text,
                                                 std::size_t index) noexcept {
    if (text.compare(index, kDollar.size(), kDollar) == 0) {
        return kDollar.size();
    }
    if (text.compare(index, kPound.size(), kPound) == 0) {
        return kPound.size();
    }
    if (text.compare(index, kEuro.size(), kEuro) == 0) {
        return kEuro.size();
    }
    return 0;
}

// ---------------------------------------------------------------------------
// 字符分类
// ---------------------------------------------------------------------------

/// Python 正则的 `\w`：字母 / 数字 / 下划线（Unicode 感知）。
///
/// 非 ASCII 部分按区间近似：拉丁字母（含变音）、CJK、假名、谚文算词字符；
/// 标点、符号、emoji 不算。英文文本里这块几乎只影响 `\b` 的边界判定。
[[nodiscard]] bool is_word_codepoint(std::uint32_t cp) noexcept {
    if (cp < 0x80) {
        return (cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z') || (cp >= '0' && cp <= '9') ||
               cp == '_';
    }
    if (cp >= 0xC0 && cp <= 0xFF) {
        return cp != 0xD7 && cp != 0xF7;  // × ÷ 不是字母
    }
    if (cp >= 0x100 && cp <= 0x2FF) {
        return true;  // 拉丁扩展 A/B、修饰字母、IPA 扩展
    }
    if (cp >= 0x3040 && cp <= 0xD7FF) {
        return true;  // 平假名、片假名、谚文、CJK 部首等
    }
    if (cp >= 0xF900 && cp <= 0xFDCF) {
        return true;  // 兼容表意文字
    }
    return cp >= 0x20000;  // CJK 扩展 B 起
}

/// Python 正则的 `\s`：ASCII 空白 + 几个 Unicode 空白。
[[nodiscard]] bool is_space_codepoint(std::uint32_t cp) noexcept {
    switch (cp) {
        case ' ':
        case '\t':
        case '\n':
        case '\r':
        case '\f':
        case '\v':
        case 0x00A0:
        case 0x1680:
        case 0x2028:
        case 0x2029:
        case 0x202F:
        case 0x205F:
        case 0x3000:
            return true;
        default:
            return false;
    }
}

// 金额与分数的展开（定义在各自的规则附近）。
[[nodiscard]] std::string currency_phrase(std::string_view symbol, std::string_view amount);
[[nodiscard]] std::string fraction_phrase(std::string_view numerator, std::string_view denominator);

// ---------------------------------------------------------------------------
// 通用小工具
// ---------------------------------------------------------------------------

/// 码点是不是 Mn（组合记号）—— 上游 NFD 之后要把它们全丢掉。
[[nodiscard]] bool is_combining_mark(std::uint32_t codepoint) noexcept {
    return marks::is_combining_mark(codepoint);
}

/// text[index] 处是不是 pattern（**大小写敏感**）。
[[nodiscard]] bool match_at(std::string_view text, std::size_t index,
                            std::string_view pattern) noexcept {
    if (index + pattern.size() > text.size()) {
        return false;
    }
    return text.compare(index, pattern.size(), pattern) == 0;
}

/// 前一个字符（index - 1 处）是不是词字符。串首返回 false。
[[nodiscard]] bool prev_is_word(std::string_view text, std::size_t index) noexcept {
    if (index == 0) {
        return false;
    }
    // 往前退一个码点
    std::size_t offset = index;
    while (offset > 0) {
        --offset;
        const unsigned char c = static_cast<unsigned char>(text[offset]);
        if ((c & 0xC0) != 0x80) {  // 找到一个起始字节
            std::uint32_t cp = c;
            if (c >= 0xF0) {
                cp = c & 0x07u;
            } else if (c >= 0xE0) {
                cp = c & 0x0Fu;
            } else if (c >= 0xC0) {
                cp = c & 0x1Fu;
            }
            // 补上后续字节
            std::size_t extra = 0;
            if (c >= 0xF0) {
                extra = 3;
            } else if (c >= 0xE0) {
                extra = 2;
            } else if (c >= 0xC0) {
                extra = 1;
            }
            for (std::size_t k = 1; k <= extra && offset + k < text.size(); ++k) {
                cp = (cp << 6) | (static_cast<unsigned char>(text[offset + k]) & 0x3Fu);
            }
            return is_word_codepoint(cp);
        }
    }
    return false;
}

/// 后一个字符（index 处）是不是词字符。串尾返回 false。
[[nodiscard]] bool next_is_word(std::string_view text, std::size_t index) noexcept {
    if (index >= text.size()) {
        return false;
    }
    const unsigned char c = static_cast<unsigned char>(text[index]);
    if (c < 0x80) {
        return is_word_codepoint(c);
    }
    std::uint32_t cp = c;
    std::size_t extra = 0;
    if (c >= 0xF0) {
        cp = c & 0x07u;
        extra = 3;
    } else if (c >= 0xE0) {
        cp = c & 0x0Fu;
        extra = 2;
    } else if (c >= 0xC0) {
        cp = c & 0x1Fu;
        extra = 1;
    }
    for (std::size_t k = 1; k <= extra && index + k < text.size(); ++k) {
        cp = (cp << 6) | (static_cast<unsigned char>(text[index + k]) & 0x3Fu);
    }
    return is_word_codepoint(cp);
}

/// Python `\b` 的语义：**词字符与非词字符的交界处**才算边界。
///
/// 注意不是"前后只要有一个不是词字符就算" —— Python 的 `\b` 是
/// "两侧恰好一侧是词字符"。串尾同理：串尾没有字符，非词字符，
/// 所以 "abc" 末尾是边界、"abc." 末尾**不是**边界。
///
/// 上游的正则大量依赖这个语义：`\b([01]?\d|2[0-3]):([0-5]\d)...`
/// 在 "3:30 pm." 上会回溯 —— 先试着吃掉 "pm."，发现末尾没有边界，
/// 于是退回去只吃 "pm"，句号因此被保留。
[[nodiscard]] bool word_boundary_at(std::string_view text, std::size_t index) noexcept {
    return prev_is_word(text, index) != next_is_word(text, index);
}

/// `\b` 的左边：位置 index 之前是不是边界。
[[nodiscard]] bool word_boundary_left(std::string_view text, std::size_t index) noexcept {
    return word_boundary_at(text, index);
}

/// `\b` 的右边：位置 index 之后是不是边界。
[[nodiscard]] bool word_boundary_right(std::string_view text, std::size_t index) noexcept {
    return word_boundary_at(text, index);
}

/// Python 的否定先行断言 `(?<!\w)`：前一个字符**不是**词字符。
///
/// 与 `\b` 不是一回事 —— `\b` 看的是"两侧是否恰好一侧是词字符"，
/// 而 `(?<!\w)` 只看左边。例："A-12" 里 "-" 前面是 "A"（词字符），
/// 所以 `(?<!\w)` 不成立，负号不被算进数字里。
[[nodiscard]] bool prev_not_word(std::string_view text, std::size_t index) noexcept {
    return !prev_is_word(text, index);
}

/// Python 的否定后顾断言 `(?!\w)`：后一个字符**不是**词字符。
[[nodiscard]] bool next_not_word(std::string_view text, std::size_t index) noexcept {
    return !next_is_word(text, index);
}

/// 读一个十进制数字（至少一位）。返回长度。
[[nodiscard]] std::size_t scan_digits(std::string_view text, std::size_t index) noexcept {
    std::size_t i = index;
    while (i < text.size() && text[i] >= '0' && text[i] <= '9') {
        ++i;
    }
    return i - index;
}

/// 把 text[index] 起的 n 个数字读成整数（溢出时返回 0）。
[[nodiscard]] long long parse_digits(std::string_view text, std::size_t index,
                                     std::size_t n) noexcept {
    long long value = 0;
    for (std::size_t i = 0; i < n && index + i < text.size(); ++i) {
        const char c = text[index + i];
        if (c < '0' || c > '9') {
            return 0;
        }
        value = value * 10 + (c - '0');
    }
    return value;
}

/// 全是十进制数字吗（空串算 false）。
[[nodiscard]] bool is_digit_string(std::string_view text) noexcept {
    if (text.empty()) {
        return false;
    }
    for (const char c : text) {
        if (c < '0' || c > '9') {
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// 数字 → 单词（对齐上游 `_number_to_words_custom`）
// ---------------------------------------------------------------------------

/// 小于 1000 的数转单词。
[[nodiscard]] std::string convert_less_than_thousand(long long n) {
    static constexpr std::array kUnits = std::to_array<std::string_view>(
        {"", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"});
    static constexpr std::array kTeens = std::to_array<std::string_view>(
        {"ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
         "eighteen", "nineteen"});
    static constexpr std::array kTens = std::to_array<std::string_view>(
        {"", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"});
    if (n <= 0) {
        return {};
    }
    if (n < 10) {
        return std::string(kUnits[static_cast<std::size_t>(n)]);
    }
    if (n < 20) {
        return std::string(kTeens[static_cast<std::size_t>(n - 10)]);
    }
    if (n < 100) {
        std::string out(kTens[static_cast<std::size_t>(n / 10)]);
        if (n % 10 != 0) {
            out.push_back(' ');
            out += kUnits[static_cast<std::size_t>(n % 10)];
        }
        return out;
    }
    std::string out(kUnits[static_cast<std::size_t>(n / 100)]);
    out += " hundred";
    if (n % 100 != 0) {
        out.push_back(' ');
        out += convert_less_than_thousand(n % 100);
    }
    return out;
}

/// 整数转单词（对齐上游 `_number_to_words_custom`）。
[[nodiscard]] std::string number_to_words(long long num) {
    static constexpr std::array kThousands =
        std::to_array<std::string_view>({"", "thousand", "million", "billion", "trillion"});
    if (num == 0) {
        return "zero";
    }
    std::vector<std::string> words;
    long long rest = num;
    std::size_t scale = 0;
    while (rest > 0) {
        if (rest % 1000 != 0) {
            std::string part = convert_less_than_thousand(rest % 1000);
            if (scale < kThousands.size()) {
                part.push_back(' ');
                part += kThousands[scale];
            }
            words.insert(words.begin(), std::move(part));
        }
        rest /= 1000;
        ++scale;
    }
    std::string out;
    for (std::size_t i = 0; i < words.size(); ++i) {
        if (i != 0) {
            out.push_back(' ');
        }
        out += words[i];
    }
    // 上游是 `" ".join(words).strip()` —— 每段都带着一个尾随空格，
    // 最后统一 strip 掉。
    const std::size_t begin = out.find_first_not_of(' ');
    if (begin == std::string::npos) {
        return out;
    }
    const std::size_t end = out.find_last_not_of(' ');
    return out.substr(begin, end - begin + 1);
}

/// 数字串转单词（非数字串原样返回，对齐上游）。
[[nodiscard]] std::string number_to_words_of(std::string_view digits) {
    if (!is_digit_string(digits)) {
        return std::string(digits);
    }
    long long value = 0;
    for (const char c : digits) {
        value = value * 10 + (c - '0');
        if (value > 999999999999LL) {
            break;  // 超大数字：截断（上游 int 也会溢出）
        }
    }
    return number_to_words(value);
}

/// 上游 `_expand_number_positive`。
[[nodiscard]] std::string expand_number_positive(std::string_view digits) {
    if (!is_digit_string(digits)) {
        return std::string(digits);
    }
    long long num = 0;
    for (const char c : digits) {
        num = num * 10 + (c - '0');
        if (num > 999999999999LL) {
            break;
        }
    }
    if (num >= 2000 && num < 2010) {
        return "two thousand and " + number_to_words(num % 100);
    }
    if (num >= 1100 && num < 2100 && num % 100 != 0) {
        return number_to_words(num / 100) + " " + number_to_words(num % 100);
    }
    return number_to_words(num);
}

/// 上游 `_ordinal_custom`：序数词（"1" → "first"，"21" → "twenty first"）。
[[nodiscard]] std::string ordinal_of(std::string_view digits) {
    if (!is_digit_string(digits)) {
        return std::string(digits);
    }
    long long num = 0;
    for (const char c : digits) {
        num = num * 10 + (c - '0');
        if (num > 999999999999LL) {
            break;
        }
    }
    const long long mod100 = num % 100;
    std::string suffix = "th";
    if (mod100 >= 10 && mod100 <= 20) {
        suffix = "th";
    } else {
        switch (num % 10) {
            case 1:
                suffix = "st";
                break;
            case 2:
                suffix = "nd";
                break;
            case 3:
                suffix = "rd";
                break;
            default:
                suffix = "th";
                break;
        }
    }
    return number_to_words(num) + suffix;
}

// ---------------------------------------------------------------------------
// 规则 1：NFD 分解 + 丢弃组合记号
// ---------------------------------------------------------------------------

[[nodiscard]] std::string decompose_and_strip_marks(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    std::size_t offset = 0;
    utf8::Char ch{};
    while (utf8::decode_next(text, offset, ch)) {
        if (is_combining_mark(ch.codepoint)) {
            continue;  // 上游：丢掉所有 Mn
        }
        // 预组合拉丁字母 → 基字符（等价于 NFD 之后再丢组合记号）。
        // 基字符可能是多个字母（"Æ" → "AE"）✓
        bool replaced = false;
        for (const auto& rule : latin::kRules) {
            if (rule.composed == ch.codepoint) {
                out += rule.base;
                replaced = true;
                break;
            }
        }
        if (!replaced) {
            out.append(text.substr(ch.offset, ch.length));
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// 规则 2：`@` → " at "
// ---------------------------------------------------------------------------

[[nodiscard]] std::string replace_at_sign(std::string_view text) {
    std::string out;
    out.reserve(text.size() + 8);
    for (const char c : text) {
        if (c == '@') {
            out += " at ";
        } else {
            out.push_back(c);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// 规则 3：缩写展开
// ---------------------------------------------------------------------------

struct Abbreviation {
    std::string_view pattern;  // 大小写不敏感
    std::string_view replacement;
};

const std::array kAbbreviations = std::to_array<Abbreviation>({
    {"Mr.", "Mister"},
    {"Mrs.", "Missus"},
    {"Dr.", "Doctor"},
    {"Prof.", "Professor"},
    {"St.", "Street"},
    {"Co.", "Company"},
    {"Ltd.", "Limited"},
    {"e.g.", "for example"},
    {"i.e.", "that is"},
});

[[nodiscard]] bool equals_ignore_case(std::string_view a, std::string_view b) noexcept {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        const char ca = static_cast<char>(std::tolower(static_cast<unsigned char>(a[i])));
        const char cb = static_cast<char>(std::tolower(static_cast<unsigned char>(b[i])));
        if (ca != cb) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] std::string expand_abbreviations(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        bool matched = false;
        for (const auto& abbr : kAbbreviations) {
            // 上游这条正则带 IGNORECASE —— "MR." / "Dr." 都认。
            if (!equals_ignore_case(text.substr(i, abbr.pattern.size()), abbr.pattern)) {
                continue;
            }
            // 上游正则：`\bMr\.(?=[\s,.]|\Z)` —— 前面是词边界，
            // 后面必须跟空白 / 逗号 / 句点，或者就是串尾。
            if (!word_boundary_left(text, i)) {
                continue;
            }
            const std::size_t after = i + abbr.pattern.size();
            const bool lookahead_ok = after >= text.size() || text[after] == ' ' ||
                                      text[after] == '\t' || text[after] == '\n' ||
                                      text[after] == '\r' || text[after] == ',' ||
                                      text[after] == '.';
            if (!lookahead_ok) {
                continue;
            }
            out += abbr.replacement;
            i = after;
            matched = true;
            break;
        }
        if (!matched) {
            out.push_back(text[i]);
            ++i;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// 规则 4：金额后缀（"$3 million" → "three million dollars"）
// ---------------------------------------------------------------------------

/// 上游 `_currency_suffix_re`：`([£$€])([\d,.]*\d)\s*(million|billion|thousand)\b`
[[nodiscard]] std::string expand_currency_suffix(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        const std::size_t symbol_length = currency_symbol_length(text, i);
        if (symbol_length != 0) {
            std::size_t j = i + symbol_length;
            // [\d,.]*\d —— 数字、逗号、点组成的串，最后一位必须是数字
            std::size_t k = j;
            while (k < text.size() &&
                   (text[k] == ',' || text[k] == '.' || (text[k] >= '0' && text[k] <= '9'))) {
                ++k;
            }
            if (k > j && text[k - 1] >= '0' && text[k - 1] <= '9') {
                const std::string_view amount = text.substr(j, k - j);
                // \s*
                std::size_t m = k;
                while (m < text.size() && is_space_codepoint(static_cast<unsigned char>(text[m]))) {
                    ++m;
                }
                // (million|billion|thousand) —— 大小写不敏感
                static constexpr std::array kSuffixes =
                    std::to_array<std::string_view>({"million", "billion", "thousand"});
                for (const auto suffix : kSuffixes) {
                    if (m + suffix.size() > text.size()) {
                        continue;
                    }
                    if (!equals_ignore_case(text.substr(m, suffix.size()), suffix)) {
                        continue;
                    }
                    // \b
                    if (!word_boundary_right(text, m + suffix.size())) {
                        continue;
                    }
                    std::string digits;
                    for (const char ch : amount) {
                        if (ch != ',') {
                            digits.push_back(ch);
                        }
                    }
                    const char* currency =
                        symbol_length == kDollar.size()
                            ? "dollars"
                            : (text.compare(i, kPound.size(), kPound) == 0 ? "pounds" : "euros");
                    out += number_to_words_of(digits);
                    out.push_back(' ');
                    // 上游用的是**原始文本**的后缀（`f"{amount_word} {suffix} ..."`），
                    // 所以 "$3 MILLION" 会留下 "MILLION" —— 后面再被缩写规则
                    // 拆成 "m i l l i o n"。这里照抄。
                    out.append(text.substr(m, suffix.size()));
                    out.push_back(' ');
                    out += currency;
                    i = m + suffix.size();
                    goto next;
                }
            }
        }
        out.push_back(text[i]);
        ++i;
    next:
        continue;
    }
    return out;
}

// ---------------------------------------------------------------------------
// 规则 5：电话号码
// ---------------------------------------------------------------------------

/// 上游 `_phone_re`：`(\+?\d{1,3}-)?\b(\d{3})-(?:(\d{3})-)?(\d{4})\b`
[[nodiscard]] std::string expand_phone_number(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        // 可选的国家码：`\+?\d{1,3}-`
        //
        // 上游的正则里这个分组是**可选**的：正则引擎先试它，不行就回退到
        // "没有国家码"。所以 "555-1234" 里，"555-" 先被当成国家码试试，
        // 后面接不上 "\b\d{3}-" → 回退 → "555-" 当区号、"1234" 当号码。
        std::size_t start = i;
        std::string_view country;
        std::size_t after_country = i;
        bool with_country = false;
        {
            std::size_t c = i;
            if (c < text.size() && text[c] == '+') {
                ++c;
            }
            const std::size_t digit_start = c;
            while (c < text.size() && text[c] >= '0' && text[c] <= '9' && c - digit_start < 3) {
                ++c;
            }
            if (c > digit_start && c < text.size() && text[c] == '-') {
                country = text.substr(start, c - start + 1);
                after_country = c + 1;
                with_country = true;
            }
        }
        // `\b(\d{3})-`
        std::size_t p = after_country;
        const bool need_boundary = word_boundary_left(text, p);
        const bool area_matches = need_boundary && p + 4 <= text.size() && text[p] >= '0' &&
                                  text[p + 1] >= '0' && text[p + 2] >= '0' && text[p + 3] == '-';
        if (!area_matches && with_country) {
            // 回退：把 "555-" 重新当成区号的一部分
            country = {};
            after_country = i;
            p = i;
        }
        if (area_matches ||
            (word_boundary_left(text, p) && p + 4 <= text.size() && text[p] >= '0' &&
             text[p + 1] >= '0' && text[p + 2] >= '0' && text[p + 3] == '-')) {
            // `(?:(\d{3})-)?`
            std::string_view exchange;
            std::size_t q = p + 4;
            if (q + 4 <= text.size() && text[q] >= '0' && text[q] <= '9' && text[q + 1] >= '0' &&
                text[q + 1] <= '9' && text[q + 2] >= '0' && text[q + 2] <= '9' &&
                text[q + 3] == '-') {
                exchange = text.substr(q, 3);
                q += 4;
            }
            // `(\d{4})\b`
            if (q + 4 <= text.size() && text[q] >= '0' && text[q] <= '9' && text[q + 1] >= '0' &&
                text[q + 1] <= '9' && text[q + 2] >= '0' && text[q + 2] <= '9' &&
                text[q + 3] >= '0' && text[q + 3] <= '9' && word_boundary_right(text, q + 4)) {
                // 命中，替换
                std::vector<std::string> parts;
                if (!country.empty()) {
                    std::vector<std::string> country_words;
                    if (country.front() == '+') {
                        country_words.emplace_back("plus");
                    }
                    std::string digits;
                    for (const char ch : country) {
                        if (ch >= '0' && ch <= '9') {
                            digits.push_back(ch);
                        }
                    }
                    if (!digits.empty()) {
                        std::string joined;
                        for (const char ch : digits) {
                            if (!joined.empty()) {
                                joined.push_back(' ');
                            }
                            joined += number_to_words_of(std::string_view(&ch, 1));
                        }
                        country_words.push_back(joined);
                    }
                    std::string joined_country;
                    for (const auto& w : country_words) {
                        if (!joined_country.empty()) {
                            joined_country.push_back(' ');
                        }
                        joined_country += w;
                    }
                    parts.push_back(joined_country);
                }
                const std::string_view area = text.substr(p, 3);
                std::string area_words;
                for (std::size_t k = 0; k < 3; ++k) {
                    if (!area_words.empty()) {
                        area_words.push_back(' ');
                    }
                    area_words += number_to_words_of(area.substr(k, 1));
                }
                parts.push_back(area_words);
                if (!exchange.empty()) {
                    std::string exch_words;
                    for (std::size_t k = 0; k < exchange.size(); ++k) {
                        if (!exch_words.empty()) {
                            exch_words.push_back(' ');
                        }
                        exch_words += number_to_words_of(exchange.substr(k, 1));
                    }
                    parts.push_back(exch_words);
                }
                const std::string_view line = text.substr(q, 4);
                std::string line_words;
                for (std::size_t k = 0; k < 4; ++k) {
                    if (!line_words.empty()) {
                        line_words.push_back(' ');
                    }
                    line_words += number_to_words_of(line.substr(k, 1));
                }
                parts.push_back(line_words);
                for (std::size_t k = 0; k < parts.size(); ++k) {
                    if (k > 0) {
                        out += ", ";
                    }
                    out += parts[k];
                }
                i = q + 4;
                continue;
            }
        }
        out.push_back(text[i]);
        ++i;
    }
    return out;
}

// ---------------------------------------------------------------------------
// 规则 6：尺寸（"3 x 4" → "three by four"）
// ---------------------------------------------------------------------------

/// 上游 `_dimension_re`：`\b(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)(?:\s*x\s*(\d+(?:\.\d+)?))?\b`
[[nodiscard]] std::string expand_dimension(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        if (text[i] >= '0' && text[i] <= '9' && word_boundary_left(text, i)) {
            std::size_t p = i;
            const std::size_t int_len = scan_digits(text, p);
            p += int_len;
            std::string first = std::string(text.substr(i, int_len));
            if (p < text.size() && text[p] == '.') {
                const std::size_t frac_start = p + 1;
                const std::size_t frac_len = scan_digits(text, frac_start);
                if (frac_len > 0) {
                    first += std::string(text.substr(p, frac_start + frac_len - p));
                    p = frac_start + frac_len;
                }
            }
            std::size_t q = p;
            while (q < text.size() && is_space_codepoint(static_cast<unsigned char>(text[q]))) {
                ++q;
            }
            if (q < text.size() && (text[q] == 'x' || text[q] == 'X')) {
                std::size_t r = q + 1;
                while (r < text.size() && is_space_codepoint(static_cast<unsigned char>(text[r]))) {
                    ++r;
                }
                if (r < text.size() && text[r] >= '0' && text[r] <= '9') {
                    const std::size_t int2 = scan_digits(text, r);
                    std::string second = std::string(text.substr(r, int2));
                    std::size_t s = r + int2;
                    if (s < text.size() && text[s] == '.') {
                        const std::size_t frac_start = s + 1;
                        const std::size_t frac_len = scan_digits(text, frac_start);
                        if (frac_len > 0) {
                            second += std::string(text.substr(s, frac_start + frac_len - s));
                            s = frac_start + frac_len;
                        }
                    }
                    // 第三段（可选）
                    std::string third;
                    std::size_t end = s;
                    std::size_t t = s;
                    while (t < text.size() &&
                           is_space_codepoint(static_cast<unsigned char>(text[t]))) {
                        ++t;
                    }
                    if (t < text.size() && (text[t] == 'x' || text[t] == 'X')) {
                        std::size_t u = t + 1;
                        while (u < text.size() &&
                               is_space_codepoint(static_cast<unsigned char>(text[u]))) {
                            ++u;
                        }
                        if (u < text.size() && text[u] >= '0' && text[u] <= '9') {
                            const std::size_t int3 = scan_digits(text, u);
                            third = std::string(text.substr(u, int3));
                            std::size_t v = u + int3;
                            if (v < text.size() && text[v] == '.') {
                                const std::size_t frac_start = v + 1;
                                const std::size_t frac_len = scan_digits(text, frac_start);
                                if (frac_len > 0) {
                                    third += std::string(text.substr(v, frac_start + frac_len - v));
                                    v = frac_start + frac_len;
                                }
                            }
                            end = v;
                        }
                    }
                    if (word_boundary_right(text, end)) {
                        out += number_to_words_of(first);
                        out += " by ";
                        out += number_to_words_of(second);
                        if (!third.empty()) {
                            out += " by ";
                            out += number_to_words_of(third);
                        }
                        i = end;
                        continue;
                    }
                }
            }
        }
        out.push_back(text[i]);
        ++i;
    }
    return out;
}

// ---------------------------------------------------------------------------
// 规则 7：罗马数字
// ---------------------------------------------------------------------------

/// 上游 `_roman_re`：长的排前面（`XIX|XVIII|…`）。
[[nodiscard]] std::string expand_roman(std::string_view text) {
    static const std::array kMap = std::to_array<std::pair<std::string_view, std::string_view>>({
        {"xix", "nineteen"},
        {"xviii", "eighteen"},
        {"xvii", "seventeen"},
        {"xvi", "sixteen"},
        {"xv", "fifteen"},
        {"xiv", "fourteen"},
        {"xiii", "thirteen"},
        {"xii", "twelve"},
        {"xi", "eleven"},
        {"x", "ten"},
        {"ix", "nine"},
        {"viii", "eight"},
        {"vii", "seven"},
        {"vi", "six"},
        {"v", "five"},
        {"iv", "four"},
        {"iii", "three"},
        {"ii", "two"},
    });
    std::string out;
    out.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        bool matched = false;
        for (const auto& entry : kMap) {
            // 上游的正则是 IGNORECASE —— "X" 和 "x" 都要认。
            if (!equals_ignore_case(text.substr(i, entry.first.size()), entry.first)) {
                continue;
            }
            if (!word_boundary_left(text, i) ||
                !word_boundary_right(text, i + entry.first.size())) {
                continue;
            }
            out += entry.second;
            i += entry.first.size();
            matched = true;
            break;
        }
        if (!matched) {
            out.push_back(text[i]);
            ++i;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// 规则 8：年代（"1980s" → "nineteen eighties"）
// ---------------------------------------------------------------------------

/// 上游 `_decade_re`：`\b((?:1[89]|20)\d0)s\b`
[[nodiscard]] std::string expand_decade(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        if (word_boundary_left(text, i) && i + 5 <= text.size() && text[i + 4] == 's' &&
            word_boundary_right(text, i + 5)) {
            bool prefix_ok = false;
            if ((text[i] == '1' && (text[i + 1] == '8' || text[i + 1] == '9')) ||
                (text[i] == '2' && text[i + 1] == '0')) {
                prefix_ok = true;
            }
            if (prefix_ok && text[i + 2] >= '0' && text[i + 2] <= '9' && text[i + 3] == '0') {
                const std::string year(text.substr(i, 4));
                std::string words = expand_number_positive(year);
                if (words.size() >= 2 && words.compare(words.size() - 2, 2, "ty") == 0) {
                    words = words.substr(0, words.size() - 1) + "ies";
                } else {
                    words += "s";
                }
                out += words;
                i += 5;
                continue;
            }
        }
        out.push_back(text[i]);
        ++i;
    }
    return out;
}

// ---------------------------------------------------------------------------
// 规则 9：比分（"3-2" → "three to two"）
// ---------------------------------------------------------------------------

/// 上游 `_score_re`：`\b(\d{1,2})-(\d{1,2})\b`
[[nodiscard]] std::string expand_score(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        if (word_boundary_left(text, i) && text[i] >= '0' && text[i] <= '9') {
            std::size_t len1 = 0;
            while (len1 < 2 && i + len1 < text.size() && text[i + len1] >= '0' &&
                   text[i + len1] <= '9') {
                ++len1;
            }
            if (len1 >= 1 && i + len1 < text.size() && text[i + len1] == '-') {
                std::size_t len2 = 0;
                while (len2 < 2 && i + len1 + 1 + len2 < text.size() &&
                       text[i + len1 + 1 + len2] >= '0' && text[i + len1 + 1 + len2] <= '9') {
                    ++len2;
                }
                if (len2 >= 1 && word_boundary_right(text, i + len1 + 1 + len2)) {
                    const std::string a(text.substr(i, len1));
                    const std::string b(text.substr(i + len1 + 1, len2));
                    out += number_to_words_of(a);
                    out += " to ";
                    out += number_to_words_of(b);
                    i += len1 + 1 + len2;
                    continue;
                }
            }
        }
        out.push_back(text[i]);
        ++i;
    }
    return out;
}

// ---------------------------------------------------------------------------
// 规则 10：日期（"3/15/2024" → "March fifteenth, two thousand and twenty-four"）
// ---------------------------------------------------------------------------

[[nodiscard]] std::string month_name(long long month) {
    static const std::array kMonths =
        std::to_array({"January", "February", "March", "April", "May", "June", "July", "August",
                       "September", "October", "November", "December"});
    if (month < 1 || month > 12) {
        return "";
    }
    return std::string(kMonths[static_cast<std::size_t>(month - 1)]);
}

/// 上游 `_date_re`：`\b(0?[1-9]|1[0-2])/([0-2]?\d|3[01])/(\d{2,4})\b`
[[nodiscard]] std::string expand_date(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        if (word_boundary_left(text, i)) {
            std::size_t month_len = 0;
            if (text[i] == '0') {
                month_len = 2;  // 0?[1-9]
                if (i + 2 >= text.size() || text[i + 1] < '1' || text[i + 1] > '9') {
                    month_len = 0;
                }
            } else if (text[i] >= '1' && text[i] <= '9') {
                month_len = 1;
            } else if (text[i] == '1' && i + 1 < text.size() && text[i + 1] == '0') {
                month_len = 2;  // 1[0-2]
                if (i + 2 >= text.size() || text[i + 2] < '0' || text[i + 2] > '2') {
                    month_len = 0;
                }
            }
            if (month_len > 0 && i + month_len < text.size() && text[i + month_len] == '/') {
                std::size_t day_start = i + month_len + 1;
                std::size_t day_len = 0;
                if (day_start < text.size() && text[day_start] == '3' &&
                    day_start + 1 < text.size() && text[day_start + 1] == '0' &&
                    day_start + 2 < text.size() && text[day_start + 2] == '1') {
                    day_len = 2;  // 3[01]
                } else if (day_start < text.size() && text[day_start] >= '0' &&
                           text[day_start] <= '2') {
                    day_len = 1;  // [0-2]?\d
                    if (day_start + 1 < text.size() && text[day_start + 1] >= '0' &&
                        text[day_start + 1] <= '9') {
                        day_len = 2;
                    }
                }
                if (day_len > 0 && day_start + day_len < text.size() &&
                    text[day_start + day_len] == '/') {
                    const std::size_t year_start = day_start + day_len + 1;
                    std::size_t year_len = scan_digits(text, year_start);
                    if (year_len >= 2 && year_len <= 4 &&
                        word_boundary_right(text, year_start + year_len)) {
                        const std::string month_text(text.substr(i, month_len));
                        const std::string day_text(text.substr(day_start, day_len));
                        std::string year_text(text.substr(year_start, year_len));
                        long long year = parse_digits(text, year_start, year_len);
                        if (year_text.size() == 2) {
                            year += year < 50 ? 2000 : 1900;
                        }
                        out += month_name(parse_digits(text, i, month_len));
                        out.push_back(' ');
                        out += ordinal_of(day_text);
                        out += ", ";
                        out += expand_number_positive(std::to_string(year));
                        i = year_start + year_len;
                        continue;
                    }
                }
            }
        }
        out.push_back(text[i]);
        ++i;
    }
    return out;
}

// ---------------------------------------------------------------------------
// 规则 11：时间（"3:30 pm" → "three thirty pm"）
// ---------------------------------------------------------------------------

/// 上游 `_time_re`：
/// `\b([01]?\d|2[0-3]):([0-5]\d)(?::([0-5]\d))?(\s*(?:a\.?m\.?|p\.?m\.?))?\b`
[[nodiscard]] std::string expand_time(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        if (word_boundary_left(text, i) && text[i] >= '0' && text[i] <= '9') {
            std::size_t hour_len = 1;
            if (text[i] == '0' || text[i] == '1') {
                hour_len = 2;
                if (i + 1 >= text.size() || text[i + 1] < '0' || text[i + 1] > '9') {
                    hour_len = 1;
                }
            } else if (text[i] == '2') {
                hour_len = 2;
                if (i + 1 >= text.size() || text[i + 1] < '0' || text[i + 1] > '3') {
                    hour_len = 1;
                }
            }
            if (i + hour_len < text.size() && text[i + hour_len] == ':') {
                const std::size_t min_start = i + hour_len + 1;
                if (min_start + 2 <= text.size() && text[min_start] >= '0' &&
                    text[min_start] <= '5' && text[min_start + 1] >= '0' &&
                    text[min_start + 1] <= '9') {
                    // 可选的秒
                    std::string second_text;
                    std::size_t end = min_start + 2;
                    if (end + 3 <= text.size() && text[end] == ':' && text[end + 1] >= '0' &&
                        text[end + 1] <= '5' && text[end + 2] >= '0' && text[end + 2] <= '9') {
                        second_text = std::string(text.substr(end + 1, 2));
                        end += 3;
                    }
                    // 可选的 am/pm。
                    //
                    // 上游正则是 `(\s*(?:a\.?m\.?|p\.?m\.?))?\b` —— 贪婪匹配
                    // 之后还要满足末尾的 `\b`，不满足就**回溯**：
                    // "3:30 pm." 先试 "pm."（末尾没边界）→ 退回 "pm" ✓。
                    // 所以这里要按最长优先逐个候选试，命中第一个满足边界的。
                    std::size_t p = end;
                    while (p < text.size() &&
                           is_space_codepoint(static_cast<unsigned char>(text[p]))) {
                        ++p;
                    }
                    std::string am_pm;
                    // 默认：am/pm 组不参与（匹配在分钟/秒之后结束）。
                    std::size_t after = end;
                    {
                        static constexpr const char* kAmPmForms[] = {"a.m.", "a.m", "am.", "am",
                                                                     "p.m.", "p.m", "pm.", "pm"};
                        for (const char* form : kAmPmForms) {
                            const std::size_t len = std::strlen(form);
                            // 上游这条正则是 IGNORECASE 的 —— "AM" / "Pm" 都认。
                            bool same = p + len <= text.size();
                            for (std::size_t k = 0; same && k < len; ++k) {
                                same = std::tolower(static_cast<unsigned char>(text[p + k])) ==
                                       form[k];
                            }
                            if (!same) {
                                continue;
                            }
                            if (!word_boundary_right(text, p + len)) {
                                continue;  // 末尾边界不成立 → 继续回溯
                            }
                            am_pm = (form[0] == 'a') ? "am" : "pm";
                            after = p + len;
                            break;
                        }
                    }
                    if (word_boundary_right(text, after)) {
                        const long long hour = parse_digits(text, i, hour_len);
                        const long long minute = parse_digits(text, min_start, 2);
                        long long hour_word_value = hour;
                        if (hour > 12 && !am_pm.empty()) {
                            hour_word_value = hour - 12;
                        }
                        if (hour == 0 && !am_pm.empty()) {
                            hour_word_value = 12;
                        }
                        std::string result = number_to_words(hour_word_value);
                        if (minute > 0) {
                            result += minute < 10 ? " oh " : " ";
                            result += number_to_words(minute);
                        }
                        if (!second_text.empty()) {
                            result += " and " +
                                      number_to_words(parse_digits(text, min_start + 3, 2)) +
                                      " seconds";
                        }
                        if (!am_pm.empty()) {
                            result += " " + am_pm;
                        }
                        out += result;
                        i = after;
                        continue;
                    }
                }
            }
        }
        out.push_back(text[i]);
        ++i;
    }
    return out;
}

// ---------------------------------------------------------------------------
// 规则 12：序数（"3. " → "third, "）
// ---------------------------------------------------------------------------

/// 上游 `_ordinal_number_re`：`\b(\d+)\. `（注意末尾**必须有空格**）
[[nodiscard]] std::string convert_ordinal_number(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        if (word_boundary_left(text, i) && text[i] >= '0' && text[i] <= '9') {
            const std::size_t len = scan_digits(text, i);
            if (len > 0 && i + len < text.size() && text[i + len] == '.' &&
                i + len + 1 < text.size() && text[i + len + 1] == ' ') {
                out += ordinal_of(text.substr(i, len));
                out += ", ";
                i += len + 2;
                continue;
            }
        }
        out.push_back(text[i]);
        ++i;
    }
    return out;
}

// ---------------------------------------------------------------------------
// 规则 13：去千分位逗号（"1,234" → "1234"）
// ---------------------------------------------------------------------------

/// 上游 `_comma_number_re`：`(\d[\d,]+d)`
[[nodiscard]] std::string remove_commas(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        if (text[i] >= '0' && text[i] <= '9') {
            std::size_t j = i + 1;
            bool has_comma = false;
            while (j < text.size() && (text[j] == ',' || (text[j] >= '0' && text[j] <= '9'))) {
                if (text[j] == ',') {
                    has_comma = true;
                }
                ++j;
            }
            // 至少要有 "数字 , 数字" 三段
            if (has_comma && j - i >= 3 && text[j - 1] >= '0' && text[j - 1] <= '9') {
                bool has_letters = false;
                for (std::size_t k = i; k < j; ++k) {
                    if (text[k] == ',') {
                        continue;
                    }
                    if (text[k] < '0' || text[k] > '9') {
                        has_letters = true;
                        break;
                    }
                }
                if (!has_letters) {
                    for (std::size_t k = i; k < j; ++k) {
                        if (text[k] != ',') {
                            out.push_back(text[k]);
                        }
                    }
                    i = j;
                    continue;
                }
            }
        }
        out.push_back(text[i]);
        ++i;
    }
    return out;
}

// ---------------------------------------------------------------------------
// 规则 14：货币（"$1,234.56" → "one thousand two hundred thirty four dollars and fifty six cents"）
// ---------------------------------------------------------------------------

/// 上游 `_currency_re`：`([£$€])(\d*\.?\d+)|(\d*\.?\d+)\s*([£$€])`
[[nodiscard]] std::string expand_currency(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        const char c = text[i];
        const std::size_t symbol_length = currency_symbol_length(text, i);
        if (symbol_length != 0) {
            std::size_t j = i + symbol_length;
            // 上游的金额模式是 `\d*\.?\d+` —— 小数点后必须跟数字，
            // 所以句末那个孤立的 "." 不属于金额（"\$1.50." → 金额是 "1.50"）。
            std::size_t k = j;
            while (k < text.size() && text[k] >= '0' && text[k] <= '9') {
                ++k;
            }
            if (k < text.size() && text[k] == '.') {
                std::size_t frac = k + 1;
                while (frac < text.size() && text[frac] >= '0' && text[frac] <= '9') {
                    ++frac;
                }
                if (frac > k + 1) {
                    k = frac;
                }
            }
            // `\d*\.?\d+` —— 至少一位数字
            if (k > j) {
                const std::string amount(text.substr(j, k - j));
                out += currency_phrase(text.substr(i, symbol_length), amount);
                i = k;
                continue;
            }
        } else if (c >= '0' && c <= '9') {
            // `(\d*\.?\d+)\s*([£$€])`
            std::size_t k = i;
            while (k < text.size() && (text[k] == '.' || (text[k] >= '0' && text[k] <= '9'))) {
                ++k;
            }
            std::size_t p = k;
            while (p < text.size() && is_space_codepoint(static_cast<unsigned char>(text[p]))) {
                ++p;
            }
            const std::size_t trailing_symbol = currency_symbol_length(text, p);
            if (trailing_symbol != 0 && k > i) {
                const std::string amount(text.substr(i, k - i));
                out += currency_phrase(text.substr(p, trailing_symbol), amount);
                i = p + trailing_symbol;
                continue;
            }
        }
        out.push_back(text[i]);
        ++i;
    }
    return out;
}

[[nodiscard]] std::string currency_phrase(std::string_view symbol, std::string_view amount) {
    std::string digits;
    for (const char ch : amount) {
        if (ch != ',') {
            digits.push_back(ch);
        }
    }
    if (!digits.empty() && digits.front() == '.') {
        digits.insert(digits.begin(), '0');
    }
    const bool is_dollar = symbol == kDollar;
    const bool is_pound = symbol == kPound;
    const char* major_singular = is_dollar ? "dollar" : (is_pound ? "pound" : "euro");
    const char* major_plural = is_dollar ? "dollars" : (is_pound ? "pounds" : "euros");
    const char* minor_singular = is_dollar ? "cent" : (is_pound ? "penny" : "cent");
    const char* minor_plural = is_dollar ? "cents" : (is_pound ? "pence" : "cents");

    const std::size_t dot = digits.find('.');
    const std::string major_part = dot == std::string::npos ? digits : digits.substr(0, dot);
    std::string minor_part = dot == std::string::npos ? "" : digits.substr(dot + 1);
    while (minor_part.size() < 2) {
        minor_part.push_back('0');
    }
    if (minor_part.size() > 2) {
        minor_part = minor_part.substr(0, 2);
    }
    const long long major_value =
        is_digit_string(major_part) && !major_part.empty() ? std::stoll(major_part) : 0;
    const long long minor_value = is_digit_string(minor_part) ? std::stoll(minor_part) : 0;

    std::vector<std::string> parts;
    if (major_value > 0) {
        parts.push_back(number_to_words(major_value) + " " +
                        (major_value == 1 ? major_singular : major_plural));
    }
    if (minor_value > 0) {
        parts.push_back(number_to_words(minor_value) + " " +
                        (minor_value == 1 ? minor_singular : minor_plural));
    }
    if (parts.empty()) {
        return std::string("zero ") + major_plural;
    }
    std::string out;
    for (std::size_t k = 0; k < parts.size(); ++k) {
        if (k > 0) {
            out += " and ";
        }
        out += parts[k];
    }
    return out;
}

// ---------------------------------------------------------------------------
// 规则 15：度量（"5cm" → "five centimeters"）
// ---------------------------------------------------------------------------

struct Measurement {
    std::string_view unit;
    std::string_view singular;
    std::string_view plural;
};

/// 上游 `_measurement_map`。单位在匹配时按**长度降序**（上游用
/// `sorted(keys, key=len, reverse=True)`），所以 "km/h" 必须排在 "km" 前面。
// 用 std::to_array —— 定长 std::array 写错元素数不会报错，多出来的元素会被
// 值初始化成空串，而空串单位会匹配任何东西（这是本项目踩过的坑）。
const std::array kMeasurements = std::to_array<Measurement>({
    {"km/h", "kilometer per hour", "kilometers per hour"},
    {"mph", "mile per hour", "miles per hour"},
    {"°C", "degree celsius", "degrees celsius"},
    {"°F", "degree fahrenheit", "degrees fahrenheit"},
    {"tbsp", "tablespoon", "tablespoons"},
    {"tsp", "teaspoon", "teaspoons"},
    {"km", "kilometer", "kilometers"},
    {"kg", "kilogram", "kilograms"},
    {"min", "minute", "minutes"},
    {"ft", "foot", "feet"},
    {"cm", "centimeter", "centimeters"},
    {"m", "meter", "meters"},
    {"L", "liter", "liters"},
    {"h", "hour", "hours"},
    {"s", "second", "seconds"},
});

/// 上游 `_measurement_re`：`(?<!\w)(-?(?:\d+/\d+|\d+(?:\.\d+)?))\s*(units)\b`
[[nodiscard]] std::string expand_measurement(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        // 上游是 `(?<!\w)` —— 只看左边，不是 `\b`。
        const bool can_start =
            prev_not_word(text, i) && (text[i] == '-' || (text[i] >= '0' && text[i] <= '9'));
        if (can_start) {
            std::size_t p = i;
            std::string number_text;
            if (text[p] == '-') {
                number_text.push_back('-');
                ++p;
            }
            const std::size_t int_len = scan_digits(text, p);
            if (int_len > 0) {
                number_text += std::string(text.substr(p, int_len));
                p += int_len;
                if (p < text.size() && text[p] == '/') {
                    const std::size_t den_len = scan_digits(text, p + 1);
                    if (den_len > 0) {
                        number_text += "/";
                        number_text += std::string(text.substr(p + 1, den_len));
                        p += 1 + den_len;
                    }
                } else if (p < text.size() && text[p] == '.') {
                    const std::size_t frac_len = scan_digits(text, p + 1);
                    if (frac_len > 0) {
                        number_text += ".";
                        number_text += std::string(text.substr(p + 1, frac_len));
                        p += 1 + frac_len;
                    }
                }
                std::size_t q = p;
                while (q < text.size() && is_space_codepoint(static_cast<unsigned char>(text[q]))) {
                    ++q;
                }
                // 单位：按长度降序匹配
                for (const auto& measurement : kMeasurements) {
                    if (!match_at(text, q, measurement.unit)) {
                        continue;
                    }
                    if (!word_boundary_right(text, q + measurement.unit.size())) {
                        continue;
                    }
                    const bool negative = !number_text.empty() && number_text.front() == '-';
                    std::string value = number_text;
                    if (negative) {
                        value = value.substr(1);
                    }
                    std::string number_word;
                    bool plural = true;
                    if (value.find('/') != std::string::npos) {
                        const std::size_t slash = value.find('/');
                        const std::string num = value.substr(0, slash);
                        const std::string den = value.substr(slash + 1);
                        number_word = fraction_phrase(num, den);
                        plural = true;
                    } else {
                        number_word = number_to_words_of(value);
                        // 上游：`is_plural = float(num_str) != 1`
                        // —— "1"、"1.0"、"1.00" 都算单数。
                        double numeric = 0.0;
                        try {
                            numeric = std::stod(value);
                        } catch (const std::exception&) {
                            numeric = 1.0;
                        }
                        plural = numeric != 1.0;
                    }
                    out += negative ? "minus " : "";
                    out += number_word;
                    out += " ";
                    out += plural ? measurement.plural : measurement.singular;
                    i = q + measurement.unit.size();
                    goto next;
                }
            }
        }
        out.push_back(text[i]);
        ++i;
    next:
        continue;
    }
    return out;
}

// ---------------------------------------------------------------------------
// 规则 16：分数（"3/4" → "three quarters"）
// ---------------------------------------------------------------------------

/// 上游 `_expand_fraction`。
[[nodiscard]] std::string fraction_phrase(std::string_view numerator,
                                          std::string_view denominator) {
    if (!is_digit_string(numerator) || !is_digit_string(denominator)) {
        return std::string(numerator) + "/" + std::string(denominator);
    }
    const long long n = std::stoll(std::string(numerator));
    const long long d = std::stoll(std::string(denominator));
    if (d == 0) {
        return std::string(numerator) + "/" + std::string(denominator);
    }
    if (n == 1 && d == 2) {
        return "one half";
    }
    if (n == 1 && d == 4) {
        return "one quarter";
    }
    if (n == 3 && d == 4) {
        return "three quarters";
    }
    return number_to_words(n) + " over " + number_to_words(d);
}

/// 上游 `_fraction_re`：`\b(\d+)/(\d+)\b`
[[nodiscard]] std::string expand_fraction(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        if (word_boundary_left(text, i) && text[i] >= '0' && text[i] <= '9') {
            const std::size_t num_len = scan_digits(text, i);
            if (num_len > 0 && i + num_len < text.size() && text[i + num_len] == '/') {
                const std::size_t den_len = scan_digits(text, i + num_len + 1);
                if (den_len > 0 && word_boundary_right(text, i + num_len + 1 + den_len)) {
                    out += fraction_phrase(text.substr(i, num_len),
                                           text.substr(i + num_len + 1, den_len));
                    i += num_len + 1 + den_len;
                    continue;
                }
            }
        }
        out.push_back(text[i]);
        ++i;
    }
    return out;
}

// ---------------------------------------------------------------------------
// 规则 17：小数（"3.14" → "three point one four"）
// ---------------------------------------------------------------------------

/// 上游 `_decimal_number_re`：`(\d+\.\d+)`
[[nodiscard]] std::string expand_decimal(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        if (text[i] >= '0' && text[i] <= '9') {
            const std::size_t int_len = scan_digits(text, i);
            if (int_len > 0 && i + int_len < text.size() && text[i + int_len] == '.') {
                const std::size_t frac_len = scan_digits(text, i + int_len + 1);
                if (frac_len > 0) {
                    const std::string integer_part(text.substr(i, int_len));
                    const std::string frac_part(text.substr(i + int_len + 1, frac_len));
                    out += number_to_words_of(integer_part);
                    out += " point";
                    for (const char ch : frac_part) {
                        out.push_back(' ');
                        out += number_to_words_of(std::string_view(&ch, 1));
                    }
                    i += int_len + 1 + frac_len;
                    continue;
                }
            }
        }
        out.push_back(text[i]);
        ++i;
    }
    return out;
}

// ---------------------------------------------------------------------------
// 规则 18：序数后缀（"21st" → "twenty first"）
// ---------------------------------------------------------------------------

/// 上游 `_ordinal_re`：`\b\d+(st|nd|rd|th)\b`
[[nodiscard]] std::string expand_ordinal_suffix(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        if (word_boundary_left(text, i) && text[i] >= '0' && text[i] <= '9') {
            const std::size_t len = scan_digits(text, i);
            if (len > 0 && i + len + 2 <= text.size()) {
                const std::string suffix(text.substr(i + len, 2));
                if ((suffix == "st" || suffix == "nd" || suffix == "rd" || suffix == "th") &&
                    word_boundary_right(text, i + len + 2)) {
                    out += ordinal_of(text.substr(i, len));
                    i += len + 2;
                    continue;
                }
            }
        }
        out.push_back(text[i]);
        ++i;
    }
    return out;
}

// ---------------------------------------------------------------------------
// 规则 19：字母数字混排（"H2O" → "H 2 O"）
// ---------------------------------------------------------------------------

/// 上游 `_alphanumeric_re`：`\b([a-zA-Z]+[0-9]+|[0-9]+[a-zA-Z]+)\b`
[[nodiscard]] std::string expand_alphanumeric(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        if (word_boundary_left(text, i)) {
            const bool alpha =
                (text[i] >= 'a' && text[i] <= 'z') || (text[i] >= 'A' && text[i] <= 'Z');
            const bool digit = text[i] >= '0' && text[i] <= '9';
            if (alpha || digit) {
                std::size_t j = i;
                while (j < text.size() &&
                       ((text[j] >= 'a' && text[j] <= 'z') || (text[j] >= 'A' && text[j] <= 'Z') ||
                        (text[j] >= '0' && text[j] <= '9'))) {
                    ++j;
                }
                const std::string_view token = text.substr(i, j - i);
                bool has_alpha = false;
                bool has_digit = false;
                for (const char ch : token) {
                    if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z')) {
                        has_alpha = true;
                    } else {
                        has_digit = true;
                    }
                }
                if (has_alpha && has_digit && word_boundary_right(text, j)) {
                    // 上游 `re.findall(r'[a-zA-Z]+|[0-9]+', text)`
                    std::string expanded;
                    std::size_t k = 0;
                    while (k < token.size()) {
                        std::size_t run = 0;
                        const bool run_alpha = (token[k] >= 'a' && token[k] <= 'z') ||
                                               (token[k] >= 'A' && token[k] <= 'Z');
                        while (k + run < token.size()) {
                            const char ch = token[k + run];
                            const bool is_alpha =
                                (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z');
                            if (is_alpha != run_alpha) {
                                break;
                            }
                            ++run;
                        }
                        const std::string_view part = token.substr(k, run);
                        if (!expanded.empty()) {
                            expanded.push_back(' ');
                        }
                        if (run_alpha) {
                            for (const char ch : part) {
                                if (!expanded.empty()) {
                                    expanded.push_back(' ');
                                }
                                expanded.push_back(ch);
                            }
                        } else {
                            for (const char ch : part) {
                                if (!expanded.empty()) {
                                    expanded.push_back(' ');
                                }
                                expanded += number_to_words_of(std::string_view(&ch, 1));
                            }
                        }
                        k += run;
                    }
                    out += expanded;
                    i = j;
                    continue;
                }
            }
        }
        out.push_back(text[i]);
        ++i;
    }
    return out;
}

// ---------------------------------------------------------------------------
// 规则 20：首字母缩写（"NASA" → "N A S A"）
// ---------------------------------------------------------------------------

/// 上游 `_acronym_re`：`\b[A-Z]{2,}\b`
[[nodiscard]] std::string expand_acronym(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        if (word_boundary_left(text, i) && text[i] >= 'A' && text[i] <= 'Z') {
            std::size_t j = i;
            while (j < text.size() && text[j] >= 'A' && text[j] <= 'Z') {
                ++j;
            }
            if (j - i >= 2 && word_boundary_right(text, j)) {
                for (std::size_t k = i; k < j; ++k) {
                    if (k > i) {
                        out.push_back(' ');
                    }
                    out.push_back(text[k]);
                }
                i = j;
                continue;
            }
        }
        out.push_back(text[i]);
        ++i;
    }
    return out;
}

// ---------------------------------------------------------------------------
// 规则 21：数字（"42" → "forty two"，"-3" → "minus three"）
// ---------------------------------------------------------------------------

/// 上游 `_number_re`：`(?<!\w)-?\d+(?!\w)`
[[nodiscard]] std::string expand_number(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        const bool digit = text[i] >= '0' && text[i] <= '9';
        // 上游 `(?<!\w)-?\d+(?!\w)`：负号只有在**前面不是词字符**时才算数字的一部分。
        // "A-12" 的 "-" 前面是 "A" → 负号不算，数字从 "1" 开始。
        const bool lookbehind_ok = prev_not_word(text, i);
        const bool negative = text[i] == '-' && lookbehind_ok && i + 1 < text.size() &&
                              text[i + 1] >= '0' && text[i + 1] <= '9';
        if (lookbehind_ok && (digit || negative)) {
            std::size_t p = i;
            std::string sign;
            if (negative) {
                sign = "-";
                ++p;
            }
            const std::size_t len = scan_digits(text, p);
            // 上游的 `(?!\w)`：数字后面不能紧跟词字符。
            if (len > 0 && next_not_word(text, p + len)) {
                const std::string digits(text.substr(p, len));
                if (negative) {
                    out += "minus " + expand_number_positive(digits);
                } else {
                    out += expand_number_positive(digits);
                }
                i = p + len;
                continue;
            }
        }
        out.push_back(text[i]);
        ++i;
    }
    return out;
}

// ---------------------------------------------------------------------------
// 规则 22–24：小写化、百分号、域名
// ---------------------------------------------------------------------------

[[nodiscard]] std::string replace_percent(std::string_view text) {
    std::string out;
    out.reserve(text.size() + 8);
    for (const char c : text) {
        if (c == '%') {
            out += " percent";
        } else {
            out.push_back(c);
        }
    }
    return out;
}

/// 上游的域名规则：`\b([a-z0-9-]+)\.([a-z]{2,})\b` → `\1 dot \2`，循环到不再匹配。
[[nodiscard]] std::string expand_domains(std::string_view text) {
    std::string current(text);
    for (int guard = 0; guard < 64; ++guard) {
        bool changed = false;
        std::string out;
        out.reserve(current.size());
        std::size_t i = 0;
        while (i < current.size()) {
            // 匹配 `[a-z0-9-]+\.[a-z]{2,}`
            std::size_t start = i;
            while (i < current.size() &&
                   ((current[i] >= 'a' && current[i] <= 'z') ||
                    (current[i] >= '0' && current[i] <= '9') || current[i] == '-')) {
                ++i;
            }
            const bool has_label = i > start;
            if (has_label && i < current.size() && current[i] == '.' &&
                word_boundary_left(current, start)) {
                const std::size_t tld_start = i + 1;
                std::size_t tld_end = tld_start;
                while (tld_end < current.size() && current[tld_end] >= 'a' &&
                       current[tld_end] <= 'z') {
                    ++tld_end;
                }
                if (tld_end - tld_start >= 2 && word_boundary_right(current, tld_end)) {
                    out += std::string(current.substr(start, i - start));
                    out += " dot ";
                    out += std::string(current.substr(tld_start, tld_end - tld_start));
                    i = tld_end;
                    changed = true;
                    continue;
                }
            }
            if (!has_label) {
                out.push_back(current[i]);
                ++i;
            } else {
                // 标签没跟域名 —— 原样吐出
                out.append(current.substr(start, i - start));
            }
        }
        current = out;
        if (!changed) {
            break;
        }
    }
    return current;
}

// ---------------------------------------------------------------------------
// 规则 25–27：字符白名单、空白折叠、去首尾空白
// ---------------------------------------------------------------------------

/// 上游 `re.sub(r"[^a-z0-9'.,?!:;-]", " ", text)`
[[nodiscard]] std::string filter_characters(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    std::size_t offset = 0;
    utf8::Char ch{};
    while (utf8::decode_next(text, offset, ch)) {
        const std::uint32_t cp = ch.codepoint;
        bool keep = false;
        if (cp < 0x80) {
            const char c = static_cast<char>(cp);
            keep = (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '\'' || c == '.' ||
                   c == ',' || c == '?' || c == '!' || c == ':' || c == ';' || c == '-';
        }
        if (keep) {
            out.append(text.substr(ch.offset, ch.length));
        } else {
            out.push_back(' ');
        }
    }
    return out;
}

/// 上游 `re.sub(r"\s+", " ", text)`
[[nodiscard]] std::string collapse_whitespace(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    bool in_space = false;
    for (const char c : text) {
        if (is_space_codepoint(static_cast<unsigned char>(c))) {
            if (!in_space) {
                out.push_back(' ');
                in_space = true;
            }
        } else {
            out.push_back(c);
            in_space = false;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// 上游 EnglishG2P.py 里的三步包装
// ---------------------------------------------------------------------------

/// 上游 `REP_MAP`。
///
/// **注意**：上游的键是**字面字符串**，不是字符集合 ——
/// `re.escape` 会把 "[;:：，；]" 整体转义掉，所以它匹配的是
/// **这五个字面字符连在一起**的那一段，而不是单个的 ";" 或 ":"。
/// 同理 `'["’]'` 匹配的是 " 引号 + 右单引号 两个字面字符。
///
/// 这一点很要紧：单个的 ":" 不该被替换 —— URL 的 "https://" 里的
/// 冒号必须原样保留（否则 "https,//..." 里的逗号会进到音素序列）。
[[nodiscard]] std::string apply_rep_map(std::string_view text) {
    // 上游 REP_MAP 的键（按出现顺序匹配；上游是 `|` 连接的 alternation，
    // 左边优先）。
    // 裸数组：大小自动推导，不会出现"写错元素数"的问题。
    static const std::pair<std::string_view, std::string_view> kReplacements[] = {
        {"[;:\xEF\xBC\x9A\xEF\xBC\x8C\xEF\xBC\x9B]", ","},  // "[;:：，；]"
        {"\"\xE2\x80\x99", "'"},                            // "\"’"
        {"\xE3\x80\x82", "."},                              // "。"
        {"\xEF\xBC\x81", "!"},                              // "！"
        {"\xEF\xBC\x9F", "?"},                              // "？"
    };

    std::string out;
    out.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        bool matched = false;
        for (const auto& entry : kReplacements) {
            if (text.compare(i, entry.first.size(), entry.first) == 0) {
                out += entry.second;
                i += entry.first.size();
                matched = true;
                break;
            }
        }
        if (!matched) {
            out.push_back(text[i]);
            ++i;
        }
    }
    return out;
}

/// 上游 `replace_consecutive_punctuation`：
/// `([!?,…. \s])([!?,….])+` → `\1`。注意第一个分组里**含空白**，
/// 所以连续空白也会被压成一个空格。
[[nodiscard]] std::string collapse_consecutive_punctuation(std::string_view text) {
    // "…" 是 3 字节 UTF-8 —— 不能写成字符字面量（multi-char constant）。
    constexpr std::string_view kEllipsis = "\xE2\x80\xA6";
    auto is_group = [&](std::string_view source, std::size_t index) {
        const char c = source[index];
        if (c == '!' || c == '?' || c == ',' || c == '.') {
            return true;
        }
        if (static_cast<unsigned char>(c) >= 0x80) {
            if (source.compare(index, kEllipsis.size(), kEllipsis) == 0) {
                return true;
            }
        }
        return is_space_codepoint(static_cast<unsigned char>(c));
    };
    std::string out;
    out.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        if (is_group(text, i)) {
            std::size_t j = i + 1;
            while (j < text.size() && is_group(text, j) &&
                   !is_space_codepoint(static_cast<unsigned char>(text[j]))) {
                ++j;
            }
            if (j > i + 1) {
                out.push_back(text[i]);
                i = j;
                continue;
            }
        }
        out.push_back(text[i]);
        ++i;
    }
    return out;
}

}  // namespace

std::string normalize(std::string_view text) {
    const bool trace = std::getenv("GENIE_TRACE_NORMALIZE") != nullptr;
    auto show = [&](const char* name, const std::string& v) {
        if (trace) {
            std::fprintf(stderr, "  %-22s %s\n", name, v.c_str());
        }
    };
    // 上游 EnglishG2P.text_normalize 的三步。
    std::string step = apply_rep_map(text);
    show("rep_map", step);
    step = decompose_and_strip_marks(step);
    show("nfd", step);
    step = replace_at_sign(step);
    show("at_sign", step);
    step = expand_abbreviations(step);
    show("abbrev", step);
    step = expand_currency_suffix(step);
    show("currency_suffix", step);
    step = expand_phone_number(step);
    show("phone", step);
    step = expand_dimension(step);
    show("dimension", step);
    step = expand_roman(step);
    show("roman", step);
    step = expand_decade(step);
    show("decade", step);
    step = expand_score(step);
    show("score", step);
    step = expand_date(step);
    show("date", step);
    step = expand_time(step);
    show("time", step);
    step = convert_ordinal_number(step);
    show("ordinal_number", step);
    step = remove_commas(step);
    show("comma_number", step);
    step = expand_currency(step);
    show("currency", step);
    step = expand_measurement(step);
    show("measurement", step);
    step = expand_fraction(step);
    show("fraction", step);
    step = expand_decimal(step);
    show("decimal", step);
    step = expand_ordinal_suffix(step);
    show("ordinal", step);
    step = expand_alphanumeric(step);
    show("alphanumeric", step);
    step = expand_acronym(step);
    show("acronym", step);
    step = expand_number(step);
    show("number", step);
    // 下游统一小写
    for (char& c : step) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    step = replace_percent(step);
    show("percent", step);
    step = expand_domains(step);
    show("domains", step);
    step = filter_characters(step);
    show("filter", step);
    step = collapse_whitespace(step);
    show("whitespace", step);
    // strip
    std::size_t begin = step.find_first_not_of(" \t\n\r\f\v");
    if (begin == std::string::npos) {
        return {};
    }
    const std::size_t end = step.find_last_not_of(" \t\n\r\f\v");
    return step.substr(begin, end - begin + 1);
}

std::string text_normalize(std::string_view text) {
    // 上游 EnglishG2P.text_normalize 的三步：
    //   REP_MAP → normalize() → 连续标点折叠
    const bool trace = std::getenv("GENIE_TRACE_NORMALIZE") != nullptr;
    std::string step = apply_rep_map(text);
    if (trace) {
        std::fprintf(stderr, "  %-22s %s\n", "rep_map", step.c_str());
    }
    step = normalize(step);
    step = collapse_consecutive_punctuation(step);
    if (trace) {
        std::fprintf(stderr, "  %-22s %s\n", "consecutive_punctuation", step.c_str());
    }
    return step;
}

}  // namespace genie::text::english
