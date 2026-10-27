// src/text/normalizer.cpp
//
// 中文文本规范化 —— 逐条对齐上游 Genie-TTS 的
//     G2P/Chinese/Normalization/text_normlization.py
// 里的 `TextNormalizer`（源头 PaddlePaddle / WeTextProcessing）。
//
// 为什么要一比一对齐：少了这一步，"2024年" 会被读成 "二千零二十四年"、
// 手机号会被读成 "一百三十八亿一千二百三十四万五千六百七十八"，
// 而上游分别读作 "二零二四年" 和 "幺三八幺二三四五六七八"。
//
// **替换顺序会影响结果**，所以下面的顺序与上游逐行对应，不要随手调换。
//
// 上游用 Python 正则实现，这里用**手写扫描** —— 不用 std::regex：
//   - 上游有两个模式带**反向先行断言**（`(?<!\d)`），ECMAScript 不支持
//   - MSVC 的 std::regex 慢且实现差异多
//   - 本项目一贯自己实现（JSON / SHA-256 / UTF-8 / WAV 都是）
//
// 行为验证：tests/unit/test_normalizer.cpp 用上游 Python 生成的黄金语料比对，
// 语料生成方式见 tools/verify_normalizer.py。

#include "genie/normalizer.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "genie/utf8.hpp"
#include "text/char_convert_table.hpp"
#include "text/punctuation.hpp"

namespace genie::text {
namespace {

/// 一对字符串。
using Pair = std::pair<std::string_view, std::string_view>;

// ===========================================================================
// 基础工具
// ===========================================================================

/// 中文数字。
///
/// 注意：必须**按字符**取，不能用 `kDigitChars[i]` —— 那是**字节**下标，
/// "二"占 3 字节，取到的是半个汉字，输出直接成乱码。
constexpr std::array<std::string_view, 10> kDigitNames = {
    "零", "一", "二", "三", "四", "五", "六", "七", "八", "九",
};

/// 上游 `UNITS`（OrderedDict）：键是"位权对应的数字个数"。
constexpr auto kUnits = std::to_array({
    std::pair{1, std::string_view{"十"}},
    std::pair{2, std::string_view{"百"}},
    std::pair{3, std::string_view{"千"}},
    std::pair{4, std::string_view{"万"}},
    std::pair{8, std::string_view{"亿"}},
});

[[nodiscard]] bool is_ascii_digit(char c) noexcept {
    return c >= '0' && c <= '9';
}

// ===========================================================================
// 码点辅助
//
// 上游的正则都是**按字符**匹配的，而 UTF-8 的中文 / 上标 / 希腊字母都是
// 多字节的。逐字节比会错位，所以这几处一律按码点处理。
// ===========================================================================

/// 常用字符的 UTF-8 字节序列。
namespace literals {
constexpr std::string_view kYi = "\xE4\xB8\x80";       // 一
constexpr std::string_view kYao = "\xE5\xB9\xBA";      // 幺
constexpr std::string_view kDuo = "\xE5\xA4\x9A";      // 多
constexpr std::string_view kYu = "\xE4\xBD\x99";       // 余
constexpr std::string_view kJi = "\xE5\x87\xA0";       // 几
constexpr std::string_view kTimes = "\xC3\x97";        // ×
constexpr std::string_view kDivide = "\xC3\xB7";       // ÷
constexpr std::string_view kDegree = "\xC2\xB0";       // °
constexpr std::string_view kCelsius = "\xE2\x84\x83";  // ℃
constexpr std::string_view kShe = "\xE6\x91\x84";      // 摄
constexpr std::string_view kShi = "\xE6\xB0\x8F";      // 氏
constexpr std::string_view kDu = "\xE5\xBA\xA6";       // 度
constexpr std::string_view kNian = "\xE5\xB9\xB4";     // 年
constexpr std::string_view kYue = "\xE6\x9C\x88";      // 月
constexpr std::string_view kRi = "\xE6\x97\xA5";       // 日
constexpr std::string_view kHao = "\xE5\x8F\xB7";      // 号
}  // namespace literals

/// 取 `text[offset]` 处的码点。
[[nodiscard]] char32_t codepoint_at(std::string_view text, std::size_t offset) {
    std::size_t cursor = offset;
    utf8::Char ch;
    if (!utf8::decode_next(text, cursor, ch)) {
        return 0;
    }
    return ch.codepoint;
}

/// 取 `text[pos]` **之前**那个码点（可能多字节）。
[[nodiscard]] char32_t codepoint_before(std::string_view text, std::size_t pos) {
    if (pos == 0 || pos > text.size()) {
        return 0;
    }
    std::size_t start = pos - 1;
    // 往前跳过 UTF-8 的续字节（10xxxxxx）
    while (start > 0 && (static_cast<unsigned char>(text[start]) & 0xC0) == 0x80) {
        --start;
    }
    return codepoint_at(text, start);
}

/// 上游 `verbalize_digit`：逐位读，alt_one 时把"一"读成"幺"。
[[nodiscard]] std::string verbalize_digit(std::string_view digits, bool alt_one) {
    std::string out;
    out.reserve(digits.size());
    for (const char c : digits) {
        if (!is_ascii_digit(c)) {
            continue;
        }
        out += kDigitNames[static_cast<std::size_t>(c - '0')];
    }
    if (alt_one) {
        // 上游是 result.replace("一", "幺") —— 按**字符**替换，不是按字节。
        std::size_t pos = 0;
        while ((pos = out.find(literals::kYi, pos)) != std::string::npos) {
            out.replace(pos, literals::kYi.size(), literals::kYao);
            pos += literals::kYao.size();
        }
    }
    return out;
}

/// 上游 `_get_value`：把数字串拆成"数字 + 单位"的序列。
void collect_value(std::string_view value, std::vector<std::string>& out) {
    std::size_t first = 0;
    while (first < value.size() && value[first] == '0') {
        ++first;
    }
    const std::string_view stripped = value.substr(first);
    if (stripped.empty()) {
        return;
    }
    if (stripped.size() == 1) {
        // 上游：use_zero 且去掉前导零后变短 → 前面补一个"零"
        if (stripped.size() < value.size()) {
            out.emplace_back("零");
        }
        out.emplace_back(std::string(kDigitNames[static_cast<std::size_t>(stripped[0] - '0')]));
        return;
    }

    // largest_unit = next(power for power in reversed(UNITS) if power < len(stripped))
    std::string_view unit;
    int largest = 0;
    for (auto it = kUnits.rbegin(); it != kUnits.rend(); ++it) {
        if (it->first < static_cast<int>(stripped.size())) {
            largest = it->first;
            unit = it->second;
            break;
        }
    }
    if (largest <= 0) {
        return;
    }
    collect_value(value.substr(0, value.size() - static_cast<std::size_t>(largest)), out);
    out.emplace_back(std::string(unit));
    collect_value(value.substr(value.size() - static_cast<std::size_t>(largest)), out);
}

/// 上游 `verbalize_cardinal`：整数读法。
[[nodiscard]] std::string verbalize_cardinal(std::string_view value) {
    if (value.empty()) {
        return {};
    }
    std::size_t first = 0;
    while (first < value.size() && value[first] == '0') {
        ++first;
    }
    const std::string_view stripped = value.substr(first);
    if (stripped.empty()) {
        return "零";
    }

    std::vector<std::string> symbols;
    collect_value(stripped, symbols);

    // 以"一十"开头 → 缩写成"十"（十点而不是一十点）
    if (symbols.size() >= 2 && symbols[0] == "一" && symbols[1] == "十") {
        symbols.erase(symbols.begin());
    }

    std::string out;
    for (const auto& symbol : symbols) {
        out += symbol;
    }
    return out;
}

/// 上游 `num2str`：数字 → 中文读法（支持小数）。
[[nodiscard]] std::string num2str(std::string_view value) {
    const auto dot = value.find('.');
    std::string_view integer = value;
    std::string_view decimal;
    if (dot != std::string_view::npos) {
        integer = value.substr(0, dot);
        decimal = value.substr(dot + 1);
    }

    std::string result = verbalize_cardinal(integer);

    // 上游：末尾是 0 → rstrip 后补一个 0（"3.20" → "三点二零"）；
    //       否则直接 rstrip（"3.2" → "三点二"）。
    std::string dec(decimal);
    std::size_t end = dec.size();
    while (end > 0 && dec[end - 1] == '0') {
        --end;
    }
    dec = dec.substr(0, end);
    if (!decimal.empty() && decimal.back() == '0') {
        dec += '0';
    }

    if (!dec.empty()) {
        if (result.empty()) {
            result = "零";
        }
        result += "点";
        result += verbalize_digit(dec, false);
    }
    return result;
}

// ===========================================================================
// 数字扫描
// ===========================================================================

/// 一个数字的匹配结果。
struct NumberSpan {
    std::size_t begin = 0;  ///< 匹配起点（含负号）
    std::size_t end = 0;    ///< 匹配终点
    bool negative = false;
    bool pure_decimal = false;  ///< ".5" 形式（没有整数部分，也没有负号）
    std::string integer;
    std::string fraction;
};

/// 在 `text[pos]` 处匹配一个数字。
///
/// 对应上游的正则 `(-?)((\d+)(\.\d+)?)|(\.(\d+))`：
///   备选 1 = 带符号、整数开头（可带小数）
///   备选 2 = 纯小数 `.digits`（**不带符号**）
///
/// @param require_fraction true 时要求有小数部分（对应 RE_DECIMAL_NUM）
[[nodiscard]] std::optional<NumberSpan> match_number(std::string_view text, std::size_t pos,
                                                     bool require_fraction) {
    // 备选 1
    std::size_t p = pos;
    bool negative = false;
    if (p < text.size() && text[p] == '-') {
        negative = true;
        ++p;
    }
    const std::size_t int_start = p;
    while (p < text.size() && is_ascii_digit(text[p])) {
        ++p;
    }
    if (p > int_start) {
        std::size_t frac_start = 0;
        bool has_fraction = false;
        if (p < text.size() && text[p] == '.') {
            std::size_t q = p + 1;
            const std::size_t fs = q;
            while (q < text.size() && is_ascii_digit(text[q])) {
                ++q;
            }
            if (q > fs) {
                has_fraction = true;
                frac_start = fs;
                p = q;
            }
        }
        if (!require_fraction || has_fraction) {
            NumberSpan span;
            span.begin = pos;
            span.end = p;
            span.negative = negative;
            const std::size_t int_len = (has_fraction ? frac_start - 1 : p) - int_start;
            span.integer = std::string(text.substr(int_start, int_len));
            if (has_fraction) {
                span.fraction = std::string(text.substr(frac_start, p - frac_start));
            }
            return span;
        }
    }

    // 备选 2：纯小数
    if (pos < text.size() && text[pos] == '.') {
        std::size_t q = pos + 1;
        const std::size_t fs = q;
        while (q < text.size() && is_ascii_digit(text[q])) {
            ++q;
        }
        if (q > fs) {
            NumberSpan span;
            span.begin = pos;
            span.end = q;
            span.pure_decimal = true;
            span.fraction = std::string(text.substr(fs, q - fs));
            return span;
        }
    }
    return std::nullopt;
}

/// 上游 `replace_number` / `replace_vrsion_num` 用的替换。
std::string replace_number(std::string_view text, bool require_fraction) {
    std::string out;
    out.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        if (is_ascii_digit(text[i]) || text[i] == '-' || text[i] == '.') {
            const auto match = match_number(text, i, require_fraction);
            if (match) {
                if (match->pure_decimal) {
                    out += num2str("." + match->fraction);
                } else {
                    if (match->negative) {
                        out += "负";
                    }
                    out +=
                        num2str(match->integer +
                                (match->fraction.empty() ? std::string{} : "." + match->fraction));
                }
                i = match->end;
                continue;
            }
        }
        out.push_back(text[i]);
        ++i;
    }
    return out;
}

// ===========================================================================
// 量词
// ===========================================================================

/// 上游 `COM_QUANTIFIERS` 展开后的**有序**候选表。
///
/// 顺序很重要：Python 的 alternation 是"**先匹配哪个用哪个**"，
/// 不是最长匹配。例：`(亿|千万|百万|万|千|百)` 遇到 "千万" 会命中 "千万"，
/// 因为它在 "万" 前面。
///
/// 展开方式：把嵌套组按原顺序摊平（`(千|毫|微)克` → 千克 / 毫克 / 微克）。
constexpr auto kQuantifiers = std::to_array({
    std::string_view{"处"},     std::string_view{"台"},     std::string_view{"架"},
    std::string_view{"枚"},     std::string_view{"趟"},     std::string_view{"幅"},
    std::string_view{"平"},     std::string_view{"方"},     std::string_view{"堵"},
    std::string_view{"间"},     std::string_view{"床"},     std::string_view{"株"},
    std::string_view{"批"},     std::string_view{"项"},     std::string_view{"例"},
    std::string_view{"列"},     std::string_view{"篇"},     std::string_view{"栋"},
    std::string_view{"注"},     std::string_view{"亩"},     std::string_view{"封"},
    std::string_view{"艘"},     std::string_view{"把"},     std::string_view{"目"},
    std::string_view{"套"},     std::string_view{"段"},     std::string_view{"人"},
    std::string_view{"所"},     std::string_view{"朵"},     std::string_view{"匹"},
    std::string_view{"张"},     std::string_view{"座"},     std::string_view{"回"},
    std::string_view{"场"},     std::string_view{"尾"},     std::string_view{"条"},
    std::string_view{"个"},     std::string_view{"首"},     std::string_view{"阙"},
    std::string_view{"阵"},     std::string_view{"网"},     std::string_view{"炮"},
    std::string_view{"顶"},     std::string_view{"丘"},     std::string_view{"棵"},
    std::string_view{"只"},     std::string_view{"支"},     std::string_view{"袭"},
    std::string_view{"辆"},     std::string_view{"挑"},     std::string_view{"担"},
    std::string_view{"颗"},     std::string_view{"壳"},     std::string_view{"窠"},
    std::string_view{"曲"},     std::string_view{"墙"},     std::string_view{"群"},
    std::string_view{"腔"},     std::string_view{"砣"},     std::string_view{"座"},
    std::string_view{"客"},     std::string_view{"贯"},     std::string_view{"扎"},
    std::string_view{"捆"},     std::string_view{"刀"},     std::string_view{"令"},
    std::string_view{"打"},     std::string_view{"手"},     std::string_view{"罗"},
    std::string_view{"坡"},     std::string_view{"山"},     std::string_view{"岭"},
    std::string_view{"江"},     std::string_view{"溪"},     std::string_view{"钟"},
    std::string_view{"队"},     std::string_view{"单"},     std::string_view{"双"},
    std::string_view{"对"},     std::string_view{"出"},     std::string_view{"口"},
    std::string_view{"头"},     std::string_view{"脚"},     std::string_view{"板"},
    std::string_view{"跳"},     std::string_view{"枝"},     std::string_view{"件"},
    std::string_view{"贴"},     std::string_view{"针"},     std::string_view{"线"},
    std::string_view{"管"},     std::string_view{"名"},     std::string_view{"位"},
    std::string_view{"身"},     std::string_view{"堂"},     std::string_view{"课"},
    std::string_view{"本"},     std::string_view{"页"},     std::string_view{"家"},
    std::string_view{"户"},     std::string_view{"层"},     std::string_view{"丝"},
    std::string_view{"毫"},     std::string_view{"厘"},     std::string_view{"分"},
    std::string_view{"钱"},     std::string_view{"两"},     std::string_view{"斤"},
    std::string_view{"担"},     std::string_view{"铢"},     std::string_view{"石"},
    std::string_view{"钧"},     std::string_view{"锱"},     std::string_view{"忽"},
    std::string_view{"千克"},   std::string_view{"毫克"},   std::string_view{"微克"},
    std::string_view{"毫"},     std::string_view{"厘"},     std::string_view{"公分"},
    std::string_view{"分"},     std::string_view{"寸"},     std::string_view{"尺"},
    std::string_view{"丈"},     std::string_view{"里"},     std::string_view{"寻"},
    std::string_view{"常"},     std::string_view{"铺"},     std::string_view{"程"},
    std::string_view{"千米"},   std::string_view{"分米"},   std::string_view{"厘米"},
    std::string_view{"毫米"},   std::string_view{"微米"},   std::string_view{"米"},
    std::string_view{"撮"},     std::string_view{"勺"},     std::string_view{"合"},
    std::string_view{"升"},     std::string_view{"斗"},     std::string_view{"石"},
    std::string_view{"盘"},     std::string_view{"碗"},     std::string_view{"碟"},
    std::string_view{"叠"},     std::string_view{"桶"},     std::string_view{"笼"},
    std::string_view{"盆"},     std::string_view{"盒"},     std::string_view{"杯"},
    std::string_view{"钟"},     std::string_view{"斛"},     std::string_view{"锅"},
    std::string_view{"簋"},     std::string_view{"篮"},     std::string_view{"盘"},
    std::string_view{"桶"},     std::string_view{"罐"},     std::string_view{"瓶"},
    std::string_view{"壶"},     std::string_view{"卮"},     std::string_view{"盏"},
    std::string_view{"箩"},     std::string_view{"箱"},     std::string_view{"煲"},
    std::string_view{"啖"},     std::string_view{"袋"},     std::string_view{"钵"},
    std::string_view{"年"},     std::string_view{"月"},     std::string_view{"日"},
    std::string_view{"季"},     std::string_view{"刻"},     std::string_view{"时"},
    std::string_view{"周"},     std::string_view{"天"},     std::string_view{"秒"},
    std::string_view{"分"},     std::string_view{"小时"},   std::string_view{"旬"},
    std::string_view{"纪"},     std::string_view{"岁"},     std::string_view{"世"},
    std::string_view{"更"},     std::string_view{"夜"},     std::string_view{"春"},
    std::string_view{"夏"},     std::string_view{"秋"},     std::string_view{"冬"},
    std::string_view{"代"},     std::string_view{"伏"},     std::string_view{"辈"},
    std::string_view{"丸"},     std::string_view{"泡"},     std::string_view{"粒"},
    std::string_view{"颗"},     std::string_view{"幢"},     std::string_view{"堆"},
    std::string_view{"条"},     std::string_view{"根"},     std::string_view{"支"},
    std::string_view{"道"},     std::string_view{"面"},     std::string_view{"片"},
    std::string_view{"张"},     std::string_view{"颗"},     std::string_view{"块"},
    std::string_view{"元"},     std::string_view{"亿"},     std::string_view{"千万"},
    std::string_view{"百万"},   std::string_view{"万"},     std::string_view{"千"},
    std::string_view{"百"},     std::string_view{"亿元"},   std::string_view{"千万元"},
    std::string_view{"百万元"}, std::string_view{"万元"},   std::string_view{"千元"},
    std::string_view{"百元"},   std::string_view{"美元"},   std::string_view{"元"},
    std::string_view{"亿吨"},   std::string_view{"千万吨"}, std::string_view{"百万吨"},
    std::string_view{"万吨"},   std::string_view{"千吨"},   std::string_view{"百吨"},
    std::string_view{"十吨"},   std::string_view{"吨"},     std::string_view{"亿块"},
    std::string_view{"千万块"}, std::string_view{"百万块"}, std::string_view{"万块"},
    std::string_view{"千块"},   std::string_view{"百块"},   std::string_view{"块"},
    std::string_view{"角"},     std::string_view{"毛"},     std::string_view{"分"},
});

/// 在 `text[pos]` 处匹配量词（按候选表的顺序，先匹配哪个用哪个）。
///
/// @return 量词长度；0 表示没匹配上
[[nodiscard]] std::size_t match_quantifier(std::string_view text, std::size_t pos) {
    for (const auto candidate : kQuantifiers) {
        if (candidate.empty()) {
            continue;
        }
        if (text.compare(pos, candidate.size(), candidate) == 0) {
            return candidate.size();
        }
    }
    return 0;
}

/// 上游 `replace_positive_quantifier`：正整数 + 量词。
std::string replace_positive_quantifier(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        if (is_ascii_digit(text[i])) {
            std::size_t p = i;
            while (p < text.size() && is_ascii_digit(text[p])) {
                ++p;
            }
            const std::string number(text.substr(i, p - i));

            // 可选的 [多余几+]
            std::string extra;
            if (p < text.size()) {
                const char32_t c = codepoint_at(text, p);
                if (c == U'+') {
                    extra = "多";
                    ++p;
                } else if (text.compare(p, literals::kDuo.size(), literals::kDuo) == 0) {
                    extra = "多";
                    p += literals::kDuo.size();
                } else if (text.compare(p, literals::kYu.size(), literals::kYu) == 0) {
                    extra = "余";
                    p += literals::kYu.size();
                } else if (text.compare(p, literals::kJi.size(), literals::kJi) == 0) {
                    extra = "几";
                    p += literals::kJi.size();
                }
            }

            const std::size_t quantifier = match_quantifier(text, p);
            if (quantifier > 0) {
                std::string spoken = num2str(number);
                // 上游：二 → 两
                if (spoken == "二") {
                    spoken = "两";
                }
                out += spoken;
                out += extra;
                out.append(text.substr(p, quantifier));
                i = p + quantifier;
                continue;
            }
        }
        out.push_back(text[i]);
        ++i;
    }
    return out;
}

/// 上游 `replace_default_num`：三位以上的纯数字逐位读（"00078" → "零零零七八"）。
std::string replace_default_num(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        if (is_ascii_digit(text[i])) {
            std::size_t p = i;
            while (p < text.size() && is_ascii_digit(text[p])) {
                ++p;
            }
            if (p - i >= 3) {
                out += verbalize_digit(text.substr(i, p - i), true);
                i = p;
                continue;
            }
        }
        out.push_back(text[i]);
        ++i;
    }
    return out;
}

/// 上游 `replace_negative_num`：`-12` → "负十二"。
std::string replace_negative_num(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        if (text[i] == '-') {
            std::size_t p = i + 1;
            const std::size_t ds = p;
            while (p < text.size() && is_ascii_digit(text[p])) {
                ++p;
            }
            if (p > ds) {
                out += "负";
                out += num2str(text.substr(ds, p - ds));
                i = p;
                continue;
            }
        }
        out.push_back(text[i]);
        ++i;
    }
    return out;
}

/// 上游 `replace_frac`：`3/4` → "四分之三"。
std::string replace_frac(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        if (is_ascii_digit(text[i]) || text[i] == '-') {
            const auto match = match_number(text, i, false);
            if (match && !match->pure_decimal) {
                // 需要 "数字/数字"
                std::size_t p = match->end;
                if (p < text.size() && text[p] == '/') {
                    std::size_t q = p + 1;
                    const std::size_t ds = q;
                    while (q < text.size() && is_ascii_digit(text[q])) {
                        ++q;
                    }
                    if (q > ds) {
                        if (match->negative) {
                            out += "负";
                        }
                        out += num2str(text.substr(ds, q - ds));
                        out += "分之";
                        out += num2str(match->integer);
                        i = q;
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

/// 上游 `replace_percentage`：`50%` → "百分之五十"。
std::string replace_percentage(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        if (is_ascii_digit(text[i]) || text[i] == '-') {
            const auto match = match_number(text, i, false);
            if (match && !match->pure_decimal) {
                std::size_t p = match->end;
                if (p < text.size() && text[p] == '%') {
                    if (match->negative) {
                        out += "负";
                    }
                    out += "百分之";
                    out +=
                        num2str(match->integer +
                                (match->fraction.empty() ? std::string{} : "." + match->fraction));
                    i = p + 1;
                    continue;
                }
            }
        }
        out.push_back(text[i]);
        ++i;
    }
    return out;
}

/// 上游 `replace_vrsion_num`：`1.2.3` → "一点二点三"。
std::string replace_version_num(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        if (is_ascii_digit(text[i])) {
            // 至少两个点才算版本号
            std::size_t p = i;
            std::size_t dots = 0;
            while (p < text.size() && (is_ascii_digit(text[p]) || text[p] == '.')) {
                if (text[p] == '.') {
                    ++dots;
                }
                ++p;
            }
            if (dots >= 2) {
                // 末尾的点不算（"1.2." → "1.2"）
                std::size_t end = p;
                while (end > i && text[end - 1] == '.') {
                    --end;
                }
                for (std::size_t k = i; k < end; ++k) {
                    if (text[k] == '.') {
                        out += "点";
                    } else {
                        out += num2str(text.substr(k, 1));
                    }
                }
                i = end;
                continue;
            }
        }
        out.push_back(text[i]);
        ++i;
    }
    return out;
}

// ===========================================================================
// 区间 / 次方 / 四则运算
// ===========================================================================

/// 上游 `RE_RANGE` 的负向断言字符集（按**码点**判断）。
[[nodiscard]] bool is_range_guard(char32_t c) noexcept {
    if (c >= U'0' && c <= U'9') {
        return true;
    }
    switch (c) {
        case U'+':
        case U'-':
        case U'=':
        case 0x00D7:  // ×
        case 0x00F7:  // ÷
            return true;
        default:
            return false;
    }
}

/// 上游 `replace_range`：`3-5` → "三到五"。
std::string replace_range(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        // 负向先行断言：前一个**字符**不能是数字/运算符
        const bool guarded = i > 0 && is_range_guard(codepoint_before(text, i));
        if (!guarded && (is_ascii_digit(text[i]) || text[i] == '-' || text[i] == '.')) {
            const auto first = match_number(text, i, false);
            if (first && !first->pure_decimal) {
                std::size_t p = first->end;
                if (p < text.size() && (text[p] == '-' || text[p] == '~')) {
                    const auto second = match_number(text, p + 1, false);
                    if (second && !second->pure_decimal) {
                        const bool tail_guarded = second->end < text.size() &&
                                                  is_range_guard(codepoint_at(text, second->end));
                        if (!tail_guarded) {
                            if (first->negative) {
                                out += "负";
                            }
                            out += num2str(first->integer + (first->fraction.empty()
                                                                 ? std::string{}
                                                                 : "." + first->fraction));
                            out += "到";
                            if (second->negative) {
                                out += "负";
                            }
                            out += num2str(second->integer + (second->fraction.empty()
                                                                  ? std::string{}
                                                                  : "." + second->fraction));
                            i = second->end;
                            continue;
                        }
                    }
                }
            }
        }
        out.push_back(text[i]);
        ++i;
    }
    return out;
}

/// 上游 `RE_TO_RANGE` 用的单位（有序，先匹配哪个用哪个）。
constexpr auto kRangeUnits = std::to_array({
    std::string_view{"%"},   std::string_view{"°C"},     std::string_view{"℃"},
    std::string_view{"度"},  std::string_view{"摄氏度"}, std::string_view{"cm2"},
    std::string_view{"cm²"}, std::string_view{"cm3"},    std::string_view{"cm³"},
    std::string_view{"cm"},  std::string_view{"db"},     std::string_view{"ds"},
    std::string_view{"kg"},  std::string_view{"km"},     std::string_view{"m2"},
    std::string_view{"m²"},  std::string_view{"m³"},     std::string_view{"m3"},
    std::string_view{"ml"},  std::string_view{"m"},      std::string_view{"mm"},
    std::string_view{"s"},
});

/// 上游 `replace_to_range`：`20~30cm` → "20至30cm"（只把 ~ 换成"至"）。
std::string replace_to_range(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        if (is_ascii_digit(text[i]) || text[i] == '-' || text[i] == '.') {
            const auto first = match_number(text, i, false);
            if (first) {
                std::size_t p = first->end;
                // 单位
                std::size_t unit_len = 0;
                for (const auto unit : kRangeUnits) {
                    if (text.compare(p, unit.size(), unit) == 0) {
                        unit_len = unit.size();
                        break;
                    }
                }
                if (unit_len > 0 && p + unit_len < text.size() && text[p + unit_len] == '~') {
                    const auto second = match_number(text, p + unit_len + 1, false);
                    if (second) {
                        std::size_t q = second->end;
                        std::size_t unit2_len = 0;
                        for (const auto unit : kRangeUnits) {
                            if (text.compare(q, unit.size(), unit) == 0) {
                                unit2_len = unit.size();
                                break;
                            }
                        }
                        if (unit2_len > 0) {
                            out.append(text.substr(i, p + unit_len - i));
                            out += "至";
                            out.append(
                                text.substr(p + unit_len + 1, q + unit2_len - (p + unit_len + 1)));
                            i = q + unit2_len;
                            continue;
                        }
                    }
                }
            }
        }
        out.push_back(text[i]);
        ++i;
    }
    return out;
}

/// 上游 `RE_POWER` 的上标字符集（按**码点**判断）。
[[nodiscard]] bool is_superscript(char32_t c) noexcept {
    switch (c) {
        case 0x2070:  // ⁰
        case 0x00B9:  // ¹
        case 0x00B2:  // ²
        case 0x00B3:  // ³
        case 0x2074:  // ⁴
        case 0x2075:  // ⁵
        case 0x2076:  // ⁶
        case 0x2077:  // ⁷
        case 0x2078:  // ⁸
        case 0x2079:  // ⁹
        case 0x02E3:  // ˣ
        case 0x02B8:  // ʸ
        case 0x207F:  // ⁿ
            return true;
        default:
            return false;
    }
}

/// 上游 `power_map`。
[[nodiscard]] char superscript_to_ascii(char32_t c) noexcept {
    switch (c) {
        case 0x2070:
            return '0';
        case 0x00B9:
            return '1';
        case 0x00B2:
            return '2';
        case 0x00B3:
            return '3';
        case 0x2074:
            return '4';
        case 0x2075:
            return '5';
        case 0x2076:
            return '6';
        case 0x2077:
            return '7';
        case 0x2078:
            return '8';
        case 0x2079:
            return '9';
        case 0x02E3:
            return 'x';
        case 0x02B8:
            return 'y';
        case 0x207F:
            return 'n';
        default:
            return 0;
    }
}

/// 上游 `replace_power`：`x²` → "x的2次方"。
std::string replace_power(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        const char32_t first = codepoint_at(text, i);
        if (is_superscript(first)) {
            std::size_t p = i;
            std::string ascii;
            while (p < text.size()) {
                const char32_t c = codepoint_at(text, p);
                if (!is_superscript(c)) {
                    break;
                }
                ascii.push_back(superscript_to_ascii(c));
                p += utf8::encoded_length(c);
            }
            out += "的";
            out += ascii;
            out += "次方";
            i = p;
            continue;
        }
        out.push_back(text[i]);
        ++i;
    }
    return out;
}

/// 上游 `asmd_map`。
[[nodiscard]] std::string_view asmd_operator(char32_t c) noexcept {
    switch (c) {
        case U'+':
            return "加";
        case U'-':
            return "减";
        case 0x00D7:
            return "乘";  // ×
        case 0x00F7:
            return "除";  // ÷
        case U'=':
            return "等于";
        default:
            return {};
    }
}

/// 上游 `RE_ASMD`：算式两边可以是 数字 / 字母（带上标）。
///
/// 扫描一个"操作数"：数字（可带上标）或字母（带上标）。
[[nodiscard]] std::size_t match_operand(std::string_view text, std::size_t pos) {
    if (pos >= text.size()) {
        return 0;
    }
    std::size_t p = pos;
    if (is_ascii_digit(text[p]) || text[p] == '-' || text[p] == '.') {
        const auto number = match_number(text, pos, false);
        if (!number) {
            return 0;
        }
        p = number->end;
    } else if ((text[p] >= 'A' && text[p] <= 'Z') || (text[p] >= 'a' && text[p] <= 'z')) {
        ++p;
    } else {
        return 0;
    }
    while (p < text.size() && is_superscript(codepoint_at(text, p))) {
        p += utf8::encoded_length(codepoint_at(text, p));
    }
    return p - pos;
}

/// 上游 `replace_asmd`：`1+2` → "1加2"（两边保留原样）。
std::string replace_asmd(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        const std::size_t left = match_operand(text, i);
        if (left > 0) {
            const std::size_t op_pos = i + left;
            if (op_pos < text.size()) {
                const auto op = asmd_operator(codepoint_at(text, op_pos));
                if (!op.empty()) {
                    const std::size_t right = match_operand(text, op_pos + 1);
                    if (right > 0) {
                        out.append(text.substr(i, left));
                        out += op;
                        out.append(text.substr(op_pos + 1, right));
                        i = op_pos + 1 + right;
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

// ===========================================================================
// 日期 / 时间
// ===========================================================================

/// 上游 `RE_DATE`：`(\d{4}|\d{2})年((0?[1-9]|1[0-2])月)?(日期)?`
std::string replace_date(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        if (is_ascii_digit(text[i])) {
            // 年：4 位或 2 位
            std::size_t p = i;
            while (p < text.size() && is_ascii_digit(text[p])) {
                ++p;
            }
            const std::size_t len = p - i;
            if ((len == 4 || len == 2) && p < text.size() &&
                text.compare(p, literals::kNian.size(), literals::kNian) == 0) {
                const std::string year(text.substr(i, len));
                std::size_t q = p + literals::kNian.size();

                // 月
                std::string month;
                if (q < text.size() && is_ascii_digit(text[q])) {
                    std::size_t m = q;
                    while (m < text.size() && is_ascii_digit(text[m])) {
                        ++m;
                    }
                    const std::size_t mlen = m - q;
                    if ((mlen == 1 && text[q] != '0') || (mlen == 2 && text[q] == '1') ||
                        (mlen == 2 && text[q] == '0' && text[q + 1] >= '1' && text[q + 1] <= '9')) {
                        month = std::string(text.substr(q, mlen));
                        q = m;
                        if (q < text.size() &&
                            text.compare(q, literals::kYue.size(), literals::kYue) == 0) {
                            q += literals::kYue.size();
                        }
                    }
                }

                // 日
                std::string day;
                std::string_view suffix;
                if (q < text.size() && is_ascii_digit(text[q])) {
                    std::size_t d = q;
                    while (d < text.size() && is_ascii_digit(text[d])) {
                        ++d;
                    }
                    const std::size_t dlen = d - q;
                    int value = 0;
                    bool valid = false;
                    if (dlen == 1) {
                        value = text[q] - '0';
                        valid = value >= 1 && value <= 9;
                    } else if (dlen == 2) {
                        value = std::atoi(std::string(text.substr(q, 2)).c_str());
                        valid = (value >= 10 && value <= 29) || value == 30 || value == 31;
                    }
                    const bool has_ri = d < text.size() &&
                                        text.compare(d, literals::kRi.size(), literals::kRi) == 0;
                    const bool has_hao = d < text.size() && text.compare(d, literals::kHao.size(),
                                                                         literals::kHao) == 0;
                    if (valid && (has_ri || has_hao)) {
                        day = std::string(text.substr(q, dlen));
                        suffix = has_ri ? literals::kRi : literals::kHao;
                        q = d + (has_ri ? literals::kRi.size() : literals::kHao.size());
                    }
                }

                out += verbalize_digit(year, false);
                out += "年";
                if (!month.empty()) {
                    out += verbalize_cardinal(month);
                    out += "月";
                }
                if (!day.empty()) {
                    out += verbalize_cardinal(day);
                    out += suffix;
                }
                i = q;
                continue;
            }
        }
        out.push_back(text[i]);
        ++i;
    }
    return out;
}

/// 上游 `RE_DATE2`：`2024/03/15`（分隔符要一致）。
std::string replace_date2(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        if (is_ascii_digit(text[i])) {
            std::size_t p = i;
            while (p < text.size() && is_ascii_digit(text[p])) {
                ++p;
            }
            if (p - i == 4 && p < text.size()) {
                const char sep = text[p];
                if (sep == '-' || sep == '/' || sep == ' ' || sep == '.') {
                    std::size_t q = p + 1;
                    const std::size_t ms = q;
                    while (q < text.size() && is_ascii_digit(text[q])) {
                        ++q;
                    }
                    if (q - ms == 2) {
                        const int month = std::atoi(std::string(text.substr(ms, 2)).c_str());
                        if (month >= 1 && month <= 12 && q < text.size() && text[q] == sep) {
                            std::size_t r = q + 1;
                            const std::size_t ds = r;
                            while (r < text.size() && is_ascii_digit(text[r])) {
                                ++r;
                            }
                            if (r - ds == 2) {
                                const int day = std::atoi(std::string(text.substr(ds, 2)).c_str());
                                if (day >= 1 && day <= 31) {
                                    out += verbalize_digit(std::string(text.substr(i, 4)), false);
                                    out += "年";
                                    out += verbalize_cardinal(std::string(text.substr(ms, 2)));
                                    out += "月";
                                    out += verbalize_cardinal(std::string(text.substr(ds, 2)));
                                    out += "日";
                                    i = r;
                                    continue;
                                }
                            }
                        }
                    }
                }
            }
        }
        out.push_back(text[i]);
        ++i;
    }
    return out;
}

/// 上游 `_time_num2str`：时间里的数字（保留前导零）。
[[nodiscard]] std::string time_num2str(std::string_view value) {
    std::string result =
        num2str(value.substr(std::min<std::size_t>(value.find_first_not_of('0'), value.size())));
    if (!value.empty() && value[0] == '0') {
        result = "零" + result;
    }
    return result;
}

/// 上游 `RE_TIME` / `RE_TIME_RANGE`。
std::string replace_time(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        // 小时
        std::size_t p = i;
        std::size_t hlen = 0;
        if (p < text.size() && is_ascii_digit(text[p])) {
            if (p + 1 < text.size() && is_ascii_digit(text[p + 1])) {
                const int hour = (text[p] - '0') * 10 + (text[p + 1] - '0');
                if (hour <= 23) {
                    hlen = 2;
                }
            }
            if (hlen == 0) {
                const int hour = text[p] - '0';
                if (hour <= 23) {
                    hlen = 1;
                }
            }
        }
        if (hlen > 0 && p + hlen < text.size() && text[p + hlen] == ':') {
            const std::size_t ms = p + hlen + 1;
            std::size_t q = ms;
            while (q < text.size() && is_ascii_digit(text[q])) {
                ++q;
            }
            if (q - ms == 2) {
                const int minute = std::atoi(std::string(text.substr(ms, 2)).c_str());
                if (minute <= 59) {
                    std::size_t end = q;
                    std::string second;
                    if (q < text.size() && text[q] == ':') {
                        const std::size_t ss = q + 1;
                        std::size_t r = ss;
                        while (r < text.size() && is_ascii_digit(text[r])) {
                            ++r;
                        }
                        if (r - ss == 2) {
                            const int s = std::atoi(std::string(text.substr(ss, 2)).c_str());
                            if (s <= 59) {
                                second = std::string(text.substr(ss, 2));
                                end = r;
                            }
                        }
                    }
                    const std::string hour(text.substr(p, hlen));
                    const std::string minute_text(text.substr(ms, 2));

                    // 时间范围？
                    std::string hour2;
                    std::string minute2;
                    std::string second2;
                    if (end < text.size() && (text[end] == '~' || text[end] == '-')) {
                        std::size_t r = end + 1;
                        std::size_t h2len = 0;
                        if (r < text.size() && is_ascii_digit(text[r])) {
                            if (r + 1 < text.size() && is_ascii_digit(text[r + 1])) {
                                const int h2 = (text[r] - '0') * 10 + (text[r + 1] - '0');
                                if (h2 <= 23) {
                                    h2len = 2;
                                }
                            }
                            if (h2len == 0 && text[r] - '0' <= 23) {
                                h2len = 1;
                            }
                        }
                        if (h2len > 0 && r + h2len < text.size() && text[r + h2len] == ':') {
                            const std::size_t m2s = r + h2len + 1;
                            std::size_t m2e = m2s;
                            while (m2e < text.size() && is_ascii_digit(text[m2e])) {
                                ++m2e;
                            }
                            if (m2e - m2s == 2) {
                                const int m2 = std::atoi(std::string(text.substr(m2s, 2)).c_str());
                                if (m2 <= 59) {
                                    hour2 = std::string(text.substr(r, h2len));
                                    minute2 = std::string(text.substr(m2s, 2));
                                    std::size_t e2 = m2e;
                                    if (m2e < text.size() && text[m2e] == ':') {
                                        const std::size_t s2s = m2e + 1;
                                        std::size_t s2e = s2s;
                                        while (s2e < text.size() && is_ascii_digit(text[s2e])) {
                                            ++s2e;
                                        }
                                        if (s2e - s2s == 2) {
                                            const int s2 =
                                                std::atoi(std::string(text.substr(s2s, 2)).c_str());
                                            if (s2 <= 59) {
                                                second2 = std::string(text.substr(s2s, 2));
                                                e2 = s2e;
                                            }
                                        }
                                    }
                                    end = e2;
                                }
                            }
                        }
                    }

                    out += num2str(hour);
                    out += "点";
                    if (minute_text.find_first_not_of('0') != std::string::npos) {
                        if (minute == 30) {
                            out += "半";
                        } else {
                            out += time_num2str(minute_text);
                            out += "分";
                        }
                    }
                    if (!second.empty() && second.find_first_not_of('0') != std::string::npos) {
                        out += time_num2str(second);
                        out += "秒";
                    }
                    if (!hour2.empty()) {
                        out += "至";
                        out += num2str(hour2);
                        out += "点";
                        if (minute2.find_first_not_of('0') != std::string::npos) {
                            // 上游这里判断的是**前一个** minute（原文如此，保留）
                            if (minute == 30) {
                                out += "半";
                            } else {
                                out += time_num2str(minute2);
                                out += "分";
                            }
                        }
                        if (!second2.empty() &&
                            second2.find_first_not_of('0') != std::string::npos) {
                            out += time_num2str(second2);
                            out += "秒";
                        }
                    }
                    i = end;
                    continue;
                }
            }
        }
        out.push_back(text[i]);
        ++i;
    }
    return out;
}

// ===========================================================================
// 电话
// ===========================================================================

/// 上游 `phone2str`：逐位读，"一"读成"幺"，用"，"分隔。
[[nodiscard]] std::string phone2str(std::string_view phone, bool mobile) {
    std::vector<std::string> parts;
    if (mobile) {
        // 上游：strip("+").split() —— 按空白切
        std::string_view trimmed = phone;
        while (!trimmed.empty() && (trimmed.front() == '+' || trimmed.front() == ' ')) {
            trimmed.remove_prefix(1);
        }
        while (!trimmed.empty() && trimmed.back() == ' ') {
            trimmed.remove_suffix(1);
        }
        std::size_t start = 0;
        while (start <= trimmed.size()) {
            const auto pos = trimmed.find(' ', start);
            const auto piece = trimmed.substr(
                start, pos == std::string_view::npos ? std::string_view::npos : pos - start);
            if (!piece.empty()) {
                parts.push_back(verbalize_digit(piece, true));
            }
            if (pos == std::string_view::npos) {
                break;
            }
            start = pos + 1;
        }
    } else {
        std::size_t start = 0;
        while (true) {
            const auto pos = phone.find('-', start);
            const auto piece = phone.substr(
                start, pos == std::string_view::npos ? std::string_view::npos : pos - start);
            if (!piece.empty()) {
                parts.push_back(verbalize_digit(piece, true));
            }
            if (pos == std::string_view::npos) {
                break;
            }
            start = pos + 1;
        }
    }
    std::string out;
    for (std::size_t i = 0; i < parts.size(); ++i) {
        if (i > 0) {
            out += "，";
        }
        out += parts[i];
    }
    return out;
}

/// 上游 `RE_MOBILE_PHONE` + `RE_TELEPHONE` + `RE_NATIONAL_UNIFORM_NUMBER`。
std::string replace_phone(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        const char c = text[i];
        if (is_ascii_digit(c) || c == '+') {
            // 负向先行断言：前一个字符不是数字
            const bool preceded_by_digit = i > 0 && is_ascii_digit(text[i - 1]);

            // 400 号
            if (!preceded_by_digit && text.compare(i, 3, "400") == 0) {
                std::size_t p = i + 3;
                std::size_t digits = 0;
                while (p < text.size() && (is_ascii_digit(text[p]) || text[p] == '-')) {
                    if (is_ascii_digit(text[p])) {
                        ++digits;
                    }
                    ++p;
                }
                // 上游：(400)(-)?\d{3}(-)?\d{4}
                if (digits == 7) {
                    const std::string_view piece = text.substr(i, p - i);
                    if (p >= text.size() || !is_ascii_digit(text[p])) {
                        out += phone2str(piece, false);
                        i = p;
                        continue;
                    }
                }
            }

            if (!preceded_by_digit) {
                // 手机号：可选的 +86 前缀，然后 1 开头
                std::size_t p = i;
                if (c == '+') {
                    // +86 ?
                    if (text.compare(p, 3, "+86") == 0) {
                        p += 3;
                        if (p < text.size() && text[p] == ' ') {
                            ++p;
                        }
                    }
                }
                if (p < text.size() && text[p] == '1') {
                    // 第二位：3/8/5(0-3,5-9)/7(6,7,8)/9(8,9)
                    // 上游：`1([38]\d|5[0-35-9]|7[678]|9[89])\d{8}`
                    // → "1" + 2 位前缀 + 8 位数字 = 11 位。
                    const auto prefix_ok = [&]() {
                        if (p + 2 >= text.size()) {
                            return false;
                        }
                        const char second = text[p + 1];
                        if (second == '3' || second == '8') {
                            return true;
                        }
                        if (second == '5') {
                            return (text[p + 2] >= '0' && text[p + 2] <= '3') ||
                                   (text[p + 2] >= '5' && text[p + 2] <= '9');
                        }
                        if (second == '7') {
                            return text[p + 2] >= '6' && text[p + 2] <= '8';
                        }
                        if (second == '9') {
                            return text[p + 2] >= '8' && text[p + 2] <= '9';
                        }
                        return false;
                    };
                    if (prefix_ok()) {
                        std::size_t q = p + 3;  // "1" + 2 位前缀
                        // 还要 8 位数字
                        std::size_t count = 0;
                        while (q < text.size() && is_ascii_digit(text[q]) && count < 8) {
                            ++q;
                            ++count;
                        }
                        if (count == 8 && (q >= text.size() || !is_ascii_digit(text[q]))) {
                            out += phone2str(text.substr(i, q - i), true);
                            i = q;
                            continue;
                        }
                    }
                }
            }
        }
        out.push_back(text[i]);
        ++i;
    }
    return out;
}

// ===========================================================================
// 单位 / 温度
// ===========================================================================

/// 上游 `measure_dict`（**有序** —— "cm2" 必须在 "cm" 前面，否则 "cm2" 会被拆坏）。
constexpr auto kMeasures = std::to_array({
    Pair{"cm2", "平方厘米"},
    Pair{"cm²", "平方厘米"},
    Pair{"cm3", "立方厘米"},
    Pair{"cm³", "立方厘米"},
    Pair{"cm", "厘米"},
    Pair{"db", "分贝"},
    Pair{"ds", "毫秒"},
    Pair{"kg", "千克"},
    Pair{"km", "千米"},
    Pair{"m2", "平方米"},
    Pair{"m²", "平方米"},
    Pair{"m³", "立方米"},
    Pair{"m3", "立方米"},
    Pair{"ml", "毫升"},
    Pair{"m", "米"},
    Pair{"mm", "毫米"},
    Pair{"s", "秒"},
});

/// 上游 `replace_measure`。
std::string replace_measure(std::string_view text) {
    std::string out(text);
    for (const auto& [notation, spoken] : kMeasures) {
        std::size_t pos = 0;
        int guard = 0;
        while ((pos = out.find(notation, pos)) != std::string::npos) {
            if (++guard > 5) {
                std::fflush(stderr);
            }
            out.replace(pos, notation.size(), spoken);
            pos += spoken.size();
        }
    }
    return out;
}

/// 上游 `RE_TEMPERATURE` + `replace_temperature`。
std::string replace_temperature(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        if (is_ascii_digit(text[i]) || text[i] == '-') {
            const auto match = match_number(text, i, false);
            if (match && !match->pure_decimal) {
                std::size_t p = match->end;
                std::string unit;
                static const std::string kSheDu = std::string(literals::kShe) +
                                                  std::string(literals::kShi) +
                                                  std::string(literals::kDu);
                // 上游的单位顺序：°C | ℃ | 度 | 摄氏度
                if (text.compare(p, 3,
                                 "\xC2\xB0"
                                 "C") == 0) {
                    unit =
                        "\xC2\xB0"
                        "C";
                } else if (text.compare(p, 3, literals::kCelsius) == 0) {
                    unit = std::string(literals::kCelsius);
                } else if (text.compare(p, 3, literals::kDu) == 0) {
                    unit = std::string(literals::kDu);
                } else if (text.compare(p, kSheDu.size(), kSheDu) == 0) {
                    unit = std::string(kSheDu);
                }
                if (!unit.empty()) {
                    if (match->negative) {
                        out += "零下";
                    }
                    out +=
                        num2str(match->integer +
                                (match->fraction.empty() ? std::string{} : "." + match->fraction));
                    // 上游的 replace_temperature 取的是 group(3) —— 那是**小数**组，
                    // 不是单位组（单位是 group(4)）。所以它永远输出 "度"，
                    // 连 "摄氏度" 也读成 "度"。
                    //
                    // 这是上游的 group 编号 bug，但**行为要对齐**：
                    // "-3摄氏度" 上游读作 "零下三度"。
                    out += std::string(literals::kDu);
                    i = p + unit.size();
                    continue;
                }
            }
        }
        out.push_back(text[i]);
        ++i;
    }
    return out;
}

// ===========================================================================
// 全角 → 半角
// ===========================================================================

/// 把一段文本按码点做全角 → 半角映射。
///
/// @param convert 码点 → 码点的映射；返回 0 表示不转换
[[nodiscard]] std::string map_codepoints(std::string_view text, char32_t (*convert)(char32_t)) {
    std::string out;
    out.reserve(text.size());
    std::size_t offset = 0;
    utf8::Char ch;
    while (utf8::decode_next(text, offset, ch)) {
        const char32_t converted = convert(ch.codepoint);
        out += utf8::encode(converted != 0 ? converted : ch.codepoint);
    }
    return out;
}

/// 上游 `F2H_ASCII_LETTERS`：全角字母 → 半角。
[[nodiscard]] char32_t halfwidth_letter(char32_t codepoint) noexcept {
    if (codepoint >= 0xFF21 && codepoint <= 0xFF3A) {
        return codepoint - 0xFF21 + U'A';
    }
    if (codepoint >= 0xFF41 && codepoint <= 0xFF5A) {
        return codepoint - 0xFF41 + U'a';
    }
    return 0;
}

/// 上游 `F2H_DIGITS`：全角数字 → 半角。
[[nodiscard]] char32_t halfwidth_digit(char32_t codepoint) noexcept {
    if (codepoint >= 0xFF10 && codepoint <= 0xFF19) {
        return codepoint - 0xFF10 + U'0';
    }
    return 0;
}

/// 上游 `F2H_SPACE`：表意空格 → 半角空格。
[[nodiscard]] char32_t halfwidth_space(char32_t codepoint) noexcept {
    if (codepoint == 0x3000) {
        return U' ';
    }
    return 0;
}

[[nodiscard]] std::string to_halfwidth_letters(std::string_view text) {
    return map_codepoints(text, &halfwidth_letter);
}

[[nodiscard]] std::string to_halfwidth_digits(std::string_view text) {
    return map_codepoints(text, &halfwidth_digit);
}

[[nodiscard]] std::string to_halfwidth_space(std::string_view text) {
    return map_codepoints(text, &halfwidth_space);
}

// ===========================================================================
// 繁简转换
// ===========================================================================

/// 繁体 → 简体 的查表（建一次）。
const std::unordered_map<char32_t, char32_t>& traditional_to_simplified_map() {
    static const std::unordered_map<char32_t, char32_t> map = [] {
        std::unordered_map<char32_t, char32_t> result;
        const auto simplified = simplified_chars();
        const auto traditional = traditional_chars();
        std::size_t s_offset = 0;
        std::size_t t_offset = 0;
        utf8::Char s_char;
        utf8::Char t_char;
        while (utf8::decode_next(simplified, s_offset, s_char) &&
               utf8::decode_next(traditional, t_offset, t_char)) {
            result.emplace(t_char.codepoint, s_char.codepoint);
        }
        return result;
    }();
    return map;
}

/// 上游 `tranditional_to_simplified`。
[[nodiscard]] std::string traditional_to_simplified(std::string_view text) {
    const auto& map = traditional_to_simplified_map();
    std::string out;
    out.reserve(text.size());
    std::size_t offset = 0;
    utf8::Char ch;
    while (utf8::decode_next(text, offset, ch)) {
        const auto it = map.find(ch.codepoint);
        if (it != map.end()) {
            out += utf8::encode(it->second);
        } else {
            out += utf8::encode(ch.codepoint);
        }
    }
    return out;
}

// ===========================================================================
// 标点与收尾
// ===========================================================================

/// 上游要**丢弃**的字符（按码点判断）。
///
/// 上游用的是字符类 `[——《》【】<>{}()（）#&@"”^_\|]`，是**按字符**匹配的。
/// 这里必须按码点 —— 按字节过滤会把多字节汉字的字节删掉，
/// 把"月"（E6 9C 88）删成 E6 9C，结果是乱码。
///
/// @param post true 时用 `_post_replace` 的集合（多一个 "-" 和 "="）
[[nodiscard]] bool is_dropped_codepoint(char32_t c, bool post) noexcept {
    switch (c) {
        case 0x2014:  // —
        case 0x300A:  // 《
        case 0x300B:  // 》
        case 0x3010:  // 【
        case 0x3011:  // 】
        case U'<':
        case U'>':
        case U'{':
        case U'}':
        case U'(':
        case U')':
        case 0xFF08:  // （
        case 0xFF09:  // ）
        case U'#':
        case U'&':
        case U'@':
        case 0x201C:  // "
        case 0x201D:  // "
        case U'^':
        case U'_':
        case U'|':
        case U'\\':
            return true;
        case U'-':
        case U'=':
            // 只有 _post_replace 那一步才丢这两个
            return post;
        default:
            return false;
    }
}

/// 上游 `PUNCTUATION_REPLACEMENTS`（一次扫描替换，不重复触发）。
///
/// 返回替换后的**字节串**；空串表示不替换。
[[nodiscard]] std::string_view punctuation_replacement(char32_t codepoint) noexcept {
    switch (codepoint) {
        case 0xFF1A:
            return ",";  // ：
        case 0xFF1B:
            return ",";  // ；
        case 0xFF0C:
            return ",";  // ，
        case 0x3002:
            return ".";  // 。
        case 0xFF01:
            return "!";  // ！
        case 0xFF1F:
            return "?";  // ？
        case 0x3007:
            return ",";  // ·
        case 0x3001:
            return ",";  // 、
        case 0x0024:
            return ".";  // $
        case 0x002F:
            return ",";  // /
        case 0x2014:
            return "-";  // —
        case 0x007E:
            return "\xE2\x80\xA6";  // ~  → …
        case 0xFF5E:
            return "\xE2\x80\xA6";  // ～ → …
        default:
            return {};
    }
}

/// 上游 `_replace_punctuation`：标点映射 + 过滤掉非中文非标点的字符。
[[nodiscard]] std::string replace_punctuation(std::string_view text) {
    // 上游先做 text.replace("...", "…")
    std::string prepared;
    prepared.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        if (text.compare(i, 3, "...") == 0) {
            prepared += "\xE2\x80\xA6";  // …
            i += 3;
            continue;
        }
        prepared.push_back(text[i]);
        ++i;
    }

    // 标点映射（一次扫描）
    std::string mapped;
    mapped.reserve(prepared.size());
    std::size_t offset = 0;
    utf8::Char ch;
    while (utf8::decode_next(prepared, offset, ch)) {
        const auto replacement = punctuation_replacement(ch.codepoint);
        if (!replacement.empty()) {
            mapped += replacement;
        } else {
            mapped += utf8::encode(ch.codepoint);
        }
    }

    // 过滤：只保留中文（U+4E00–U+9FA5）与六个标点
    const auto allowed = [](char32_t codepoint) {
        if (codepoint >= 0x4E00 && codepoint <= 0x9FA5) {
            return true;
        }
        switch (codepoint) {
            case '!':
            case '?':
            case 0x2026:  // …
            case ',':
            case '.':
            case '-':
                return true;
            default:
                return false;
        }
    };

    std::string filtered;
    filtered.reserve(mapped.size());
    std::size_t offset2 = 0;
    utf8::Char ch2;
    while (utf8::decode_next(mapped, offset2, ch2)) {
        if (allowed(ch2.codepoint)) {
            filtered += utf8::encode(ch2.codepoint);
        }
    }
    return filtered;
}

/// 上游 `pattern_consecutive`：连续相同标点折叠成一个。
///
/// 实现见 text/punctuation.hpp —— **必须按码点**折叠。
/// 按字节折叠会把"四"（E5 9B 9B）吃掉后两个字节，输出成乱码。
using detail::collapse_punctuation;

/// 上游 `_post_replace`：数学运算符、希腊字母、带圈数字，以及最后的字符过滤。
[[nodiscard]] std::string post_replace(std::string_view text) {
    std::string out(text);

    // 上游 _post_replace 的第一句："/" → "每"（"每/天" → "每每天"）
    {
        std::size_t pos = 0;
        while ((pos = out.find('/', pos)) != std::string::npos) {
            out.replace(pos, 1, "每");
            pos += 3;  // "每" 是 3 字节
        }
    }

    // 带圈数字
    static constexpr auto kCircled = std::to_array({
        Pair{"①", "一"},
        Pair{"②", "二"},
        Pair{"③", "三"},
        Pair{"④", "四"},
        Pair{"⑤", "五"},
        Pair{"⑥", "六"},
        Pair{"⑦", "七"},
        Pair{"⑧", "八"},
        Pair{"⑨", "九"},
        Pair{"⑩", "十"},
    });
    for (const auto& [from, to] : kCircled) {
        std::size_t pos = 0;
        while ((pos = out.find(from, pos)) != std::string::npos) {
            out.replace(pos, from.size(), to);
            pos += to.size();
        }
    }

    // 希腊字母
    static constexpr auto kGreek = std::to_array({
        Pair{"α", "阿尔法"}, Pair{"β", "贝塔"},   Pair{"γ", "伽玛"},     Pair{"Γ", "伽玛"},
        Pair{"δ", "德尔塔"}, Pair{"Δ", "德尔塔"}, Pair{"ε", "艾普西龙"}, Pair{"ζ", "捷塔"},
        Pair{"η", "依塔"},   Pair{"θ", "西塔"},   Pair{"Θ", "西塔"},     Pair{"ι", "艾欧塔"},
        Pair{"κ", "喀帕"},   Pair{"λ", "拉姆达"}, Pair{"Λ", "拉姆达"},   Pair{"μ", "缪"},
        Pair{"ν", "拗"},     Pair{"ξ", "克西"},   Pair{"Ξ", "克西"},     Pair{"ο", "欧米克伦"},
        Pair{"π", "派"},     Pair{"Π", "派"},     Pair{"ρ", "肉"},       Pair{"ς", "西格玛"},
        Pair{"Σ", "西格玛"}, Pair{"σ", "西格玛"}, Pair{"τ", "套"},       Pair{"υ", "宇普西龙"},
        Pair{"φ", "服艾"},   Pair{"Φ", "服艾"},   Pair{"χ", "器"},       Pair{"ψ", "普赛"},
        Pair{"Ψ", "普赛"},   Pair{"ω", "欧米伽"}, Pair{"Ω", "欧米伽"},
    });
    for (const auto& [from, to] : kGreek) {
        std::size_t pos = 0;
        while ((pos = out.find(from, pos)) != std::string::npos) {
            out.replace(pos, from.size(), to);
            pos += to.size();
        }
    }

    // 兜底数学运算
    static constexpr auto kMath = std::to_array({
        Pair{"+", "加"},
        Pair{"-", "减"},
        Pair{literals::kTimes, "乘"},
        Pair{literals::kDivide, "除"},
        Pair{"=", "等"},
    });
    for (const auto& [from, to] : kMath) {
        std::size_t pos = 0;
        while ((pos = out.find(from, pos)) != std::string::npos) {
            out.replace(pos, from.size(), to);
            pos += to.size();
        }
    }

    // 上游最后的字符过滤（比 _split 那次多一个 "-" 和 "="）
    std::string filtered;
    filtered.reserve(out.size());
    std::size_t offset = 0;
    utf8::Char ch;
    while (utf8::decode_next(out, offset, ch)) {
        if (!is_dropped_codepoint(ch.codepoint, /*post=*/true)) {
            filtered += utf8::encode(ch.codepoint);
        }
    }
    return filtered;
}

/// 上游 `_split`：按句末标点切句。
[[nodiscard]] std::vector<std::string> split_sentences(std::string_view text) {
    // 上游：text.replace(" ", "") —— 只删半角空格
    std::string prepared;
    prepared.reserve(text.size());
    for (const char c : text) {
        if (c != ' ') {
            prepared.push_back(c);
        }
    }
    // 上游：过滤 [——《》【】<>{}()（）#&@"”^_|\]
    std::string filtered;
    filtered.reserve(prepared.size());
    {
        std::size_t offset = 0;
        utf8::Char ch;
        while (utf8::decode_next(prepared, offset, ch)) {
            if (!is_dropped_codepoint(ch.codepoint, /*post=*/false)) {
                filtered += utf8::encode(ch.codepoint);
            }
        }
    }

    // 句末标点：：、，；。？！,;?!
    // 上游的正则是**按字符**匹配的，所以这里按码点判断。
    // 上游的 SENTENCE_SPLITOR 是 `([：、，；。？！,;?!][”’]?)` ——
    // 冒号是**全角**的 "："；ASCII 的 ":" 不在列表里。
    // （把 ASCII 冒号也算进去会把 "15:30:45" 从中间切开。）
    const auto is_terminator = [](char32_t c) {
        switch (c) {
            case 0xFF1A:  // ：
            case 0x3001:  // 、
            case U',':
            case U';':
            case 0x3002:  // 。
            case U'?':
            case 0xFF1F:  // ？
            case U'!':
            case 0xFF01:  // ！
                return true;
            default:
                return false;
        }
    };

    std::vector<std::string> sentences;
    std::string current;
    std::size_t i = 0;
    while (i < filtered.size()) {
        const char32_t c = codepoint_at(filtered, i);
        if (is_terminator(c)) {
            current += utf8::encode(c);
            i += utf8::encoded_length(c);
            // 可选的右引号
            const char32_t quote = codepoint_at(filtered, i);
            if (quote == 0x201D || quote == 0x2019) {
                current += utf8::encode(quote);
                i += utf8::encoded_length(quote);
            }
            sentences.push_back(current);
            current.clear();
            continue;
        }
        current += utf8::encode(c);
        i += utf8::encoded_length(c);
    }
    if (!current.empty()) {
        sentences.push_back(current);
    }

    // 上游：strip 后按 \n+ 切（空的丢掉）
    std::vector<std::string> result;
    for (auto& sentence : sentences) {
        std::size_t begin = 0;
        std::size_t end = sentence.size();
        while (begin < end && std::isspace(static_cast<unsigned char>(sentence[begin]))) {
            ++begin;
        }
        while (end > begin && std::isspace(static_cast<unsigned char>(sentence[end - 1]))) {
            --end;
        }
        if (end > begin) {
            result.push_back(sentence.substr(begin, end - begin));
        }
    }
    return result;
}

/// 上游 `normalize_sentence`。
[[nodiscard]] std::string normalize_sentence(std::string_view sentence) {
    std::string s = traditional_to_simplified(sentence);

    // 全角 → 半角。
    //
    // 注意只转**字母 / 数字 / 空格**三种 —— 上游的 F2H_PUNCTUATIONS
    // 定义了但**没用**。全角标点（"，" "！"）是由后面 _replace_punctuation
    // 的替换表处理的，路径不一样，结果可能不同。
    s = to_halfwidth_letters(s);
    s = to_halfwidth_digits(s);
    s = to_halfwidth_space(s);

    s = replace_date(s);
    s = replace_date2(s);
    s = replace_time(s);
    s = replace_to_range(s);
    s = replace_temperature(s);
    s = replace_measure(s);

    // 上游是 while 循环：`1+2+3` 要替换成 "1加2加3" 需要迭代
    // （第一次替换出 "1加2"，剩下 "+3" 再替换一轮）
    for (int guard = 0; guard < 64; ++guard) {
        const std::string next = replace_asmd(s);
        if (next == s) {
            break;
        }
        s = std::move(next);
    }

    s = replace_power(s);
    s = replace_frac(s);
    s = replace_percentage(s);
    s = replace_phone(s);
    s = replace_range(s);
    s = replace_negative_num(s);
    s = replace_version_num(s);
    s = replace_number(s, /*require_fraction=*/true);
    s = replace_positive_quantifier(s);
    s = replace_default_num(s);
    s = replace_number(s, /*require_fraction=*/false);
    s = post_replace(s);
    return s;
}

}  // namespace

std::string normalize_chinese(std::string_view text) {
    const auto sentences = split_sentences(text);
    std::string joined;
    for (const auto& sentence : sentences) {
        joined += replace_punctuation(normalize_sentence(sentence));
    }
    return collapse_punctuation(joined);
}

}  // namespace genie::text
