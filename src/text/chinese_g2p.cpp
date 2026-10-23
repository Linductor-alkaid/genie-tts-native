// src/text/chinese_g2p.cpp
//
// 拼音 → 音素的规则实现。
//
// 这里的每一条规则都对应 pypinyin 的一个函数（因为上游用的就是
// pypinyin 的 to_initials / to_finals_tone3），并且已对 41,923 个汉字
// 全量比对验证 —— 见 tools/g2p/verify_pinyin_rules.py。

#include "genie/pinyin.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <map>
#include <mutex>
#include <string>

#include "genie/pinyin_table.hpp"

namespace genie::pinyin {
namespace {

// "ü" 的 UTF-8 编码。pypinyin 内部用带调的字符，本实现统一用不带调的 "ü"。
constexpr std::string_view kUmlaut = "\xC3\xBC";

// ---------------------------------------------------------------------------
// 声母表。顺序即匹配顺序 —— "zh" 必须排在 "z" 前面，否则 "zhong" 会被切成
// "z" + "hong"。这与 pypinyin 的 _INITIALS 顺序一致。
// ---------------------------------------------------------------------------
constexpr std::array<std::string_view, 21> kInitialsStrict = {
    "b", "p", "m", "f",  "d",  "t",  "n", "l", "g", "k", "h",
    "j", "q", "x", "zh", "ch", "sh", "r", "z", "c", "s",
};

// 非严格模式：把 y / w 也当声母（pypinyin 的 _INITIALS_NOT_STRICT）。
constexpr std::array<std::string_view, 23> kInitialsLoose = {
    "b", "p", "m",  "f",  "d",  "t", "n", "l", "g", "k", "h", "j",
    "q", "x", "zh", "ch", "sh", "r", "z", "c", "s", "y", "w",
};

// 韵母表（pypinyin 的 _FINALS）。
constexpr std::array<std::string_view, 38> kFinals = {
    "i",
    "u",
    "\xC3\xBC",
    "a",
    "ia",
    "ua",
    "o",
    "uo",
    "e",
    "ie",
    "\xC3\xBC\x65",
    "ai",
    "uai",
    "ei",
    "uei",
    "ao",
    "iao",
    "ou",
    "iou",
    "an",
    "ian",
    "uan",
    "\xC3\xBC\x61\x6E",
    "en",
    "in",
    "uen",
    "\xC3\xBC\x6E",
    "ang",
    "iang",
    "uang",
    "eng",
    "ing",
    "ueng",
    "ong",
    "iong",
    "er",
    "\xC3\xAA",
};

[[nodiscard]] bool is_final(std::string_view value) noexcept {
    for (const auto final : kFinals) {
        if (final == value) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool is_lower_alpha(std::string_view value) noexcept {
    if (value.empty()) {
        return false;
    }
    return std::all_of(value.begin(), value.end(), [](char ch) { return ch >= 'a' && ch <= 'z'; });
}

/// 声母前缀匹配。pypinyin 的 get_initials()。
[[nodiscard]] std::string get_initials(std::string_view pinyin, bool strict) noexcept {
    const auto match = [pinyin](std::string_view initial) -> std::string {
        if (pinyin.size() >= initial.size() && pinyin.compare(0, initial.size(), initial) == 0) {
            return std::string(initial);
        }
        return {};
    };
    if (strict) {
        for (const auto initial : kInitialsStrict) {
            if (const auto found = match(initial); !found.empty()) {
                return found;
            }
        }
    } else {
        for (const auto initial : kInitialsLoose) {
            if (const auto found = match(initial); !found.empty()) {
                return found;
            }
        }
    }
    return {};
}

/// 把 value 里的所有 from 替换成 to。
void replace_all(std::string& value, std::string_view from, std::string_view to) {
    if (from.empty()) {
        return;
    }
    std::size_t pos = 0;
    while ((pos = value.find(from, pos)) != std::string::npos) {
        value.replace(pos, from.size(), to);
        pos += to.size();
    }
}

/// 零声母还原。pypinyin 的 convert_zero_consonant()。
///
///   y 行：yu→ü、yi→i、y→i
///   w 行：wu→u、w→u
[[nodiscard]] std::string convert_zero_consonant(std::string pinyin) {
    const std::string raw = pinyin;

    if (!pinyin.empty() && pinyin.front() == 'y') {
        const std::string no_y = pinyin.substr(1);
        const std::string first = no_y.substr(0, 1);
        if (first == "u" || first == kUmlaut) {
            // UV_MAP['u'] == 'ü'
            pinyin = std::string(kUmlaut) + pinyin.substr(2);
        } else if (first == "i") {
            pinyin = no_y;
        } else {
            pinyin = "i" + no_y;
        }
    }

    // 注意：pypinyin 这里判断的是**原始**拼音的首字符。
    if (!raw.empty() && raw.front() == 'w') {
        const std::string first = raw.substr(1, 1);
        if (first == "u" || first == kUmlaut) {
            pinyin = raw.substr(1);
        } else {
            pinyin = "u" + raw.substr(1);
        }
    }

    // 确保不会出现韵母表中不存在的韵母。
    if (!is_final(pinyin)) {
        pinyin = raw;
    }
    return pinyin;
}

/// ü 还原。pypinyin 的 convert_uv()：^(j|q|x)(u|ü)(.*)$ → g1 + ü + g3。
[[nodiscard]] std::string convert_uv(std::string pinyin) {
    if (pinyin.size() < 2) {
        return pinyin;
    }
    const char lead = pinyin.front();
    if (lead != 'j' && lead != 'q' && lead != 'x') {
        return pinyin;
    }
    const std::string rest = pinyin.substr(1);
    if (rest.size() >= 2 && rest.compare(0, 2, kUmlaut) == 0) {
        return pinyin.substr(0, 1) + std::string(kUmlaut) + rest.substr(2);
    }
    if (rest.front() == 'u') {
        return pinyin.substr(0, 1) + std::string(kUmlaut) + rest.substr(1);
    }
    return pinyin;
}

/// iou 还原。pypinyin 的 convert_iou()：^([a-z]+)(iu)$ → g1 + "iou"。
[[nodiscard]] std::string convert_iou(const std::string& pinyin) {
    if (pinyin.size() < 3 || pinyin.compare(pinyin.size() - 2, 2, "iu") != 0) {
        return pinyin;
    }
    const std::string head = pinyin.substr(0, pinyin.size() - 2);
    if (!is_lower_alpha(head)) {
        return pinyin;
    }
    return head + "iou";
}

/// uei 还原。pypinyin 的 convert_uei()：^([a-z]+)(ui)$ → g1 + "uei"。
[[nodiscard]] std::string convert_uei(const std::string& pinyin) {
    if (pinyin.size() < 3 || pinyin.compare(pinyin.size() - 2, 2, "ui") != 0) {
        return pinyin;
    }
    const std::string head = pinyin.substr(0, pinyin.size() - 2);
    if (!is_lower_alpha(head)) {
        return pinyin;
    }
    return head + "uei";
}

/// uen 还原。pypinyin 的 convert_uen()：^([a-z]+)(un)$ → g1 + "uen"。
[[nodiscard]] std::string convert_uen(const std::string& pinyin) {
    if (pinyin.size() < 3 || pinyin.compare(pinyin.size() - 2, 2, "un") != 0) {
        return pinyin;
    }
    const std::string head = pinyin.substr(0, pinyin.size() - 2);
    if (!is_lower_alpha(head)) {
        return pinyin;
    }
    return head + "uen";
}

/// 韵母切分。pypinyin 的 get_finals()（严格模式）。
///
/// @return 韵母（已还原为 ü 形式）；无法切分时返回空串。
[[nodiscard]] std::string get_finals(std::string pinyin) {
    pinyin = convert_zero_consonant(std::move(pinyin));
    pinyin = convert_uv(std::move(pinyin));
    pinyin = convert_iou(pinyin);
    pinyin = convert_uei(pinyin);
    pinyin = convert_uen(pinyin);

    std::string initials = get_initials(pinyin, /*strict=*/true);
    std::string finals = pinyin.substr(initials.size());
    if (!is_final(finals)) {
        initials = get_initials(pinyin, /*strict=*/false);
        finals = pinyin.substr(initials.size());
        if (!is_final(finals)) {
            return {};
        }
    }
    // pypinyin 的 _fix_v_u：韵母里的 "ü" 最终写回 "v"（符号表用 v 表示 ü）。
    replace_all(finals, kUmlaut, "v");
    return finals;
}

/// 拼音词典：汉字 → tone3 拼音。
[[nodiscard]] const std::map<std::string, std::string, std::less<>>& pinyin_table() {
    static const std::map<std::string, std::string, std::less<>> kTable = [] {
        std::map<std::string, std::string, std::less<>> table;
        std::string_view text = detail::kPinyinTable;
        while (!text.empty()) {
            const auto newline = text.find('\n');
            const std::string_view line =
                text.substr(0, newline == std::string_view::npos ? text.size() : newline);
            const auto space = line.find(' ');
            if (space != std::string_view::npos) {
                table.emplace(std::string(line.substr(0, space)),
                              std::string(line.substr(space + 1)));
            }
            if (newline == std::string_view::npos) {
                break;
            }
            text = text.substr(newline + 1);
        }
        return table;
    }();
    return kTable;
}

}  // namespace

Syllable split(std::string_view tone3) noexcept {
    Syllable result;
    if (tone3.empty()) {
        return result;
    }

    // 分离声调。tone3 的声调数字在末尾；没有数字视为轻声（5）。
    std::string base;
    std::string tone = "5";
    const char last = tone3.back();
    if (last >= '1' && last <= '5') {
        base.assign(tone3.substr(0, tone3.size() - 1));
        tone.assign(1, last);
    } else {
        base.assign(tone3);
    }

    // 'v' → "ü"（与 pypinyin 的 to_finals 内部处理一致）。
    std::string normalized;
    normalized.reserve(base.size() + 4);
    for (const char ch : base) {
        if (ch == 'v') {
            normalized.append(kUmlaut);
        } else {
            normalized.push_back(ch);
        }
    }

    // 声母：直接对原始拼音做前缀匹配（pypinyin 的 to_initials 不做零声母还原）。
    result.initial = get_initials(base, /*strict=*/true);
    // 韵母：内部会做零声母还原。
    const std::string final = get_finals(normalized);
    if (final.empty()) {
        result.initial.clear();
        return result;
    }
    result.final_with_tone = final + tone;
    return result;
}

std::optional<std::string> lookup(std::string_view utf8_char) {
    const auto& table = pinyin_table();
    const auto it = table.find(utf8_char);
    if (it == table.end()) {
        return std::nullopt;
    }
    return it->second;
}

std::vector<std::string> syllable_to_phones(const Syllable& syllable) {
    const std::string& initial = syllable.initial;
    const std::string& final_with_tone = syllable.final_with_tone;
    if (final_with_tone.empty()) {
        return {};
    }

    const std::string tone = final_with_tone.substr(final_with_tone.size() - 1);
    const std::string final = final_with_tone.substr(0, final_with_tone.size() - 1);

    if (!initial.empty()) {
        // 有声母：韵母收缩成书写形式（opencpop 词表里的那一套）。
        std::string contracted = final;
        if (contracted == "uei") {
            contracted = "ui";
        } else if (contracted == "iou") {
            contracted = "iu";
        } else if (contracted == "uen") {
            contracted = "un";
        }
        return {initial, contracted + tone};
    }

    // 零声母：还原书写形式（上游的 pinyin_rep_map / single_rep_map）。
    std::string written;
    if (final == "ing") {
        written = "ying";
    } else if (final == "i") {
        written = "yi";
    } else if (final == "in") {
        written = "yin";
    } else if (final == "u") {
        written = "wu";
    } else if (!final.empty()) {
        switch (final.front()) {
            case 'v':
                written = "yu" + final.substr(1);
                break;
            case 'e':
                written = "e" + final.substr(1);
                break;
            case 'i':
                written = "y" + final.substr(1);
                break;
            case 'u':
                written = "w" + final.substr(1);
                break;
            default:
                written = final;
                break;
        }
    }

    if (written.empty()) {
        return {};
    }
    if (written.front() == 'y') {
        return {"y", written.substr(1) + tone};
    }
    if (written.front() == 'w') {
        return {"w", written.substr(1) + tone};
    }
    // 真正的元音开头：opencpop 词表里写作 "<元音> <整个音节>"，
    // 例如 "a" → "a a"、"ao" → "a ao"。
    // "er" 是特例 —— opencpop 里写作 "er er"。
    if (written == "er") {
        return {"er", "er" + tone};
    }
    return {written.substr(0, 1), written + tone};
}

}  // namespace genie::pinyin
