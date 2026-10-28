// src/text/korean/korean.cpp —— 韩语 G2P 的主流程
//
// 上游 `G2P/Korean/KoreanG2P.py` + g2pk2（g2pk2 的规则表数据见
// src/text/korean/tables.hpp）。
//
// 与上游的偏差（**只有一处实质偏差**）：
//   g2pk2 的第 3 步是 mecab-ko 形态标注（给 /J /E /P /B 打标记）。
//   mecab 是 C++ 形态分析器 + 100 MB 词典，本移植不带它 —— 标注不做，
//   依赖标注的四条规则（rieulgiyeok / rieulbieub / verb_nieun /
//   modifying_rieul）因此不会触发。详见 docs/upstream-audit.md §18。

#include "text/korean/korean.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

#include "genie/symbols.hpp"
#include "genie/utf8.hpp"
#include "text/korean/jamo.hpp"
#include "text/korean/tables.hpp"

namespace genie::text::korean {
namespace {

// ---------------------------------------------------------------------------
// 通用小工具
// ---------------------------------------------------------------------------

/// 把 text 按**码点**切成一段一段（避免按字节切多字节字符 ✓）。
[[nodiscard]] std::vector<std::string_view> split_codepoints(std::string_view text) {
    std::vector<std::string_view> out;
    std::size_t offset = 0;
    utf8::Char ch{};
    while (utf8::decode_next(text, offset, ch)) {
        out.emplace_back(text.substr(ch.offset, ch.length));
    }
    return out;
}

/// 字面量替换（全部出现都换 ✓）。
void replace_all(std::string& text, std::string_view from, std::string_view to) {
    if (from.empty()) {
        return;
    }
    std::size_t pos = 0;
    while ((pos = text.find(from, pos)) != std::string::npos) {
        text.replace(pos, from.size(), to);
        pos += to.size();
    }
}

/// "模式 → 替换" 的字面量规则。
struct LiteralRule {
    std::string_view from;
    std::string_view to;
};

/// 依次应用字面量规则 ✓（顺序与上游一致 ✓）。
void apply_literal_rules(std::string& text, const LiteralRule* rules, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        replace_all(text, rules[i].from, rules[i].to);
    }
}

/// 是不是"词字符"（Python 的 `\w`：字母、数字、下划线 ✓）。
[[nodiscard]] bool is_word_codepoint(std::uint32_t cp) noexcept {
    if (cp < 0x80) {
        return (cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z') || (cp >= '0' && cp <= '9') ||
               cp == '_';
    }
    // 非 ASCII：韩文、日文、汉字、字母都算词字符；标点符号不算。
    if (cp >= 0x2000 && cp <= 0x206F) {
        return false;  // 通用标点
    }
    if (cp >= 0x20A0 && cp <= 0x20CF) {
        return false;  // 货币符号
    }
    if (cp >= 0x3000 && cp <= 0x303F) {
        return false;  // CJK 标点
    }
    if (cp >= 0xFF01 && cp <= 0xFF20) {
        return false;  // 全角标点
    }
    return true;
}

// ---------------------------------------------------------------------------
// 上游 KoreanG2P 的辅助函数
// ---------------------------------------------------------------------------

/// 上游 `_latin_to_hangul`：逐个拉丁字母换成韩文读法（大小写不敏感 ✓）。
[[nodiscard]] std::string latin_to_hangul(std::string_view text) {
    struct Entry {
        char latin;
        std::string_view hangul;
    };
    static constexpr std::array kTable = std::to_array<Entry>({
        {'a', "에이"}, {'b', "비"},   {'c', "시"},     {'d', "디"},   {'e', "이"},
        {'f', "에프"}, {'g', "지"},   {'h', "에이치"}, {'i', "아이"}, {'j', "제이"},
        {'k', "케이"}, {'l', "엘"},   {'m', "엠"},     {'n', "엔"},   {'o', "오"},
        {'p', "피"},   {'q', "큐"},   {'r', "아르"},   {'s', "에스"}, {'t', "티"},
        {'u', "유"},   {'v', "브이"}, {'w', "더블유"}, {'x', "엑스"}, {'y', "와이"},
        {'z', "제트"},
    });
    std::string out;
    out.reserve(text.size() * 3);
    std::size_t offset = 0;
    utf8::Char ch{};
    while (utf8::decode_next(text, offset, ch)) {
        if (ch.codepoint < 0x80) {
            const char lower = static_cast<char>(std::tolower(static_cast<int>(ch.codepoint)));
            bool replaced = false;
            for (const auto& entry : kTable) {
                if (entry.latin == lower) {
                    out += entry.hangul;
                    replaced = true;
                    break;
                }
            }
            if (replaced) {
                continue;
            }
        }
        out.append(text.substr(ch.offset, ch.length));
    }
    return out;
}

/// 上游 `_hangul_number`：数字串 → 韩文（汉字数词 / 固有数词）。
[[nodiscard]] std::string hangul_number(std::string_view number, bool sino) {
    std::string num;
    num.reserve(number.size());
    for (const char c : number) {
        if (c != ',') {
            num.push_back(c);
        }
    }
    if (num == "0") {
        return "영";
    }
    if (!sino && num == "20") {
        return "스무";
    }
    static constexpr std::string_view kDigits = "123456789";
    static constexpr std::string_view kNames = "일이삼사오육칠팔구";
    static constexpr std::array kModifiers = std::to_array<std::string_view>(
        {"한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉"});
    static constexpr std::array kDecimals = std::to_array<std::string_view>(
        {"열", "스물", "서른", "마흔", "쉰", "예순", "일흔", "여든", "아흔"});

    // 上游用列表拼名字，这里照抄（"0" 的处理要看最后三段是不是全空 ✓）。
    std::vector<std::string> spelled;
    const std::size_t length = num.size();
    for (std::size_t k = 0; k < length; ++k) {
        const char digit = num[k];
        const std::size_t i = length - k - 1;
        std::string name;
        if (sino) {
            if (i == 0) {
                const std::size_t index = kDigits.find(digit);
                if (index != std::string_view::npos) {
                    name = std::string(kNames.substr(index * 3, 3));
                }
            } else if (i == 1) {
                const std::size_t index = kDigits.find(digit);
                if (index != std::string_view::npos) {
                    name = std::string(kNames.substr(index * 3, 3));
                }
                name += "십";
                if (name == "일십") {
                    name = "십";
                }
            }
        } else {
            if (i == 0) {
                const std::size_t index = kDigits.find(digit);
                if (index != std::string_view::npos && index < kModifiers.size()) {
                    name = kModifiers[index];
                }
            } else if (i == 1) {
                const std::size_t index = kDigits.find(digit);
                if (index != std::string_view::npos && index < kDecimals.size()) {
                    name = kDecimals[index];
                }
            }
        }
        if (digit == '0') {
            if (i % 4 == 0) {
                const std::size_t take = std::min<std::size_t>(3, spelled.size());
                bool all_empty = true;
                for (std::size_t j = spelled.size() - take; j < spelled.size(); ++j) {
                    if (!spelled[j].empty()) {
                        all_empty = false;
                        break;
                    }
                }
                if (all_empty) {
                    spelled.emplace_back();
                    continue;
                }
            } else {
                spelled.emplace_back();
                continue;
            }
        }
        if (i >= 2) {
            const std::size_t index = kDigits.find(digit);
            const std::string base = index != std::string_view::npos
                                         ? std::string(kNames.substr(index * 3, 3))
                                         : std::string();
            switch (i) {
                case 2:
                    name = base + "백";
                    if (name == "일백") {
                        name = "백";
                    }
                    break;
                case 3:
                    name = base + "천";
                    if (name == "일천") {
                        name = "천";
                    }
                    break;
                case 4:
                    name = base + "만";
                    if (name == "일만") {
                        name = "만";
                    }
                    break;
                case 5:
                    name = base + "십";
                    if (name == "일십") {
                        name = "십";
                    }
                    break;
                case 6:
                    name = base + "백";
                    if (name == "일백") {
                        name = "백";
                    }
                    break;
                case 7:
                    name = base + "천";
                    if (name == "일천") {
                        name = "천";
                    }
                    break;
                case 8:
                    name = base + "억";
                    break;
                case 9:
                    name = base + "십";
                    break;
                case 10:
                    name = base + "백";
                    break;
                case 11:
                    name = base + "천";
                    break;
                case 12:
                    name = base + "조";
                    break;
                case 13:
                    name = base + "십";
                    break;
                case 14:
                    name = base + "백";
                    break;
                case 15:
                    name = base + "천";
                    break;
                default:
                    break;
            }
        }
        spelled.push_back(std::move(name));
    }
    std::string out;
    for (const auto& part : spelled) {
        out += part;
    }
    return out;
}

/// 上游 `_number_to_hangul`：数字 → 韩文。
///
/// 后面跟着量词的用固有数词（"한 개"），其余用汉字数词（"일"）。
/// 上游用 `re.findall(r"(\d[\d,]*)([\uac00-\ud71f]+)")` 找"数字 + 韩文"，
/// 这里按码点扫描 ✓（不引正则 ✓）。
[[nodiscard]] std::string number_to_hangul(std::string_view text) {
    static constexpr std::string_view kClassifiers =
        "군데 권 개 그루 닢 대 두 마리 모 모금 뭇 발 발짝 방 번 벌 보루 살 수 술 시 쌈 움큼 정 짝 "
        "채 척 첩 축 켤레 톨 통";

    const auto pieces = split_codepoints(text);
    std::string out;
    std::size_t i = 0;
    while (i < pieces.size()) {
        // 先看是不是"数字 + 韩文" 的组合 ✓。
        std::size_t digits_end = i;
        bool has_digit = false;
        while (digits_end < pieces.size() &&
               (pieces[digits_end] == "," ||
                (pieces[digits_end].size() == 1 && pieces[digits_end][0] >= '0' &&
                 pieces[digits_end][0] <= '9'))) {
            if (pieces[digits_end] != ",") {
                has_digit = true;
            }
            ++digits_end;
        }
        if (has_digit && digits_end < pieces.size()) {
            // 后面连续的韩文（兼容字母 U+AC00–U+D71F ✓）。
            std::size_t hangul_end = digits_end;
            while (hangul_end < pieces.size()) {
                std::size_t offset = 0;
                utf8::Char ch{};
                if (!utf8::decode_next(pieces[hangul_end], offset, ch)) {
                    break;
                }
                if (ch.codepoint < 0xAC00 || ch.codepoint > 0xD71F) {
                    break;
                }
                ++hangul_end;
            }
            if (hangul_end > digits_end) {
                std::string number;
                for (std::size_t k = i; k < digits_end; ++k) {
                    number += pieces[k];
                }
                std::string classifier;
                for (std::size_t k = digits_end; k < hangul_end; ++k) {
                    classifier += pieces[k];
                }
                // 上游的判定：`classifier[:2] in _korean_classifiers or
                // classifier[0] in _korean_classifiers` —— Python 的 `in` 对
                // 字符串是**子串**判断 ✓，而且切片按字符算（不是字节 ✓）。
                const auto classifier_pieces = split_codepoints(classifier);
                const std::string first_one =
                    classifier_pieces.empty() ? std::string() : std::string(classifier_pieces[0]);
                std::string first_two = first_one;
                if (classifier_pieces.size() >= 2) {
                    first_two += std::string(classifier_pieces[1]);
                }
                const bool native = kClassifiers.find(first_two) != std::string_view::npos ||
                                    kClassifiers.find(first_one) != std::string_view::npos;
                out += hangul_number(number, !native);
                out += classifier;
                i = hangul_end;
                continue;
            }
        }
        out += pieces[i];
        ++i;
    }
    // 剩下的数字逐位读 ✓。
    static constexpr std::string_view kDigitNames = "영일이삼사오육칠팔구";
    std::string result;
    result.reserve(out.size() * 2);
    for (const char c : out) {
        if (c >= '0' && c <= '9') {
            result += std::string(kDigitNames.substr(static_cast<std::size_t>(c - '0') * 3, 3));
        } else {
            result.push_back(c);
        }
    }
    return result;
}

/// 上游 `_divide_hangul`：组合字母 → 兼容字母，再把复合元音拆开。
[[nodiscard]] std::string divide_hangul(std::string_view text) {
    std::string out = j2hcj(h2j(text));
    static constexpr std::array kRules = std::to_array<LiteralRule>({
        {"ㅘ", "ㅗㅏ"},
        {"ㅙ", "ㅗㅐ"},
        {"ㅚ", "ㅗㅣ"},
        {"ㅝ", "ㅜㅓ"},
        {"ㅞ", "ㅜㅔ"},
        {"ㅟ", "ㅜㅣ"},
        {"ㅢ", "ㅡㅣ"},
        {"ㅑ", "ㅣㅏ"},
        {"ㅒ", "ㅣㅐ"},
        {"ㅕ", "ㅣㅓ"},
        {"ㅖ", "ㅣㅔ"},
        {"ㅛ", "ㅣㅗ"},
        {"ㅠ", "ㅣㅜ"},
    });
    apply_literal_rules(out, kRules.data(), kRules.size());
    return out;
}

/// 上游 `_fix_g2pk2_error`：修正 g2pk2 的 "ㅇㅡㄹ ㄹ" → "ㅇㅡㄹ ㄴ"。
[[nodiscard]] std::string fix_g2pk2_error(std::string_view text) {
    const auto pieces = split_codepoints(text);
    std::string out;
    std::size_t i = 0;
    while (i + 4 < pieces.size()) {
        const std::string triple =
            std::string(pieces[i]) + std::string(pieces[i + 1]) + std::string(pieces[i + 2]);
        if ((triple == "ㅇㅡㄹ" || triple == "ㄹㅡㄹ") && pieces[i + 3] == " " &&
            pieces[i + 4] == "ㄹ") {
            out += triple;
            out += " ";
            out += "ㄴ";
            i += 5;
            continue;
        }
        out += pieces[i];
        ++i;
    }
    for (; i < pieces.size(); ++i) {
        out += pieces[i];
    }
    return out;
}

/// 上游 `_post_replace_phoneme`：单字符的替换 + 符号表核对。
[[nodiscard]] std::string post_replace_phoneme(std::string_view ph) {
    static constexpr std::array kRepMap = std::to_array<LiteralRule>({
        {"：", ","},
        {"；", ","},
        {"，", ","},
        {"。", "."},
        {"！", "!"},
        {"？", "?"},
        {"\n", "."},
        {"·", ","},
        {"、", ","},
        {" ", "空"},
    });
    std::string value(ph);
    for (const auto& rule : kRepMap) {
        if (value == rule.from) {
            value = rule.to;
            break;
        }
    }
    if (genie::symbols::contains(value)) {
        return value;
    }
    return "停";
}

// ---------------------------------------------------------------------------
// g2pk2 的流水线（不含 mecab 形态标注那一步）
// ---------------------------------------------------------------------------

/// 第 1 步：惯用语表（上游逐条做正则替换 —— 这些模式都是字面量 ✓）。
[[nodiscard]] std::string apply_idioms(std::string_view text) {
    std::string out(text);
    for (const auto& idiom : tables::kIdioms) {
        replace_all(out, idiom.from, idiom.to);
    }
    return out;
}

/// 第 6 步：特殊规则（上游 g2pk2/special.py）。
///
/// 注意上游的文本在这一步是 **h2j 之后的组合字母**（U+11xx），
/// 不是兼容字母 —— 规则里的 "ᄌᄍ쳐" 都是组合字母 ✓。
[[nodiscard]] std::string apply_special_rules(std::string_view text) {
    std::string out(text);

    // jyeo: ([ᄌᄍᄎ])ᅧ → \1ᅥ
    replace_all(out, "져", "저");
    replace_all(out, "쪄", "쩌");
    replace_all(out, "쳐", "처");

    // consonant_ui: ([ᄀᄁᄂᄃᄄᄅᄆᄇᄈᄉᄊᄌᄍᄎᄏᄐᄑᄒ])ᅴ → \1ᅵ
    {
        static constexpr std::string_view kConsonants = "ᄀᄁᄂᄃᄄᄅᄆᄇᄈᄉᄊᄌᄍᄎᄏᄐᄑᄒ";
        for (std::size_t i = 0; i + 3 <= kConsonants.size(); i += 3) {
            const std::string consonant(kConsonants.substr(i, 3));
            replace_all(out, consonant + "ᅴ", consonant + "ᅵ");
        }
    }

    // jamo: 귿ᄋ → 그ቻ，으[ᆽᆾᇀᇂ]ᄋ → 으ቻ，으ᇿᄋ → 으ᄀ，읖ᄋ → 으ᄇ
    replace_all(out, "귿ᄋ", "그ቻ");
    replace_all(out, "읒ᄋ", "으ቻ");
    replace_all(out, "읓ᄋ", "으ቻ");
    replace_all(out, "읕ᄋ", "으ቻ");
    replace_all(out, "읗ᄋ", "으ቻ");
    replace_all(out, "으ᇿᄋ", "으ᄀ");
    replace_all(out, "읖ᄋ", "으ᄇ");

    // rieulgiyeok: ᆰ/P([ᄀᄁ]) → ᆯᄁ（依赖形态标注 ✓）
    replace_all(out, "ᆰ/Pᄀ", "ᆯᄁ");
    replace_all(out, "ᆰ/Pᄁ", "ᆯᄁ");

    // rieulbieub: ([ᆲᆴ])/Pᄀ → \1ᄁ …
    static constexpr std::array kRieulbieub = std::to_array<LiteralRule>({
        {"ᆲ/Pᄀ", "ᆲᄁ"},
        {"ᆲ/Pᄃ", "ᆲᄄ"},
        {"ᆲ/Pᄉ", "ᆲᄊ"},
        {"ᆲ/Pᄌ", "ᆲᄍ"},
        {"ᆴ/Pᄀ", "ᆴᄁ"},
        {"ᆴ/Pᄃ", "ᆴᄄ"},
        {"ᆴ/Pᄉ", "ᆴᄊ"},
        {"ᆴ/Pᄌ", "ᆴᄍ"},
    });
    apply_literal_rules(out, kRieulbieub.data(), kRieulbieub.size());

    // verb_nieun: ([ᆫᆷ])/Pᄀ → \1ᄁ …
    static constexpr std::array kVerbNieun = std::to_array<LiteralRule>({
        {"ᆫ/Pᄀ", "ᆫᄁ"},
        {"ᆫ/Pᄃ", "ᆫᄄ"},
        {"ᆫ/Pᄉ", "ᆫᄊ"},
        {"ᆫ/Pᄌ", "ᆫᄍ"},
        {"ᆷ/Pᄀ", "ᆷᄁ"},
        {"ᆷ/Pᄃ", "ᆷᄄ"},
        {"ᆷ/Pᄉ", "ᆷᄊ"},
        {"ᆷ/Pᄌ", "ᆷᄍ"},
        {"ᆬ/Pᄀ", "ᆫᄁ"},
        {"ᆬ/Pᄃ", "ᆫᄄ"},
        {"ᆬ/Pᄉ", "ᆫᄊ"},
        {"ᆬ/Pᄌ", "ᆫᄍ"},
        {"ᆱ/Pᄀ", "ᆷᄁ"},
        {"ᆱ/Pᄃ", "ᆷᄄ"},
        {"ᆱ/Pᄉ", "ᆷᄊ"},
        {"ᆱ/Pᄌ", "ᆷᄍ"},
    });
    apply_literal_rules(out, kVerbNieun.data(), kVerbNieun.size());

    // balb: (바)ᆲ($|[^ᄋᄒ]) → \1ᆸ\2
    //      (너)ᆲ([ᄌᄍ]ᅮ|[ᄃᄄ]ᅮ) → \1ᆸ\2
    {
        std::string result;
        std::size_t i = 0;
        while (i < out.size()) {
            const std::size_t hit = out.find("밟", i);
            if (hit == std::string::npos) {
                result += out.substr(i);
                break;
            }
            result += out.substr(i, hit - i);
            const std::size_t after = hit + 6;  // "바"(3) + "ᆲ"(3)
            if (after >= out.size()) {
                // 行尾：`$` 也匹配 ✓
                result += "밥";
                i = after;
                continue;
            }
            const unsigned char next = static_cast<unsigned char>(out[after]);
            if (next >= 0xE3) {
                // ᄋ（U+110B）或 ᄒ（U+1112）—— 不换 ✓
                result += "밟";
                i = after;
                continue;
            }
            result += "밥";
            result.push_back(out[after]);
            i = after + 1;
        }
        out = result;
    }
    {
        static constexpr std::array kAfter =
            std::to_array<std::string_view>({"주", "쭈", "두", "뚜"});
        std::string result;
        std::size_t i = 0;
        while (i < out.size()) {
            const std::size_t hit = out.find("넓", i);
            if (hit == std::string::npos) {
                result += out.substr(i);
                break;
            }
            result += out.substr(i, hit - i);
            const std::size_t after = hit + 6;
            bool matched = false;
            for (const auto& tail : kAfter) {
                if (out.compare(after, tail.size(), tail) == 0) {
                    result += "넙";
                    result += tail;
                    i = after + tail.size();
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                result += "넓";
                i = after;
            }
        }
        out = result;
    }

    // palatalize: ᆮᄋ([ᅵᅧ]) → ᄌ\1，ᇀᄋ([ᅵᅧ]) → ᄎ\1，ᆴᄋ([ᅵᅧ]) → ᆯᄎ\1
    //             ᆮᄒ([ᅵ]) → 치
    {
        static constexpr std::array kVowel = std::to_array<std::string_view>({"ᅵ", "ᅧ"});
        for (const auto& vowel : kVowel) {
            replace_all(out, "ᆮᄋ" + std::string(vowel), "ᄌ" + std::string(vowel));
            replace_all(out, "ᇀᄋ" + std::string(vowel), "ᄎ" + std::string(vowel));
            replace_all(out, "ᆴᄋ" + std::string(vowel), "ᆯᄎ" + std::string(vowel));
        }
        replace_all(out, "ᆮ히", "치");
    }

    // modifying_rieul: ᆯ/E ᄀ → ᆯ ᄁ …
    static constexpr std::array kModifying = std::to_array<LiteralRule>({
        {"ᆯ/E ᄀ", "ᆯ ᄁ"},
        {"ᆯ/E ᄃ", "ᆯ ᄄ"},
        {"ᆯ/E ᄇ", "ᆯ ᄈ"},
        {"ᆯ/E ᄉ", "ᆯ ᄊ"},
        {"ᆯ/E ᄌ", "ᆯ ᄍ"},
        {"ᆯ걸", "ᆯ껄"},
        {"ᆯ밖에", "ᆯ빠께"},
        {"ᆯ세라", "ᆯ쎄라"},
        {"ᆯ수록", "ᆯ쑤록"},
        {"ᆯ지라도", "ᆯ찌라도"},
        {"ᆯ지언정", "ᆯ찌언정"},
        {"ᆯ진대", "ᆯ찐대"},
    });
    apply_literal_rules(out, kModifying.data(), kModifying.size());

    // josa_ui / vowel_ui / ye：上游在 descriptive 模式下才改，默认只去掉标记 ✓
    replace_all(out, "/J", "");

    return out;
}

/// 第 7 步："받침 + 起首辅音" 的规则表。
///
/// 上游的正则形如 `ᆨ( ?)ᄂ`：coda + 一个**可选空格** + onset，
/// 替换里的 `\1` 指代那个可选空格。
/// 最后一列的 onset 是 `(\W|$)`：后面是非词字符或字符串结尾 ✓。
[[nodiscard]] std::string apply_regular_table(std::string_view text) {
    std::string out(text);
    for (const auto& rule : tables::kRegularRules) {
        const bool is_tail_rule = rule.onset == "(\\W|$)";
        // "( ?)X" → 可选空格 + X ✓
        std::string onset_char;
        if (!is_tail_rule) {
            onset_char = rule.onset.compare(0, 4, "( ?)") == 0 ? rule.onset.substr(4) : rule.onset;
        }
        std::string result;
        std::size_t i = 0;
        while (i < out.size()) {
            const std::size_t hit = out.find(rule.coda, i);
            if (hit == std::string::npos) {
                result += out.substr(i);
                break;
            }
            result += out.substr(i, hit - i);
            const std::size_t after = hit + rule.coda.size();
            std::string group1;  // 正则的分组 1 ✓
            std::size_t next = after;
            if (is_tail_rule) {
                // (\W|$)：后面必须是非词字符或结尾 ✓
                if (after >= out.size()) {
                    // $ —— 匹配成功（分组为空 ✓）
                } else {
                    std::size_t offset = 0;
                    utf8::Char ch{};
                    if (!utf8::decode_next(out.substr(after), offset, ch)) {
                        result += out.substr(after);
                        i = out.size();
                        break;
                    }
                    if (is_word_codepoint(ch.codepoint)) {
                        result += rule.coda;
                        i = after;
                        continue;
                    }
                    group1 = out.substr(after, ch.length);
                    next = after + ch.length;
                }
            } else {
                if (next < out.size() && out[next] == ' ') {
                    group1 = " ";
                    ++next;
                }
                if (out.compare(next, onset_char.size(), onset_char) != 0) {
                    result += rule.coda;
                    i = after;
                    continue;
                }
                next += onset_char.size();
            }
            std::string replacement(rule.replacement);
            const std::size_t marker = replacement.find("\\1");
            if (marker != std::string::npos) {
                replacement.replace(marker, 2, group1);
            }
            result += replacement;
            i = next;
        }
        out = result;
    }
    return out;
}

/// 第 8 步：连音规则（上游 g2pk2/regular.py）。
[[nodiscard]] std::string apply_link_rules(std::string_view text) {
    std::string out(text);
    static constexpr std::array kLink1 = std::to_array<LiteralRule>({
        {"ᆨᄋ", "ᄀ"},
        {"ᆩᄋ", "ᄁ"},
        {"ᆫᄋ", "ᄂ"},
        {"ᆮᄋ", "ᄃ"},
        {"ᆯᄋ", "ᄅ"},
        {"ᆷᄋ", "ᄆ"},
        {"ᆸᄋ", "ᄇ"},
        {"ᆺᄋ", "ᄉ"},
        {"ᆻᄋ", "ᄊ"},
        {"ᆽᄋ", "ᄌ"},
        {"ᆾᄋ", "ᄎ"},
        {"ᆿᄋ", "ᄏ"},
        {"ᇀᄋ", "ᄐ"},
        {"ᇁᄋ", "ᄑ"},
    });
    static constexpr std::array kLink2 = std::to_array<LiteralRule>({
        {"ᆪᄋ", "ᆨᄊ"},
        {"ᆬᄋ", "ᆫᄌ"},
        {"ᆰᄋ", "ᆯᄀ"},
        {"ᆱᄋ", "ᆯᄆ"},
        {"ᆲᄋ", "ᆯᄇ"},
        {"ᆳᄋ", "ᆯᄊ"},
        {"ᆴᄋ", "ᆯᄐ"},
        {"ᆵᄋ", "ᆯᄑ"},
        {"ᆹᄋ", "ᆸᄊ"},
    });
    static constexpr std::array kLink4 = std::to_array<LiteralRule>({
        {"ᇂᄋ", "ᄋ"},
        {"ᆭᄋ", "ᄂ"},
        {"ᆶᄋ", "ᄅ"},
    });
    apply_literal_rules(out, kLink1.data(), kLink1.size());
    apply_literal_rules(out, kLink2.data(), kLink2.size());
    apply_literal_rules(out, kLink4.data(), kLink4.size());
    return out;
}

/// g2pk2 的主流程（少了 mecab 形态标注那一步 ✓）。
[[nodiscard]] std::string g2pk2(std::string_view text) {
    // 调试用：设 GENIE_TRACE_KOREAN 环境变量可以看每一步的中间结果 ✓
    const bool trace = std::getenv("GENIE_TRACE_KOREAN") != nullptr;
    const auto show = [&](const char* name, const std::string& value) {
        if (trace) {
            std::fprintf(stderr, "  %-16s %s\n", name, value.c_str());
        }
    };
    std::string step = apply_idioms(text);
    show("idioms", step);
    step = h2j(step);
    show("h2j", step);
    step = apply_special_rules(step);
    show("special", step);
    step = apply_regular_table(step);
    show("regular", step);
    step = apply_link_rules(step);
    show("link", step);
    step = compose(step);
    show("compose", step);
    return step;
}

}  // namespace

std::vector<std::string> g2p(std::string_view text) {
    if (text.find_first_not_of(" \t\n\r\f\v") == std::string_view::npos) {
        return {};
    }
    std::string step = latin_to_hangul(text);
    step = number_to_hangul(step);
    step = g2pk2(step);
    step = divide_hangul(step);
    step = fix_g2pk2_error(step);
    // 结尾如果是兼容字母，后面补一个 "." ✓（上游的正则 ✓）
    {
        const auto pieces = split_codepoints(step);
        if (!pieces.empty()) {
            const std::string last(pieces.back());
            std::size_t offset = 0;
            utf8::Char ch{};
            if (utf8::decode_next(last, offset, ch) && ch.codepoint >= 0x3131 &&
                ch.codepoint <= 0x3163) {
                step += ".";
            }
        }
    }
    std::vector<std::string> phonemes;
    for (const auto& piece : split_codepoints(step)) {
        phonemes.push_back(post_replace_phoneme(piece));
    }
    return phonemes;
}

}  // namespace genie::text::korean
