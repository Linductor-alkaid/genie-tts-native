// src/text/korean/jamo.cpp —— 韩文字母的拆解与组装
//
// 移植 python-jamo（MIT）的三个函数：
//   h2j     —— 组合字（가）→ 组合字母（ᄀ ᅡ ᆫ）
//   j2hcj   —— 组合字母（ᄀ ᅡ ᆫ）→ 兼容字母（ㄱ ㅏ ㄴ）
//   compose —— 组合字母 → 组合字（g2pk2 的 assemble 步骤）
//
// jamo 靠 Unicode 名字匹配两套字母，本仓库把映射固化成表
// （src/text/korean/tables.hpp，由 tools/g2p/generate_korean_tables.py 生成）。

#include "text/korean/jamo.hpp"

#include <string>
#include <string_view>

#include "genie/utf8.hpp"
#include "text/korean/tables.hpp"

namespace genie::text::korean {
namespace {

constexpr std::uint32_t kHangulSyllableBase = 0xAC00;  // 가
// jamo 的偏移量：下标从 1 开始，所以基准比首字母小 1 ✓
// （ᄀ = U+1100 = 1 + 0x10FF ✓）。
constexpr std::uint32_t kLeadBase = 0x10FF;   // ᄀ
constexpr std::uint32_t kVowelBase = 0x1160;  // ᅡ
constexpr std::uint32_t kTailBase = 0x11A7;   // ᆨ

/// 组合字母 → 兼容字母（二分查找；找不到就原样返回）。
[[nodiscard]] std::uint32_t modern_to_compat(std::uint32_t modern) noexcept {
    std::size_t low = 0;
    std::size_t high = tables::kJamoPairs.size();
    while (low < high) {
        const std::size_t mid = low + (high - low) / 2;
        const auto& pair = tables::kJamoPairs[mid];
        if (pair.modern < modern) {
            low = mid + 1;
        } else if (pair.modern > modern) {
            high = mid;
        } else {
            return pair.compat;
        }
    }
    return modern;
}

}  // namespace

std::string h2j(std::string_view text) {
    std::string out;
    out.reserve(text.size() * 2);
    std::size_t offset = 0;
    utf8::Char ch{};
    while (utf8::decode_next(text, offset, ch)) {
        const std::uint32_t cp = ch.codepoint;
        if (cp < kHangulSyllableBase || cp > kHangulSyllableBase + 0xD7FF) {
            // 不是组合字：原样保留（jamo 的行为）。
            out.append(text.substr(ch.offset, ch.length));
            continue;
        }
        const std::uint32_t rem = cp - kHangulSyllableBase;
        const std::uint32_t tail = rem % 28;
        const std::uint32_t vowel = 1 + ((rem - tail) % 588) / 28;
        const std::uint32_t lead = 1 + rem / 588;
        out += utf8::encode(lead + kLeadBase);
        out += utf8::encode(vowel + kVowelBase);
        if (tail != 0) {
            out += utf8::encode(tail + kTailBase);
        }
    }
    return out;
}

std::string j2hcj(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    std::size_t offset = 0;
    utf8::Char ch{};
    while (utf8::decode_next(text, offset, ch)) {
        out += utf8::encode(modern_to_compat(ch.codepoint));
    }
    return out;
}

std::string compose(std::string_view text) {
    // 上游 compose() 的三步：
    //   1. 在"前面不是起首辅音"的位置给元音补一个占位的 ᄋ
    //   2. 把 "辅音 + 元音 + 收音" 组装成音节
    //   3. 把 "辅音 + 元音" 组装成音节
    std::string marked;
    marked.reserve(text.size() * 2);
    std::size_t offset = 0;
    utf8::Char ch{};
    bool previous_is_lead = false;
    while (utf8::decode_next(text, offset, ch)) {
        const bool is_vowel = ch.codepoint >= 0x1161 && ch.codepoint <= 0x1175;
        if (is_vowel && !previous_is_lead) {
            marked += utf8::encode(0x110B);  // ᄋ 占位
        }
        marked += utf8::encode(ch.codepoint);
        previous_is_lead = ch.codepoint >= 0x1100 && ch.codepoint <= 0x1112;
    }

    // 把一段组合字母组装成一个音节（返回消耗的字节数与音节码位 ✓）。
    const auto assemble = [](std::string& source, std::size_t at, bool with_tail,
                             std::size_t& length, std::uint32_t& syllable) -> bool {
        std::size_t cursor = at;
        utf8::Char lead{};
        if (!utf8::decode_next(source, cursor, lead)) {
            return false;
        }
        if (lead.codepoint < 0x1100 || lead.codepoint > 0x1112) {
            return false;
        }
        const std::size_t after_lead = cursor;
        utf8::Char vowel{};
        if (!utf8::decode_next(source, cursor, vowel) || vowel.codepoint < 0x1161 ||
            vowel.codepoint > 0x1175) {
            return false;
        }
        const std::size_t after_vowel = cursor;
        std::uint32_t tail = 0;
        std::size_t end = after_vowel;
        if (with_tail) {
            utf8::Char tail_char{};
            if (!utf8::decode_next(source, cursor, tail_char) || tail_char.codepoint < 0x11A8 ||
                tail_char.codepoint > 0x11C2) {
                return false;
            }
            tail = tail_char.codepoint;
            end = cursor;
        }
        // Unicode 的组合公式：S = 0xAC00 + (L*21 + V)*28 + T。
        // 注意收音的下标从 1 起（ᆨ = 1，0 表示"无收音"）——
        // jamo 包的 _JAMO_TAIL_OFFSET = 0x11A7 就是这个约定 ✓。
        syllable = 0xAC00 + (lead.codepoint - 0x1100) * 588 + (vowel.codepoint - 0x1161) * 28 +
                   (with_tail ? (tail - 0x11A7) : 0);
        length = end - at;
        (void)after_lead;
        return true;
    };

    for (const bool with_tail : {true, false}) {
        for (;;) {
            bool changed = false;
            std::size_t offset2 = 0;
            utf8::Char current{};
            while (utf8::decode_next(marked, offset2, current)) {
                std::size_t length = 0;
                std::uint32_t syllable = 0;
                if (assemble(marked, current.offset, with_tail, length, syllable)) {
                    marked.replace(current.offset, length, utf8::encode(syllable));
                    changed = true;
                    break;
                }
            }
            if (!changed) {
                break;
            }
        }
    }
    return marked;
}

}  // namespace genie::text::korean
