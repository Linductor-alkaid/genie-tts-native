// src/text/chinese_frontend.cpp
//
// 中文文本前端：汉字 → 音素符号 → 符号 ID。
//
// 链路：
//   文本 → normalize() → 逐字取拼音 → 切分声母/韵母 → 音素符号
//        → 按符号表过滤 → 符号 ID
//
// 与上游的差异见 docs/upstream-audit.md，主要有三条：
//   1. 多音字取最常用读音（上游用 g2pM 的上下文判定）
//   2. 不做变调（ToneSandhi）与儿化（Erhua）处理
//   3. word2ph 按单字统计（上游按 jieba 分词结果统计）

#include <memory>
#include <string>
#include <vector>

#include "genie/pinyin.hpp"
#include "genie/symbols.hpp"
#include "genie/text.hpp"
#include "genie/utf8.hpp"

namespace genie::text {
namespace {

/// 中文前端。
class ChineseFrontend final : public TextFrontend {
public:
    [[nodiscard]] PhonemeSequence convert(std::string_view text) const override {
        PhonemeSequence sequence;
        const std::string normalized = normalize(text, Language::Chinese);

        std::size_t offset = 0;
        utf8::Char ch;
        while (utf8::decode_next(normalized, offset, ch)) {
            const std::string_view piece = normalized.substr(ch.offset, ch.length);

            if (utf8::is_han(ch.codepoint)) {
                const auto pinyin = pinyin::lookup(piece);
                if (!pinyin) {
                    // 词典里没有这个字。跳过 —— 上游同样会跳过查不到拼音的字。
                    continue;
                }
                const auto syllable = pinyin::split(*pinyin);
                if (!syllable.valid()) {
                    continue;
                }
                const auto phones = pinyin::syllable_to_phones(syllable);

                std::int32_t count = 0;
                for (const auto& phone : phones) {
                    const auto id = symbols::id_of(phone);
                    if (id < 0) {
                        // 不在符号表里的音素直接丢掉（上游的 `if ph in symbols_v2`）。
                        continue;
                    }
                    sequence.symbols.push_back(phone);
                    sequence.ids.push_back(id);
                    ++count;
                }
                if (count > 0) {
                    sequence.word2ph.push_back(count);
                }
                continue;
            }

            // 标点：符号就是它本身（"!" "?" "…" "," "." "-" 都在符号表里）。
            const auto id = symbols::id_of(piece);
            if (id >= 0) {
                sequence.symbols.emplace_back(piece);
                sequence.ids.push_back(id);
                sequence.word2ph.push_back(1);
            }
        }
        return sequence;
    }

    [[nodiscard]] Language language() const noexcept override { return Language::Chinese; }
};

}  // namespace

std::unique_ptr<TextFrontend> make_chinese_frontend() {
    return std::make_unique<ChineseFrontend>();
}

}  // namespace genie::text
