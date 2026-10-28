// src/text/chinese_frontend.cpp
//
// 中文文本前端：汉字 → 音素符号 → 符号 ID。
//
// 链路（与上游 ChineseG2P.g2p 逐步对应）：
//   文本 → normalize() → 按汉字连续段分组 → 词表分词 → pre_merge
//        → 变调（ToneSandhi）→ 儿化（Erhua）→ 切分声母/韵母 → 音素符号
//        → 按符号表过滤 → 符号 ID
//
// 与上游的差异见 docs/upstream-audit.md §9.4 / §9.6，主要有三条：
//   1. 多音字取最常用读音（上游用 g2pM 的上下文判定）
//   2. 分词用上游词表的贪心最长匹配（上游用 jieba + POS），POS 依赖
//      的变调分支取默认行为
//   3. word2ph 按单字统计（上游按 jieba 分词结果统计）

#include <memory>
#include <string>
#include <vector>

#include "genie/pinyin.hpp"
#include "genie/symbols.hpp"
#include "genie/text.hpp"
#include "genie/utf8.hpp"
#include "text/tone_sandhi.hpp"

namespace genie::text {
namespace {

/// 一个汉字段的变调处理结果直接写进音素序列。
///
/// 上游 g2p 对每个标点分段做 jieba 分词；标点本身不参与变调，
/// 所以这里按"连续汉字段"分组 —— 段内走完整流程，语义等价。
void process_han_run(const std::vector<std::string>& chars, PhonemeSequence& sequence) {
    // 1) 词表分词（上游 jieba 的替代，见 tone_sandhi.hpp 顶部说明）。
    std::vector<sandhi::Unit> units = sandhi::segment(chars);

    // 2) 填每字带调韵母（上游由 g2pM 给出；我们查单字拼音表）。
    {
        std::size_t index = 0;
        for (auto& unit : units) {
            for (auto& final_with_tone : unit.finals) {
                if (const auto pinyin = pinyin::lookup(chars[index]); pinyin) {
                    if (const auto syllable = pinyin::split(*pinyin); syllable.valid()) {
                        final_with_tone = syllable.final_with_tone;
                    }
                }
                ++index;
            }
        }
    }

    // 3) 变调前合并 + 变调 + 儿化（上游 pre_merge_for_modify /
    //    modified_tone / merge_erhua 的顺序）。
    sandhi::pre_merge_for_modify(units);
    for (auto& unit : units) {
        sandhi::modified_tone(unit);
        sandhi::merge_erhua(unit);
    }

    // 4) 音素转换。变调只改 final_with_tone 的声调数字，
    //    opencpop 查表键（声母+韵母）不受影响。
    for (const auto& unit : units) {
        for (std::size_t k = 0; k < unit.chars.size(); ++k) {
            const auto pinyin = pinyin::lookup(unit.chars[k]);
            if (!pinyin) {
                // 拼音表缺字：跳过 —— 上游同样跳过给不出拼音的字。
                continue;
            }
            auto syllable = pinyin::split(*pinyin);
            if (!syllable.valid()) {
                continue;
            }
            // 用变调后的韵母覆盖（声母不变）。
            syllable.final_with_tone = unit.finals[k];
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
        }
    }
}

/// 中文前端。
class ChineseFrontend final : public TextFrontend {
public:
    [[nodiscard]] PhonemeSequence convert(std::string_view text) const override {
        PhonemeSequence sequence;
        const std::string normalized = normalize(text, Language::Chinese);

        std::vector<std::string> han_run;
        const auto flush_run = [&]() {
            if (!han_run.empty()) {
                process_han_run(han_run, sequence);
                han_run.clear();
            }
        };

        std::size_t offset = 0;
        utf8::Char ch;
        while (utf8::decode_next(normalized, offset, ch)) {
            // 注意：不能把 `normalized.substr(...)` 的结果直接绑给 string_view ——
            // substr() 返回的是**临时 std::string**，语句结束就析构，
            // string_view 随即指向已释放的内存。
            //
            // 这个坑是 Windows CI 抓到的：MSVC 上这块内存被回收后查字典必然落空，
            // 音素数变成 0；Linux 上则可能"碰巧"还能查到，ASAN 也看不见
            // （单个汉字只有 3 字节，走 SSO，数据在对象内部，没有堆释放可查）。
            // 直接从 normalized 里取视图，没有临时对象。
            const std::string_view piece(normalized.data() + ch.offset, ch.length);

            if (utf8::is_han(ch.codepoint)) {
                han_run.emplace_back(piece);
                continue;
            }

            // 标点打断汉字段（变调不跨标点），标点自身照旧走符号表。
            flush_run();
            const auto id = symbols::id_of(piece);
            if (id >= 0) {
                sequence.symbols.emplace_back(piece);
                sequence.ids.push_back(id);
                sequence.word2ph.push_back(1);
            }
        }
        flush_run();
        return sequence;
    }

    [[nodiscard]] Language language() const noexcept override { return Language::Chinese; }
};

}  // namespace

std::unique_ptr<TextFrontend> make_chinese_frontend() {
    return std::make_unique<ChineseFrontend>();
}

}  // namespace genie::text
