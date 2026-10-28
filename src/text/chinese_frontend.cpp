// src/text/chinese_frontend.cpp
//
// 中文文本前端：汉字 → 音素符号 → 符号 ID。
//
// 链路（与上游 ChineseG2P.g2p 逐步对应）：
//   文本 → normalize() → 按汉字连续段分组 → 词表分词 → pre_merge
//        → 变调（ToneSandhi）→ 儿化（Erhua）→ 切分声母/韵母 → 音素符号
//        → 按符号表过滤 → 符号 ID
//
// 混读路径（文本含 ASCII 字母，上游 'hybrid-chinese-english'）：
//   split_language 按语言分块 → 英文块丢弃（无英文 G2P，见 audit §9.4）
//   → 每个中文块各自走一遍完整 normalize + 上述链路
//   → 记录 BERT 分块（每块单独跑 RoBERTa，见 include/genie/text.hpp）
//
// 与上游的差异见 docs/upstream-audit.md §9.4 / §9.6，主要有四条：
//   1. 多音字取最常用读音（上游用 g2pM 的上下文判定）
//   2. 分词用上游词表的贪心最长匹配（上游用 jieba + POS），POS 依赖
//      的变调分支取默认行为
//   3. word2ph 按单字统计（上游按 jieba 分词结果统计）
//   4. 混读路径的英文块丢弃（上游产出英文音素）

#include <cctype>
#include <memory>
#include <string>
#include <vector>

#include "genie/pinyin.hpp"
#include "genie/symbols.hpp"
#include "genie/text.hpp"
#include "genie/utf8.hpp"
#include "text/language_split.hpp"
#include "text/polyphone.hpp"
#include "text/tone_sandhi.hpp"

namespace genie::text {
namespace {

/// 单个汉字的读音：全拼与切分结果。读音不可得时 valid 为 false。
struct Reading {
    pinyin::Syllable syllable;
    bool valid = false;
};

/// 一个汉字段的变调处理结果直接写进音素序列。
///
/// 上游 g2p 对每个标点分段做 jieba 分词；标点本身不参与变调，
/// 所以这里按"连续汉字段"分组 —— 段内走完整流程，语义等价。
void process_han_run(const std::vector<std::string>& chars, const Polyphone* polyphone,
                     PhonemeSequence& sequence) {
    // 1) 词表分词（上游 jieba 的替代，见 tone_sandhi.hpp 顶部说明）。
    std::vector<sandhi::Unit> units = sandhi::segment(chars);

    // 2) 每字读音。有 Polyphone（g2pM）时整段推理取上下文读音 ——
    //    对齐上游 `g2pm(seg, char_split=True)`，多音字按句内语境判定；
    //    模型不认识的字回退单字拼音表。没有 Polyphone 时全部查表
    //    （最常用读音，音色包不带 g2pm 模型时的降级路径）。
    std::vector<Reading> readings(chars.size());
    if (polyphone != nullptr) {
        std::string run;
        for (const auto& ch : chars) {
            run += ch;
        }
        // 上游 ChineseG2P 在 g2pm 输出上统一把 "u:" 换成 "v"，
        // 然后用词表做词级读音修正（correct_pronunciation），再进音素。
        auto pinyins = polyphone->pinyins_of(run);
        if (pinyins.size() == chars.size()) {
            for (auto& pron : pinyins) {
                if (std::size_t at = pron.find("u:"); at != std::string::npos) {
                    pron.replace(at, 2, "v");
                }
            }
            polyphone->correct_pronunciation(chars, pinyins);
            for (std::size_t i = 0; i < pinyins.size(); ++i) {
                const std::string& pron = pinyins[i];
                if (!pron.empty() && std::isalpha(static_cast<unsigned char>(pron[0])) != 0) {
                    auto syllable = pinyin::split(pron);
                    readings[i].valid = syllable.valid();
                    readings[i].syllable = std::move(syllable);
                }
            }
        }
    }
    for (std::size_t i = 0; i < readings.size(); ++i) {
        if (readings[i].valid) {
            continue;
        }
        // 回退：单字拼音表的最常用读音。
        if (const auto pinyin = pinyin::lookup(chars[i]); pinyin) {
            if (const auto syllable = pinyin::split(*pinyin); syllable.valid()) {
                readings[i].valid = true;
                readings[i].syllable = std::move(syllable);
            }
        }
    }

    // 3) 把带调韵母写进分词单元（变调只动声调数字）。
    {
        std::size_t index = 0;
        for (auto& unit : units) {
            for (auto& final_with_tone : unit.finals) {
                if (readings[index].valid) {
                    final_with_tone = readings[index].syllable.final_with_tone;
                }
                ++index;
            }
        }
    }

    // 4) 变调前合并 + 变调 + 儿化（上游 pre_merge_for_modify /
    //    modified_tone / merge_erhua 的顺序）。
    sandhi::pre_merge_for_modify(units);
    for (auto& unit : units) {
        sandhi::modified_tone(unit);
        sandhi::merge_erhua(unit);
    }

    // 5) 音素转换。变调只改 final_with_tone 的声调数字，
    //    opencpop 查表键（声母+韵母）不受影响。
    std::size_t index = 0;
    for (const auto& unit : units) {
        for (std::size_t k = 0; k < unit.chars.size(); ++k) {
            const auto& reading = readings[index++];
            if (!reading.valid) {
                // 读音不可得（拼音表缺字 / g2pM 不认识）：跳过 ——
                // 上游同样跳过给不出拼音的字。
                continue;
            }
            // 用变调后的韵母覆盖（声母不变）。
            auto syllable = reading.syllable;
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

/// 一段**规范化后**的中文文本 → 音素序列。
///
/// 汉字段走完整流水线（分词 → 变调 → 儿化 → 音素），标点走符号表。
/// 单块与混读路径共用这里。
void process_normalized(const std::string& normalized, const Polyphone* polyphone,
                        PhonemeSequence& sequence) {
    std::vector<std::string> han_run;
    const auto flush_run = [&]() {
        if (!han_run.empty()) {
            process_han_run(han_run, polyphone, sequence);
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
}

/// 中文前端。
///
/// @param polyphone 多音字消歧（g2pM）；为空时退化为单字拼音表
///                  最常用读音（音色包不带 g2pm 模型的降级路径）。
class ChineseFrontend final : public TextFrontend {
public:
    explicit ChineseFrontend(const Polyphone* polyphone = nullptr) : polyphone_(polyphone) {}
    [[nodiscard]] PhonemeSequence convert(std::string_view text) const override {
        PhonemeSequence sequence;

        // 与上游 GetPhonesAndBert.py 同一个分叉点：文本含 ASCII 字母才切到
        // 'hybrid-chinese-english'（按语言分块），纯中文走单块路径。
        // 混读路径里每个中文块各自走一遍完整的 TextNormalizer（上游对
        // split_language 切出的每个 chinese 块单独调 chinese_to_phones），
        // 块边界同时也是 RoBERTa 的编码边界（块间没有跨块注意力）。
        if (lang_split::has_ascii_letter(text)) {
            for (const auto& chunk : lang_split::split_by_language(text)) {
                // 偏差（docs/upstream-audit.md §9.4）：上游英文块走
                // english_to_phones 产出真实音素；本实现还没有英文 G2P，
                // 英文块暂且丢弃 —— 与旧版"过滤字母"的行为一致。
                if (chunk.language != lang_split::Chunk::Language::Chinese) {
                    continue;
                }
                const std::string normalized = normalize(chunk.content, Language::Chinese);
                const std::size_t word_offset = sequence.word2ph.size();
                process_normalized(normalized, polyphone_, sequence);
                if (sequence.word2ph.size() > word_offset) {
                    // 产出为空的块不进 BERT 分块：没有音素也就没有特征行
                    //（上游对空 phone 列表展开同样是 0 行）。
                    sequence.bert_chunks.push_back(
                        BertChunk{normalized, word_offset, sequence.word2ph.size() - word_offset});
                }
            }
            return sequence;
        }

        process_normalized(normalize(text, Language::Chinese), polyphone_, sequence);
        return sequence;
    }

    [[nodiscard]] Language language() const noexcept override { return Language::Chinese; }

private:
    const Polyphone* polyphone_ = nullptr;
};

}  // namespace

std::unique_ptr<TextFrontend> make_chinese_frontend(const Polyphone* polyphone) {
    return std::make_unique<ChineseFrontend>(polyphone);
}

}  // namespace genie::text
