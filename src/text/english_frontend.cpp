// src/text/english_frontend.cpp —— 英文文本前端。
//
// 链路（与上游 EnglishG2P.g2p 逐步对应）：
//   文本 → REP_MAP + 规范化（20 多条规则）→ 折叠连续标点
//        → TweetTokenizer 分词 → POS 标注 → CMU 词典查发音
//        → 按符号表过滤 → 符号 ID
//
// 与上游的差异见 docs/upstream-audit.md §17：
//   1. 词典数据来自 GPT-SoVITS 的 engdict_cache.pickle + engdict-hot.rep
//      （上游的 English_G2P 目录内容同源）；
//   2. TweetTokenizer 的表情符号 / 国旗 / emoji 组合分支没有移植
//      （它们产出的 token 同样查不到音素）；
//   3. ngram 分词词典不随仓库发布 —— 没有它时 OOV 词直接交给 GRU。
//
// 上游的英文路径**不产出 RoBERTa 特征**（text_bert 全零），
// 所以这里 bert_chunks 恒为空 —— 引擎据此走全零特征分支。

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "genie/symbols.hpp"
#include "genie/text.hpp"
#include "text/english/english.hpp"

namespace genie::text {
namespace {

class EnglishFrontend : public TextFrontend {
public:
    /// 资源由前端**自己持有** —— 不能只留引用（加载函数返回后
    /// 局部的 unique_ptr 就析构了，前端会指向已释放的内存）。
    EnglishFrontend(english::CmuDict dict, english::Seq2SeqG2p seq2seq,
                    english::WordSegmenter segmenter, std::optional<english::PosTagger> tagger)
        : dict_(std::move(dict)),
          seq2seq_(std::move(seq2seq)),
          segmenter_(std::move(segmenter)),
          tagger_(std::move(tagger)) {}

    [[nodiscard]] PhonemeSequence convert(std::string_view text) const override {
        PhonemeSequence sequence;
        const auto phonemes =
            english::phonemes(text, dict_, seq2seq_, segmenter_, tagger_ ? &*tagger_ : nullptr);
        sequence.symbols.reserve(phonemes.size());
        sequence.ids.reserve(phonemes.size());
        for (const auto& phoneme : phonemes) {
            const auto id = symbols::id_of(phoneme);
            if (id < 0) {
                continue;  // 不在符号表里 —— 上游的 replace_phs 同样丢掉
            }
            sequence.symbols.push_back(phoneme);
            sequence.ids.push_back(id);
            sequence.word2ph.push_back(1);
        }
        return sequence;
    }

    [[nodiscard]] Language language() const noexcept override { return Language::English; }

private:
    english::CmuDict dict_;
    english::Seq2SeqG2p seq2seq_;
    english::WordSegmenter segmenter_;
    std::optional<english::PosTagger> tagger_;
};

}  // namespace

std::unique_ptr<TextFrontend> make_english_frontend(const EnglishG2pResources& resources) {
    if (resources.cmu_dict.empty()) {
        return nullptr;
    }
    // 词典是硬依赖 —— 读不出来就报错，不静默降级（降级等于"英文没声音"）。
    english::CmuDict dict = english::CmuDict::load(resources.cmu_dict);

    // 其余都是可选资源：缺失时对应的能力降级（见 docs/upstream-audit.md §17）。
    english::Seq2SeqG2p seq2seq;
    if (!resources.g2p_model.empty()) {
        (void)seq2seq.load(resources.g2p_model);
    }
    english::WordSegmenter segmenter;
    if (!resources.word_segment_dir.empty()) {
        (void)segmenter.load(resources.word_segment_dir);
    }
    std::optional<english::PosTagger> tagger;
    if (!resources.pos_tagger.empty()) {
        tagger = english::PosTagger::load(resources.pos_tagger);
    }

    return std::make_unique<EnglishFrontend>(std::move(dict), std::move(seq2seq),
                                             std::move(segmenter), std::move(tagger));
}

}  // namespace genie::text
