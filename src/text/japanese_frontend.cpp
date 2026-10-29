// src/text/japanese_frontend.cpp —— 日语文本前端。
//
// 链路（与上游 JapaneseG2P.g2p 逐步对应）：
//   文本 → 规范化（% → パーセント、折叠连续标点、转小写）
//        → 按非日语字符切分 → OpenJTalk 全上下文标签
//        → 音素 + 韵律符号（# [ ]）→ 符号表过滤 → 符号 ID
//
// OpenJTalk 的词典（103 MB）不在仓库里 —— 由音色包给出路径 ✓。
// 上游的日文路径**不产出 RoBERTa 特征**（text_bert 全零），
// 所以这里 bert_chunks 恒为空 —— 引擎据此走全零特征分支。

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "genie/symbols.hpp"
#include "genie/text.hpp"
#include "text/japanese/japanese.hpp"
#include "text/japanese/openjtalk.hpp"

namespace genie::text {
namespace {

class JapaneseFrontend : public TextFrontend {
public:
    /// 词典由前端自己持有 ✓（103 MB ✓ —— 跟着前端走 ✓）。
    explicit JapaneseFrontend(std::unique_ptr<japanese::OpenJTalk> engine)
        : engine_(std::move(engine)) {}

    [[nodiscard]] PhonemeSequence convert(std::string_view text) const override {
        PhonemeSequence sequence;
        if (engine_ == nullptr) {
            return sequence;
        }
        const japanese::JapaneseG2p g2p(*engine_);
        const auto phonemes = g2p.g2p(text);
        sequence.symbols.reserve(phonemes.size());
        sequence.ids.reserve(phonemes.size());
        for (const auto& phoneme : phonemes) {
            const auto id = symbols::id_of(phoneme);
            if (id < 0) {
                continue;  // 不在符号表里 —— 上游的 japanese_to_phones 同样丢掉
            }
            sequence.symbols.push_back(phoneme);
            sequence.ids.push_back(id);
            sequence.word2ph.push_back(1);
        }
        return sequence;
    }

    [[nodiscard]] Language language() const noexcept override { return Language::Japanese; }

private:
    std::unique_ptr<japanese::OpenJTalk> engine_;
};

}  // namespace

std::unique_ptr<TextFrontend> make_japanese_frontend(const JapaneseG2pResources& resources) {
    if (resources.dictionary_dir.empty()) {
        return nullptr;
    }
    auto engine = std::make_unique<japanese::OpenJTalk>();
    if (!engine->load(resources.dictionary_dir)) {
        return nullptr;
    }
    return std::make_unique<JapaneseFrontend>(std::move(engine));
}

}  // namespace genie::text
