// src/text/korean_frontend.cpp —— 韩语文本前端。
//
// 链路（与上游 KoreanG2P.g2p 逐步对应）：
//   文本 → 拉丁字母转韩文 → 数字转韩文 → g2pk2（惯用语 / 拆解 / 音变规则）
//        → 兼容字母 → 符号表过滤 → 符号 ID
//
// 与上游的差异见 docs/upstream-audit.md §18：
//   g2pk2 的 mecab-ko 形态标注那一步没有移植（mecab 是 C++ 库 + 100 MB
//   词典），依赖标注的四条音变规则因此不触发。
//
// 上游的韩文路径**不产出 RoBERTa 特征**（text_bert 全零），
// 所以这里 bert_chunks 恒为空 —— 引擎据此走全零特征分支。

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "genie/symbols.hpp"
#include "genie/text.hpp"
#include "text/korean/korean.hpp"

namespace genie::text {
namespace {

class KoreanFrontend : public TextFrontend {
public:
    [[nodiscard]] PhonemeSequence convert(std::string_view text) const override {
        PhonemeSequence sequence;
        const auto phonemes = korean::g2p(text);
        sequence.symbols.reserve(phonemes.size());
        sequence.ids.reserve(phonemes.size());
        for (const auto& phoneme : phonemes) {
            const auto id = symbols::id_of(phoneme);
            if (id < 0) {
                continue;  // 不在符号表里 —— 上游的 korean_to_phones 同样丢掉
            }
            sequence.symbols.push_back(phoneme);
            sequence.ids.push_back(id);
            sequence.word2ph.push_back(1);
        }
        return sequence;
    }

    [[nodiscard]] Language language() const noexcept override { return Language::Korean; }
};

}  // namespace

std::unique_ptr<TextFrontend> make_korean_frontend() {
    return std::make_unique<KoreanFrontend>();
}

}  // namespace genie::text
