// src/engine/roberta.cpp — RoBERTa 韵律特征的运行时装配
//
// 输入布局对齐上游 `_build_roberta_inputs`：input_ids / attention_mask /
// token_type_ids 三个 int64 输入（token_type_ids 本模型也需要，全零）。
// 输出布局的归一化见 expand_by_word2ph。

#include "engine/roberta.hpp"

#include <algorithm>
#include <numeric>

#include "genie/errors.hpp"
#include "genie/reference.hpp"  // kBertDimension

namespace genie::roberta {

Tensor expand_by_word2ph(std::span<const float> token_bert,
                         const std::vector<std::int32_t>& word2ph, std::size_t num_phones) {
    constexpr std::int64_t kDim = kBertDimension;
    const auto token_count = token_bert.size() / static_cast<std::size_t>(kDim);
    if (token_bert.size() % static_cast<std::size_t>(kDim) != 0) {
        throw_error(ErrorCode::InvalidArgument,
                    "roberta output is not a multiple of the bert dimension");
    }

    const auto word_total =
        static_cast<std::size_t>(std::accumulate(word2ph.begin(), word2ph.end(), std::int64_t{0}));

    // 三种布局归一化到「按 word2ph 重复」或「直接使用」。
    const float* source = token_bert.data();
    std::size_t source_rows = token_count;
    std::vector<float> stripped;

    if (source_rows == num_phones) {
        // 已经是音素级：直接搬运。
        return Tensor::from_float32({static_cast<std::int64_t>(num_phones), kDim},
                                    std::vector<float>(token_bert.begin(), token_bert.end()));
    }
    if (source_rows == word2ph.size() + 2) {
        // 去掉 [CLS] / [SEP] 两行。
        stripped.assign(token_bert.begin() + static_cast<std::ptrdiff_t>(kDim),
                        token_bert.end() - static_cast<std::ptrdiff_t>(kDim));
        source = stripped.data();
        source_rows = stripped.size() / static_cast<std::size_t>(kDim);
    } else if (source_rows != word2ph.size()) {
        throw_error(ErrorCode::InvalidArgument,
                    "unsupported roberta output layout: got " + std::to_string(source_rows) +
                        " token rows, expected " + std::to_string(num_phones) + " phones or " +
                        std::to_string(word2ph.size()) + "/" + std::to_string(word2ph.size() + 2) +
                        " token features");
    }

    if (word_total != num_phones) {
        throw_error(ErrorCode::InvalidArgument, "word2ph expands to " + std::to_string(word_total) +
                                                    " phones, but the sequence has " +
                                                    std::to_string(num_phones));
    }

    std::vector<float> expanded;
    expanded.reserve(num_phones * static_cast<std::size_t>(kDim));
    for (std::size_t w = 0; w < word2ph.size(); ++w) {
        for (std::int32_t repeat = 0; repeat < word2ph[w]; ++repeat) {
            expanded.insert(expanded.end(), source + w * static_cast<std::size_t>(kDim),
                            source + (w + 1) * static_cast<std::size_t>(kDim));
        }
    }
    return Tensor::from_float32({static_cast<std::int64_t>(num_phones), kDim}, std::move(expanded));
}

Tensor compute_text_bert(const text::RobertaTokenizer& tokenizer, ModelSession& model,
                         std::string_view text, const std::vector<std::int32_t>& word2ph,
                         std::size_t num_phones) {
    const auto encoding = tokenizer.encode(text);
    const auto length = static_cast<std::int64_t>(encoding.ids.size());

    TensorFeed feed;
    feed.emplace_back("input_ids", Tensor::from_int64({1, length}, encoding.ids));
    feed.emplace_back("attention_mask", Tensor::from_int64({1, length}, encoding.attention_mask));
    // token_type_ids：单句全零（BERT 的 segment A）。
    feed.emplace_back("token_type_ids", Tensor::zeros(ElementType::Int64, {1, length}));

    const auto outputs = model.run(feed);
    if (outputs.empty() || outputs.front().type() != ElementType::Float32) {
        throw_error(ErrorCode::InferenceError, "roberta produced no float output");
    }

    // 上游 _expand_roberta_output 的第一步是 [1, T, 1024] → text_bert[0]；
    // batch 维为 1 时数据是连续的，去不去这一维内存布局都一样，
    // expand_by_word2ph 按行数展开即可。
    return expand_by_word2ph(outputs.front().floats(), word2ph, num_phones);
}

}  // namespace genie::roberta
