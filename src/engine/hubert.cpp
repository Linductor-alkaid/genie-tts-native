// src/engine/hubert.cpp — HuBERT 参考音频特征的运行时装配

#include "engine/hubert.hpp"

#include "genie/errors.hpp"

namespace genie::hubert {

Tensor compute_ssl(ModelSession& model, const AudioBuffer& audio_16k) {
    TensorFeed feed;
    feed.emplace_back("input_values",
                      Tensor::from_float32({1, static_cast<std::int64_t>(audio_16k.samples.size())},
                                           audio_16k.samples));

    const auto outputs = model.run(feed);
    if (outputs.empty() || outputs.front().type() != ElementType::Float32) {
        throw_error(ErrorCode::InferenceError, "hubert produced no float output");
    }
    // 上游把 run(...)[0] 原样喂给 t2s_encoder 的 ssl_content ——
    // 图内已是 [1, 768, T']，这里不做任何形状处理。
    return outputs.front();
}

}  // namespace genie::hubert
