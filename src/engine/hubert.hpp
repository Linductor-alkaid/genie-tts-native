// src/engine/hubert.hpp — HuBERT 参考音频特征（内部实现）
//
// 上游参考音频的 ssl_content 由 chinese-hubert-base 对 16 kHz 波形推理
// 产出（ReferenceAudio.__init__：
// `cn_hubert.run(None, {'input_values': self.audio_16k})[0]`），
// 输出**直接**喂给 t2s_encoder —— 转置在导出时已烘进图里
// （last_hidden_state: [1, 768, T']）。缺失时上游/本实现同样全零降级。

#ifndef GENIE_SRC_ENGINE_HUBERT_HPP
#define GENIE_SRC_ENGINE_HUBERT_HPP

#include "genie/audio.hpp"
#include "genie/session.hpp"
#include "genie/tensor.hpp"

namespace genie::hubert {

/// 16 kHz 参考波形 → ssl_content [1, 768, T']。
///
/// @param model      chinese-hubert 会话
/// @param audio_16k  16 kHz 单声道参考波形
/// @throws genie::Error 模型无浮点输出或推理失败
[[nodiscard]] Tensor compute_ssl(ModelSession& model, const AudioBuffer& audio_16k);

}  // namespace genie::hubert

#endif  // GENIE_SRC_ENGINE_HUBERT_HPP
