// include/genie/reference.hpp
//
// 参考音频条件。
//
// T2S 编码器不只吃待合成文本，还要吃**参考音频**的条件：
//
//   ref_seq      参考音频的音素序列（int64）
//   ref_bert     参考音频的韵律特征（float32 [..., 1024]）
//   ssl_content  参考音频的 HuBERT 特征（float32 [1, 768, T]）
//
// 声码器还要吃原始参考波形（`ref_audio`）。
//
// 上游这些特征分别来自：音素 → 文本前端；韵律 → Chinese RoBERTa；
// HuBERT → chinese-hubert-base.onnx。**后两个依赖可选模型**，
// 缺模型时上游退化为**全零特征**（见上游 `GetPhonesAndBert.py`）。
// 本实现照搬这个降级路径 —— 没有可选模型也能合成，只是韵律差一点。
//
// 关键点：**特征张量的形状必须对**。形状错了模型不会报错，
// 只会输出垃圾。所以这里所有张量都显式构造，不留空的默认张量。

#ifndef GENIE_REFERENCE_HPP
#define GENIE_REFERENCE_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "genie/audio.hpp"
#include "genie/tensor.hpp"
#include "genie/text.hpp"

namespace genie {

/// 参考条件的形状常量。
///
/// 与上游模型一致（见 docs/upstream-audit.md §3.1）：
/// - 韵律特征（RoBERTa）的维度是 **1024**（chinese-roberta-wwm-ext-large）
/// - HuBERT 特征的通道数是 **768**
inline constexpr std::int64_t kBertDimension = 1024;
inline constexpr std::int64_t kSslChannels = 768;

/// 参考条件。
struct ReferenceCondition {
    /// 参考音频的音素符号 ID（喂给 t2s_encoder 的 `ref_seq`）。
    std::vector<std::int64_t> phoneme_ids;
    /// 参考音频的韵律特征（`ref_bert`）。没有 RoBERTa 时是零张量。
    Tensor bert;
    /// 参考音频的 HuBERT 特征（`ssl_content`）。没有 HuBERT 时是零张量。
    Tensor ssl;
    /// 参考波形，32 kHz 单声道（喂给 vits 的 `ref_audio`）。
    AudioBuffer audio;
    /// 参考波形，16 kHz 单声道。
    ///
    /// 上游把参考音频同时喂给三个模型，各要一个采样率：
    /// - HuBERT      → 16 kHz（`input_values`）
    /// - 说话人验证   → 16 kHz（`waveform`）
    /// - prompt_encoder / vits → 32 kHz（`ref_audio`）
    AudioBuffer audio_16k;
    /// 参考文本（用于生成 phoneme_ids）。
    std::string text;

    /// 条件是否可用（有参考波形）。
    [[nodiscard]] bool usable() const noexcept { return !audio.empty(); }
};

/// 从音色包目录装配参考条件。
///
/// 约定（都在 `<package>/reference/` 下）：
/// - `audio.wav` —— 参考音频。任意采样率，会重采样到 32 kHz。
/// - `text.txt`  —— 参考文本（可选）。有的话走文本前端得到 `ref_seq`。
///
/// @param root 音色包根目录
/// @param frontend 文本前端（用于把参考文本转成音素）
/// @param sample_rate 目标采样率（默认 32000）
///
/// @throws genie::Error IoError —— 参考音频不存在或无法解码
[[nodiscard]] ReferenceCondition load_reference(const std::string& root,
                                                const text::TextFrontend& frontend,
                                                int sample_rate = 32000);

}  // namespace genie

#endif  // GENIE_REFERENCE_HPP
