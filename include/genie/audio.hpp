// genie/audio.hpp — 音频缓冲与 WAV 编解码
//
// 上游事实（见 docs/upstream-audit.md）：
//   - 参考音频以 32 kHz 为主链路（VITS 输出 32 kHz），16 kHz 用于 HuBERT / 说话人向量。
//   - 上游用 soundfile 读 wav（多声道取均值转单声道），用 soxr 做高质量重采样。
//   - 本移植用自带的 RIFF/PCM 读写器（不引入第三方音频库），
//     重采样策略见 docs/architecture.md「重采样一致性」一节。

#ifndef GENIE_AUDIO_HPP
#define GENIE_AUDIO_HPP

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace genie {

/// 单声道 float32 音频缓冲。采样值域为 [-1, 1]。
struct AudioBuffer {
    std::vector<float> samples;
    int sample_rate = 32000;

    [[nodiscard]] std::size_t frame_count() const noexcept { return samples.size(); }
    [[nodiscard]] bool empty() const noexcept { return samples.empty(); }
    [[nodiscard]] double duration_seconds() const noexcept;

    /// 峰值幅度。全零缓冲返回 0。
    [[nodiscard]] float peak() const noexcept;

    /// 缩放到峰值 1.0。非静音缓冲会被原地修改。
    void normalize_peak() noexcept;

    /// 截断到 [-1, 1]。原地。
    void clamp() noexcept;
};

/// WAV 文件的格式信息。
struct WavInfo {
    int sample_rate = 0;
    int channels = 0;
    int bits_per_sample = 0;
    std::size_t frame_count = 0;
};

/// WAV 编解码器。
///
/// 支持读取：PCM 8/16/24/32-bit 与 IEEE float 32/64-bit，多声道会被降为单声道
/// （与上游 soundfile 的 `np.mean(wav, axis=1)` 行为一致）。
/// 支持写入：16-bit PCM 与 32-bit float。
class WavCodec {
public:
    /// 读取 WAV 文件。
    ///
    /// @throws genie::Error IoError —— 文件不存在或不是合法 RIFF/WAVE
    [[nodiscard]] static AudioBuffer read(const std::string& path);

    /// 从内存读取 WAV。
    [[nodiscard]] static AudioBuffer read_from_memory(std::span<const std::byte> data);

    /// 读取 WAV 并保留多声道信息（不降为单声道）。
    ///
    /// @throws genie::Error IoError
    [[nodiscard]] static WavInfo inspect(const std::string& path);

    /// 写 16-bit PCM WAV。
    ///
    /// @throws genie::Error IoError —— 写入失败
    static void write_pcm16(const std::string& path, const AudioBuffer& audio);

    /// 写 32-bit float WAV。
    ///
    /// @throws genie::Error IoError —— 写入失败
    static void write_float32(const std::string& path, const AudioBuffer& audio);
};

/// 重采样。
///
/// 采用多相滤波（polyphase FIR），默认使用窗函数法设计的线性相位低通，
/// 与上游 soxr 的 'hq' 档在通带响应上接近；差异在
/// docs/architecture.md「重采样一致性」一节中量化说明。
///
/// @param input        输入采样
/// @param input_rate   输入采样率
/// @param output_rate  目标采样率
/// @throws genie::Error InvalidArgument —— 采样率非法
[[nodiscard]] std::vector<float> resample(std::span<const float> input, int input_rate,
                                          int output_rate);

/// 线性插值重采样。用于对延迟不敏感的低成本路径。
[[nodiscard]] std::vector<float> resample_linear(std::span<const float> input, int input_rate,
                                                 int output_rate);

}  // namespace genie

#endif  // GENIE_AUDIO_HPP
