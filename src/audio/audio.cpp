// src/audio/audio.cpp — 音频缓冲、WAV 编解码、重采样

#include "genie/audio.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numbers>
#include <span>

#include "genie/errors.hpp"

namespace genie {
namespace {

constexpr double kPi = std::numbers::pi;

// ------------------------------------------------------------------ RIFF 基础工具

std::uint32_t read_u32(std::span<const std::byte> data, std::size_t offset) {
    std::uint32_t value = 0;
    std::memcpy(&value, data.data() + offset, sizeof(value));
    return value;
}

std::uint16_t read_u16(std::span<const std::byte> data, std::size_t offset) {
    std::uint16_t value = 0;
    std::memcpy(&value, data.data() + offset, sizeof(value));
    return value;
}

void append_u32(std::vector<std::byte>& out, std::uint32_t value) {
    const auto offset = out.size();
    out.resize(offset + sizeof(value));
    std::memcpy(out.data() + offset, &value, sizeof(value));
}

void append_u16(std::vector<std::byte>& out, std::uint16_t value) {
    const auto offset = out.size();
    out.resize(offset + sizeof(value));
    std::memcpy(out.data() + offset, &value, sizeof(value));
}

void append_tag(std::vector<std::byte>& out, const char* tag) {
    for (int i = 0; i < 4; ++i) {
        out.push_back(static_cast<std::byte>(tag[i]));
    }
}

std::string tag_at(std::span<const std::byte> data, std::size_t offset) {
    return std::string(reinterpret_cast<const char*>(data.data() + offset), 4);
}

// ------------------------------------------------------------------ 采样格式

enum class SampleFormat {
    PcmU8,
    Pcm16,
    Pcm24,
    Pcm32,
    Float32,
    Float64,
};

SampleFormat format_from_tag(std::uint16_t tag, std::uint16_t bits) {
    if (tag == 1) {  // WAVE_FORMAT_PCM
        switch (bits) {
            case 8:
                return SampleFormat::PcmU8;
            case 16:
                return SampleFormat::Pcm16;
            case 24:
                return SampleFormat::Pcm24;
            case 32:
                return SampleFormat::Pcm32;
            default:
                break;
        }
    } else if (tag == 3) {  // WAVE_FORMAT_IEEE_FLOAT
        if (bits == 32) {
            return SampleFormat::Float32;
        }
        if (bits == 64) {
            return SampleFormat::Float64;
        }
    }
    throw_error(ErrorCode::Unsupported, "unsupported WAV format: tag=" + std::to_string(tag) +
                                            " bits=" + std::to_string(bits));
}

/// 把一个采样帧的所有声道求平均，得到单声道值。
double decode_sample(std::span<const std::byte> data, std::size_t byte_offset, SampleFormat format,
                     int channels) {
    double accumulator = 0.0;
    for (int ch = 0; ch < channels; ++ch) {
        const std::size_t offset =
            byte_offset + static_cast<std::size_t>(ch) * (format == SampleFormat::Pcm24   ? 3
                                                          : format == SampleFormat::Pcm16 ? 2
                                                          : format == SampleFormat::PcmU8 ? 1
                                                          : format == SampleFormat::Pcm32 ? 4
                                                                                          : 4);
        double value = 0.0;
        switch (format) {
            case SampleFormat::PcmU8:
                value = (static_cast<double>(data[offset]) - 128.0) / 128.0;
                break;
            case SampleFormat::Pcm16: {
                std::int16_t raw = 0;
                std::memcpy(&raw, data.data() + offset, sizeof(raw));
                value = static_cast<double>(raw) / 32768.0;
                break;
            }
            case SampleFormat::Pcm24: {
                std::int32_t raw = static_cast<std::int32_t>(data[offset]) |
                                   (static_cast<std::int32_t>(data[offset + 1]) << 8) |
                                   (static_cast<std::int32_t>(data[offset + 2]) << 16);
                if ((raw & 0x800000) != 0) {
                    raw |= static_cast<std::int32_t>(0xFF000000);
                }
                value = static_cast<double>(raw) / 8388608.0;
                break;
            }
            case SampleFormat::Pcm32: {
                std::int32_t raw = 0;
                std::memcpy(&raw, data.data() + offset, sizeof(raw));
                value = static_cast<double>(raw) / 2147483648.0;
                break;
            }
            case SampleFormat::Float32: {
                float raw = 0.0F;
                std::memcpy(&raw, data.data() + offset, sizeof(raw));
                value = static_cast<double>(raw);
                break;
            }
            case SampleFormat::Float64: {
                double raw = 0.0;
                std::memcpy(&raw, data.data() + offset, sizeof(raw));
                value = raw;
                break;
            }
        }
        accumulator += value;
    }
    return accumulator / static_cast<double>(channels);
}

int bytes_per_sample(SampleFormat format) {
    switch (format) {
        case SampleFormat::PcmU8:
            return 1;
        case SampleFormat::Pcm16:
            return 2;
        case SampleFormat::Pcm24:
            return 3;
        case SampleFormat::Pcm32:
            return 4;
        case SampleFormat::Float32:
            return 4;
        case SampleFormat::Float64:
            return 8;
    }
    return 0;
}

// ------------------------------------------------------------------ 重采样

/// 归一化 sinc。
double sinc(double x) {
    if (std::abs(x) < 1e-9) {
        return 1.0;
    }
    const double pix = kPi * x;
    return std::sin(pix) / pix;
}

/// Blackman 窗，x ∈ [-1, 1]。
double blackman(double x) {
    if (x <= -1.0 || x >= 1.0) {
        return 0.0;
    }
    return 0.42 + 0.5 * std::cos(kPi * x) + 0.08 * std::cos(2.0 * kPi * x);
}

/// 带限 sinc 插值重采样。
///
/// y(t) = Σ_k x[k] · fc · sinc(fc·(t-k)) · w((t-k)/half)
///
/// 这是标准的窗函数法带限插值：fc 是相对于输入奈奎斯特频率的截止系数
/// （降采样时 <1，抑制混叠），w 是 Blackman 窗。
///
/// @param half 每侧的抽头数（半长）。
std::vector<float> resample_windowed_sinc(std::span<const float> input, int input_rate,
                                          int output_rate, int half) {
    const double ratio = static_cast<double>(output_rate) / static_cast<double>(input_rate);
    const double fc = std::min(1.0, ratio);
    const double step = 1.0 / ratio;  // 每个输出样本对应的输入位置步长

    const auto output_size =
        static_cast<std::size_t>(std::llround(static_cast<double>(input.size()) * ratio));
    std::vector<float> output(output_size);

    const auto n = static_cast<double>(input.size());
    for (std::size_t out_index = 0; out_index < output_size; ++out_index) {
        const double position = static_cast<double>(out_index) * step;
        const auto center = static_cast<long long>(std::floor(position));

        double accumulator = 0.0;
        for (long long k = center - half + 1; k <= center + half; ++k) {
            const double distance = position - static_cast<double>(k);
            const double weight = fc * sinc(fc * distance) * blackman(distance / half);
            if (weight == 0.0) {
                continue;
            }
            if (k < 0 || k >= static_cast<long long>(n)) {
                continue;  // 边界外按零填充
            }
            accumulator += static_cast<double>(input[static_cast<std::size_t>(k)]) * weight;
        }
        output[out_index] = static_cast<float>(accumulator);
    }
    return output;
}

}  // namespace

// ------------------------------------------------------------------ AudioBuffer

double AudioBuffer::duration_seconds() const noexcept {
    if (sample_rate <= 0) {
        return 0.0;
    }
    return static_cast<double>(samples.size()) / static_cast<double>(sample_rate);
}

float AudioBuffer::peak() const noexcept {
    float peak_value = 0.0F;
    for (const float sample : samples) {
        peak_value = std::max(peak_value, std::abs(sample));
    }
    return peak_value;
}

void AudioBuffer::normalize_peak() noexcept {
    const float peak_value = peak();
    if (peak_value <= 0.0F) {
        return;
    }
    const float gain = 1.0F / peak_value;
    for (float& sample : samples) {
        sample *= gain;
    }
}

void AudioBuffer::clamp() noexcept {
    for (float& sample : samples) {
        sample = std::clamp(sample, -1.0F, 1.0F);
    }
}

// ------------------------------------------------------------------ WavCodec

WavInfo WavCodec::inspect(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw_error(ErrorCode::IoError, "cannot open WAV file: " + path);
    }
    file.seekg(0, std::ios::end);
    const auto size = static_cast<std::size_t>(file.tellg());
    file.seekg(0, std::ios::beg);
    std::vector<std::byte> data(size);
    if (size > 0) {
        file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size));
    }

    if (data.size() < 12 || tag_at(data, 0) != "RIFF" || tag_at(data, 8) != "WAVE") {
        throw_error(ErrorCode::IoError, "not a RIFF/WAVE file: " + path);
    }

    std::size_t offset = 12;
    std::uint16_t format_tag = 0;
    std::uint16_t channels = 0;
    std::uint32_t sample_rate = 0;
    std::uint16_t bits = 0;
    std::size_t data_bytes = 0;

    while (offset + 8 <= data.size()) {
        const std::string chunk_id = tag_at(data, offset);
        const std::uint32_t chunk_size = read_u32(data, offset + 4);
        const std::size_t body = offset + 8;

        if (chunk_id == "fmt " && body + 16 <= data.size()) {
            format_tag = read_u16(data, body);
            channels = read_u16(data, body + 2);
            sample_rate = read_u32(data, body + 4);
            bits = read_u16(data, body + 14);
            if (format_tag == 0xFFFE && body + 26 <= data.size()) {
                // WAVE_FORMAT_EXTENSIBLE：真实格式在 SubFormat GUID 的前两字节。
                format_tag = read_u16(data, body + 24);
            }
        } else if (chunk_id == "data") {
            data_bytes = std::min<std::size_t>(chunk_size, data.size() - body);
        }

        offset = body + chunk_size + (chunk_size % 2);  // chunk 按偶数字节对齐
    }

    if (channels == 0 || sample_rate == 0) {
        throw_error(ErrorCode::IoError, "WAV file has no usable fmt chunk: " + path);
    }

    const SampleFormat format = format_from_tag(format_tag, bits);
    const int frame_bytes = bytes_per_sample(format) * channels;

    WavInfo info;
    info.sample_rate = static_cast<int>(sample_rate);
    info.channels = channels;
    info.bits_per_sample = bits;
    info.frame_count = frame_bytes > 0 ? data_bytes / static_cast<std::size_t>(frame_bytes) : 0;
    return info;
}

AudioBuffer WavCodec::read_from_memory(std::span<const std::byte> data) {
    if (data.size() < 12 || tag_at(data, 0) != "RIFF" || tag_at(data, 8) != "WAVE") {
        throw_error(ErrorCode::IoError, "not a RIFF/WAVE buffer");
    }

    std::size_t offset = 12;
    std::uint16_t format_tag = 0;
    std::uint16_t channels = 0;
    std::uint32_t sample_rate = 0;
    std::uint16_t bits = 0;
    std::size_t data_offset = 0;
    std::size_t data_bytes = 0;

    while (offset + 8 <= data.size()) {
        const std::string chunk_id = tag_at(data, offset);
        const std::uint32_t chunk_size = read_u32(data, offset + 4);
        const std::size_t body = offset + 8;

        if (chunk_id == "fmt " && body + 16 <= data.size()) {
            format_tag = read_u16(data, body);
            channels = read_u16(data, body + 2);
            sample_rate = read_u32(data, body + 4);
            bits = read_u16(data, body + 14);
            if (format_tag == 0xFFFE && body + 26 <= data.size()) {
                format_tag = read_u16(data, body + 24);
            }
        } else if (chunk_id == "data") {
            data_offset = body;
            data_bytes = std::min<std::size_t>(chunk_size, data.size() - body);
        }

        offset = body + chunk_size + (chunk_size % 2);
    }

    if (channels == 0 || sample_rate == 0) {
        throw_error(ErrorCode::IoError, "WAV buffer has no usable fmt chunk");
    }

    const SampleFormat format = format_from_tag(format_tag, bits);
    const int sample_bytes = bytes_per_sample(format);
    const auto frame_bytes = static_cast<std::size_t>(sample_bytes) * channels;
    if (frame_bytes == 0) {
        throw_error(ErrorCode::IoError, "WAV buffer has zero-sized frames");
    }

    const std::size_t frames = data_bytes / frame_bytes;
    AudioBuffer audio;
    audio.sample_rate = static_cast<int>(sample_rate);
    audio.samples.resize(frames);
    for (std::size_t i = 0; i < frames; ++i) {
        audio.samples[i] = static_cast<float>(
            decode_sample(data, data_offset + i * frame_bytes, format, channels));
    }
    return audio;
}

AudioBuffer WavCodec::read(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw_error(ErrorCode::IoError, "cannot open WAV file: " + path);
    }
    file.seekg(0, std::ios::end);
    const auto size = static_cast<std::size_t>(file.tellg());
    file.seekg(0, std::ios::beg);
    std::vector<std::byte> data(size);
    if (size > 0) {
        file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size));
        if (!file) {
            throw_error(ErrorCode::IoError, "failed to read WAV file: " + path);
        }
    }
    return read_from_memory(std::span<const std::byte>(data.data(), data.size()));
}

void WavCodec::write_pcm16(const std::string& path, const AudioBuffer& audio) {
    const std::uint16_t channels = 1;
    const std::uint16_t bits = 16;
    const std::uint16_t format_tag = 1;
    const auto sample_rate = static_cast<std::uint32_t>(audio.sample_rate);
    const std::uint32_t byte_rate = sample_rate * channels * (bits / 8);
    const std::uint16_t block_align = channels * (bits / 8);
    const auto data_bytes = static_cast<std::uint32_t>(audio.samples.size() * 2);

    std::vector<std::byte> out;
    out.reserve(44 + data_bytes);

    append_tag(out, "RIFF");
    append_u32(out, 36 + data_bytes);
    append_tag(out, "WAVE");
    append_tag(out, "fmt ");
    append_u32(out, 16);
    append_u16(out, format_tag);
    append_u16(out, channels);
    append_u32(out, sample_rate);
    append_u32(out, byte_rate);
    append_u16(out, block_align);
    append_u16(out, bits);
    append_tag(out, "data");
    append_u32(out, data_bytes);

    for (const float sample : audio.samples) {
        const float clamped = std::clamp(sample, -1.0F, 1.0F);
        const auto value =
            static_cast<std::int16_t>(std::lround(static_cast<double>(clamped) * 32767.0));
        const auto offset = out.size();
        out.resize(offset + 2);
        std::memcpy(out.data() + offset, &value, sizeof(value));
    }

    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw_error(ErrorCode::IoError, "cannot write WAV file: " + path);
    }
    file.write(reinterpret_cast<const char*>(out.data()), static_cast<std::streamsize>(out.size()));
    if (!file) {
        throw_error(ErrorCode::IoError, "failed to write WAV file: " + path);
    }
}

void WavCodec::write_float32(const std::string& path, const AudioBuffer& audio) {
    const std::uint16_t channels = 1;
    const std::uint16_t bits = 32;
    const std::uint16_t format_tag = 3;
    const auto sample_rate = static_cast<std::uint32_t>(audio.sample_rate);
    const std::uint32_t byte_rate = sample_rate * channels * (bits / 8);
    const std::uint16_t block_align = channels * (bits / 8);
    const auto data_bytes = static_cast<std::uint32_t>(audio.samples.size() * 4);

    std::vector<std::byte> out;
    out.reserve(44 + data_bytes);

    append_tag(out, "RIFF");
    append_u32(out, 36 + data_bytes);
    append_tag(out, "WAVE");
    append_tag(out, "fmt ");
    append_u32(out, 16);
    append_u16(out, format_tag);
    append_u16(out, channels);
    append_u32(out, sample_rate);
    append_u32(out, byte_rate);
    append_u16(out, block_align);
    append_u16(out, bits);
    append_tag(out, "data");
    append_u32(out, data_bytes);

    for (const float sample : audio.samples) {
        const auto offset = out.size();
        out.resize(offset + 4);
        std::memcpy(out.data() + offset, &sample, sizeof(sample));
    }

    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw_error(ErrorCode::IoError, "cannot write WAV file: " + path);
    }
    file.write(reinterpret_cast<const char*>(out.data()), static_cast<std::streamsize>(out.size()));
    if (!file) {
        throw_error(ErrorCode::IoError, "failed to write WAV file: " + path);
    }
}

// ------------------------------------------------------------------ 重采样

std::vector<float> resample(std::span<const float> input, int input_rate, int output_rate) {
    if (input_rate <= 0 || output_rate <= 0) {
        throw_error(ErrorCode::InvalidArgument, "sample rates must be positive");
    }
    if (input_rate == output_rate || input.empty()) {
        return std::vector<float>(input.begin(), input.end());
    }
    // 16 抽头/侧的 Blackman 窗带限 sinc，对语音频段（<8 kHz）误差远低于听阈。
    return resample_windowed_sinc(input, input_rate, output_rate, 16);
}

std::vector<float> resample_linear(std::span<const float> input, int input_rate, int output_rate) {
    if (input_rate <= 0 || output_rate <= 0) {
        throw_error(ErrorCode::InvalidArgument, "sample rates must be positive");
    }
    if (input_rate == output_rate || input.empty()) {
        return std::vector<float>(input.begin(), input.end());
    }
    const double ratio = static_cast<double>(output_rate) / static_cast<double>(input_rate);
    const auto output_size =
        static_cast<std::size_t>(std::llround(static_cast<double>(input.size()) * ratio));
    std::vector<float> output(output_size);
    for (std::size_t i = 0; i < output_size; ++i) {
        const double position = static_cast<double>(i) / ratio;
        const auto index = static_cast<std::size_t>(position);
        if (index + 1 >= input.size()) {
            output[i] = input.back();
            continue;
        }
        const double frac = position - static_cast<double>(index);
        output[i] = static_cast<float>(static_cast<double>(input[index]) * (1.0 - frac) +
                                       static_cast<double>(input[index + 1]) * frac);
    }
    return output;
}

}  // namespace genie
