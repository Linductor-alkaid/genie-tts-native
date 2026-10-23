// tests/unit/test_audio.cpp

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <string>
#include <vector>

#include "genie/audio.hpp"
#include "genie/errors.hpp"

using namespace genie;

namespace {

std::filesystem::path temp_path(const std::string& name) {
    return std::filesystem::temp_directory_path() / ("genie_test_" + name);
}

std::vector<float> sine(double frequency, int rate, double seconds) {
    const auto count = static_cast<std::size_t>(rate * seconds);
    std::vector<float> samples(count);
    for (std::size_t i = 0; i < count; ++i) {
        samples[i] =
            static_cast<float>(0.5 * std::sin(2.0 * std::numbers::pi * frequency *
                                              static_cast<double>(i) / static_cast<double>(rate)));
    }
    return samples;
}

}  // namespace

TEST(Audio, DurationAndPeak) {
    AudioBuffer audio;
    audio.sample_rate = 32000;
    audio.samples = {0.0F, 0.5F, -0.25F};
    EXPECT_DOUBLE_EQ(audio.duration_seconds(), 3.0 / 32000.0);
    EXPECT_FLOAT_EQ(audio.peak(), 0.5F);
}

TEST(Audio, NormalizePeakScalesToUnity) {
    AudioBuffer audio;
    audio.sample_rate = 32000;
    audio.samples = {0.1F, -0.2F, 0.05F};
    audio.normalize_peak();
    EXPECT_NEAR(audio.peak(), 1.0F, 1e-6);
}

TEST(Audio, ClampKeepsSamplesInRange) {
    AudioBuffer audio;
    audio.sample_rate = 32000;
    audio.samples = {1.5F, -2.0F, 0.25F};
    audio.clamp();
    EXPECT_FLOAT_EQ(audio.samples[0], 1.0F);
    EXPECT_FLOAT_EQ(audio.samples[1], -1.0F);
    EXPECT_FLOAT_EQ(audio.samples[2], 0.25F);
}

TEST(Audio, WavRoundTripPcm16) {
    const auto path = temp_path("roundtrip.wav");
    AudioBuffer original;
    original.sample_rate = 32000;
    original.samples = sine(440.0, 32000, 0.05);

    WavCodec::write_pcm16(path.string(), original);
    const AudioBuffer loaded = WavCodec::read(path.string());

    EXPECT_EQ(loaded.sample_rate, original.sample_rate);
    ASSERT_EQ(loaded.samples.size(), original.samples.size());
    for (std::size_t i = 0; i < original.samples.size(); ++i) {
        // 16-bit 量化误差上界约为 1/32768。
        EXPECT_NEAR(loaded.samples[i], original.samples[i], 1.0 / 32767.0);
    }
    std::filesystem::remove(path);
}

TEST(Audio, WavInspectReportsFormat) {
    const auto path = temp_path("inspect.wav");
    AudioBuffer audio;
    audio.sample_rate = 16000;
    audio.samples = sine(300.0, 16000, 0.01);
    WavCodec::write_pcm16(path.string(), audio);

    const WavInfo info = WavCodec::inspect(path.string());
    EXPECT_EQ(info.sample_rate, 16000);
    EXPECT_EQ(info.channels, 1);
    EXPECT_EQ(info.bits_per_sample, 16);
    EXPECT_EQ(info.frame_count, audio.samples.size());
    std::filesystem::remove(path);
}

TEST(Audio, MissingFileThrowsIoError) {
    EXPECT_THROW(WavCodec::read("/nonexistent/definitely/not/here.wav"), Error);
}

TEST(Audio, NonWavFileThrowsIoError) {
    const auto path = temp_path("not_a_wav.bin");
    {
        std::ofstream file(path, std::ios::binary);
        file << "this is definitely not a RIFF file";
    }
    EXPECT_THROW(WavCodec::read(path.string()), Error);
    std::filesystem::remove(path);
}

TEST(Audio, ResampleSameRateIsIdentity) {
    const auto input = sine(440.0, 32000, 0.01);
    const auto output = resample(input, 32000, 32000);
    ASSERT_EQ(output.size(), input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        EXPECT_FLOAT_EQ(output[i], input[i]);
    }
}

TEST(Audio, ResampleHalvesLength) {
    const auto input = sine(440.0, 32000, 0.1);
    const auto output = resample(input, 32000, 16000);
    EXPECT_NEAR(static_cast<double>(output.size()), static_cast<double>(input.size()) / 2.0, 2.0);
}

TEST(Audio, ResamplePreservesFrequency) {
    // 1 kHz 正弦：32 kHz → 16 kHz 后，周期仍应为 16 个采样。
    const double frequency = 1000.0;
    const int input_rate = 32000;
    const int output_rate = 16000;
    const auto input = sine(frequency, input_rate, 0.1);
    const auto output = resample(input, input_rate, output_rate);

    ASSERT_GT(output.size(), 32U);
    // 统计过零率来估计频率。
    std::size_t crossings = 0;
    for (std::size_t i = 1; i < output.size(); ++i) {
        if ((output[i - 1] < 0.0F) != (output[i] < 0.0F)) {
            ++crossings;
        }
    }
    const double seconds = static_cast<double>(output.size()) / output_rate;
    const double estimated = static_cast<double>(crossings) / (2.0 * seconds);
    EXPECT_NEAR(estimated, frequency, 20.0);
}

TEST(Audio, ResampleRejectsInvalidRates) {
    const std::vector<float> input(16, 0.0F);
    EXPECT_THROW(resample(input, 0, 16000), Error);
    EXPECT_THROW(resample(input, 16000, -1), Error);
}
