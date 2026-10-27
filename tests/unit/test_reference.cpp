// tests/unit/test_reference.cpp
//
// 参考音频条件的测试。
//
// 这里的重点是**形状** —— 形状错了模型不会报错，只会输出垃圾。

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "genie/audio.hpp"
#include "genie/errors.hpp"
#include "genie/reference.hpp"
#include "genie/text.hpp"

namespace genie {
namespace {

/// 临时音色包，带参考音频。
class TempPackage {
public:
    explicit TempPackage(int sample_rate = 32000, int frames = 32000) {
        static int counter = 0;
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        root_ = std::filesystem::temp_directory_path() /
                ("genie_ref_" + std::to_string(stamp) + "_" + std::to_string(counter++));
        std::filesystem::create_directories(root_ / "reference");
        root_str_ = root_.string();

        AudioBuffer audio;
        audio.sample_rate = sample_rate;
        // 440 Hz 的正弦 —— 重采样后能看出频率对不对。
        for (int i = 0; i < frames; ++i) {
            const double t = static_cast<double>(i) / sample_rate;
            audio.samples.push_back(
                static_cast<float>(0.5 * std::sin(2.0 * 3.14159265358979 * 440.0 * t)));
        }
        WavCodec::write_pcm16((root_ / "reference" / "audio.wav").string(), audio);

        std::ofstream text(root_ / "reference" / "text.txt");
        text << "\xE5\x8F\x82\xE8\x80\x83";  // 参考
    }

    ~TempPackage() {
        std::error_code ec;
        std::filesystem::remove_all(root_, ec);
    }

    TempPackage(const TempPackage&) = delete;
    TempPackage& operator=(const TempPackage&) = delete;

    [[nodiscard]] const std::string& root() const noexcept { return root_str_; }

    /// 删掉参考音频。
    void remove_audio() {
        std::error_code ec;
        std::filesystem::remove(root_ / "reference" / "audio.wav", ec);
    }

private:
    std::filesystem::path root_;
    std::string root_str_;
};

std::unique_ptr<text::TextFrontend> chinese() {
    return text::make_frontend(Language::Chinese);
}

TEST(Reference, LoadsAudioAndText) {
    TempPackage package;
    const auto frontend = chinese();
    ASSERT_NE(frontend, nullptr);

    const auto condition = load_reference(package.root(), *frontend);

    EXPECT_TRUE(condition.usable());
    EXPECT_EQ(condition.audio.sample_rate, 32000);
    EXPECT_EQ(condition.audio.frame_count(), 32000U);
    EXPECT_EQ(condition.text, "\xE5\x8F\x82\xE8\x80\x83");
    // 参考文本 "参考" → 4 个音素：
    //   参 = c + an1，考 = k + ao3
    EXPECT_EQ(condition.phoneme_ids.size(), 4U);
}

TEST(Reference, ResamplesToTargetRate) {
    // 44.1 kHz 的参考音频要重采样到 32 kHz。
    TempPackage package(44100, 44100);
    const auto frontend = chinese();
    ASSERT_NE(frontend, nullptr);

    const auto condition = load_reference(package.root(), *frontend);

    EXPECT_EQ(condition.audio.sample_rate, 32000);
    // 1 秒的音频 → 32000 个采样。
    EXPECT_NEAR(static_cast<double>(condition.audio.frame_count()), 32000.0, 2.0);
}

TEST(Reference, FeatureShapesAreExplicit) {
    TempPackage package;
    const auto frontend = chinese();
    ASSERT_NE(frontend, nullptr);

    const auto condition = load_reference(package.root(), *frontend);

    // ref_bert：[音素数, 1024] —— **秩 2**，上游 ONNX 图就是这么声明的。
    ASSERT_EQ(condition.bert.rank(), 2);
    EXPECT_EQ(condition.bert.dim(0), static_cast<std::int64_t>(condition.phoneme_ids.size()));
    EXPECT_EQ(condition.bert.dim(1), kBertDimension);

    // ssl_content：[1, 768, 帧数]
    //
    // 帧数语义与上游 HuBERT 对齐：16 kHz 波形、步长 320 → 50 帧/秒。
    // 1 秒 32 kHz 的参考音频重采样到 16 kHz 后是 16000 采样 → 50 帧。
    // 这里钉死数值：之前按 32 kHz 帧数除 320 会多出一倍（100 帧）。
    ASSERT_EQ(condition.ssl.rank(), 3);
    EXPECT_EQ(condition.ssl.dim(0), 1);
    EXPECT_EQ(condition.ssl.dim(1), kSslChannels);
    EXPECT_NEAR(condition.ssl.dim(2), 50, 1);
}

TEST(Reference, MissingAudioIsReported) {
    TempPackage package;
    package.remove_audio();
    const auto frontend = chinese();
    ASSERT_NE(frontend, nullptr);

    // 上游的 tts() 必须有 prompt_audio —— 缺参考音频就该在加载期报错，
    // 而不是等到推理时才发现。
    EXPECT_THROW(load_reference(package.root(), *frontend), Error);
}

TEST(Reference, MissingTextIsAllowed) {
    // 参考文本是可选的：没有它就没有 ref_seq，但参考波形照常可用。
    TempPackage package;
    std::error_code ec;
    std::filesystem::remove(package.root() + "/reference/text.txt", ec);

    const auto frontend = chinese();
    ASSERT_NE(frontend, nullptr);

    const auto condition = load_reference(package.root(), *frontend);
    EXPECT_TRUE(condition.usable());
    EXPECT_TRUE(condition.phoneme_ids.empty());
    // 音素数缺失时 bert 的长度维也要有值（不能是 0）。
    EXPECT_EQ(condition.bert.dim(0), 1);
}

}  // namespace
}  // namespace genie
