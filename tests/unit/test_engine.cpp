// tests/unit/test_engine.cpp
//
// 合成引擎的测试。
//
// 用假会话 + 临时音色包跑完整条流水线：
//   文本 → 音素 → t2s_encoder → first stage → stage ×N → vits → PCM
//
// 不需要 ONNX Runtime，也不需要模型权重。

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "genie/audio.hpp"
#include "genie/engine.hpp"
#include "genie/errors.hpp"
#include "genie/symbols.hpp"
#include "support/fake_session.hpp"

namespace genie {
namespace {

using test::FakeSession;

/// 临时音色包。模型文件是占位的 —— 假会话不看文件内容。
class TempVoicePackage {
public:
    /// @param v2proplus 是否构造 V2ProPlus 音色包（多两个模型 + 能力开关）
    explicit TempVoicePackage(bool v2proplus = false) : v2proplus_(v2proplus) {
        static int counter = 0;
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        root_ = std::filesystem::temp_directory_path() /
                ("genie_engine_" + std::to_string(stamp) + "_" + std::to_string(counter++));
        std::filesystem::create_directories(root_ / "models");
        root_str_ = root_.string();

        write_model("t2s_encoder.onnx");
        write_model("t2s_first_stage_decoder.onnx");
        write_model("t2s_stage_decoder.onnx");
        write_model("vits.onnx");
        if (v2proplus_) {
            write_model("prompt_encoder.onnx");
            write_model("speaker_verification.onnx");
        }

        // 参考音频与参考文本 —— T2S 编码器需要它们（上游的 prompt_audio）。
        std::filesystem::create_directories(root_ / "reference");
        {
            genie::AudioBuffer audio;
            audio.sample_rate = 32000;
            audio.samples.assign(32000, 0.1F);  // 1 秒
            genie::WavCodec::write_pcm16((root_ / "reference" / "audio.wav").string(), audio);
        }
        {
            std::ofstream reference_text(root_ / "reference" / "text.txt");
            reference_text << "\xE5\x8F\x82\xE8\x80\x83";  // 参考
        }

        std::ofstream manifest(root_ / "manifest.json");
        manifest << R"({
  "name": "fake",
  "format_version": 1,
  "model_version": )";
        if (v2proplus_) {
            manifest << R"("v2proplus",
  "language": "chinese",
  "sample_rate": 32000,
  "capabilities": {"v2proplus": true},
  "models": [
    {"role": "t2s_encoder", "file": "models/t2s_encoder.onnx"},
    {"role": "t2s_first_stage_decoder", "file": "models/t2s_first_stage_decoder.onnx"},
    {"role": "t2s_stage_decoder", "file": "models/t2s_stage_decoder.onnx"},
    {"role": "vits", "file": "models/vits.onnx"},
    {"role": "prompt_encoder", "file": "models/prompt_encoder.onnx"},
    {"role": "speaker_verification", "file": "models/speaker_verification.onnx"}
  ]
})";
        } else {
            manifest << R"("v2",
  "language": "chinese",
  "sample_rate": 32000,
  "models": [
    {"role": "t2s_encoder", "file": "models/t2s_encoder.onnx"},
    {"role": "t2s_first_stage_decoder", "file": "models/t2s_first_stage_decoder.onnx"},
    {"role": "t2s_stage_decoder", "file": "models/t2s_stage_decoder.onnx"},
    {"role": "vits", "file": "models/vits.onnx"}
  ]
})";
        }
    }

    ~TempVoicePackage() {
        std::error_code ec;
        std::filesystem::remove_all(root_, ec);
    }

    TempVoicePackage(const TempVoicePackage&) = delete;
    TempVoicePackage& operator=(const TempVoicePackage&) = delete;

    [[nodiscard]] const std::string& root() const noexcept { return root_str_; }

private:
    void write_model(const std::string& name) {
        std::ofstream file(root_ / "models" / name);
        file << "placeholder";
    }

    std::filesystem::path root_;
    std::string root_str_;
    bool v2proplus_ = false;
};

/// 按角色分发的假会话工厂。
SessionFactory fake_factory(int stop_after = 3) {
    return [stop_after](const std::string& path, const SessionConfig&) {
        std::unique_ptr<ModelSession> session;
        if (path.find("t2s_encoder") != std::string::npos) {
            session = FakeSession::encoder();
        } else if (path.find("first_stage") != std::string::npos) {
            session = FakeSession::first_stage();
        } else if (path.find("stage_decoder") != std::string::npos) {
            session = FakeSession::stage(stop_after);
        } else if (path.find("vits") != std::string::npos) {
            session = FakeSession::vocoder(64);
        }
        return session;
    };
}

/// V2ProPlus 的假会话工厂。
SessionFactory fake_factory_v2proplus(int stop_after = 3) {
    return [stop_after](const std::string& path, const SessionConfig&) {
        std::unique_ptr<ModelSession> session;
        if (path.find("t2s_encoder") != std::string::npos) {
            session = FakeSession::encoder();
        } else if (path.find("first_stage") != std::string::npos) {
            session = FakeSession::first_stage();
        } else if (path.find("stage_decoder") != std::string::npos) {
            session = FakeSession::stage(stop_after);
        } else if (path.find("prompt_encoder") != std::string::npos) {
            session = FakeSession::prompt_encoder();
        } else if (path.find("speaker_verification") != std::string::npos) {
            session = FakeSession::speaker_verification();
        } else if (path.find("vits") != std::string::npos) {
            session = FakeSession::vocoder_v2proplus(64);
        }
        return session;
    };
}

TEST(Engine, V2ProPlusComputesGlobalEmbeddings) {
    TempVoicePackage package(/*v2proplus=*/true);
    Engine engine(package.root(), fake_factory_v2proplus());

    // V2ProPlus 走的是 ge / ge_advanced，**不是** ref_audio。
    // 上游的顺序是 sv_emb → prompt_encoder → ge / ge_advanced。
    const auto result = engine.synthesize("你好");
    EXPECT_GT(result.audio.frame_count(), 0U);
}

TEST(Engine, V2ProPlusFeedsGlobalEmbeddingsToVocoder) {
    TempVoicePackage package(/*v2proplus=*/true);

    // 自己抓一次 vocoder 的输入绑定。
    // 引擎持有会话，但引擎还没析构时这个裸指针是有效的。
    FakeSession* vocoder = nullptr;
    const auto factory = [&vocoder](const std::string& path, const SessionConfig&) {
        std::unique_ptr<ModelSession> session;
        if (path.find("t2s_encoder") != std::string::npos) {
            session = FakeSession::encoder();
        } else if (path.find("first_stage") != std::string::npos) {
            session = FakeSession::first_stage();
        } else if (path.find("stage_decoder") != std::string::npos) {
            session = FakeSession::stage(3);
        } else if (path.find("prompt_encoder") != std::string::npos) {
            session = FakeSession::prompt_encoder();
        } else if (path.find("speaker_verification") != std::string::npos) {
            session = FakeSession::speaker_verification();
        } else if (path.find("vits") != std::string::npos) {
            auto fake = FakeSession::vocoder_v2proplus(64);
            vocoder = fake.get();
            session = std::move(fake);
        }
        return session;
    };

    Engine engine(package.root(), factory);
    (void)engine.synthesize("你好");

    ASSERT_NE(vocoder, nullptr);
    ASSERT_FALSE(vocoder->last_feed().empty());

    // ge 必须是 [1, 1024, 1]、ge_advanced 必须是 [1, 512, 1]。
    bool has_ge = false;
    bool has_ge_advanced = false;
    for (const auto& [name, tensor] : vocoder->last_feed()) {
        if (name == "ge") {
            has_ge = true;
            ASSERT_EQ(tensor.rank(), 3);
            EXPECT_EQ(tensor.dim(0), 1);
            EXPECT_EQ(tensor.dim(1), 1024);
            EXPECT_EQ(tensor.dim(2), 1);
        }
        if (name == "ge_advanced") {
            has_ge_advanced = true;
            ASSERT_EQ(tensor.rank(), 3);
            EXPECT_EQ(tensor.dim(0), 1);
            EXPECT_EQ(tensor.dim(1), 512);
            EXPECT_EQ(tensor.dim(2), 1);
        }
    }
    EXPECT_TRUE(has_ge);
    EXPECT_TRUE(has_ge_advanced);
}

TEST(Engine, V2ProPlusWithoutSpeakerModelFailsAtLoad) {
    // 说话人验证模型是 V2ProPlus 的硬依赖 ——
    // 算不出 ge / ge_advanced，V2ProPlus 分支根本没法用。
    // 必须在**加载期**报清楚，而不是等到推理时炸。
    TempVoicePackage package(/*v2proplus=*/true);
    std::filesystem::remove(package.root() + "/models/speaker_verification.onnx");

    EXPECT_THROW(Engine(package.root(), fake_factory_v2proplus()), Error);
}

TEST(Engine, SynthesizesEndToEnd) {
    TempVoicePackage package;
    Engine engine(package.root(), fake_factory());

    const auto result = engine.synthesize("\xE4\xBD\xA0\xE5\xA5\xBD");

    EXPECT_EQ(result.audio.samples.size(), 64U);
    EXPECT_EQ(result.audio.sample_rate, 32000);
    EXPECT_FALSE(result.phoneme_ids.empty());
    EXPECT_FALSE(result.semantic_tokens.empty());
    EXPECT_EQ(result.language, Language::Chinese);
}

TEST(Engine, PrependsUpstreamSentenceMark) {
    // 上游 `tts()` 第一行是 `text = '。' + text`。
    // 音素序列因此会比原文多出句号的音素。
    TempVoicePackage package;
    Engine engine(package.root(), fake_factory());

    const auto result = engine.synthesize("\xE4\xBD\xA0");
    // 句号 → 1 个音素；你 → 2 个音素。
    EXPECT_EQ(result.phoneme_ids.size(), 3U);
    EXPECT_EQ(result.phoneme_ids.front(), symbols::id_of("."));
}

TEST(Engine, RejectsEmptyText) {
    TempVoicePackage package;
    Engine engine(package.root(), fake_factory());
    EXPECT_THROW(engine.synthesize(""), Error);
}

TEST(Engine, ReportsManifest) {
    TempVoicePackage package;
    Engine engine(package.root(), fake_factory());
    EXPECT_EQ(engine.manifest().name, "fake");
    EXPECT_EQ(engine.manifest().language, Language::Chinese);
    EXPECT_EQ(engine.sample_rate(), 32000);
}

TEST(Engine, StreamingDeliversAudioAndCloses) {
    TempVoicePackage package;
    Engine engine(package.root(), fake_factory());

    auto channel = engine.synthesize_streaming("\xE4\xBD\xA0\xE5\xA5\xBD");

    AudioBuffer chunk;
    bool got_audio = false;
    // 通道在合成结束后会关闭 —— 读到"取不出且已关闭"即结束。
    for (int i = 0; i < 200; ++i) {
        if (channel->try_pop(chunk)) {
            got_audio = !chunk.empty();
            continue;
        }
        if (channel->closed()) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    EXPECT_TRUE(got_audio);
    EXPECT_TRUE(channel->closed());
}

TEST(Engine, MissingModelFailsAtConstruction) {
    TempVoicePackage package;
    // 工厂对未知路径返回空 → 构造期就该报错。
    EXPECT_THROW(Engine(package.root(),
                        [](const std::string&, const SessionConfig&) {
                            return std::unique_ptr<ModelSession>();
                        }),
                 Error);
}

TEST(Engine, CancelsStreaming) {
    TempVoicePackage package;
    // 停止条件永不触发 → 合成会一直跑，可以被取消。
    Engine engine(package.root(), fake_factory(/*stop_after=*/-1));
    auto channel = engine.synthesize_streaming("\xE4\xBD\xA0\xE5\xA5\xBD");
    EXPECT_TRUE(engine.cancel_streaming());
    (void)channel;
}

}  // namespace
}  // namespace genie
