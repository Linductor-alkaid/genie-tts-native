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

#include <cmath>
#include <cstdlib>
#include "genie/audio.hpp"
#include "genie/engine.hpp"

#include "genie/errors.hpp"
#include "genie/reference.hpp"
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

TEST(Engine, ReportsStageTiming) {
    TempVoicePackage package;
    Engine engine(package.root(), fake_factory());

    // 加载报告：构造期就有内容（voice_package + 四个模型 + reference）。
    const auto& load_report = engine.load_timing();
    EXPECT_FALSE(load_report.stages().empty());
    EXPECT_GT(load_report.total_milliseconds(), 0.0);

    const auto result = engine.synthesize("\xE4\xBD\xA0\xE5\xA5\xBD");
    const auto& timing = result.timing;

    const auto find = [&timing](const std::string& name) {
        for (const auto& stage : timing.stages()) {
            if (stage.name == name) {
                return &stage;
            }
        }
        return static_cast<const StageTiming*>(nullptr);
    };

    // 顺序阶段一个不缺；t2s 的三个子阶段也要在。
    for (const char* name : {"language_detect", "frontend", "t2s", "t2s/encoder", "t2s/first_stage",
                             "t2s/stage_decoder", "vocoder"}) {
        EXPECT_NE(find(name), nullptr) << "missing stage: " << name;
    }

    // 这个包没有 RoBERTa 角色 —— text_bert 不应出现。
    EXPECT_EQ(find("text_bert"), nullptr);

    // 每个阶段都不超过端到端总时长；自回归循环至少跑了一步。
    const double total = timing.total_milliseconds();
    for (const auto& stage : timing.stages()) {
        EXPECT_LE(stage.milliseconds, total) << "stage: " << stage.name;
        EXPECT_GT(stage.count, 0U) << "stage: " << stage.name;
    }
    const auto* stage_decoder = find("t2s/stage_decoder");
    ASSERT_NE(stage_decoder, nullptr);
    EXPECT_GE(stage_decoder->count, 1U);

    // 同一份报告的两种呈现都要有内容（CLI / 基准工具依赖）。
    EXPECT_FALSE(timing.to_string().empty());
    EXPECT_NE(timing.to_json().find("\"total_ms\""), std::string::npos);
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

TEST(Engine, RobertaFeaturesFlowIntoEncoder) {
    // 音色包带 RoBERTa 时，text_bert / ref_bert 必须换成真特征
    // （假会话的第 j 行 = token id，据此核对"特征确实经过了
    // 分词与展开"），而不是零张量降级。
    TempVoicePackage package;

    // RoBERTa 模型占位文件 + 精简词表（含参考文本"参考"与"你好"所需 token）。
    {
        std::ofstream roberta_model(package.root() + "/models/roberta.onnx");
        roberta_model << "placeholder";
    }
    {
        std::ofstream vocab(package.root() + "/models/vocab.txt");
        vocab << "[PAD]\n[UNK]\n[CLS]\n[SEP]\n\xE5\x8F\x82\n\xE8\x80\x83\n"
                 "\xE4\xBD\xA0\n\xE5\xA5\xBD\n,\n.\n";
    }
    {
        std::ofstream manifest(package.root() + "/manifest.json");
        manifest << R"({
  "name": "fake",
  "format_version": 1,
  "model_version": "v2",
  "language": "chinese",
  "sample_rate": 32000,
  "models": [
    {"role": "t2s_encoder", "file": "models/t2s_encoder.onnx"},
    {"role": "t2s_first_stage_decoder", "file": "models/t2s_first_stage_decoder.onnx"},
    {"role": "t2s_stage_decoder", "file": "models/t2s_stage_decoder.onnx"},
    {"role": "vits", "file": "models/vits.onnx"},
    {"role": "roberta", "file": "models/roberta.onnx"}
  ]
})";
    }

    FakeSession* encoder = nullptr;
    const auto factory = [&encoder](const std::string& path, const SessionConfig&) {
        std::unique_ptr<ModelSession> session;
        if (path.find("t2s_encoder") != std::string::npos) {
            auto fake = FakeSession::encoder();
            encoder = fake.get();
            session = std::move(fake);
        } else if (path.find("first_stage") != std::string::npos) {
            session = FakeSession::first_stage();
        } else if (path.find("stage_decoder") != std::string::npos) {
            session = FakeSession::stage(3);
        } else if (path.find("vits") != std::string::npos) {
            session = FakeSession::vocoder(64);
        } else if (path.find("roberta") != std::string::npos) {
            session = FakeSession::roberta();
        }
        return session;
    };

    Engine engine(package.root(), factory);
    const auto result = engine.synthesize("\xE4\xBD\xA0\xE5\xA5\xBD");
    ASSERT_NE(encoder, nullptr);

    // 找到 ref_bert 与 text_bert 绑定。
    bool has_ref_bert = false;
    bool has_text_bert = false;
    for (const auto& [name, tensor] : encoder->last_feed()) {
        if (name == "ref_bert") {
            has_ref_bert = true;
            ASSERT_EQ(tensor.rank(), 2);
            EXPECT_EQ(tensor.dim(1), kBertDimension);
            // 参考文本"参考" → [CLS] 参 考 [SEP]，首行是 [CLS] 的 token id。
            // 迷你词表：0=[PAD] 1=[UNK] 2=[CLS] 3=[SEP]。
            EXPECT_FLOAT_EQ(tensor.floats()[0], 2.0F);
        }
        if (name == "text_bert") {
            has_text_bert = true;
            ASSERT_EQ(tensor.rank(), 2);
            EXPECT_EQ(tensor.dim(1), kBertDimension);
            // "。你好" → 规范化为 ".你好" → [CLS] . 你 好 [SEP]，首行是 2。
            EXPECT_FLOAT_EQ(tensor.floats()[0], 2.0F);
        }
    }
    EXPECT_TRUE(has_ref_bert);
    EXPECT_TRUE(has_text_bert);
    EXPECT_GT(result.audio.frame_count(), 0U);
}

TEST(Engine, EnglishTextSkipsChineseRoberta) {
    // 上游的 RoBERTa 是**中文**模型 ✓ —— 英文 / 韩文 / 日文三条路径给的是
    // 全零特征 ✓（见上游 `_get_phones_and_bert_pure_lang` ✓）。
    //
    // 以前这里无条件跑 RoBERTa ✓ —— 英文音素与中文 RoBERTa 的 token 数
    // 对不上 ✓ → expand_by_word2ph 直接抛异常 ✓，
    // 于是"音色包里带了 RoBERTa" + "合成英文" = 合成失败 ✓。
    TempVoicePackage package;

    {
        std::ofstream roberta_model(package.root() + "/models/roberta.onnx");
        roberta_model << "placeholder";
    }
    {
        std::ofstream vocab(package.root() + "/models/vocab.txt");
        vocab << "[PAD]\n[UNK]\n[CLS]\n[SEP]\nhello\nworld\n";
    }
    {
        // 英文词典直接用仓库里那份 ✓（manifest 的 file 给绝对路径 ✓ ——
        // std::filesystem 的 operator/ 遇到绝对路径会替换 ✓）。
        std::ofstream manifest(package.root() + "/manifest.json");
        manifest << R"({
  "name": "fake",
  "format_version": 1,
  "model_version": "v2",
  "language": "chinese",
  "sample_rate": 32000,
  "models": [
    {"role": "t2s_encoder", "file": "models/t2s_encoder.onnx"},
    {"role": "t2s_first_stage_decoder", "file": "models/t2s_first_stage_decoder.onnx"},
    {"role": "t2s_stage_decoder", "file": "models/t2s_stage_decoder.onnx"},
    {"role": "vits", "file": "models/vits.onnx"},
    {"role": "roberta", "file": "models/roberta.onnx"},
    {"role": "english_cmu_dict", "file": ")";
        manifest << GENIE_ENGLISH_CMU_DICT << R"("}
  ]
})";
    }

    FakeSession* encoder = nullptr;
    const auto factory = [&encoder](const std::string& path, const SessionConfig&) {
        std::unique_ptr<ModelSession> session;
        if (path.find("t2s_encoder") != std::string::npos) {
            auto fake = FakeSession::encoder();
            encoder = fake.get();
            session = std::move(fake);
        } else if (path.find("first_stage") != std::string::npos) {
            session = FakeSession::first_stage();
        } else if (path.find("stage_decoder") != std::string::npos) {
            session = FakeSession::stage(3);
        } else if (path.find("vits") != std::string::npos) {
            session = FakeSession::vocoder(64);
        } else if (path.find("roberta") != std::string::npos) {
            session = FakeSession::roberta();
        }
        return session;
    };

    Engine engine(package.root(), factory);
    // 英文文本 → 走英文前端 ✓ —— 不应该被中文 RoBERTa 绊倒 ✓。
    const auto result = engine.synthesize("hello world");
    EXPECT_GT(result.audio.frame_count(), 0U);
    ASSERT_NE(encoder, nullptr);

    // text_bert 必须是**全零** ✓ —— 中文 RoBERTa 没有跑 ✓。
    for (const auto& [name, tensor] : encoder->last_feed()) {
        if (name != "text_bert") {
            continue;
        }
        ASSERT_EQ(tensor.rank(), 2);
        EXPECT_EQ(tensor.dim(1), kBertDimension);
        for (const float value : tensor.floats()) {
            EXPECT_FLOAT_EQ(value, 0.0F);
        }
    }
}

TEST(Engine, JapaneseWithoutDictionaryFailsLoudly) {
    // 音色包里没有 OpenJTalk 词典时 ✓，日文**不能**静默地走中文前端 ✓ ——
    // 那样的话假名会被中文前端的 pattern_filter 过滤掉 ✓，
    // 合成出来是静音 ✓，而根因（缺 103 MB 的词典 ✓）没人看得见 ✓。
    TempVoicePackage package;

    FakeSession* encoder = nullptr;
    const auto factory = [&encoder](const std::string& path, const SessionConfig&) {
        std::unique_ptr<ModelSession> session;
        if (path.find("t2s_encoder") != std::string::npos) {
            auto fake = FakeSession::encoder();
            encoder = fake.get();
            session = std::move(fake);
        } else if (path.find("first_stage") != std::string::npos) {
            session = FakeSession::first_stage();
        } else if (path.find("stage_decoder") != std::string::npos) {
            session = FakeSession::stage(3);
        } else if (path.find("vits") != std::string::npos) {
            session = FakeSession::vocoder(64);
        }
        return session;
    };

    Engine engine(package.root(), factory);
    bool threw = false;
    try {
        // "こんにちは"（假名 ✓）→ 检测为日文 ✓。
        engine.synthesize("\xE3\x81\x93\xE3\x82\x93\xE3\x81\xAB\xE3\x81\xA1\xE3\x81\xAF");
    } catch (const Error& error) {
        threw = true;
        // 报错要说清楚缺的是什么 ✓ —— 两种情况都提到 OpenJTalk ✓：
        //   编译了 OpenJTalk 但音色包没词典 → 提示 japanese_dictionary ✓；
        //   没编译 OpenJTalk（MSVC 默认 ✓）→ 提示重新构建 ✓。
        const std::string message(error.what());
        EXPECT_NE(message.find("OpenJTalk"), std::string::npos) << message;
    }
    EXPECT_TRUE(threw) << "缺日文词典时应该明确报错，而不是静默合成静音";
}

TEST(Engine, JapaneseWithDictionarySynthesizes) {
    // JapaneseWithoutDictionaryFailsLoudly 的正向对偶 ✓：音色包登记了
    // japanese_dictionary（真实 OpenJTalk 词典 ✓，约 103 MB，不入库）时，
    // 日文前端必须在**加载期**装配 ✓ —— 只声明 japanese_frontend 成员
    // 而不装配的话 ✓，包里带了词典也照样报"缺词典"✓。
    //
    // 词典路径来自 GENIE_TEST_JAPANESE_DICT —— 没设或不存在时 skip
    // （与 test_onnx_graph.cpp 的"模型缺失就 skip"同理）。
    const char* dictionary_env = std::getenv("GENIE_TEST_JAPANESE_DICT");
    if (dictionary_env == nullptr ||
        !std::filesystem::exists(std::filesystem::path(dictionary_env) / "sys.dic")) {
        GTEST_SKIP() << "未设置 GENIE_TEST_JAPANESE_DICT 或词典不存在，跳过真实 OpenJTalk 路径";
    }

    TempVoicePackage package;
    // 与 make_voice_package.py 的 --japanese-dictionary 产物等价 ✓：
    // 词典目录放进 models/ + manifest 登记 role（目录条目不带 sha256 ✓）。
    std::filesystem::create_symlink(dictionary_env, std::filesystem::path(package.root()) /
                                                        "models" / "open_jtalk_dic_utf_8-1.11");
    {
        std::ofstream manifest(std::filesystem::path(package.root()) / "manifest.json");
        manifest << R"({
  "name": "fake",
  "format_version": 1,
  "model_version": "v2",
  "language": "chinese",
  "sample_rate": 32000,
  "models": [
    {"role": "t2s_encoder", "file": "models/t2s_encoder.onnx"},
    {"role": "t2s_first_stage_decoder", "file": "models/t2s_first_stage_decoder.onnx"},
    {"role": "t2s_stage_decoder", "file": "models/t2s_stage_decoder.onnx"},
    {"role": "vits", "file": "models/vits.onnx"},
    {"role": "japanese_dictionary", "file": "models/open_jtalk_dic_utf_8-1.11"}
  ]
}
)";
    }

    Engine engine(package.root(), fake_factory());
    // "こんにちは"（假名 ✓）→ 检测为日文 ✓ → 日文前端（真实 OpenJTalk）。
    const auto result =
        engine.synthesize("\xE3\x81\x93\xE3\x82\x93\xE3\x81\xAB\xE3\x81\xA1\xE3\x81\xAF");
    EXPECT_EQ(result.language, Language::Japanese);
    // 引擎给文本加"。"前缀（上游 tts() 的第一行 ✓）→ pau + 10 个音素 = 11，
    // 与真实模型 CLI 实测一致 ✓。
    EXPECT_EQ(result.phoneme_ids.size(), 11U);
    EXPECT_FALSE(result.semantic_tokens.empty());
    EXPECT_FALSE(result.audio.samples.empty());
}

/// 记录每次 run 的 input_ids 宽度的 RoBERTa 包装。
class RecordingRoberta final : public ModelSession {
public:
    explicit RecordingRoberta(std::unique_ptr<ModelSession> inner) : inner_(std::move(inner)) {}

    [[nodiscard]] const std::vector<TensorSpec>& inputs() const override {
        return inner_->inputs();
    }
    [[nodiscard]] const std::vector<TensorSpec>& outputs() const override {
        return inner_->outputs();
    }
    [[nodiscard]] const std::string& model_path() const override { return inner_->model_path(); }
    [[nodiscard]] std::vector<Tensor> run(const TensorFeed& feed) override {
        for (const auto& [name, tensor] : feed) {
            if (name == "input_ids") {
                widths_.push_back(static_cast<std::size_t>(tensor.dim(1)));
            }
        }
        return inner_->run(feed);
    }

    [[nodiscard]] const std::vector<std::size_t>& widths() const noexcept { return widths_; }

private:
    std::unique_ptr<ModelSession> inner_;
    std::vector<std::size_t> widths_;
};

TEST(Engine, LongChineseTextEncodesBertInChunks) {
    // 缺陷回归（2026-11-10）：长中文文本整段编码撞 RoBERTa 的 512 位置上限
    //（真模型报 "512 by 531" 广播错误）。前端现在按标点切 bert_chunks，
    // 引擎逐块编码 —— 每次 run 的 input_ids 宽度都必须 ≤ 512。
    TempVoicePackage package;
    {
        std::ofstream roberta_model(package.root() + "/models/roberta.onnx");
        roberta_model << "placeholder";
    }
    {
        // 词表覆盖测试文本用到的字（二零二五年秋天原生推理引擎完成了迁移）
        // 与标点 —— 缺字会变成 [UNK]，宽度不变，不影响本测试的断言。
        std::ofstream vocab(package.root() + "/models/vocab.txt");
        vocab << "[PAD]\n[UNK]\n[CLS]\n[SEP]\n\xE4\xBA\x8C\n\xE9\x9B\xB6\n"
                 "\xE4\xBA\x94\n\xE5\xB9\xB4\n\xE7\xA7\x8B\n\xE5\xA4\xA9\n"
                 ",\n.\n\xE5\x8E\x9F\n\xE7\x94\x9F\n\xE6\x8E\xA8\n\xE7\x90\x86\n"
                 "\xE5\xBC\x95\n\xE6\x93\x8E\n\xE5\xAE\x8C\n\xE6\x88\x90\n"
                 "\xE4\xBA\x86\n\xE8\xBF\x81\n\xE7\xA7\xBB\n";
    }
    {
        std::ofstream manifest(package.root() + "/manifest.json");
        manifest << R"({
  "name": "fake",
  "format_version": 1,
  "model_version": "v2",
  "language": "chinese",
  "sample_rate": 32000,
  "models": [
    {"role": "t2s_encoder", "file": "models/t2s_encoder.onnx"},
    {"role": "t2s_first_stage_decoder", "file": "models/t2s_first_stage_decoder.onnx"},
    {"role": "t2s_stage_decoder", "file": "models/t2s_stage_decoder.onnx"},
    {"role": "vits", "file": "models/vits.onnx"},
    {"role": "roberta", "file": "models/roberta.onnx"}
  ]
})";
    }

    auto recording = std::make_unique<RecordingRoberta>(FakeSession::roberta());
    auto* roberta = recording.get();
    const auto factory = [&recording](const std::string& path,
                                      const SessionConfig&) -> std::unique_ptr<ModelSession> {
        if (path.find("t2s_encoder") != std::string::npos) {
            return FakeSession::encoder();
        }
        if (path.find("first_stage") != std::string::npos) {
            return FakeSession::first_stage();
        }
        if (path.find("stage_decoder") != std::string::npos) {
            return FakeSession::stage(3);
        }
        if (path.find("vits") != std::string::npos) {
            return FakeSession::vocoder(64);
        }
        if (path.find("roberta") != std::string::npos) {
            return std::move(recording);  // NOLINT —— 移动后 roberta 指针仍有效（引擎持有）
        }
        return nullptr;
    };

    Engine engine(package.root(), factory);
    // 30 句 × 18 条 ≈ 540 个条目 > 512，必须分块。
    std::string text;
    for (int i = 0; i < 30; ++i) {
        text +=
            "\xE4\xBA\x8C\xE9\x9B\xB6\xE4\xBA\x8C\xE4\xBA\x94"
            "\xE5\xB9\xB4\xE7\xA7\x8B\xE5\xA4\xA9"
            ",\xE5\x8E\x9F\xE7\x94\x9F\xE6\x8E\xA8\xE7\x90\x86"
            "\xE5\xBC\x95\xE6\x93\x8E\xE5\xAE\x8C\xE6\x88\x90"
            "\xE4\xBA\x86\xE8\xBF\x81\xE7\xA7\xBB.";
    }
    const auto result = engine.synthesize(text);
    EXPECT_GT(result.audio.frame_count(), 0U);

    ASSERT_GT(roberta->widths().size(), 1U) << "长文本必须分多次编码";
    for (const auto width : roberta->widths()) {
        EXPECT_LE(width, 512U) << "单次编码的 token 数不得超过 RoBERTa 位置上限";
    }
}

TEST(FakeSession, HubertWithEmptyWaveDoesNotDivideByZero) {
    // 空波形不能拿来当除数 ✓ —— 否则特征里全是 inf/NaN ✓，
    // 后面会以"数值不对"的形式失败 ✓，根因却完全看不出来 ✓。
    auto session = genie::test::FakeSession::chinese_hubert();
    TensorFeed feed;
    feed.emplace_back("input_values", genie::Tensor::from_float32({1, 0}, {}));
    const auto outputs = session->run(feed);
    ASSERT_EQ(outputs.size(), 1U);
    for (const float value : outputs.front().floats()) {
        EXPECT_TRUE(std::isfinite(value)) << "特征里出现了 inf / NaN";
    }
}

TEST(Engine, HubertSslFlowsIntoEncoder) {
    // 音色包带 chinese_hubert 角色时，ssl_content 必须换成真特征
    // （假会话按波形均值填充，非零），而不是零张量降级。
    TempVoicePackage package;

    {
        std::ofstream hubert_model(package.root() + "/models/hubert.onnx");
        hubert_model << "placeholder";
    }
    {
        std::ofstream manifest(package.root() + "/manifest.json");
        manifest << R"({
  "name": "fake",
  "format_version": 1,
  "model_version": "v2",
  "language": "chinese",
  "sample_rate": 32000,
  "models": [
    {"role": "t2s_encoder", "file": "models/t2s_encoder.onnx"},
    {"role": "t2s_first_stage_decoder", "file": "models/t2s_first_stage_decoder.onnx"},
    {"role": "t2s_stage_decoder", "file": "models/t2s_stage_decoder.onnx"},
    {"role": "vits", "file": "models/vits.onnx"},
    {"role": "chinese_hubert", "file": "models/hubert.onnx"}
  ]
})";
    }

    FakeSession* encoder = nullptr;
    const auto factory = [&encoder](const std::string& path, const SessionConfig&) {
        std::unique_ptr<ModelSession> session;
        if (path.find("t2s_encoder") != std::string::npos) {
            auto fake = FakeSession::encoder();
            encoder = fake.get();
            session = std::move(fake);
        } else if (path.find("first_stage") != std::string::npos) {
            session = FakeSession::first_stage();
        } else if (path.find("stage_decoder") != std::string::npos) {
            session = FakeSession::stage(3);
        } else if (path.find("vits") != std::string::npos) {
            session = FakeSession::vocoder(64);
        } else if (path.find("hubert") != std::string::npos) {
            session = FakeSession::chinese_hubert();
        }
        return session;
    };

    Engine engine(package.root(), factory);
    const auto result = engine.synthesize("\xE4\xBD\xA0\xE5\xA5\xBD");
    ASSERT_NE(encoder, nullptr);

    bool has_ssl = false;
    for (const auto& [name, tensor] : encoder->last_feed()) {
        if (name == "ssl_content") {
            has_ssl = true;
            ASSERT_EQ(tensor.rank(), 3);
            EXPECT_EQ(tensor.dim(0), 1);
            EXPECT_EQ(tensor.dim(1), kSslChannels);
            EXPECT_GT(tensor.dim(2), 0);
            // 参考波形均值 ≈0.077（TempVoicePackage 夹具音频 0.1 ×
            // 有声占比 32000/41600，含 0.3s 静音尾巴与 PCM16/重采样偏差），
            // 零张量降级时这里是 0。
            EXPECT_NEAR(tensor.floats()[0], 0.1F * 32000.0F / 41600.0F, 1e-3F);
        }
    }
    EXPECT_TRUE(has_ssl);
    EXPECT_GT(result.audio.frame_count(), 0U);
}

/// 析构时置位的会话包装：观察引擎何时释放某个会话。
class TrackedSession final : public ModelSession {
public:
    TrackedSession(std::unique_ptr<ModelSession> inner, std::shared_ptr<bool> destroyed)
        : inner_(std::move(inner)), destroyed_(std::move(destroyed)) {}
    ~TrackedSession() override { *destroyed_ = true; }

    TrackedSession(const TrackedSession&) = delete;
    TrackedSession& operator=(const TrackedSession&) = delete;

    [[nodiscard]] const std::vector<TensorSpec>& inputs() const override {
        return inner_->inputs();
    }
    [[nodiscard]] const std::vector<TensorSpec>& outputs() const override {
        return inner_->outputs();
    }
    [[nodiscard]] std::vector<Tensor> run(const TensorFeed& feed) override {
        return inner_->run(feed);
    }
    [[nodiscard]] const std::string& model_path() const override { return inner_->model_path(); }

private:
    std::unique_ptr<ModelSession> inner_;
    std::shared_ptr<bool> destroyed_;
};

TEST(Engine, ReleasesLoadOnlySessionsAfterConstruction) {
    // HuBERT / 说话人验证 / prompt_encoder 只在加载期算参考特征与全局嵌入，
    // 结果缓存后会话即释放（常驻约 0.6 GB，docs/performance.md §内存）；
    // 合成期用的会话必须保留，且释放后合成照常。
    TempVoicePackage package(/*v2proplus=*/true);
    {
        std::ofstream hubert_model(package.root() + "/models/hubert.onnx");
        hubert_model << "placeholder";
    }
    {
        std::ofstream manifest(package.root() + "/manifest.json");
        manifest << R"({
  "name": "fake",
  "format_version": 1,
  "model_version": "v2proplus",
  "language": "chinese",
  "sample_rate": 32000,
  "capabilities": {"v2proplus": true},
  "models": [
    {"role": "t2s_encoder", "file": "models/t2s_encoder.onnx"},
    {"role": "t2s_first_stage_decoder", "file": "models/t2s_first_stage_decoder.onnx"},
    {"role": "t2s_stage_decoder", "file": "models/t2s_stage_decoder.onnx"},
    {"role": "vits", "file": "models/vits.onnx"},
    {"role": "prompt_encoder", "file": "models/prompt_encoder.onnx"},
    {"role": "speaker_verification", "file": "models/speaker_verification.onnx"},
    {"role": "chinese_hubert", "file": "models/hubert.onnx"}
  ]
})";
    }

    const auto hubert_gone = std::make_shared<bool>(false);
    const auto sv_gone = std::make_shared<bool>(false);
    const auto prompt_gone = std::make_shared<bool>(false);
    const auto vocoder_gone = std::make_shared<bool>(false);
    const auto tracked = [](std::unique_ptr<ModelSession> inner, std::shared_ptr<bool> flag) {
        return std::make_unique<TrackedSession>(std::move(inner), std::move(flag));
    };
    const auto factory = [&](const std::string& path,
                             const SessionConfig&) -> std::unique_ptr<ModelSession> {
        if (path.find("t2s_encoder") != std::string::npos) {
            return FakeSession::encoder();
        }
        if (path.find("first_stage") != std::string::npos) {
            return FakeSession::first_stage();
        }
        if (path.find("stage_decoder") != std::string::npos) {
            return FakeSession::stage(3);
        }
        if (path.find("prompt_encoder") != std::string::npos) {
            return tracked(FakeSession::prompt_encoder(), prompt_gone);
        }
        if (path.find("speaker_verification") != std::string::npos) {
            return tracked(FakeSession::speaker_verification(), sv_gone);
        }
        if (path.find("hubert") != std::string::npos) {
            return tracked(FakeSession::chinese_hubert(), hubert_gone);
        }
        if (path.find("vits") != std::string::npos) {
            return tracked(FakeSession::vocoder_v2proplus(64), vocoder_gone);
        }
        return nullptr;
    };

    Engine engine(package.root(), factory);

    EXPECT_TRUE(*hubert_gone);
    EXPECT_TRUE(*sv_gone);
    EXPECT_TRUE(*prompt_gone);
    EXPECT_FALSE(*vocoder_gone);

    bool timed = false;
    for (const auto& stage : engine.load_timing().stages()) {
        timed = timed || stage.name == "load/release_load_only_models";
    }
    EXPECT_TRUE(timed);

    // 参考特征与全局嵌入已缓存：释放之后合成照常。
    const auto result = engine.synthesize("\xE4\xBD\xA0\xE5\xA5\xBD");
    EXPECT_GT(result.audio.frame_count(), 0U);
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

// 信任模式（EngineConfig::verify_voice_package）：manifest 登记了错误的
// SHA-256 时，默认构造必须报错，显式关掉校验后构造放行 —— 文件存在性
// 检查不受开关影响（模型文件是存在的，只是内容对不上校验值）。
TEST(Engine, TrustModeSkipsPackageChecksumVerification) {
    TempVoicePackage package;
    // 把 manifest 覆写为带错误校验和的版本（TempVoicePackage 默认不写 sha256，
    // 校验是空操作 —— 这里显式登记一个必然不匹配的值）。
    std::ofstream manifest(std::filesystem::path(package.root()) / "manifest.json",
                           std::ios::trunc);
    manifest << R"({
  "name": "fake",
  "format_version": 1,
  "model_version": "v2",
  "language": "chinese",
  "sample_rate": 32000,
  "models": [
    {"role": "t2s_encoder", "file": "models/t2s_encoder.onnx",
     "sha256": "0000000000000000000000000000000000000000000000000000000000000000"},
    {"role": "t2s_first_stage_decoder", "file": "models/t2s_first_stage_decoder.onnx"},
    {"role": "t2s_stage_decoder", "file": "models/t2s_stage_decoder.onnx"},
    {"role": "vits", "file": "models/vits.onnx"}
  ]
})";
    manifest.close();
    ASSERT_TRUE(manifest.good());

    EngineConfig trusted;
    trusted.verify_voice_package = false;
    EXPECT_NO_THROW(Engine(package.root(), fake_factory(), trusted));

    EngineConfig verify;
    verify.verify_voice_package = true;
    try {
        Engine(package.root(), fake_factory(), verify);
        FAIL() << "expected checksum mismatch to throw";
    } catch (const Error& error) {
        EXPECT_EQ(error.code(), ErrorCode::ModelLoadError);
    }
}

}  // namespace
}  // namespace genie
