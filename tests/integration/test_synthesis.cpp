// tests/integration/test_synthesis.cpp
//
// 端到端推理链路测试：直接用真实 ONNX 图跑通 V2 的 T2S 链路。
//
//   encoder → first_stage_decoder → stage_decoder(自回归循环) → vits
//
// 这里刻意**不**依赖 G2P 与参考音频特征：输入用合成张量填充，
// 目的不是检验语音质量，而是检验 C++ 侧的绑定逻辑、张量形状流转、
// KV cache 递推与输出解析是否正确。
//
// 语音质量的验收属于 golden dataset 里程碑（M0.2 / M4.1）。

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "genie/config.hpp"
#include "genie/errors.hpp"
#include "genie/session.hpp"
#include "genie/tensor.hpp"
#include "support/model_paths.hpp"

using namespace genie;
using namespace genie::test;

namespace {

constexpr int kRefPhonemes = 16;
constexpr int kTextPhonemes = 24;
constexpr int kSslFrames = 40;
constexpr int kBertDim = 1024;
constexpr int kSslDim = 768;

/// 构造一个可运行的 T2S 链路驱动器。
///
/// 注意：这里刻意不用 gtest 的 ASSERT_* —— 断言宏在返回值的函数里无法编译
/// （它展开成 `return;`）。契约违反一律抛异常，由测试用例捕获。
class T2sChain {
public:
    T2sChain(const std::filesystem::path& model_dir) {
        const SessionConfig config;
        encoder_ = create_session((model_dir / "v2" / "t2s_encoder_fp32.onnx").string(), config);
        first_stage_ = create_session(
            (model_dir / "v2" / "t2s_first_stage_decoder_fp32.onnx").string(), config);
        stage_decoder_ =
            create_session((model_dir / "v2" / "t2s_stage_decoder_fp32.onnx").string(), config);
    }

    /// 跑一次 encoder，返回 (x, prompts)。
    std::pair<Tensor, Tensor> encode() {
        const std::vector<std::int64_t> ref_seq(kRefPhonemes, 5);
        const std::vector<std::int64_t> text_seq(kTextPhonemes, 7);
        const std::vector<float> ref_bert(static_cast<std::size_t>(kRefPhonemes) * kBertDim, 0.0F);
        const std::vector<float> text_bert(static_cast<std::size_t>(kTextPhonemes) * kBertDim,
                                           0.0F);
        const std::vector<float> ssl(static_cast<std::size_t>(kSslDim) * kSslFrames, 0.0F);

        const auto outputs = encoder_->run({
            {"ref_seq", Tensor::from_int64({1, kRefPhonemes}, ref_seq)},
            {"text_seq", Tensor::from_int64({1, kTextPhonemes}, text_seq)},
            {"ref_bert", Tensor::from_float32({kRefPhonemes, kBertDim}, ref_bert)},
            {"text_bert", Tensor::from_float32({kTextPhonemes, kBertDim}, text_bert)},
            {"ssl_content", Tensor::from_float32({1, kSslDim, kSslFrames}, ssl)},
        });
        if (outputs.size() != 2U) {
            throw std::runtime_error("encoder did not return x and prompts");
        }
        return {outputs[0], outputs[1]};
    }

    /// 跑 first_stage + stage_decoder 循环，返回最后的语义 token。
    ///
    /// @param max_steps 循环上限（上游为 500）
    std::vector<std::int64_t> decode(std::size_t max_steps) {
        auto [x, prompts] = encode();

        auto first = first_stage_->run({{"x", x}, {"prompts", prompts}});
        if (first.size() != 50U) {
            throw std::runtime_error("first_stage_decoder output count mismatch");
        }

        Tensor y = first[0];
        Tensor y_emb = first[1];

        // present_k/v_layer_N → past_k/v_layer_N
        TensorFeed past;
        past.reserve(48);
        for (std::size_t i = 0; i < 24; ++i) {
            past.emplace_back("past_k_layer_" + std::to_string(i), first[2 + i * 2]);
            past.emplace_back("past_v_layer_" + std::to_string(i), first[3 + i * 2]);
        }

        for (std::size_t step = 0; step < max_steps; ++step) {
            TensorFeed feed;
            feed.emplace_back("iy", y);
            feed.emplace_back("iy_emb", y_emb);
            for (auto& [name, tensor] : past) {
                feed.emplace_back(name, tensor);
            }

            auto outputs = stage_decoder_->run(feed);
            if (outputs.size() != 51U) {
                throw std::runtime_error("stage_decoder output count mismatch");
            }

            y = outputs[0];
            y_emb = outputs[1];

            const auto stop = outputs[2].scalar();
            if (!stop.has_value()) {
                throw std::runtime_error("stop_condition_tensor is not a scalar");
            }
            if (std::get<bool>(*stop)) {
                break;
            }

            past.clear();
            for (std::size_t i = 0; i < 24; ++i) {
                past.emplace_back("past_k_layer_" + std::to_string(i), outputs[3 + i * 2]);
                past.emplace_back("past_v_layer_" + std::to_string(i), outputs[4 + i * 2]);
            }
        }

        // 上游：剔除 >= 1024 的非法 token。
        const auto tokens = y.ints();
        std::vector<std::int64_t> semantic;
        for (const std::int64_t token : tokens) {
            if (token >= 1024) {
                continue;
            }
            semantic.push_back(token);
        }
        if (semantic.empty()) {
            semantic.push_back(0);
        }
        return semantic;
    }

private:
    std::unique_ptr<ModelSession> encoder_;
    std::unique_ptr<ModelSession> first_stage_;
    std::unique_ptr<ModelSession> stage_decoder_;
};

class V2ChainTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!model_exists("v2/t2s_encoder_fp32.onnx")) {
            GTEST_SKIP() << "test models not available at " << model_root().string();
        }
    }
};

}  // namespace

TEST_F(V2ChainTest, EncoderProducesXAndPrompts) {
    T2sChain chain(model_root());
    auto [x, prompts] = chain.encode();

    ASSERT_GE(x.rank(), 2U);
    EXPECT_EQ(x.shape()[0], 1);
    EXPECT_EQ(x.shape()[2], 512);  // 隐层维度

    EXPECT_EQ(prompts.type(), ElementType::Int64);
    EXPECT_EQ(prompts.shape()[0], 1);
}

TEST_F(V2ChainTest, DecodeLoopProducesSemanticTokens) {
    T2sChain chain(model_root());
    const auto semantic = chain.decode(8);
    EXPECT_FALSE(semantic.empty());
    for (const std::int64_t token : semantic) {
        EXPECT_LT(token, 1024);
    }
}

TEST_F(V2ChainTest, VocoderProducesAudio) {
    T2sChain chain(model_root());
    const auto semantic = chain.decode(4);

    // pred_semantic 的形状是 [1, 1, pred_length] —— 中间有 singleton 维。
    Tensor semantic_tensor =
        Tensor::from_int64({1, 1, static_cast<std::int64_t>(semantic.size())}, semantic);

    const std::vector<std::int64_t> text_seq(kTextPhonemes, 7);
    const std::vector<float> ref_audio(32000, 0.0F);

    const SessionConfig config;
    auto vocoder = create_session((model_root() / "v2" / "vits_fp32.onnx").string(), config);
    const auto outputs = vocoder->run({
        {"text_seq", Tensor::from_int64({1, kTextPhonemes}, text_seq)},
        {"pred_semantic", semantic_tensor},
        {"ref_audio", Tensor::from_float32({1, 32000}, ref_audio)},
    });

    ASSERT_EQ(outputs.size(), 1U);
    EXPECT_EQ(outputs[0].type(), ElementType::Float32);
    EXPECT_GT(outputs[0].element_count(), 0U);
}
