// tests/component/test_model_contract.cpp
//
// 模型 I/O 契约测试。
//
// 这是 M0 审计结论的**可执行版本**：把上游六个 ONNX 模型的输入输出名字、
// 元素类型、维度固化下来。任何人替换模型、升级导出脚本，都必须同步更新这里 ——
// 否则引擎会在运行时按名字绑定失败，或者更糟：静默绑定到错误的张量。
//
// 契约来源：upstream audit，onnx 1.23 shape inference，见 docs/upstream-audit.md。

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

#include "genie/config.hpp"
#include "genie/session.hpp"
#include "support/model_paths.hpp"

using namespace genie;
using namespace genie::test;

namespace {

const TensorSpec* find_spec(const std::vector<TensorSpec>& specs, const std::string& name) {
    const auto it = std::find_if(specs.begin(), specs.end(),
                                 [&name](const TensorSpec& spec) { return spec.name == name; });
    return it == specs.end() ? nullptr : &*it;
}

}  // namespace

// ---------------------------------------------------------------------------------------
// V2
// ---------------------------------------------------------------------------------------

class V2ModelContract : public ::testing::Test {
protected:
    void SetUp() override {
        if (!model_exists("v2/t2s_encoder_fp32.onnx")) {
            GTEST_SKIP() << "test models not available at " << model_root().string();
        }
        config_ = SessionConfig{};
    }

    SessionConfig config_;
};

TEST_F(V2ModelContract, EncoderHasFiveInputsAndTwoOutputs) {
    auto session = create_session(v2_model("t2s_encoder_fp32.onnx").string(), config_);
    const auto& inputs = session->inputs();
    const auto& outputs = session->outputs();

    ASSERT_EQ(inputs.size(), 5U);
    EXPECT_EQ(inputs[0].name, "ref_seq");
    EXPECT_EQ(inputs[1].name, "text_seq");
    EXPECT_EQ(inputs[2].name, "ref_bert");
    EXPECT_EQ(inputs[3].name, "text_bert");
    EXPECT_EQ(inputs[4].name, "ssl_content");

    ASSERT_EQ(outputs.size(), 2U);
    EXPECT_EQ(outputs[0].name, "x");
    EXPECT_EQ(outputs[1].name, "prompts");

    const auto* ref_seq = find_spec(inputs, "ref_seq");
    ASSERT_NE(ref_seq, nullptr);
    EXPECT_EQ(ref_seq->type, ElementType::Int64);

    const auto* ref_bert = find_spec(inputs, "ref_bert");
    ASSERT_NE(ref_bert, nullptr);
    EXPECT_EQ(ref_bert->type, ElementType::Float32);
    ASSERT_EQ(ref_bert->shape.size(), 2U);
    EXPECT_EQ(ref_bert->shape[1], 1024);  // BERT 特征维度

    const auto* ssl = find_spec(inputs, "ssl_content");
    ASSERT_NE(ssl, nullptr);
    EXPECT_EQ(ssl->type, ElementType::Float32);
    ASSERT_EQ(ssl->shape.size(), 3U);
    EXPECT_EQ(ssl->shape[1], 768);  // HuBERT 隐层维度
}

TEST_F(V2ModelContract, FirstStageDecoderExposesKvCache) {
    auto session = create_session(v2_model("t2s_first_stage_decoder_fp32.onnx").string(), config_);
    const auto& inputs = session->inputs();
    const auto& outputs = session->outputs();

    ASSERT_EQ(inputs.size(), 2U);
    EXPECT_EQ(inputs[0].name, "x");
    EXPECT_EQ(inputs[1].name, "prompts");

    // y + y_emb + 24 层 × (k, v) = 50 个输出
    ASSERT_EQ(outputs.size(), 50U);
    EXPECT_EQ(outputs[0].name, "y");
    EXPECT_EQ(outputs[1].name, "y_emb");
    EXPECT_EQ(outputs[2].name, "present_k_layer_0");
    EXPECT_EQ(outputs[3].name, "present_v_layer_0");
    EXPECT_EQ(outputs[48].name, "present_k_layer_23");
    EXPECT_EQ(outputs[49].name, "present_v_layer_23");
}

// 关键契约：stage_decoder 的输入名是 iy / iy_emb / past_k_layer_N / past_v_layer_N。
// 上游 Python 靠 zip(input_names, [...]) 按位置传参，变量名叫 y / y_emb —— 名字对不上。
// C++ 移植必须按名字绑定，否则一旦上游调整输入顺序就会静默错位。
TEST_F(V2ModelContract, StageDecoderUsesIyNamesAndPastKv) {
    auto session = create_session(v2_model("t2s_stage_decoder_fp32.onnx").string(), config_);
    const auto& inputs = session->inputs();
    const auto& outputs = session->outputs();

    ASSERT_EQ(inputs.size(), 50U);
    EXPECT_EQ(inputs[0].name, "iy");
    EXPECT_EQ(inputs[1].name, "iy_emb");
    EXPECT_EQ(inputs[2].name, "past_k_layer_0");
    EXPECT_EQ(inputs[3].name, "past_v_layer_0");
    EXPECT_EQ(inputs[48].name, "past_k_layer_23");
    EXPECT_EQ(inputs[49].name, "past_v_layer_23");

    ASSERT_EQ(outputs.size(), 51U);
    EXPECT_EQ(outputs[0].name, "y");
    EXPECT_EQ(outputs[1].name, "y_emb");
    EXPECT_EQ(outputs[2].name, "stop_condition_tensor");
    EXPECT_EQ(outputs[3].name, "present_k_layer_0");
    EXPECT_EQ(outputs[50].name, "present_v_layer_23");

    // stop_condition_tensor 是 rank-0 bool 标量。
    const auto* stop = find_spec(outputs, "stop_condition_tensor");
    ASSERT_NE(stop, nullptr);
    EXPECT_EQ(stop->type, ElementType::Bool);
    EXPECT_TRUE(stop->shape.empty());
}

TEST_F(V2ModelContract, VitsTakesTextSemanticAndReferenceAudio) {
    auto session = create_session(v2_model("vits_fp32.onnx").string(), config_);
    const auto& inputs = session->inputs();
    const auto& outputs = session->outputs();

    ASSERT_EQ(inputs.size(), 3U);
    EXPECT_EQ(inputs[0].name, "text_seq");
    EXPECT_EQ(inputs[1].name, "pred_semantic");
    EXPECT_EQ(inputs[2].name, "ref_audio");

    // pred_semantic 是 [1, 1, pred_length] —— 中间有一个 singleton 维。
    const auto* semantic = find_spec(inputs, "pred_semantic");
    ASSERT_NE(semantic, nullptr);
    ASSERT_EQ(semantic->shape.size(), 3U);
    EXPECT_EQ(semantic->shape[0], 1);
    EXPECT_EQ(semantic->shape[1], 1);

    ASSERT_EQ(outputs.size(), 1U);
    EXPECT_EQ(outputs[0].name, "audio");
}

// ---------------------------------------------------------------------------------------
// V2ProPlus
// ---------------------------------------------------------------------------------------

class V2ProPlusModelContract : public ::testing::Test {
protected:
    void SetUp() override {
        if (!model_exists("v2ProPlus/prompt_encoder_fp32.onnx")) {
            GTEST_SKIP() << "test models not available at " << model_root().string();
        }
        config_ = SessionConfig{};
    }

    SessionConfig config_;
};

TEST_F(V2ProPlusModelContract, PromptEncoderTakesAudioAndSpeakerVector) {
    auto session = create_session(v2proplus_model("prompt_encoder_fp32.onnx").string(), config_);
    const auto& inputs = session->inputs();
    const auto& outputs = session->outputs();

    ASSERT_EQ(inputs.size(), 2U);
    EXPECT_EQ(inputs[0].name, "ref_audio");
    EXPECT_EQ(inputs[1].name, "sv_emb");

    const auto* sv_emb = find_spec(inputs, "sv_emb");
    ASSERT_NE(sv_emb, nullptr);
    EXPECT_EQ(sv_emb->type, ElementType::Float32);
    ASSERT_EQ(sv_emb->shape.size(), 2U);
    EXPECT_EQ(sv_emb->shape[1], 20480);

    ASSERT_EQ(outputs.size(), 2U);
    EXPECT_EQ(outputs[0].name, "ge");
    EXPECT_EQ(outputs[1].name, "ge_advanced");
}

TEST_F(V2ProPlusModelContract, VitsTakesGlobalEmbeddings) {
    auto session = create_session(v2proplus_model("vits_fp32.onnx").string(), config_);
    const auto& inputs = session->inputs();

    ASSERT_EQ(inputs.size(), 4U);
    EXPECT_EQ(inputs[0].name, "text_seq");
    EXPECT_EQ(inputs[1].name, "pred_semantic");
    EXPECT_EQ(inputs[2].name, "ge");
    EXPECT_EQ(inputs[3].name, "ge_advanced");

    const auto* ge = find_spec(inputs, "ge");
    ASSERT_NE(ge, nullptr);
    ASSERT_EQ(ge->shape.size(), 3U);
    EXPECT_EQ(ge->shape[1], 1024);

    const auto* ge_advanced = find_spec(inputs, "ge_advanced");
    ASSERT_NE(ge_advanced, nullptr);
    ASSERT_EQ(ge_advanced->shape.size(), 3U);
    EXPECT_EQ(ge_advanced->shape[1], 512);
}
