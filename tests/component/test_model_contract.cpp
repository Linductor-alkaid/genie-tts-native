// tests/component/test_model_contract.cpp
//
// 模型 I/O 契约测试 —— M0 审计结论的**可执行版本**。
//
// 为什么不用 ONNX Runtime 加载模型：
//   上游随仓库发布的六个 ONNX 模型是**图壳** —— 权重在外部 .bin 文件里
//   （合计约 1.1 GB），由上游的转换流程生成，不在仓库中。ORT 加载它们会报
//   "cannot get file size: No such file or directory"。
//
//   但**图结构**是完整的：输入输出的名字、元素类型、维度全都在。
//   因此本测试直接解析 ONNX protobuf（genie::read_onnx_graph），
//   不需要权重、不需要 ONNX Runtime —— 契约断言因此能在任何构建下运行。
//
// 契约来源：onnx 1.23 shape inference，见 docs/upstream-audit.md。

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

#include "genie/onnx_graph.hpp"
#include "support/model_paths.hpp"

using namespace genie;
using namespace genie::test;

namespace {

const OnnxTensorInfo* find(const std::vector<OnnxTensorInfo>& specs, const std::string& name) {
    const auto it = std::find_if(specs.begin(), specs.end(),
                                 [&name](const OnnxTensorInfo& spec) { return spec.name == name; });
    return it == specs.end() ? nullptr : &*it;
}

/// 上游模型的形状里，动态维用符号名（如 "ref_length"）。
/// 契约断言关心的是**静态**维（BERT 1024 / HuBERT 768 / 隐层 512 等），
/// 动态维只断言存在即可。
std::int64_t static_dim(const OnnxTensorInfo& spec, std::size_t index) {
    return index < spec.shape.size() ? spec.shape[index] : -1;
}

class ModelContract : public ::testing::Test {
protected:
    void SetUp() override {
        if (!model_exists("v2/t2s_encoder_fp32.onnx")) {
            GTEST_SKIP() << "test models not available at " << model_root().string();
        }
    }
};

}  // namespace

// ---------------------------------------------------------------------------------------
// V2
// ---------------------------------------------------------------------------------------

TEST_F(ModelContract, EncoderSignature) {
    const auto spec = read_onnx_graph(v2_model("t2s_encoder_fp32.onnx").string());

    ASSERT_EQ(spec.inputs.size(), 5U);
    EXPECT_EQ(spec.inputs[0].name, "ref_seq");
    EXPECT_EQ(spec.inputs[1].name, "text_seq");
    EXPECT_EQ(spec.inputs[2].name, "ref_bert");
    EXPECT_EQ(spec.inputs[3].name, "text_bert");
    EXPECT_EQ(spec.inputs[4].name, "ssl_content");

    ASSERT_EQ(spec.outputs.size(), 2U);
    EXPECT_EQ(spec.outputs[0].name, "x");
    EXPECT_EQ(spec.outputs[1].name, "prompts");

    const auto* ref_seq = find(spec.inputs, "ref_seq");
    ASSERT_NE(ref_seq, nullptr);
    EXPECT_EQ(ref_seq->type, OnnxElementType::Int64);

    const auto* ref_bert = find(spec.inputs, "ref_bert");
    ASSERT_NE(ref_bert, nullptr);
    EXPECT_EQ(ref_bert->type, OnnxElementType::Float);
    EXPECT_EQ(static_dim(*ref_bert, 1), 1024);  // BERT 特征维度

    const auto* text_bert = find(spec.inputs, "text_bert");
    ASSERT_NE(text_bert, nullptr);
    EXPECT_EQ(static_dim(*text_bert, 1), 1024);

    const auto* ssl = find(spec.inputs, "ssl_content");
    ASSERT_NE(ssl, nullptr);
    EXPECT_EQ(ssl->type, OnnxElementType::Float);
    EXPECT_EQ(static_dim(*ssl, 1), 768);  // HuBERT 隐层维度
    EXPECT_EQ(static_dim(*ssl, 0), 1);
}

TEST_F(ModelContract, FirstStageDecoderSignature) {
    const auto spec = read_onnx_graph(v2_model("t2s_first_stage_decoder_fp32.onnx").string());

    ASSERT_EQ(spec.inputs.size(), 2U);
    EXPECT_EQ(spec.inputs[0].name, "x");
    EXPECT_EQ(spec.inputs[1].name, "prompts");

    // y + y_emb + 24 层 × (k, v) = 50
    ASSERT_EQ(spec.outputs.size(), 50U);
    EXPECT_EQ(spec.outputs[0].name, "y");
    EXPECT_EQ(spec.outputs[1].name, "y_emb");
    EXPECT_EQ(spec.outputs[2].name, "present_k_layer_0");
    EXPECT_EQ(spec.outputs[3].name, "present_v_layer_0");
    EXPECT_EQ(spec.outputs[48].name, "present_k_layer_23");
    EXPECT_EQ(spec.outputs[49].name, "present_v_layer_23");

    const auto* x = find(spec.inputs, "x");
    ASSERT_NE(x, nullptr);
    EXPECT_EQ(static_dim(*x, 2), 512);  // 隐层维度
}

// 关键契约：stage_decoder 的输入名是 iy / iy_emb / past_k_layer_N / past_v_layer_N。
// 上游 Python 靠 zip(input_names, [y, y_emb, *present_key_values]) 按位置传参，
// 变量名叫 y / y_emb —— 名字对不上。按位置绑定一旦上游调整输入顺序就会静默错位。
TEST_F(ModelContract, StageDecoderSignatureUsesIyNames) {
    const auto spec = read_onnx_graph(v2_model("t2s_stage_decoder_fp32.onnx").string());

    ASSERT_EQ(spec.inputs.size(), 50U);
    EXPECT_EQ(spec.inputs[0].name, "iy");
    EXPECT_EQ(spec.inputs[1].name, "iy_emb");
    EXPECT_EQ(spec.inputs[2].name, "past_k_layer_0");
    EXPECT_EQ(spec.inputs[3].name, "past_v_layer_0");
    EXPECT_EQ(spec.inputs[48].name, "past_k_layer_23");
    EXPECT_EQ(spec.inputs[49].name, "past_v_layer_23");

    // y + y_emb + stop_condition + 24 层 × (k, v) = 51
    ASSERT_EQ(spec.outputs.size(), 51U);
    EXPECT_EQ(spec.outputs[0].name, "y");
    EXPECT_EQ(spec.outputs[1].name, "y_emb");
    EXPECT_EQ(spec.outputs[2].name, "stop_condition_tensor");
    EXPECT_EQ(spec.outputs[3].name, "present_k_layer_0");
    EXPECT_EQ(spec.outputs[50].name, "present_v_layer_23");

    // stop_condition_tensor 是 rank-0 bool 标量。
    const auto* stop = find(spec.outputs, "stop_condition_tensor");
    ASSERT_NE(stop, nullptr);
    EXPECT_EQ(stop->type, OnnxElementType::Bool);
    EXPECT_TRUE(stop->shape.empty());

    const auto* iy_emb = find(spec.inputs, "iy_emb");
    ASSERT_NE(iy_emb, nullptr);
    EXPECT_EQ(static_dim(*iy_emb, 2), 512);
}

TEST_F(ModelContract, VitsSignature) {
    const auto spec = read_onnx_graph(v2_model("vits_fp32.onnx").string());

    ASSERT_EQ(spec.inputs.size(), 3U);
    EXPECT_EQ(spec.inputs[0].name, "text_seq");
    EXPECT_EQ(spec.inputs[1].name, "pred_semantic");
    EXPECT_EQ(spec.inputs[2].name, "ref_audio");

    // pred_semantic 是 [1, 1, pred_length] —— 中间有一个 singleton 维。
    const auto* semantic = find(spec.inputs, "pred_semantic");
    ASSERT_NE(semantic, nullptr);
    EXPECT_EQ(semantic->type, OnnxElementType::Int64);
    ASSERT_EQ(semantic->shape.size(), 3U);
    EXPECT_EQ(semantic->shape[0], 1);
    EXPECT_EQ(semantic->shape[1], 1);

    ASSERT_EQ(spec.outputs.size(), 1U);
    EXPECT_EQ(spec.outputs[0].name, "audio");
    EXPECT_EQ(spec.outputs[0].type, OnnxElementType::Float);
}

// ---------------------------------------------------------------------------------------
// V2ProPlus
// ---------------------------------------------------------------------------------------

TEST_F(ModelContract, PromptEncoderSignature) {
    const auto spec = read_onnx_graph(v2proplus_model("prompt_encoder_fp32.onnx").string());

    ASSERT_EQ(spec.inputs.size(), 2U);
    EXPECT_EQ(spec.inputs[0].name, "ref_audio");
    EXPECT_EQ(spec.inputs[1].name, "sv_emb");

    const auto* sv_emb = find(spec.inputs, "sv_emb");
    ASSERT_NE(sv_emb, nullptr);
    EXPECT_EQ(static_dim(*sv_emb, 1), 20480);

    ASSERT_EQ(spec.outputs.size(), 2U);
    EXPECT_EQ(spec.outputs[0].name, "ge");
    EXPECT_EQ(spec.outputs[1].name, "ge_advanced");
}

TEST_F(ModelContract, V2ProPlusVitsSignature) {
    const auto spec = read_onnx_graph(v2proplus_model("vits_fp32.onnx").string());

    ASSERT_EQ(spec.inputs.size(), 4U);
    EXPECT_EQ(spec.inputs[0].name, "text_seq");
    EXPECT_EQ(spec.inputs[1].name, "pred_semantic");
    EXPECT_EQ(spec.inputs[2].name, "ge");
    EXPECT_EQ(spec.inputs[3].name, "ge_advanced");

    const auto* ge = find(spec.inputs, "ge");
    ASSERT_NE(ge, nullptr);
    ASSERT_EQ(ge->shape.size(), 3U);
    EXPECT_EQ(ge->shape[1], 1024);

    const auto* ge_advanced = find(spec.inputs, "ge_advanced");
    ASSERT_NE(ge_advanced, nullptr);
    ASSERT_EQ(ge_advanced->shape.size(), 3U);
    EXPECT_EQ(ge_advanced->shape[1], 512);
}

// ---------------------------------------------------------------------------------------
// 外部数据
// ---------------------------------------------------------------------------------------

// 上游模型是图壳：权重在外部 .bin 里。这条断言把"为什么 CI 里跑不了真实推理"
// 固化成可执行的事实，避免以后有人误以为模型是坏的。
TEST_F(ModelContract, UpstreamModelsUseExternalWeights) {
    const auto spec = read_onnx_graph(v2_model("t2s_encoder_fp32.onnx").string());
    EXPECT_TRUE(spec.uses_external_data);
    EXPECT_FALSE(spec.external_data_files.empty());
    EXPECT_NE(std::find(spec.external_data_files.begin(), spec.external_data_files.end(),
                        "t2s_encoder_fp32.bin"),
              spec.external_data_files.end());
}
