// tests/unit/test_onnx_graph.cpp
//
// ONNX protobuf 读取器的单元测试。
//
// 这个读取器存在的意义：上游发布的 ONNX 模型是"图壳"（权重在外部 .bin 里），
// 无法被 ONNX Runtime 加载，但图结构完整 —— I/O 契约必须能被**可执行地**锁定。

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "genie/errors.hpp"
#include "genie/onnx_graph.hpp"
#include "support/model_paths.hpp"

using namespace genie;
using namespace genie::test;

TEST(OnnxGraph, ReadsSyntheticModelSignature) {
    const auto path = synthetic_model();
    if (!std::filesystem::exists(path)) {
        GTEST_SKIP() << "synthetic model not available at " << path.string();
    }

    const auto spec = read_onnx_graph(path.string());

    ASSERT_EQ(spec.inputs.size(), 3U);
    EXPECT_EQ(spec.inputs[0].name, "alpha");
    EXPECT_EQ(spec.inputs[1].name, "beta");
    EXPECT_EQ(spec.inputs[2].name, "gamma");
    EXPECT_EQ(spec.inputs[0].type, OnnxElementType::Int64);
    EXPECT_EQ(spec.inputs[1].type, OnnxElementType::Float);
    EXPECT_EQ(spec.inputs[2].type, OnnxElementType::Float);

    ASSERT_EQ(spec.outputs.size(), 3U);
    EXPECT_EQ(spec.outputs[0].name, "out_i");
    EXPECT_EQ(spec.outputs[1].name, "out_f");
    EXPECT_EQ(spec.outputs[2].name, "out_scalar");

    // rank-0 bool 标量
    EXPECT_EQ(spec.outputs[2].type, OnnxElementType::Bool);
    EXPECT_TRUE(spec.outputs[2].shape.empty());

    // beta 的第二维是静态的 1024
    EXPECT_EQ(spec.inputs[1].shape.size(), 2U);
    EXPECT_EQ(spec.inputs[1].shape[1], 1024);

    // 自包含模型不使用外部数据
    EXPECT_FALSE(spec.uses_external_data);
}

TEST(OnnxGraph, RejectsNonOnnxData) {
    const std::vector<std::byte> garbage(64, std::byte{0x7F});
    EXPECT_THROW(read_onnx_graph_from_memory(garbage), Error);
}

TEST(OnnxGraph, MissingFileThrowsIoError) {
    EXPECT_THROW(read_onnx_graph("/definitely/not/here/model.onnx"), Error);
}

TEST(OnnxGraph, DetectsExternalDataInUpstreamModels) {
    if (!model_exists("v2/t2s_encoder_fp32.onnx")) {
        GTEST_SKIP() << "test models not available at " << model_root().string();
    }
    const auto spec = read_onnx_graph(v2_model("t2s_encoder_fp32.onnx").string());
    EXPECT_TRUE(spec.uses_external_data);
    EXPECT_FALSE(spec.external_data_files.empty());
}

TEST(OnnxGraph, ElementTypeNames) {
    EXPECT_STREQ(to_string(OnnxElementType::Float), "float32");
    EXPECT_STREQ(to_string(OnnxElementType::Int64), "int64");
    EXPECT_STREQ(to_string(OnnxElementType::Bool), "bool");
    EXPECT_STREQ(to_string(OnnxElementType::Undefined), "undefined");
}
