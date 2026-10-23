// tests/integration/test_synthesis.cpp
//
// ONNX Runtime 后端的端到端路径测试。
//
// 用**合成模型**（tests/models/synthetic/echo.onnx）而不是上游模型，原因是：
// 上游发布的六个 ONNX 模型是图壳，权重在外部 .bin 里（合计约 1.1 GB），
// 由上游转换流程生成、不在仓库中，无法被 ORT 加载。
//   → 真实模型的 I/O 契约由 tests/component/test_model_contract.cpp 断言
//     （直接解析 ONNX protobuf，不需要权重）；
//   → 这里用合成模型验证**运行时路径**：加载 → 按名字绑定 → 推理 → 解析输出。
//
// 合成模型刻意把输入/输出顺序与常见顺序错开，测试也**按错开的顺序**喂数据，
// 因此如果实现是按位置绑定而不是按名字绑定，测试立刻失败。

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "genie/config.hpp"
#include "genie/errors.hpp"
#include "genie/onnx_graph.hpp"
#include "genie/session.hpp"
#include "genie/tensor.hpp"
#include "support/model_paths.hpp"

using namespace genie;
using namespace genie::test;

namespace {

class OrtBackendTest : public ::testing::Test {
protected:
    void SetUp() override {
        const auto path = synthetic_model();
        if (!std::filesystem::exists(path)) {
            GTEST_SKIP() << "synthetic model not available at " << path.string();
        }
        if (!has_onnxruntime_backend()) {
            GTEST_SKIP() << "this build has no ONNX Runtime backend";
        }
        model_path_ = path;
        config_ = SessionConfig{};
    }

    std::filesystem::path model_path_;
    SessionConfig config_;
};

}  // namespace

TEST_F(OrtBackendTest, LoadsModelAndReportsSignature) {
    auto session = create_session(model_path_.string(), config_);

    const auto& inputs = session->inputs();
    const auto& outputs = session->outputs();

    ASSERT_EQ(inputs.size(), 3U);
    EXPECT_EQ(inputs[0].name, "alpha");
    EXPECT_EQ(inputs[1].name, "beta");
    EXPECT_EQ(inputs[2].name, "gamma");
    EXPECT_EQ(inputs[0].type, ElementType::Int64);
    EXPECT_EQ(inputs[1].type, ElementType::Float32);

    ASSERT_EQ(outputs.size(), 3U);
    EXPECT_EQ(outputs[0].name, "out_i");
    EXPECT_EQ(outputs[1].name, "out_f");
    EXPECT_EQ(outputs[2].name, "out_scalar");
    EXPECT_EQ(outputs[2].type, ElementType::Bool);
    EXPECT_TRUE(outputs[2].shape.empty());  // rank-0 标量
}

TEST_F(OrtBackendTest, BindsInputsByNameNotByPosition) {
    auto session = create_session(model_path_.string(), config_);

    constexpr int kBatch = 1;
    constexpr int kLength = 4;
    constexpr int kBert = 1024;
    constexpr int kSsl = 8;

    const std::vector<std::int64_t> alpha(kLength, 11);
    const std::vector<float> beta(static_cast<std::size_t>(kLength) * kBert, 0.5F);
    const std::vector<float> gamma(static_cast<std::size_t>(kBert == 0 ? 0 : 768) * kSsl, 0.25F);

    // 故意按 gamma → alpha → beta 的顺序喂，与模型声明顺序完全不同。
    // 若实现按位置绑定，alpha/beta 会拿到错误的张量。
    const auto outputs = session->run({
        {"gamma", Tensor::from_float32({kBatch, 768, kSsl}, gamma)},
        {"alpha", Tensor::from_int64({kBatch, kLength}, alpha)},
        {"beta", Tensor::from_float32({kLength, kBert}, beta)},
    });

    ASSERT_EQ(outputs.size(), 3U);
    // out_i = Identity(alpha)
    ASSERT_EQ(outputs[0].ints().size(), static_cast<std::size_t>(kLength));
    EXPECT_EQ(outputs[0].ints()[0], 11);
    // out_f = Identity(beta)
    ASSERT_EQ(outputs[1].floats().size(), static_cast<std::size_t>(kLength) * kBert);
    EXPECT_FLOAT_EQ(outputs[1].floats()[0], 0.5F);
    // out_scalar = 常量 false
    const auto scalar = outputs[2].scalar();
    ASSERT_TRUE(scalar.has_value());
    EXPECT_FALSE(std::get<bool>(*scalar));
}

TEST_F(OrtBackendTest, MissingModelThrowsModelLoadError) {
    EXPECT_THROW(create_session("/definitely/not/here/model.onnx", config_), Error);
}

TEST_F(OrtBackendTest, SyntheticModelGraphSpecMatchesSession) {
    // 两条独立的读取路径（protobuf 解析 vs ORT 运行时）必须给出一致的签名 ——
    // 这是"契约解析器"与"实际运行时"之间的一致性检查。
    const auto spec = read_onnx_graph(model_path_.string());
    auto session = create_session(model_path_.string(), config_);

    ASSERT_EQ(spec.inputs.size(), session->inputs().size());
    for (std::size_t i = 0; i < spec.inputs.size(); ++i) {
        EXPECT_EQ(spec.inputs[i].name, session->inputs()[i].name) << "input #" << i;
    }
    ASSERT_EQ(spec.outputs.size(), session->outputs().size());
    for (std::size_t i = 0; i < spec.outputs.size(); ++i) {
        EXPECT_EQ(spec.outputs[i].name, session->outputs()[i].name) << "output #" << i;
    }
}
