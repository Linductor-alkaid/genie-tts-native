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

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "genie/audio.hpp"
#include "genie/config.hpp"
#include "genie/engine.hpp"
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

namespace {

/// 用合成模型临时拼一个合法的音色包：四个角色都指向 echo.onnx。
///
/// Engine 构造只加载会话不推理，所以模型内容是否匹配 t2s/vits 契约无关紧要，
/// 只要它是 ORT 能加载的有效模型 —— 这正好可以隔离地测「构造路径」本身。
class TempPackage {
public:
    TempPackage() {
        static int counter = 0;
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        root_ = std::filesystem::temp_directory_path() /
                ("genie_ort_pkg_" + std::to_string(stamp) + "_" + std::to_string(counter++));
        std::filesystem::create_directories(root_ / "models");
        std::filesystem::create_directories(root_ / "reference");
        root_str_ = root_.string();

        for (const char* name : {"t2s_encoder.onnx", "t2s_first_stage_decoder.onnx",
                                 "t2s_stage_decoder.onnx", "vits.onnx"}) {
            std::filesystem::copy_file(synthetic_model(), root_ / "models" / name);
        }

        genie::AudioBuffer audio;
        audio.sample_rate = 32000;
        audio.samples.assign(32000, 0.1F);
        genie::WavCodec::write_pcm16((root_ / "reference" / "audio.wav").string(), audio);
        std::ofstream text(root_ / "reference" / "text.txt");
        text << "\xE5\x8F\x82\xE8\x80\x83";  // 参考

        std::ofstream manifest(root_ / "manifest.json");
        manifest << R"({
  "name": "synthetic-package",
  "format_version": 1,
  "model_version": "v2",
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

    ~TempPackage() {
        std::error_code ec;
        std::filesystem::remove_all(root_, ec);
    }

    TempPackage(const TempPackage&) = delete;
    TempPackage& operator=(const TempPackage&) = delete;

    [[nodiscard]] const std::string& root() const noexcept { return root_str_; }

private:
    std::filesystem::path root_;
    std::string root_str_;
};

}  // namespace

TEST_F(OrtBackendTest, EngineLoadsViaConfigConstructor) {
    // Engine(root, config) 必须把会话装配进**当前**对象。
    //
    // 曾经的缺陷：它在函数体里调用 Engine(...)，那只是构造又立刻析构了一个
    // 临时对象，当前对象的 impl_ 保持未初始化 —— 用它就是未定义行为
    // （真实模型上表现为 synth 命令在第一次触碰 impl_ 时段错误）。
    // 委托构造必须走初始化列表。
    const TempPackage package;

    Engine engine(package.root(), SessionConfig{});

    EXPECT_EQ(engine.sample_rate(), 32000);
    EXPECT_EQ(engine.manifest().name, "synthetic-package");
}
