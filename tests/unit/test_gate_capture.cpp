// tests/unit/test_gate_capture.cpp
//
// 精度门控捕获基础设施的单测（tools/gate/capture.hpp）。
//
// 用 tests/support/fake_session.hpp 的假会话充当"真模型"——装饰器只负责
// 旁路记录与落盘，它对会话的期望只有 ModelSession 接口本身。这样
// 捕获格式与截断策略的回归不需要真权重，CI 无模型也能跑。

#include <gtest/gtest.h>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "gate/capture.hpp"
#include "genie/json.hpp"
#include "genie/tensor.hpp"
#include "support/fake_session.hpp"

using genie::gate::CallRecord;
using genie::gate::capture_log_to_json;
using genie::gate::CapturingSession;
using genie::gate::kMaxCallsPerRole;
using genie::gate::role_from_path;

namespace {

/// 临时捕获目录，析构时清理。
class TempDir {
public:
    TempDir() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        root_ =
            std::filesystem::temp_directory_path() / ("genie_gate_test_" + std::to_string(stamp));
        std::filesystem::create_directories(root_);
    }
    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(root_, ec);
    }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return root_; }

private:
    std::filesystem::path root_;
};

/// 用假会话造一个捕获装饰器，跑一次推理。
///
/// feed 与 FakeSession 的契约一致（vocoder：text_seq / pred_semantic / ref_audio）。
std::vector<CallRecord> capture_one_run(const std::filesystem::path& out_dir,
                                        const std::string& role, int stage_full_steps = 1) {
    std::vector<CallRecord> log;
    std::atomic<int> phase{0};
    auto inner = genie::test::FakeSession::vocoder(64);
    CapturingSession session(std::move(inner), role, out_dir, log, phase, stage_full_steps);

    genie::TensorFeed feed;
    feed.emplace_back("text_seq", genie::Tensor::from_int64({1, 3}, {1, 2, 3}));
    feed.emplace_back("pred_semantic", genie::Tensor::from_int64({1, 1, 2}, {7, 8}));
    feed.emplace_back("ref_audio", genie::Tensor::zeros(genie::ElementType::Float32, {1, 16}));
    (void)session.run(feed);
    return log;
}

}  // namespace

TEST(GateCapture, RoleFromPathMapsAllPackageRoles) {
    EXPECT_EQ(role_from_path("/pkg/models/t2s_first_stage_decoder_fp32.onnx"),
              "t2s_first_stage_decoder");
    EXPECT_EQ(role_from_path("/pkg/models/t2s_stage_decoder_fp32.onnx"), "t2s_stage_decoder");
    EXPECT_EQ(role_from_path("/pkg/models/t2s_encoder_fp32.onnx"), "t2s_encoder");
    EXPECT_EQ(role_from_path("/pkg/models/vits_fp32.onnx"), "vits");
    EXPECT_EQ(role_from_path("/pkg/models/prompt_encoder_fp32.onnx"), "prompt_encoder");
    EXPECT_EQ(role_from_path("/pkg/models/speaker_encoder.onnx"), "speaker_verification");
    EXPECT_EQ(role_from_path("/pkg/models/chinese-hubert-base.onnx"), "chinese_hubert");
    EXPECT_EQ(role_from_path("/pkg/models/roberta.onnx"), "roberta");
    // 未知模型不静默丢失：用文件名主干兜底。
    EXPECT_EQ(role_from_path("/pkg/models/mystery.onnx"), "mystery");
}

TEST(GateCapture, DecoratorRecordsPhaseAndDumpsFiles) {
    TempDir dir;
    auto log = capture_one_run(dir.path(), "vits");

    // phase=0 时是"load"，置 1 后是"synth"。
    ASSERT_EQ(log.size(), std::size_t{1});
    EXPECT_EQ(log[0].phase, "load");
    EXPECT_EQ(log[0].role, "vits");
    EXPECT_FALSE(log[0].partial);
    EXPECT_FALSE(log[0].truncated);
    EXPECT_EQ(log[0].inputs.size(), std::size_t{3});
    EXPECT_EQ(log[0].outputs.size(), std::size_t{1});

    // 张量文件真的落盘了，字节数与张量一致（float32 → 4 字节/元素）。
    std::vector<genie::json::Value> metas(log[0].inputs.begin(), log[0].inputs.end());
    for (const auto& meta : metas) {
        const auto file = dir.path() / meta.as_object().at("file").as_string();
        ASSERT_TRUE(std::filesystem::exists(file)) << file.string();
        const auto bytes = static_cast<std::uintmax_t>(meta.as_object().at("bytes").as_int());
        EXPECT_EQ(std::filesystem::file_size(file), bytes);
    }
}

TEST(GateCapture, PhaseSwitchFollowsAtomic) {
    TempDir dir;
    std::vector<CallRecord> log;
    std::atomic<int> phase{0};
    auto inner = genie::test::FakeSession::vocoder(64);
    CapturingSession session(std::move(inner), "vits", dir.path(), log, phase, 1);

    genie::TensorFeed feed;
    feed.emplace_back("text_seq", genie::Tensor::from_int64({1, 1}, {1}));
    feed.emplace_back("pred_semantic", genie::Tensor::from_int64({1, 1, 1}, {7}));
    feed.emplace_back("ref_audio", genie::Tensor::zeros(genie::ElementType::Float32, {1, 16}));
    (void)session.run(feed);
    phase.store(1);
    (void)session.run(feed);

    ASSERT_EQ(log.size(), std::size_t{2});
    EXPECT_EQ(log[0].phase, "load");
    EXPECT_EQ(log[1].phase, "synth");
}

TEST(GateCapture, StageDecoderPartialPolicyKeepsTokenTensorsOnly) {
    TempDir dir;
    std::vector<CallRecord> log;
    std::atomic<int> phase{1};  // 直接处于合成期。
    auto inner = genie::test::FakeSession::stage(2);
    CapturingSession session(std::move(inner), "t2s_stage_decoder", dir.path(), log, phase, 1);

    auto make_feed = [] {
        genie::TensorFeed feed;
        feed.emplace_back("iy", genie::Tensor::from_int64({1, 2}, {5, 6}));
        feed.emplace_back("iy_emb", genie::Tensor::zeros(genie::ElementType::Float32, {1, 2, 512}));
        feed.emplace_back("past_k_layer_0",
                          genie::Tensor::zeros(genie::ElementType::Float32, {1, 1, 4}));
        feed.emplace_back("past_v_layer_0",
                          genie::Tensor::zeros(genie::ElementType::Float32, {1, 1, 4}));
        return feed;
    };

    // 第 0 步：全量（past_* 输入与 present_* 输出都在）。
    (void)session.run(make_feed());
    // 第 1 步：partial —— 只留 iy* 输入与 y / stop 输出，KV cache 不落盘。
    (void)session.run(make_feed());

    ASSERT_EQ(log.size(), std::size_t{2});
    EXPECT_FALSE(log[0].partial);
    std::vector<std::string> full_inputs;
    for (const auto& meta : log[0].inputs) {
        full_inputs.push_back(meta.as_object().at("name").as_string());
    }
    EXPECT_NE(std::find(full_inputs.begin(), full_inputs.end(), "past_k_layer_0"),
              full_inputs.end());

    EXPECT_TRUE(log[1].partial);
    std::vector<std::string> partial_inputs;
    std::vector<std::string> partial_outputs;
    for (const auto& meta : log[1].inputs) {
        partial_inputs.push_back(meta.as_object().at("name").as_string());
    }
    for (const auto& meta : log[1].outputs) {
        partial_outputs.push_back(meta.as_object().at("name").as_string());
    }
    // 输入只剩 iy 系；y_emb / past_* 被裁掉。
    EXPECT_EQ(partial_inputs, std::vector<std::string>({"iy", "iy_emb"}));
    // 输出只剩 y / stop；present_* 被裁掉（fake 的 y_emb 也被裁掉）。
    EXPECT_EQ(partial_outputs, std::vector<std::string>({"y", "stop_condition_tensor"}));
}

TEST(GateCapture, TruncatesAfterMaxCallsPerRole) {
    TempDir dir;
    std::vector<CallRecord> log;
    std::atomic<int> phase{1};
    auto inner = genie::test::FakeSession::vocoder(64);
    CapturingSession session(std::move(inner), "vits", dir.path(), log, phase, 1);

    genie::TensorFeed feed;
    feed.emplace_back("text_seq", genie::Tensor::from_int64({1, 1}, {1}));
    feed.emplace_back("pred_semantic", genie::Tensor::from_int64({1, 1, 1}, {7}));
    feed.emplace_back("ref_audio", genie::Tensor::zeros(genie::ElementType::Float32, {1, 16}));
    for (int i = 0; i < kMaxCallsPerRole + 3; ++i) {
        (void)session.run(feed);
    }

    // 超限的调用只记账不落盘 —— 计数完整（run 次数一目了然），目录不膨胀。
    ASSERT_EQ(log.size(), std::size_t{kMaxCallsPerRole + 3});
    for (std::size_t i = 0; i < log.size(); ++i) {
        EXPECT_EQ(log[i].truncated, i >= std::size_t{kMaxCallsPerRole}) << "call " << i;
    }
}

TEST(GateCapture, CaptureLogSerializesToValidJson) {
    TempDir dir;
    auto log = capture_one_run(dir.path(), "roberta");

    const auto root = capture_log_to_json(log, "/pkg", {"你好。"}, "chinese", 1);
    const auto text = root.dump();
    const auto parsed = genie::json::parse(text);

    ASSERT_TRUE(parsed.is_object());
    EXPECT_EQ(parsed.as_object().at("schema").as_int(), 1);
    EXPECT_EQ(parsed.as_object().at("package").as_string(), "/pkg");
    ASSERT_EQ(parsed.as_object().at("calls").as_array().size(), std::size_t{1});
    const auto& call = parsed.as_object().at("calls").as_array().front();
    EXPECT_EQ(call.as_object().at("role").as_string(), "roberta");
    // dtype 必须是字符串 —— Value 没有 const char* 重载，这里曾踩过
    // "指针→bool" 落进 Value(bool) 的坑（见 capture.hpp dump_one 注释）。
    const auto& input = call.as_object().at("inputs").as_array().front();
    EXPECT_TRUE(input.as_object().at("dtype").is_string());
    EXPECT_EQ(input.as_object().at("dtype").as_string(), "int64");
}
