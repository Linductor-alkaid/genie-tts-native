// tests/unit/test_t2s.cpp
//
// T2S 自回归解码的测试。
//
// 这是整个推理链路里最容易写错的一段，所以这里把上游那几行索引运算
// 的语义逐条钉死：
//   - `for idx in range(0, 500)` 的 idx 在正常跑完时是 499 而不是 500
//   - `y[0, -1] = 0` 把最后一个 token 置零（不是截断）
//   - `y[:, -idx:]` 在 idx == 0 时等价于"全取"（numpy 的 -0 == 0）
//   - 按**名字**绑定，而不是按位置

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "genie/errors.hpp"
#include "genie/t2s.hpp"
#include "support/fake_session.hpp"

namespace genie {
namespace {

using test::FakeSession;

/// 跑一次完整的解码循环，返回结果。
T2SResult run_loop(FakeSession& first_stage, FakeSession& stage,
                   const std::vector<Tensor>& first_outputs) {
    T2SDecoder decoder;
    return decoder.decode_loop(first_stage, stage, first_outputs);
}

TEST(T2S, DecodesUntilStopCondition) {
    auto first_stage = FakeSession::first_stage(/*prompt_len=*/3);
    auto stage = FakeSession::stage(/*stop_after=*/3);

    const auto first_outputs = first_stage->run({});
    const auto result = run_loop(*first_stage, *stage, first_outputs);

    // 停止条件在第 4 次 run 时为 true（run_count > stop_after）。
    EXPECT_EQ(stage->run_count(), 4);
    EXPECT_EQ(result.iterations, 4);
    EXPECT_FALSE(result.hit_limit);
}

TEST(T2S, FeedsFullTokenSequenceToIy) {
    auto first_stage = FakeSession::first_stage(/*prompt_len=*/3);
    auto stage = FakeSession::stage(/*stop_after=*/1);

    const auto first_outputs = first_stage->run({});
    run_loop(*first_stage, *stage, first_outputs);

    // 每次 run 的 iy 都必须是**完整**的 token 序列 —— 模型内部做的是
    // `y = Concat(iy, 新 token)`，喂标量的话输出长度就不对。
    ASSERT_FALSE(stage->feeds().empty());
    const auto& feed = stage->feeds().front();
    const auto it = std::find_if(feed.begin(), feed.end(),
                                 [](const auto& entry) { return entry.first == "iy"; });
    ASSERT_NE(it, feed.end());
    // 首阶段的 y 有 prompt_len + 1 = 4 个 token。
    EXPECT_EQ(it->second.dim(1), 4);
}

TEST(T2S, BindsKvCacheByName) {
    auto first_stage = FakeSession::first_stage(/*prompt_len=*/3, /*layers=*/2);
    auto stage = FakeSession::stage(/*stop_after=*/1, /*layers=*/2);

    const auto first_outputs = first_stage->run({});
    run_loop(*first_stage, *stage, first_outputs);

    const auto& feed = stage->last_feed();
    // 首阶段输出叫 present_k_layer_N，下一阶段输入叫 past_k_layer_N ——
    // 名字不一样，必须显式映射。
    EXPECT_TRUE(std::any_of(feed.begin(), feed.end(),
                            [](const auto& entry) { return entry.first == "past_k_layer_0"; }));
    EXPECT_TRUE(std::any_of(feed.begin(), feed.end(),
                            [](const auto& entry) { return entry.first == "past_v_layer_1"; }));
    EXPECT_FALSE(std::any_of(feed.begin(), feed.end(),
                             [](const auto& entry) { return entry.first == "present_k_layer_0"; }));
}

TEST(T2S, BindsByNameNotByPosition) {
    // 故意把 stage_decoder 的输入顺序打乱。如果实现是按位置绑定的，
    // 这里就会把 y 当成 iy 传出去。
    auto first_stage = FakeSession::first_stage();
    auto stage = FakeSession::stage(/*stop_after=*/1);

    // 交换 iy 与 iy_emb 的顺序。
    stage->swap_inputs(0, 1);

    const auto first_outputs = first_stage->run({});
    run_loop(*first_stage, *stage, first_outputs);

    const auto& feed = stage->last_feed();
    const auto iy = std::find_if(feed.begin(), feed.end(),
                                 [](const auto& entry) { return entry.first == "iy"; });
    const auto iy_emb = std::find_if(feed.begin(), feed.end(),
                                     [](const auto& entry) { return entry.first == "iy_emb"; });
    ASSERT_NE(iy, feed.end());
    ASSERT_NE(iy_emb, feed.end());
    EXPECT_EQ(iy->second.type(), ElementType::Int64);
    EXPECT_EQ(iy_emb->second.type(), ElementType::Float32);
}

TEST(T2S, ZeroesLastToken) {
    auto first_stage = FakeSession::first_stage(/*prompt_len=*/3);
    auto stage = FakeSession::stage(/*stop_after=*/2);

    const auto first_outputs = first_stage->run({});
    const auto result = run_loop(*first_stage, *stage, first_outputs);

    // 上游 `y[0, -1] = 0`：最后一个 token 被**置零**，不是被截断。
    ASSERT_FALSE(result.semantic_tokens.empty());
    EXPECT_EQ(result.semantic_tokens.back(), 0);
}

TEST(T2S, SlicesLastIdxTokens) {
    // 上游 `y[:, -idx:]`：idx = 执行次数 - 1。
    auto first_stage = FakeSession::first_stage(/*prompt_len=*/3);
    auto stage = FakeSession::stage(/*stop_after=*/2);

    const auto first_outputs = first_stage->run({});
    const auto result = run_loop(*first_stage, *stage, first_outputs);

    // 执行了 3 次 → idx = 2 → 取最后 2 个 token。
    EXPECT_EQ(result.semantic_tokens.size(), 2U);
}

TEST(T2S, ZeroIterationsKeepsEverything) {
    // numpy 的 `x[:, -0:]` 等价于 `x[:, 0:]`（全取）。
    // 停止条件在第一次 run 就为真时，idx == 0。
    auto first_stage = FakeSession::first_stage(/*prompt_len=*/3);
    auto stage = FakeSession::stage(/*stop_after=*/0);

    const auto first_outputs = first_stage->run({});
    const auto result = run_loop(*first_stage, *stage, first_outputs);

    EXPECT_EQ(result.iterations, 1);
    // 首阶段的 y 有 4 个 token（3 个 prompt + 1 个生成），循环里又追加了 1 个 → 5。
    EXPECT_EQ(result.semantic_tokens.size(), 5U);
}

TEST(T2S, HonoursMaxIterations) {
    // 停止条件永不触发 → 撞上 max_iterations。
    auto first_stage = FakeSession::first_stage();
    auto stage = FakeSession::stage(/*stop_after=*/-1);

    T2SDecoder decoder{T2SOptions{7, 1024}};
    const auto first_outputs = first_stage->run({});
    const auto result = decoder.decode_loop(*first_stage, *stage, first_outputs);

    EXPECT_EQ(result.iterations, 7);
    EXPECT_TRUE(result.hit_limit);
    // 上游 range(0, 500) 跑完时 idx = 499 而不是 500。
    EXPECT_EQ(result.semantic_tokens.size(), 6U);
}

TEST(T2S, CancellationStopsDecoding) {
    auto first_stage = FakeSession::first_stage();
    auto stage = FakeSession::stage(/*stop_after=*/-1);

    auto flag = std::make_shared<std::atomic<bool>>(true);
    CancellationToken token;
    token.bind(flag);

    T2SDecoder decoder;
    const auto first_outputs = first_stage->run({});
    const auto result = decoder.decode_loop(*first_stage, *stage, first_outputs, &token);

    EXPECT_EQ(result.semantic_tokens.empty(), true);
    EXPECT_EQ(stage->run_count(), 0);
}

TEST(T2S, RejectsMissingKvCache) {
    // 首阶段没给 KV cache，但 stage 需要 → 必须报错，不能静默跑。
    auto first_stage = FakeSession::first_stage(/*prompt_len=*/3, /*layers=*/2);
    auto stage = FakeSession::stage(/*stop_after=*/1, /*layers=*/2);

    std::vector<TensorSpec> broken_outputs;
    for (const auto& spec : first_stage->outputs()) {
        if (spec.name.rfind("present_", 0) == 0) {
            continue;  // 丢掉 KV cache
        }
        broken_outputs.push_back(spec);
    }
    std::vector<Tensor> broken_values;
    for (const auto& spec : broken_outputs) {
        broken_values.push_back(Tensor::zeros(spec.type, spec.shape));
    }
    // y 需要是 int64 序列
    broken_values[0] = Tensor::from_int64({1, 4}, {7, 7, 7, 100});
    broken_values[1] = Tensor::zeros(ElementType::Float32, {1, 3, 512});

    T2SDecoder decoder;
    EXPECT_THROW(decoder.decode_loop(*first_stage, *stage, broken_values), Error);
}

TEST(T2S, FullDecodeProducesSemanticTokens) {
    auto encoder = test::FakeSession::encoder();
    auto first_stage = test::FakeSession::first_stage();
    auto stage = test::FakeSession::stage(/*stop_after=*/3);

    TensorFeed feed;
    feed.emplace_back("ref_seq", Tensor::from_int64({1, 1}, {1}));
    feed.emplace_back("text_seq", Tensor::from_int64({1, 4}, {1, 2, 3, 4}));
    // ref_bert / text_bert 是 **秩 2**（[长度, 1024]，没有 batch 维）——
    // 上游 ONNX 图声明的就是这个形状。
    feed.emplace_back("ref_bert", Tensor::zeros(ElementType::Float32, {1, 1024}));
    feed.emplace_back("text_bert", Tensor::zeros(ElementType::Float32, {4, 1024}));
    feed.emplace_back("ssl_content", Tensor::zeros(ElementType::Float32, {1, 768, 10}));

    T2SDecoder decoder;
    const auto tokens = decoder.decode(*encoder, *first_stage, *stage, feed);

    EXPECT_FALSE(tokens.empty());
    // 最后一个 token 被置零。
    EXPECT_EQ(tokens.back(), 0);
}

}  // namespace
}  // namespace genie
