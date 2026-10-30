// include/genie/t2s.hpp
//
// T2S 自回归解码。
//
// 这是整个推理链路里最容易写错的一段，所以本文件把上游
// `Core/Inference.py` 的 `t2s_cpu()` 逐行对齐，并在每一处"为什么"写清楚。
//
// 上游的原始实现（简化）：
//
//     x, prompts = encoder.run(None, {...})
//     y, y_emb, *present_key_values = first_stage_decoder.run(None, {"x": x, "prompts": prompts})
//     for idx in range(0, 500):
//         if stop_event.is_set(): return None
//         input_feed = {name: data for name, data in zip(input_names, [y, y_emb, *present_key_values])}
//         y, y_emb, stop_condition_tensor, *present_key_values = stage_decoder.run(None, input_feed)
//         if stop_condition_tensor: break
//     y[0, -1] = 0
//     return np.expand_dims(y[:, -idx:], axis=0)
//
// 本实现与上游的**三处关键差异**（都记在 docs/upstream-audit.md）：
//
// 1. **按名字绑定**，不按位置。上游的 `zip(input_names, [y, y_emb, *kv])` 依赖
//    `get_inputs()` 的顺序恰好是 `iy, iy_emb, past_k_*...`，一旦上游调整顺序
//    就会静默错位 —— 不会报错，只会输出垃圾。
//
// 2. **KV cache 的名字要转换**。首阶段的输出叫 `present_k_layer_N`，
//    而下一阶段的输入叫 `past_k_layer_N` —— 名字不一样，得显式映射。
//
// 3. **`y[:, -idx:]` 的切片语义被保留**，包括 `idx == 0` 时 numpy 的
//    `-0` 等价于 `0`（即"全取"）这个怪癖。详见 decode() 的注释。

#ifndef GENIE_T2S_HPP
#define GENIE_T2S_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "genie/scheduler.hpp"
#include "genie/session.hpp"
#include "genie/tensor.hpp"

namespace genie {

/// T2S 解码参数。
struct T2SOptions {
    /// 最大解码步数。上游是 `range(0, 500)`。
    std::int32_t max_iterations = 500;
    /// EOS token id。模型的停止条件就是"新 token == 1024"。
    std::int64_t eos_token = 1024;
};

/// T2S 解码结果。
struct T2SResult {
    /// 语义 token 序列（已去掉 prompt 前缀与 EOS 之后的部分）。
    /// 形状对应上游的 `pred_semantic`：[1, 1, L]。
    std::vector<std::int64_t> semantic_tokens;
    /// 实际执行的解码步数。
    std::int32_t iterations = 0;
    /// 是否因为达到 max_iterations 而停下（而不是遇到 EOS）。
    bool hit_limit = false;
};

/// T2S 自回归解码器。
///
/// 线程安全：解码器本身只持有配置，所有状态都在栈上。
/// 但传入的 `ModelSession` **不是**线程安全的 —— ORT 的 `Session::Run`
/// 并发调用需要外部同步。
class T2SDecoder {
public:
    explicit T2SDecoder(T2SOptions options = {}) : options_(options) {}

    /// 跑完整的 T2S 链路：encoder → first stage → 自回归循环。
    ///
    /// @param encoder      t2s_encoder 会话
    /// @param first_stage  t2s_first_stage_decoder 会话
    /// @param stage        t2s_stage_decoder 会话
    /// @param encoder_feed encoder 的输入（ref_seq / text_seq / ref_bert /
    ///                    text_bert / ssl_content）
    /// @param cancelled    取消检查；非空时每次循环前检查一次
    ///
    /// @throws genie::Error  模型输入输出不符合契约时
    std::vector<std::int64_t> decode(ModelSession& encoder, ModelSession& first_stage,
                                     ModelSession& stage, const TensorFeed& encoder_feed,
                                     const CancellationToken* cancelled = nullptr) const;

    /// 只跑自回归循环（首阶段之后的部分）。
    ///
    /// 单独暴露出来是为了能直接测"循环"本身 —— 上游那几行索引运算
    /// （`y[:, -idx:]`、`y[0, -1] = 0`）是最容易写错的地方。
    ///
    /// @param first_stage first_stage_decoder 会话（用于读输出名）
    /// @param stage       stage_decoder 会话
    /// @param first_stage_output 首阶段的输出（y / y_emb / present_*）
    /// @return 解码结果
    T2SResult decode_loop(ModelSession& first_stage, ModelSession& stage,
                          const std::vector<Tensor>& first_stage_output,
                          const CancellationToken* cancelled = nullptr) const;

    [[nodiscard]] const T2SOptions& options() const noexcept { return options_; }

private:
    T2SOptions options_;
};

/// 从输出里按名字找张量。找不到返回 nullptr。
///
/// @param outputs 模型输出
/// @param specs   模型的输出描述（名字与顺序对应 outputs）
/// @param name    要找的名字
[[nodiscard]] const Tensor* find_output(const std::vector<Tensor>& outputs,
                                        const std::vector<TensorSpec>& specs,
                                        const std::string& name) noexcept;

}  // namespace genie

#endif  // GENIE_T2S_HPP
