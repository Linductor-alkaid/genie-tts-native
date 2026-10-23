// src/engine/t2s.cpp

#include "genie/t2s.hpp"

#include <algorithm>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "genie/errors.hpp"
#include "genie/scheduler.hpp"

namespace genie {
namespace {

/// 首阶段输出名 → 下一阶段输入名。
///
/// 首阶段输出 `present_k_layer_N`，下一阶段输入 `past_k_layer_N` —— 名字
/// 不一样，得显式映射。层数不写死：从模型的实际输出名里扫出来。
constexpr std::string_view kPresentPrefix = "present_";
constexpr std::string_view kPastPrefix = "past_";

/// 一轮解码的状态。
struct StageState {
    /// 完整的 token 序列（首阶段是 prompt + 首个生成 token，之后每轮 +1）。
    std::vector<std::int64_t> tokens;
    /// 词嵌入。
    Tensor y_emb;
    /// KV cache：present_k_layer_N / present_v_layer_N。
    std::vector<std::pair<std::string, Tensor>> kv_cache;
};

/// 从输出里按名字取张量。找不到返回 nullptr。
[[nodiscard]] const Tensor* by_name(const std::vector<Tensor>& outputs,
                                    const std::vector<TensorSpec>& specs,
                                    std::string_view name) noexcept {
    for (std::size_t i = 0; i < outputs.size() && i < specs.size(); ++i) {
        if (specs[i].name == name) {
            return &outputs[i];
        }
    }
    return nullptr;
}

/// 读 `y`（int64 序列）。
[[nodiscard]] std::vector<std::int64_t> read_tokens(const Tensor& tensor) {
    const auto ints = tensor.ints();
    return std::vector<std::int64_t>(ints.begin(), ints.end());
}

/// 拆解模型输出：token 序列 / y_emb / KV cache。
[[nodiscard]] StageState read_output(const std::vector<Tensor>& outputs,
                                     const std::vector<TensorSpec>& specs, const char* who) {
    StageState state;
    bool has_y = false;
    bool has_y_emb = false;
    for (std::size_t i = 0; i < outputs.size() && i < specs.size(); ++i) {
        const auto& name = specs[i].name;
        if (name == "y") {
            state.tokens = read_tokens(outputs[i]);
            has_y = true;
        } else if (name == "y_emb") {
            state.y_emb = outputs[i];
            has_y_emb = true;
        } else if (name.rfind(kPresentPrefix, 0) == 0) {
            state.kv_cache.emplace_back(name, outputs[i]);
        }
    }
    if (!has_y) {
        throw_error(ErrorCode::InvalidArgument, std::string("t2s ") + who + " has no 'y' output");
    }
    if (!has_y_emb) {
        throw_error(ErrorCode::InvalidArgument,
                    std::string("t2s ") + who + " has no 'y_emb' output");
    }
    return state;
}

/// 组装下一轮 stage_decoder 的输入。
///
/// **按名字**绑定 —— 名字取自模型的实际输入描述。
///
/// `iy` 传的是**完整**的 token 序列：模型内部做的是
/// `y = Concat(iy, 新 token)`，所以喂标量的话输出长度就不对了。
[[nodiscard]] TensorFeed build_stage_feed(ModelSession& stage, const StageState& state) {
    TensorFeed feed;
    const auto& inputs = stage.inputs();
    feed.reserve(inputs.size());

    for (const auto& spec : inputs) {
        if (spec.name == "iy") {
            feed.emplace_back(
                spec.name, Tensor::from_int64({1, static_cast<std::int64_t>(state.tokens.size())},
                                              state.tokens));
        } else if (spec.name == "iy_emb") {
            feed.emplace_back(spec.name, state.y_emb);
        } else if (spec.name.rfind(kPastPrefix, 0) == 0) {
            const auto suffix = spec.name.substr(kPastPrefix.size());
            const auto it = std::find_if(
                state.kv_cache.begin(), state.kv_cache.end(), [&suffix](const auto& entry) {
                    return entry.first.rfind(kPresentPrefix, 0) == 0 &&
                           entry.first.substr(kPresentPrefix.size()) == suffix;
                });
            if (it == state.kv_cache.end()) {
                throw_error(ErrorCode::InvalidArgument,
                            "t2s stage decoder needs KV cache '" + spec.name +
                                "' but the previous stage did not provide it");
            }
            feed.emplace_back(spec.name, it->second);
        }
        // 契约之外的输入不喂 —— 由模型自己决定缺省值。
    }
    return feed;
}

/// 读 stage_decoder 的停止条件（标量 bool）。
[[nodiscard]] bool read_stop_condition(const std::vector<Tensor>& outputs,
                                       const std::vector<TensorSpec>& specs) {
    const auto* tensor = by_name(outputs, specs, "stop_condition_tensor");
    if (tensor == nullptr) {
        return false;
    }
    if (const auto scalar = tensor->scalar(); scalar.has_value()) {
        if (const auto* flag = std::get_if<bool>(&scalar.value())) {
            return *flag;
        }
        if (const auto* value = std::get_if<std::int64_t>(&scalar.value())) {
            return *value != 0;
        }
    }
    return false;
}

}  // namespace

const Tensor* find_output(const std::vector<Tensor>& outputs, const std::vector<TensorSpec>& specs,
                          const std::string& name) noexcept {
    return by_name(outputs, specs, name);
}

T2SResult T2SDecoder::decode_loop(ModelSession& first_stage, ModelSession& stage,
                                  const std::vector<Tensor>& first_stage_output,
                                  const CancellationToken* cancelled) const {
    const auto& first_specs = first_stage.outputs();
    const auto& stage_specs = stage.outputs();

    auto state = read_output(first_stage_output, first_specs, "first stage");

    T2SResult result;
    bool stopped = false;

    // 上游是 `for idx in range(0, 500)`：
    //   - 正常跑完时 idx 停在 499（**不是** 500）
    //   - break 时 idx 停在 break 的那次下标
    // 两种情况都等于"执行次数 - 1"，但循环变量的取值方式要照抄，
    // 否则切片会差一个。
    std::int32_t idx = 0;
    std::int32_t executions = 0;
    while (executions < options_.max_iterations) {
        idx = executions;
        ++executions;  // 这一次 run 算数
        if (cancelled != nullptr && cancelled->cancelled()) {
            // 上游遇到 stop_event 直接返回 None（放弃整次合成）。
            result.semantic_tokens.clear();
            result.iterations = executions - 1;
            result.hit_limit = false;
            return result;
        }

        const auto feed = build_stage_feed(stage, state);
        const auto outputs = stage.run(feed);

        // 上游是 `y, y_emb, stop_condition_tensor, *present_key_values = outputs`，
        // 即每一轮都用**新输出**整体覆盖旧输出。
        state = read_output(outputs, stage_specs, "stage decoder");
        stopped = read_stop_condition(outputs, stage_specs);

        if (stopped) {
            break;
        }
    }

    if (executions >= options_.max_iterations) {
        // range(0, N) 跑完时循环变量是 N-1。
        idx = options_.max_iterations - 1;
    }
    result.iterations = executions;
    result.hit_limit = !stopped && executions >= options_.max_iterations;

    // 上游：`y[0, -1] = 0` —— 把最后一个 token 置零。
    //
    // 意图是抹掉 EOS（模型在生成 EOS 的那一轮就 break 了，所以最后一个
    // token 往往就是 EOS），但**不是**截断。上游随后的
    // `np.where(semantic_tokens >= 1024)` 因此不会命中，序列末尾会留一个 0。
    // 这个怪癖照搬 —— 改了会影响听感。
    if (!state.tokens.empty()) {
        state.tokens.back() = 0;
    }

    // 上游：`np.expand_dims(y[:, -idx:], axis=0)`。
    //
    // numpy 里 `x[:, -0:]` 等价于 `x[:, 0:]`（**全取**）—— `-0` 就是 `0`。
    // 所以 idx == 0 时拿到的是整个 y（含 prompt）。这个语义必须保留：
    // 它是上游的实际行为，不是笔误。
    const std::size_t total = state.tokens.size();
    if (idx > 0) {
        const auto keep = std::min<std::size_t>(static_cast<std::size_t>(idx), total);
        result.semantic_tokens.assign(state.tokens.end() - static_cast<std::ptrdiff_t>(keep),
                                      state.tokens.end());
    } else {
        result.semantic_tokens = state.tokens;
    }
    return result;
}

std::vector<std::int64_t> T2SDecoder::decode(ModelSession& encoder, ModelSession& first_stage,
                                             ModelSession& stage, const TensorFeed& encoder_feed,
                                             const CancellationToken* cancelled) const {
    // 1) Encoder：产出 x 与 prompts。
    const auto encoder_outputs = encoder.run(encoder_feed);
    const auto& encoder_specs = encoder.outputs();
    const auto* x = find_output(encoder_outputs, encoder_specs, "x");
    const auto* prompts = find_output(encoder_outputs, encoder_specs, "prompts");
    if (x == nullptr || prompts == nullptr) {
        throw_error(ErrorCode::InvalidArgument, "t2s encoder must output both 'x' and 'prompts'");
    }

    // 2) First stage：产出首个 token、embedding 与 KV cache。
    TensorFeed first_feed;
    first_feed.emplace_back("x", *x);
    first_feed.emplace_back("prompts", *prompts);
    const auto first_outputs = first_stage.run(first_feed);

    // 3) 自回归循环。
    return decode_loop(first_stage, stage, first_outputs, cancelled).semantic_tokens;
}

}  // namespace genie
