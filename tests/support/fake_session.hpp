// tests/support/fake_session.hpp
//
// 一个假的 ModelSession，用来在没有 ONNX Runtime、没有模型权重的环境里
// 测试推理链路的**控制流**。
//
// 它不是随便造的数据，而是**照着上游六个模型的真实 I/O 契约**实现的：
// 输入输出的名字、类型、形状都跟真模型一致（见 docs/upstream-audit.md §3）。
// 这样测出来的行为才有意义 —— 比如"按名字绑定"这件事，
// 如果 fake 的名字跟真模型不一样，测试就白测了。

#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "genie/session.hpp"
#include "genie/tensor.hpp"

namespace genie::test {

/// 假会话的行为配置。
struct FakeSessionConfig {
    /// 输出名 → 输出张量。名字必须与真模型一致。
    std::vector<std::pair<std::string, Tensor>> outputs;

    /// 输出描述。为空时从 outputs 推导。
    ///
    /// 有些输出（比如 stage_decoder 的 y）是**动态**的 —— 值取决于输入，
    /// 只有形状是固定的。这类输出在这里声明形状，值由 run() 现算。
    std::vector<TensorSpec> output_specs;

    /// 每次 run() 追加到输出 token 序列上的 token。默认 1。
    std::int64_t next_token = 1;

    /// 第几次 run() 之后返回停止条件为 true。-1 = 永不停止。
    int stop_after = -1;

    /// run() 被调用的次数。
    int run_count = 0;

    /// 最近一次 run() 的输入绑定。
    TensorFeed last_feed;

    /// 历次 run() 的输入绑定。
    std::vector<TensorFeed> feeds;
};

/// 照着契约实现的假会话。
class FakeSession final : public ModelSession {
public:
    FakeSession(std::vector<TensorSpec> inputs, FakeSessionConfig config)
        : inputs_(std::move(inputs)), config_(std::move(config)) {
        if (!config_.output_specs.empty()) {
            output_specs_ = config_.output_specs;
        } else {
            for (const auto& [name, tensor] : config_.outputs) {
                TensorSpec spec;
                spec.name = name;
                spec.type = tensor.type();
                spec.shape = tensor.shape();
                output_specs_.push_back(std::move(spec));
            }
        }
    }

    /// 构造一个 t2s_encoder 风格的会话。
    static std::unique_ptr<FakeSession> encoder() {
        std::vector<TensorSpec> inputs;
        inputs.push_back({"ref_seq", ElementType::Int64, {1, -1}});
        inputs.push_back({"text_seq", ElementType::Int64, {1, -1}});
        // ref_bert / text_bert 是 **秩 2**（[长度, 1024]，没有 batch 维）。
        inputs.push_back({"ref_bert", ElementType::Float32, {-1, 1024}});
        inputs.push_back({"text_bert", ElementType::Float32, {-1, 1024}});
        inputs.push_back({"ssl_content", ElementType::Float32, {1, 768, -1}});

        FakeSessionConfig config;
        config.outputs.emplace_back("x", Tensor::zeros(ElementType::Float32, {1, 4, 512}));
        config.outputs.emplace_back("prompts", Tensor::zeros(ElementType::Int64, {1, 3}));
        return std::make_unique<FakeSession>(std::move(inputs), std::move(config));
    }

    /// 构造一个 t2s_first_stage_decoder 风格的会话。
    ///
    /// @param prompt_len prompt 长度
    /// @param layers     KV cache 的层数
    static std::unique_ptr<FakeSession> first_stage(int prompt_len = 3, int layers = 2) {
        std::vector<TensorSpec> inputs;
        inputs.push_back({"x", ElementType::Float32, {1, -1, 512}});
        inputs.push_back({"prompts", ElementType::Int64, {1, -1}});

        FakeSessionConfig config;
        // y = Concat(prompts, 首个生成的 token) → 长度 prompt_len + 1
        std::vector<std::int64_t> tokens(static_cast<std::size_t>(prompt_len), 7);
        tokens.push_back(100);
        config.outputs.emplace_back(
            "y", Tensor::from_int64({1, static_cast<std::int64_t>(tokens.size())}, tokens));
        config.outputs.emplace_back(
            "y_emb",
            Tensor::zeros(ElementType::Float32, {1, static_cast<std::int64_t>(prompt_len), 512}));
        for (int i = 0; i < layers; ++i) {
            config.outputs.emplace_back("present_k_layer_" + std::to_string(i),
                                        Tensor::zeros(ElementType::Float32, {1, 1, 512}));
            config.outputs.emplace_back("present_v_layer_" + std::to_string(i),
                                        Tensor::zeros(ElementType::Float32, {1, 1, 512}));
        }
        return std::make_unique<FakeSession>(std::move(inputs), std::move(config));
    }

    /// 构造一个 t2s_stage_decoder 风格的会话。
    ///
    /// 行为：每次 run() 把输入的 `iy` 序列**追加一个 token** 后作为 `y` 输出
    /// （跟真模型 `y = Concat(iy, 新 token)` 一致）。
    static std::unique_ptr<FakeSession> stage(int stop_after = 3, int layers = 2) {
        std::vector<TensorSpec> inputs;
        inputs.push_back({"iy", ElementType::Int64, {1, -1}});
        inputs.push_back({"iy_emb", ElementType::Float32, {1, -1, 512}});
        for (int i = 0; i < layers; ++i) {
            inputs.push_back(
                {"past_k_layer_" + std::to_string(i), ElementType::Float32, {-1, 1, -1}});
            inputs.push_back(
                {"past_v_layer_" + std::to_string(i), ElementType::Float32, {-1, 1, -1}});
        }

        FakeSessionConfig config;
        config.stop_after = stop_after;
        // 输出与真模型一致：y / y_emb / stop_condition_tensor / present_*。
        config.output_specs.push_back({"y", ElementType::Int64, {1, -1}});
        config.output_specs.push_back({"y_emb", ElementType::Float32, {1, -1, 512}});
        config.output_specs.push_back({"stop_condition_tensor", ElementType::Bool, {}});
        for (int i = 0; i < layers; ++i) {
            config.output_specs.push_back(
                {"present_k_layer_" + std::to_string(i), ElementType::Float32, {-1, 1, -1}});
            config.output_specs.push_back(
                {"present_v_layer_" + std::to_string(i), ElementType::Float32, {-1, 1, -1}});
        }
        return std::make_unique<FakeSession>(std::move(inputs), std::move(config));
    }

    /// 构造一个 vits 风格的会话。
    static std::unique_ptr<FakeSession> vocoder(int samples = 16) {
        std::vector<TensorSpec> inputs;
        inputs.push_back({"text_seq", ElementType::Int64, {1, -1}});
        inputs.push_back({"pred_semantic", ElementType::Int64, {1, 1, -1}});
        inputs.push_back({"ref_audio", ElementType::Float32, {1, -1}});

        FakeSessionConfig config;
        config.outputs.emplace_back(
            "audio", Tensor::zeros(ElementType::Float32, {1, static_cast<std::int64_t>(samples)}));
        return std::make_unique<FakeSession>(std::move(inputs), std::move(config));
    }

    /// 构造一个 V2ProPlus 声码器（吃 ge / ge_advanced，不吃 ref_audio）。
    static std::unique_ptr<FakeSession> vocoder_v2proplus(int samples = 16) {
        std::vector<TensorSpec> inputs;
        inputs.push_back({"text_seq", ElementType::Int64, {1, -1}});
        inputs.push_back({"pred_semantic", ElementType::Int64, {1, 1, -1}});
        inputs.push_back({"ge", ElementType::Float32, {1, 1024, 1}});
        inputs.push_back({"ge_advanced", ElementType::Float32, {1, 512, 1}});

        FakeSessionConfig config;
        config.outputs.emplace_back(
            "audio", Tensor::zeros(ElementType::Float32, {1, static_cast<std::int64_t>(samples)}));
        return std::make_unique<FakeSession>(std::move(inputs), std::move(config));
    }

    /// 构造一个说话人验证模型：16 kHz 波形 → sv_emb [1, 20480]。
    static std::unique_ptr<FakeSession> speaker_verification() {
        std::vector<TensorSpec> inputs;
        inputs.push_back({"waveform", ElementType::Float32, {1, -1}});

        FakeSessionConfig config;
        config.outputs.emplace_back("sv_emb", Tensor::zeros(ElementType::Float32, {1, 20480}));
        return std::make_unique<FakeSession>(std::move(inputs), std::move(config));
    }

    /// 构造一个 prompt_encoder：32 kHz 波形 + sv_emb → ge / ge_advanced。
    static std::unique_ptr<FakeSession> prompt_encoder() {
        std::vector<TensorSpec> inputs;
        inputs.push_back({"ref_audio", ElementType::Float32, {1, -1}});
        inputs.push_back({"sv_emb", ElementType::Float32, {1, 20480}});

        FakeSessionConfig config;
        config.outputs.emplace_back("ge", Tensor::zeros(ElementType::Float32, {1, 1024, 1}));
        config.outputs.emplace_back("ge_advanced",
                                    Tensor::zeros(ElementType::Float32, {1, 512, 1}));
        return std::make_unique<FakeSession>(std::move(inputs), std::move(config));
    }

    /// 校验喂进来的张量是否符合声明的契约。
    ///
    /// 这一步**不是装饰** —— 形状错位（秩不对、长度不对）在真模型上
    /// 要么报错、要么安静地输出垃圾。假会话照着真模型的 spec 校验，
    /// 就能在单测阶段把这类错误抓出来。
    ///
    /// spec 里 -1 表示动态维（那一维不校验）。
    void validate_feed(const TensorFeed& feed) const {
        for (const auto& [name, tensor] : feed) {
            const auto it = std::find_if(inputs_.begin(), inputs_.end(),
                                         [&name](const auto& spec) { return spec.name == name; });
            if (it == inputs_.end()) {
                throw std::runtime_error("fake session: unknown input '" + name + "'");
            }
            if (it->type != tensor.type()) {
                throw std::runtime_error("fake session: input '" + name + "' has wrong dtype");
            }
            if (it->shape.size() != tensor.rank()) {
                throw std::runtime_error("fake session: input '" + name + "' has rank " +
                                         std::to_string(tensor.rank()) + ", expected " +
                                         std::to_string(it->shape.size()));
            }
            for (std::size_t d = 0; d < it->shape.size(); ++d) {
                if (it->shape[d] >= 0 && it->shape[d] != tensor.dim(d)) {
                    throw std::runtime_error("fake session: input '" + name + "' dim " +
                                             std::to_string(d) + " is " +
                                             std::to_string(tensor.dim(d)) + ", expected " +
                                             std::to_string(it->shape[d]));
                }
            }
        }
    }

    [[nodiscard]] const std::vector<TensorSpec>& inputs() const override { return inputs_; }
    [[nodiscard]] const std::vector<TensorSpec>& outputs() const override { return output_specs_; }

    [[nodiscard]] std::vector<Tensor> run(const TensorFeed& feed) override {
        validate_feed(feed);
        ++config_.run_count;
        config_.last_feed = feed;
        config_.feeds.push_back(feed);

        std::vector<Tensor> result;
        result.reserve(output_specs_.size());

        for (const auto& spec : output_specs_) {
            // 静态输出优先（首阶段的 y、y_emb、KV cache 是固定值）。
            const auto static_it =
                std::find_if(config_.outputs.begin(), config_.outputs.end(),
                             [&spec](const auto& entry) { return entry.first == spec.name; });
            if (static_it != config_.outputs.end()) {
                result.push_back(static_it->second);
                continue;
            }
            if (spec.name == "y") {
                // y = Concat(iy, 新 token)
                std::vector<std::int64_t> tokens;
                const auto it = std::find_if(feed.begin(), feed.end(),
                                             [](const auto& entry) { return entry.first == "iy"; });
                if (it != feed.end()) {
                    const auto ints = it->second.ints();
                    tokens.assign(ints.begin(), ints.end());
                }
                tokens.push_back(config_.next_token);
                result.push_back(
                    Tensor::from_int64({1, static_cast<std::int64_t>(tokens.size())}, tokens));
            } else if (spec.name == "y_emb") {
                std::int64_t length = 1;
                const auto it = std::find_if(feed.begin(), feed.end(), [](const auto& entry) {
                    return entry.first == "iy_emb";
                });
                if (it != feed.end()) {
                    length = it->second.dim(1);
                }
                result.push_back(Tensor::zeros(ElementType::Float32, {1, length, 512}));
            } else if (spec.name == "stop_condition_tensor") {
                const bool stop = config_.stop_after >= 0 && config_.run_count > config_.stop_after;
                result.push_back(Tensor::from_bool({}, {stop}));
            } else {
                // KV cache：原样透传输入（真模型会把它变长）。
                Tensor value = Tensor::zeros(spec.type, spec.shape);
                for (const auto& entry : feed) {
                    if (entry.first == "past_" + spec.name.substr(std::string("present_").size())) {
                        value = entry.second;
                    }
                }
                result.push_back(std::move(value));
            }
        }
        return result;
    }

    [[nodiscard]] const std::string& model_path() const override { return path_; }

    [[nodiscard]] int run_count() const noexcept { return config_.run_count; }
    [[nodiscard]] const TensorFeed& last_feed() const noexcept { return config_.last_feed; }
    [[nodiscard]] const std::vector<TensorFeed>& feeds() const noexcept { return config_.feeds; }

    /// 改写某个输出的值（用于构造边界情况）。
    void set_output(const std::string& name, Tensor tensor) {
        for (auto& [output_name, value] : config_.outputs) {
            if (output_name == name) {
                value = std::move(tensor);
                return;
            }
        }
        config_.outputs.emplace_back(name, std::move(tensor));
    }

    /// 交换两个输入的顺序（用于验证"按名字绑定"而不是"按位置绑定"）。
    void swap_inputs(std::size_t a, std::size_t b) { std::swap(inputs_[a], inputs_[b]); }

    /// 改写输出描述（用于模拟"契约不符"）。
    void set_output_spec(const std::string& name, TensorSpec spec) {
        for (auto& existing : output_specs_) {
            if (existing.name == name) {
                existing = std::move(spec);
                return;
            }
        }
        output_specs_.push_back(std::move(spec));
    }

private:
    std::vector<TensorSpec> inputs_;
    std::vector<TensorSpec> output_specs_;
    FakeSessionConfig config_;
    std::string path_;
};

}  // namespace genie::test
