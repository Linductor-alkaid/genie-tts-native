// src/runtime/ort_backend.cpp — ONNX Runtime CPU Execution Provider 后端
//
// 关键实现细节（来自 M0 上游审计，详见 docs/upstream-audit.md）：
//
//   1. 绑定必须**按名字**进行。上游 Python 用
//        zip(input_names, [y, y_emb, *present_key_values])
//      靠位置对应，而 stage_decoder 的真实输入名是
//        iy / iy_emb / past_k_layer_0..23 / past_v_layer_0..23
//      —— 与上游变量名 y / y_emb 并不一致。按名字绑定可以避免位置错位。
//
//   2. t2s_stage_decoder 的 stop_condition_tensor 是 **rank-0 bool** 标量。
//
//   3. 采样（TopK / 温度）在 ONNX 图内完成，主机侧不做 top-k/top-p。
//      唯一可控的随机性是 ORT 全局随机种子。

#include "genie/session.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "genie/errors.hpp"

namespace genie {
namespace {

/// ORT 的 ONNXTensorElementDataType → 本项目的 ElementType。
ElementType map_element_type(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return ElementType::Float32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            return ElementType::Int64;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            return ElementType::Bool;
        default:
            throw_error(ErrorCode::Unsupported, "ONNX element type " +
                                                    std::to_string(static_cast<int>(type)) +
                                                    " is not supported by the native runtime "
                                                    "(only float32 / int64 / bool are)");
    }
}

/// ORT 异常 → genie::Error。
[[noreturn]] void rethrow_ort(const Ort::Exception& e, const std::string& context) {
    throw_error(ErrorCode::InferenceError, context + ": " + e.what());
}

/// ONNX Runtime 环境下唯一的 Env 实例。
///
/// ORT 的 Env 是进程级单例语义（内部有全局状态），多实例创建虽然合法，
/// 但共用一个 Env 可以避免重复初始化日志与线程池。
Ort::Env& shared_env() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "genie-tts-native");
    return env;
}

/// 基于 ORT 的会话实现。
class OrtModelSession final : public ModelSession {
public:
    /// @param external_weights 改由内存提供的外部权重文件（fp16 上转路径；
    ///        普通加载为空）。缓冲只需在本构造期间有效 —— ORT 默认在建
    ///        会话时拷贝（AddExternalInitializersFromFilesInMemory 的文档语义，
    ///        契约由集成测试锁定：会话建完即释放缓冲，再推理校验结果）。
    OrtModelSession(const std::string& model_path, const SessionConfig& config,
                    const std::vector<ExternalWeightBuffer>& external_weights = {}) {
        try {
            auto options = make_options(config, model_path);
            if (!external_weights.empty()) {
                add_external_weights(options, external_weights);
            }
            session_ = std::make_unique<Ort::Session>(shared_env(), model_path.c_str(), options);
        } catch (const Ort::Exception& e) {
            throw_error(ErrorCode::ModelLoadError,
                        "failed to load ONNX model '" + model_path + "': " + e.what());
        }

        model_path_ = model_path;
        io_binding_enabled_ = config.enable_io_binding;
        collect_specs();
        // 学习态与输出一一对应：空着会让 learn_output_shapes 的 growth_[i]
        // 越界（默认路径也会走到 —— 这正是本分支踩过的段错误）。
        growth_.assign(outputs_.size(), std::nullopt);
    }

    [[nodiscard]] const std::vector<TensorSpec>& inputs() const override { return inputs_; }
    [[nodiscard]] const std::vector<TensorSpec>& outputs() const override { return outputs_; }
    [[nodiscard]] const std::string& model_path() const override { return model_path_; }

    [[nodiscard]] std::vector<Tensor> run(const TensorFeed& feed) override {
        // IOBinding 输出缓冲池（M5 第二轮）：预测准则绑定、预测失准自动
        // 回退重学 —— 两条路径对调用方完全等价（集成测试锁定逐位一致）。
        // 失败计数防"永久双跑"：单次调用的会话（first_stage 等）跨合成
        // 形状跳变，学到的增长永远预测不准 —— 连续两次预测失败且从未
        // 成功过的会话永久禁用绑定（代价：生命周期内最多两次廉价的重试，
        // ORT 在执行内核**之前**就拒绝形状不符的绑定）。解码循环会话
        // 首次尝试即成功（合成内增长稳定），不受影响。
        if (io_binding_enabled_ && binding_ready_ && !binding_banned_) {
            auto pooled = run_with_binding(feed);
            if (pooled.has_value()) {
                binding_failures_ = 0;
                return std::move(*pooled);
            }
            ++binding_failures_;
            if (binding_failures_ >= 2) {
                binding_banned_ = true;
            }
            // 形状跳变（会话跨合成复用等）：丢弃预测结果，重学增长规律。
            binding_ready_ = false;
            last_shapes_.clear();
            growth_.assign(outputs_.size(), std::nullopt);
        }
        auto result = run_normal(feed);
        learn_output_shapes(result);
        return result;
    }

private:
    using Shape = Tensor::Shape;

    /// 普通路径：ORT 自分配输出 + from_ort_value 逐份拷出（每输出恰好一
    /// 次 memcpy）。也是绑定路径的兜底与形状学习的数据来源。
    [[nodiscard]] std::vector<Tensor> run_normal(const TensorFeed& feed) {
        std::vector<const char*> input_names;
        std::vector<Ort::Value> input_values;
        input_names.reserve(feed.size());
        input_values.reserve(feed.size());

        for (const auto& [name, tensor] : feed) {
            input_names.push_back(name.c_str());
            input_values.push_back(to_ort_value(tensor));
        }

        std::vector<const char*> output_names;
        output_names.reserve(outputs_.size());
        for (const auto& spec : outputs_) {
            output_names.push_back(spec.name.c_str());
        }

        std::vector<Ort::Value> raw;
        try {
            raw = session_->Run(Ort::RunOptions{nullptr}, input_names.data(), input_values.data(),
                                input_values.size(), output_names.data(), output_names.size());
        } catch (const Ort::Exception& e) {
            rethrow_ort(e, "inference failed for model '" + model_path_ + "'");
        }

        std::vector<Tensor> result;
        result.reserve(raw.size());
        for (std::size_t i = 0; i < raw.size(); ++i) {
            result.push_back(from_ort_value(raw[i]));
        }
        return result;
    }

    /// 绑定路径：输出绑到自管缓冲，ORT 直接写入，Tensor **接管**缓冲
    /// （from_buffer 的 vector 指针换手）—— 宿主侧输出拷贝消失。
    ///
    /// 返回 nullopt = 预测失准（Run 拒绝形状不符的绑定，发生在写入之前，
    /// 无半成品状态），调用方回退普通路径。两次 run 的 ORT 重复执行成本
    /// 由后续几十步的免拷贝摊回。
    [[nodiscard]] std::optional<std::vector<Tensor>> run_with_binding(const TensorFeed& feed) {
        // 1) 预测本步形状：上一步实际形状 + 学到的增长（逐输出）。
        //    growth 维度 -1 = 定型不增长（预测 = 原形状，不按下标索引）。
        std::vector<Tensor::Shape> predicted;
        predicted.reserve(outputs_.size());
        for (std::size_t i = 0; i < outputs_.size(); ++i) {
            Tensor::Shape shape = last_shapes_[i];
            if (growth_[i] && growth_[i]->first >= 0) {
                shape[static_cast<std::size_t>(growth_[i]->first)] += growth_[i]->second;
            }
            predicted.push_back(std::move(shape));
        }

        // 2) 绑定：输入零拷贝借用（与普通路径同款），输出绑自管缓冲。
        Ort::IoBinding binding(*session_);
        std::vector<const char*> input_names;
        std::vector<Ort::Value> input_values;
        input_names.reserve(feed.size());
        input_values.reserve(feed.size());
        for (const auto& [name, tensor] : feed) {
            input_names.push_back(name.c_str());
            input_values.push_back(to_ort_value(tensor));
            binding.BindInput(name.c_str(), input_values.back());
        }
        std::vector<std::vector<std::byte>> blocks(outputs_.size());
        for (std::size_t i = 0; i < outputs_.size(); ++i) {
            const std::size_t element_count = product_of(predicted[i]);
            blocks[i].assign(element_count * element_size(outputs_[i].type), std::byte{0});
            binding.BindOutput(outputs_[i].name.c_str(),
                               make_output_value(blocks[i], outputs_[i].type, predicted[i]));
        }

        try {
            session_->Run(Ort::RunOptions{nullptr}, binding);
        } catch (const Ort::Exception&) {
            // 形状不符等：ORT 在写入前拒绝，无半成品。回退普通路径。
            return std::nullopt;
        }

        // 3) 按名字取回并验证实际形状 == 预测（绑定的 Ort::Value 以我们
        //    声明的形状存在，这里校验的是 ORT 实际写出的形状；不匹配说明
        //    学到的增长不再成立）。
        const auto out_names = binding.GetOutputNames();
        const auto out_values = binding.GetOutputValues();
        std::vector<Tensor> result;
        result.reserve(outputs_.size());
        std::vector<std::optional<Shape>> actual_by_index(outputs_.size());
        for (std::size_t i = 0; i < out_names.size() && i < out_values.size(); ++i) {
            for (std::size_t j = 0; j < outputs_.size(); ++j) {
                if (out_names[i] == outputs_[j].name) {
                    actual_by_index[j] = out_values[i].GetTensorTypeAndShapeInfo().GetShape();
                    break;
                }
            }
        }
        for (std::size_t i = 0; i < outputs_.size(); ++i) {
            if (!actual_by_index[i] || *actual_by_index[i] != predicted[i]) {
                return std::nullopt;
            }
        }
        for (std::size_t i = 0; i < outputs_.size(); ++i) {
            result.push_back(
                Tensor::from_buffer(std::move(blocks[i]), outputs_[i].type, predicted[i]));
            last_shapes_[i] = predicted[i];
        }
        return result;
    }

    /// 形状学习：普通路径的每次 run 观测一次输出形状；相邻两次差里学
    /// "每个输出哪个维度 + 增多少"（解码循环的 KV / token 序列每步恒定
    /// +1，是稳定增量）。三种定型：{维度, 增量} 单维稳定增长 / {-1, 0}
    /// 不变 / {-2, 0} 多维同时变（不可预测 —— 任何输出落进这一档，整个
    /// 会话放弃绑定，否则每步都要白白重跑）。上次预测失准的 reset 由
    /// run() 负责（growth_ 全空 + last_shapes_ 清空，回到首观测态）。
    void learn_output_shapes(const std::vector<Tensor>& outputs) {
        if (outputs.size() != outputs_.size()) {
            return;  // 理论不可达（输出数固定）；防御一下不把状态弄脏
        }
        if (last_shapes_.empty()) {
            last_shapes_.reserve(outputs.size());
            for (const auto& tensor : outputs) {
                last_shapes_.push_back(tensor.shape());
            }
            return;
        }
        for (std::size_t i = 0; i < outputs.size(); ++i) {
            if (growth_[i].has_value()) {
                continue;  // 已定型不再改写
            }
            const Tensor::Shape& previous = last_shapes_[i];
            const Tensor::Shape& current = outputs[i].shape();
            if (previous.size() != current.size()) {
                growth_[i] = {-2, 0};  // 秩都变了：不可预测
                continue;
            }
            std::optional<std::size_t> changed_dim;
            std::int64_t delta = 0;
            bool single_change = true;
            for (std::size_t d = 0; d < current.size(); ++d) {
                if (previous[d] != current[d]) {
                    if (changed_dim.has_value()) {
                        single_change = false;
                        break;
                    }
                    changed_dim = d;
                    delta = current[d] - previous[d];
                }
            }
            if (single_change && changed_dim.has_value() && delta > 0) {
                growth_[i] = {static_cast<int>(*changed_dim), delta};
            } else if (single_change) {
                growth_[i] = {-1, 0};  // 不增长：预测 = 原形状
            } else {
                growth_[i] = {-2, 0};  // 多维同时变：不可预测
            }
        }
        bool all_predictable = true;
        for (const auto& growth : growth_) {
            if (!growth.has_value() || growth->first == -2) {
                all_predictable = false;
                break;
            }
        }
        binding_ready_ = io_binding_enabled_ && all_predictable;
        last_shapes_.clear();
        last_shapes_.reserve(outputs.size());
        for (const auto& tensor : outputs) {
            last_shapes_.push_back(tensor.shape());
        }
    }

    /// 把自管字节缓冲按声明形状 / 类型包装为可绑定的 Ort::Value（零拷贝：
    /// ORT 直接写进缓冲）。空 shape = rank-0 标量（stop_condition_tensor）。
    static Ort::Value make_output_value(std::vector<std::byte>& block, ElementType type,
                                        const Shape& shape) {
        const std::int64_t* dims = shape.empty() ? nullptr : shape.data();
        const std::size_t rank = shape.size();
        const std::size_t element_count = product_of(shape);
        void* data = block.data();
        const auto& info = cpu_memory_info();
        switch (type) {
            case ElementType::Float32:
                return Ort::Value::CreateTensor<float>(info, static_cast<float*>(data),
                                                       element_count, dims, rank);
            case ElementType::Int64:
                return Ort::Value::CreateTensor<std::int64_t>(
                    info, static_cast<std::int64_t*>(data), element_count, dims, rank);
            case ElementType::Bool:
                return Ort::Value::CreateTensor<bool>(info, static_cast<bool*>(data), element_count,
                                                      dims, rank);
        }
        throw_error(ErrorCode::Internal, "unreachable tensor type");
    }

    static std::size_t product_of(const Shape& shape) {
        std::size_t product = 1;
        for (const std::int64_t dim : shape) {
            product *= static_cast<std::size_t>(dim < 0 ? 0 : dim);
        }
        return product;
    }
    /// 把内存里的外部权重文件挂到会话选项上（ORT ≥ 1.18）。
    ///
    /// C API 要的是 char*（非 const）：默认的拷贝模式下 ORT 只读不写，
    /// cast 掉的 const 只来自本项目"缓冲只读"的接口约定。文件名在
    /// Windows 上是宽字符（ORTCHAR_T），经 filesystem::path 转换。
    static void add_external_weights(Ort::SessionOptions& options,
                                     const std::vector<ExternalWeightBuffer>& weights) {
        std::vector<std::basic_string<ORTCHAR_T>> names;
        std::vector<char*> buffers;
        std::vector<std::size_t> lengths;
        names.reserve(weights.size());
        buffers.reserve(weights.size());
        lengths.reserve(weights.size());
        for (const auto& weight : weights) {
            names.emplace_back(std::filesystem::path(weight.location).native());
            buffers.push_back(const_cast<char*>(static_cast<const char*>(weight.data)));
            lengths.push_back(weight.size);
        }
        options.AddExternalInitializersFromFilesInMemory(names, buffers, lengths);
    }

    [[nodiscard]] static Ort::SessionOptions make_options(const SessionConfig& config,
                                                          const std::string& model_path) {
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(config.intra_op_threads);
        options.SetInterOpNumThreads(config.inter_op_threads);
        options.SetExecutionMode(config.parallel_execution ? ORT_PARALLEL : ORT_SEQUENTIAL);
        // ORT 默认开启内存模式与 CPU arena，这里按配置反向关闭。
        if (!config.enable_mem_pattern) {
            options.DisableMemPattern();
        }
        if (!config.enable_cpu_mem_arena) {
            options.DisableCpuMemArena();
        }
        // ORT 默认自旋等待（低延迟 ↔ CPU 常驻占用的交换）。引擎一次合成
        // 只串行使用一个会话，闲置会话的自旋线程纯属内耗；M5.3 的实测
        // 数据与默认值选择见 docs/performance.md。
        options.AddConfigEntry("session.intra_op.allow_spinning",
                               config.allow_spinning ? "1" : "0");
        // 预打包在 intra-op 线程池上并行。键是 ORT 1.31 才加的；更早的版本
        // 不认识它、直接忽略（等价于关闭），所以无条件写入是安全的。
        if (config.parallel_prepack) {
            options.AddConfigEntry("session.prepack.enable_parallel", "1");
        }
        // 算子级 profiler（M5 性能分析采集端）：trace 在会话析构时写出，
        // 前缀给到"目录/模型文件 stem"，ORT 自己追加时间戳与 pid —— 同进程
        // 多会话（按角色）互不覆盖。目录由调用方保证存在。EnableProfiling
        // 接 ORTCHAR_T 路径（Windows 宽字符），filesystem::path 原生形态直达。
        if (!config.profile_output_dir.empty()) {
            const std::filesystem::path profile_prefix =
                std::filesystem::path(config.profile_output_dir) /
                std::filesystem::path(model_path).stem();
            options.EnableProfiling(profile_prefix.native().c_str());
        }

        switch (config.graph_optimization_level) {
            case 0:
                options.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
                break;
            case 1:
                options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
                break;
            case 2:
                options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
                break;
            default:
                options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
                break;
        }
        return options;
    }

    void collect_specs() {
        Ort::AllocatorWithDefaultOptions allocator;
        const std::size_t input_count = session_->GetInputCount();
        inputs_.reserve(input_count);
        for (std::size_t i = 0; i < input_count; ++i) {
            inputs_.push_back(make_spec(i, true, allocator));
        }
        const std::size_t output_count = session_->GetOutputCount();
        outputs_.reserve(output_count);
        for (std::size_t i = 0; i < output_count; ++i) {
            outputs_.push_back(make_spec(i, false, allocator));
        }
    }

    TensorSpec make_spec(std::size_t index, bool is_input,
                         Ort::AllocatorWithDefaultOptions& allocator) const {
        TensorSpec spec;
        const auto name = is_input ? session_->GetInputNameAllocated(index, allocator)
                                   : session_->GetOutputNameAllocated(index, allocator);
        spec.name = name.get();

        Ort::TypeInfo type_info =
            is_input ? session_->GetInputTypeInfo(index) : session_->GetOutputTypeInfo(index);
        const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        spec.type = map_element_type(tensor_info.GetElementType());

        const std::vector<std::int64_t> shape = tensor_info.GetShape();
        spec.shape = shape.empty() ? std::vector<std::int64_t>{} : shape;
        // ORT 用 -1 表示动态维，直接沿用。
        return spec;
    }

    /// CPU / 默认分配器的内存描述符，所有张量包装共用。
    ///
    /// MemoryInfo 只是"内存在哪种分配器上"的标记，构造一次即可；
    /// 此前每个张量每次 run 都新建一个（连同 ORT 分配器），纯开销。
    static const Ort::MemoryInfo& cpu_memory_info() {
        static const Ort::MemoryInfo info =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        return info;
    }

    /// 把本项目的 Tensor **零拷贝**包装为 Ort::Value。
    ///
    /// 直接引用 Tensor 的存储缓冲：输入数据在 Run 的整个执行期间存活
    /// （feed 以 const 引用贯穿 run() 调用），ORT 的 CPU EP 不会改写
    /// 输入缓冲，因此可以安全借用。此前走"ORT 分配器分配 + memcpy"，
    /// 每步自回归解码多出一份与 KV cache 等体积的宿主拷贝
    /// （M5.2，见 docs/architecture.md 张量一节）。
    ///
    /// 历史注记：最初实现刻意绕开 const_cast（担心 strict-aliasing 与
    /// ORT 写入）。CreateTensor(info, p_data, ...) 需要非 const 指针；
    /// 这里 cast 掉的 const 只来自 run(const TensorFeed&) 的接口签名，
    /// 底层存储本来就可写（Tensor::bytes() 有非 const 重载）。
    /// strict-aliasing 与 tensor.cpp 里的 reinterpret_span 视图同款：
    /// 存储内容本来就是由 memcpy 写入的目标类型字节。若未来某 ORT
    /// 算子要就地改写输入，这个借用契约即被破坏 —— 必须回退拷贝路径。
    static Ort::Value to_ort_value(const Tensor& tensor) {
        const std::int64_t* shape = tensor.shape().data();
        const std::size_t rank = tensor.rank();
        void* data = const_cast<void*>(static_cast<const void*>(tensor.bytes().data()));
        const auto& info = cpu_memory_info();

        switch (tensor.type()) {
            case ElementType::Float32:
                return Ort::Value::CreateTensor<float>(info, static_cast<float*>(data),
                                                       tensor.element_count(), shape, rank);
            case ElementType::Int64:
                return Ort::Value::CreateTensor<std::int64_t>(
                    info, static_cast<std::int64_t*>(data), tensor.element_count(), shape, rank);
            case ElementType::Bool:
                return Ort::Value::CreateTensor<bool>(info, static_cast<bool*>(data),
                                                      tensor.element_count(), shape, rank);
        }
        throw_error(ErrorCode::Internal, "unreachable tensor type");
    }

    static Tensor from_ort_value(const Ort::Value& value) {
        const auto info = value.GetTensorTypeAndShapeInfo();
        const std::vector<std::int64_t> shape = info.GetShape();

        switch (info.GetElementType()) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                return Tensor::from_raw(ElementType::Float32, shape, value.GetTensorData<float>());
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
                return Tensor::from_raw(ElementType::Int64, shape,
                                        value.GetTensorData<std::int64_t>());
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
                // ORT 的 bool 张量每元素 1 字节，与本项目的存储约定一致。
                return Tensor::from_raw(ElementType::Bool, shape, value.GetTensorData<bool>());
            default:
                throw_error(ErrorCode::Unsupported, "unsupported ONNX output element type");
        }
    }

    // IOBinding 输出缓冲池的状态（全部只在 run() 内演进；引擎契约是
    // 单线程使用会话，见 deployment §6）。
    bool io_binding_enabled_ = false;  // SessionConfig::enable_io_binding
    bool binding_ready_ = false;       // 全部输出的形状增长已学到
    bool binding_banned_ = false;      // 反复预测失准：永久禁用（防永久双跑）
    int binding_failures_ = 0;         // 连续预测失败且无成功间隔的次数
    std::vector<Tensor::Shape> last_shapes_;  // 上一步实际输出形状
    std::vector<std::optional<std::pair<int, std::int64_t>>> growth_;  // 维度+增量

    std::string model_path_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<TensorSpec> inputs_;
    std::vector<TensorSpec> outputs_;
};

}  // namespace

const char* onnxruntime_version_string() noexcept {
    return OrtGetApiBase()->GetVersionString();
}

std::unique_ptr<ModelSession> create_session(const std::string& model_path,
                                             const SessionConfig& config) {
    return std::make_unique<OrtModelSession>(model_path, config);
}

std::unique_ptr<ModelSession> create_session(
    const std::string& model_path, const SessionConfig& config,
    const std::vector<ExternalWeightBuffer>& external_weights) {
    return std::make_unique<OrtModelSession>(model_path, config, external_weights);
}

bool has_onnxruntime_backend() noexcept {
    return true;
}

}  // namespace genie
