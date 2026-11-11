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
        collect_specs();
    }

    [[nodiscard]] const std::vector<TensorSpec>& inputs() const override { return inputs_; }
    [[nodiscard]] const std::vector<TensorSpec>& outputs() const override { return outputs_; }
    [[nodiscard]] const std::string& model_path() const override { return model_path_; }

    [[nodiscard]] std::vector<Tensor> run(const TensorFeed& feed) override {
        // 名字 → 张量 索引表，便于按名字绑定。
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

private:
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
