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
#include <cstring>
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
            throw_error(ErrorCode::Unsupported,
                        "ONNX element type " + std::to_string(static_cast<int>(type)) +
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
    OrtModelSession(const std::string& model_path, const SessionConfig& config) {
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

        try {
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
        Ort::AllocatorWithDefaultOptions allocator;

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
        (void)allocator;
        return result;
    }

private:
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

    /// 把本项目的 Tensor 转换为 Ort::Value。
    ///
    /// ORT 的 CreateTensor(info, p_data, ...) 需要非 const 指针，而输入张量
    /// 在语义上是只读的。因此改用 ORT 分配器创建张量再 memcpy 进去 ——
    /// 既避免了 const_cast，也不引入 strict-aliasing 问题。
    static Ort::Value to_ort_value(const Tensor& tensor) {
        const std::int64_t* shape = tensor.shape().data();
        const std::size_t rank = tensor.rank();
        Ort::AllocatorWithDefaultOptions allocator;

        switch (tensor.type()) {
            case ElementType::Float32: {
                Ort::Value value = Ort::Value::CreateTensor<float>(allocator, shape, rank);
                std::memcpy(value.GetTensorMutableData<float>(), tensor.floats().data(),
                            tensor.element_count() * sizeof(float));
                return value;
            }
            case ElementType::Int64: {
                Ort::Value value = Ort::Value::CreateTensor<std::int64_t>(allocator, shape, rank);
                std::memcpy(value.GetTensorMutableData<std::int64_t>(), tensor.ints().data(),
                            tensor.element_count() * sizeof(std::int64_t));
                return value;
            }
            case ElementType::Bool: {
                Ort::Value value = Ort::Value::CreateTensor<bool>(allocator, shape, rank);
                std::memcpy(value.GetTensorMutableData<bool>(), tensor.bools().data(),
                            tensor.element_count() * sizeof(bool));
                return value;
            }
        }
        throw_error(ErrorCode::Internal, "unreachable tensor type");
    }

    static Tensor from_ort_value(const Ort::Value& value) {
        const auto info = value.GetTensorTypeAndShapeInfo();
        const std::vector<std::int64_t> shape = info.GetShape();

        switch (info.GetElementType()) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
                const float* data = value.GetTensorData<float>();
                return Tensor::from_float32(shape, std::vector<float>(data, data + info.GetElementCount()));
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
                const std::int64_t* data = value.GetTensorData<std::int64_t>();
                return Tensor::from_int64(
                    shape, std::vector<std::int64_t>(data, data + info.GetElementCount()));
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: {
                const bool* data = value.GetTensorData<bool>();
                std::vector<bool> values(data, data + info.GetElementCount());
                return Tensor::from_bool(shape, std::move(values));
            }
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

std::unique_ptr<ModelSession> create_session(const std::string& model_path,
                                              const SessionConfig& config) {
    return std::make_unique<OrtModelSession>(model_path, config);
}

bool has_onnxruntime_backend() noexcept { return true; }

}  // namespace genie
