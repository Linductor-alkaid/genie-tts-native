// genie/session.hpp — 模型会话抽象
//
// 把"加载一个 ONNX 模型并跑一次推理"这件事收敛到一个接口后面。
// 上层（gpt / acoustic）只依赖本接口，具体后端由构建期决定：
//   - GENIE_ENABLE_ORT=ON  → ONNX Runtime CPU EP
//   - 否则                  → create_session 抛 BackendUnavailable
//
// 这样上层代码在两种构建下都能编译，测试也能在无 ORT 环境下跑纯逻辑部分。

#ifndef GENIE_SESSION_HPP
#define GENIE_SESSION_HPP

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "genie/config.hpp"
#include "genie/tensor.hpp"

namespace genie {

/// 一个模型输入或输出的静态描述。
struct TensorSpec {
    std::string name;
    ElementType type = ElementType::Float32;
    /// 形状。动态维用 -1 表示。
    std::vector<std::int64_t> shape;

    [[nodiscard]] std::string shape_string() const;
};

/// 一次推理的输入绑定：名字 → 张量。
using TensorFeed = std::vector<std::pair<std::string, Tensor>>;

/// 已加载的模型会话。
class ModelSession {
public:
    virtual ~ModelSession() = default;

    /// 模型输入的静态描述，顺序与模型 graph.input 一致。
    [[nodiscard]] virtual const std::vector<TensorSpec>& inputs() const = 0;

    /// 模型输出的静态描述，顺序与模型 graph.output 一致。
    [[nodiscard]] virtual const std::vector<TensorSpec>& outputs() const = 0;

    /// 执行一次推理。返回的向量顺序与 outputs() 一致。
    ///
    /// feed 中的名字必须存在于 inputs()；缺失的输入视为可选（由模型自行决定）。
    [[nodiscard]] virtual std::vector<Tensor> run(const TensorFeed& feed) = 0;

    /// 模型文件路径。
    [[nodiscard]] virtual const std::string& model_path() const = 0;
};

/// 加载模型并创建会话。
///
/// @throws genie::Error  ModelLoadError —— 文件不存在或 ONNX 解析失败
/// @throws genie::Error  BackendUnavailable —— 构建时未启用 ONNX Runtime
[[nodiscard]] std::unique_ptr<ModelSession> create_session(const std::string& model_path,
                                                           const SessionConfig& config);

/// 当前构建是否具备真实推理能力。
[[nodiscard]] bool has_onnxruntime_backend() noexcept;

}  // namespace genie

#endif  // GENIE_SESSION_HPP
