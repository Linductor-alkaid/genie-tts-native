// genie/onnx_graph.hpp — 轻量 ONNX 图元信息读取
//
// 为什么需要它：
//   上游 Genie-TTS 随仓库发布的六个 ONNX 模型是**图壳** —— 权重存放在
//   外部 .bin 文件里（总计约 1.1 GB），由上游的模型转换流程生成，不在仓库中。
//   因此：
//     - 这些模型**无法**直接被 ONNX Runtime 加载；
//     - 但图结构（输入/输出的名字、类型、维度）完整，足以作为**契约**。
//
//   本模块直接解析 ONNX 的 protobuf，只读 graph.input / graph.output，
//   不需要权重、不需要 ONNX Runtime。
//   这样"模型 I/O 契约"就是可执行断言，而不是文档里的一段文字。
//
// 范围：只实现读取 ModelProto → GraphProto → ValueInfoProto 所需的字段子集。
//       不是通用 protobuf 解析器。

#ifndef GENIE_ONNX_GRAPH_HPP
#define GENIE_ONNX_GRAPH_HPP

#include <functional>

#include <cstdint>
#include <string>
#include <vector>

namespace genie {

/// ONNX 张量元素类型（与 onnx.TensorProto.DataType 的数值一致）。
enum class OnnxElementType : int {
    Undefined = 0,
    Float = 1,
    Uint8 = 2,
    Int8 = 3,
    Uint16 = 4,
    Int16 = 5,
    Int32 = 6,
    Int64 = 7,
    String = 8,
    Bool = 9,
    Float16 = 10,
    Double = 11,
    Uint32 = 12,
    Uint64 = 13,
};

[[nodiscard]] const char* to_string(OnnxElementType type) noexcept;

/// 图中一个张量（输入或输出）的元信息。
struct OnnxTensorInfo {
    std::string name;
    OnnxElementType type = OnnxElementType::Undefined;
    /// 形状。动态维用 -1 表示（对应 ONNX 的 dim_param）。
    std::vector<std::int64_t> shape;
    /// 该维度是否是符号维（有名字而不是具体数值）。
    std::vector<bool> symbolic_dims;

    [[nodiscard]] std::string shape_string() const;
};

/// 一个 ONNX 模型的图元信息。
struct OnnxGraphSpec {
    std::int64_t ir_version = 0;
    std::vector<OnnxTensorInfo> inputs;
    std::vector<OnnxTensorInfo> outputs;
    /// 权重是否存放在外部数据文件里。
    bool uses_external_data = false;
    /// 外部数据文件的名字集合（去重）。
    std::vector<std::string> external_data_files;
};

/// 读取 ONNX 模型的图元信息。
///
/// 只解析结构，不加载权重，因此对"图壳"模型同样有效。
///
/// @throws genie::Error IoError —— 文件不存在或不是合法 ONNX
[[nodiscard]] OnnxGraphSpec read_onnx_graph(const std::string& path);

/// 从内存读取 ONNX 模型的图元信息。
///
/// @throws genie::Error IoError —— 缓冲区不是合法 ONNX
[[nodiscard]] OnnxGraphSpec read_onnx_graph_from_memory(const std::vector<std::byte>& data);

/// 外部权重提供器：按 external_data 的 location（.bin 文件名，相对模型
/// 同目录）返回整个权重文件的字节。返回 nullptr 表示该文件不可用。
using ExternalWeightSource =
    std::function<const std::vector<std::byte>*(const std::string& location)>;

/// 把模型里 external-data 的初始化器**内联**为 raw_data，返回新的模型字节。
///
/// 每个外部文件的精度由转码器按"图声明区间 vs 文件大小"自动判定：
///
/// - 文件装得下图的 fp32 区间 → 逐字节原样内联（fp32 / 整数声明类型
///   都不变，对应 fp32 bin 与自包含 fp16 张量）；
/// - 否则若**一半区间**装得下 → 按 HF 上游的 fp16 打包格式解释：
///   图声明 FLOAT、offset/length 以 fp32 字节为单位，文件里同一张量
///   在 offset/2 处占 length/2 字节 fp16 —— 拷贝时上转为 FP32 内联
///   （包体积减半，运行时仍是 FP32；这正是 HF 仓库 `*_fp16.bin` 的
///   实测布局，见 docs/architecture.md）；
/// - 两种解释都不行 → ModelLoadError，不做歧义回退。
///
/// 未触及的 protobuf 字段逐字节保留（node 属性等不透明拷贝，不依赖
/// schema）；external_data / data_location 按 ONNX 规范清除（禁止与
/// raw_data 并存，否则 ORT 会忽略 raw_data）；已有 raw_data 的初始化器
/// 原样通过。
///
/// @param model  原始模型字节（可为"图壳 + external data"形态）
/// @param source 权重文件提供器
/// @return 内联后的模型字节 —— 可直接交给支持内存加载的推理后端
///
/// @throws genie::Error ModelLoadError —— external 条目缺 length、
///         两种解释下都越界、provider 找不到文件、fp16 打包张量
///         非 4 字节对齐或声明为非 FLOAT
[[nodiscard]] std::vector<std::byte> inline_external_fp16_weights(
    const std::vector<std::byte>& model, const ExternalWeightSource& source);

}  // namespace genie

#endif  // GENIE_ONNX_GRAPH_HPP
