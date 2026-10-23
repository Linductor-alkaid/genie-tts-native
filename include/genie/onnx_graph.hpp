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

}  // namespace genie

#endif  // GENIE_ONNX_GRAPH_HPP
