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

#include <cstddef>
#include <cstdint>
#include <memory>
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

/// 一个外部权重文件在图里的引用汇总（fp16 包加载期判定存储格式用）。
struct ExternalWeightFile {
    /// external_data 的 location（相对模型所在目录）。
    std::string location;
    /// 引用它的外部张量个数（含子图 / 节点属性 / 稀疏张量里的引用）。
    std::size_t tensor_count = 0;
    /// 引用区间的上界 max(offset + length)，以图声明的 **fp32 字节**为单位。
    std::uint64_t required_bytes = 0;
    /// 第一个妨碍"按 fp16 打包上转"的原因；空 = 全部引用都可以上转。
    ///
    /// 整文件逐元素上转要求每个引用都是**主图**初始化器（推理后端的内存
    /// 替换只覆盖主图）、声明为 FLOAT、带 length、offset/length 4 字节
    /// 对齐 —— 任一条不满足，上转后的布局就对不上图。
    std::string fp16_blocker;
};

/// 流式扫描模型文件里的 external-data 张量，按外部文件汇总（首次出现顺序）。
///
/// 只读 protobuf 的结构字段，raw_data 等大字段一律跳过 —— 成本与字段数
/// 成正比、与权重体积无关：1.2 GB 的自包含模型（RoBERTa）也只读几 KB。
///
/// @return 每个外部文件一项；自包含模型返回空
/// @throws genie::Error IoError —— 文件打不开；ModelLoadError —— 结构损坏
[[nodiscard]] std::vector<ExternalWeightFile> scan_external_weights(const std::string& model_path);

/// 外部权重文件的存储格式。
enum class ExternalWeightLayout {
    /// 文件装得下图声明的 fp32 区间：原样交给推理后端。
    Fp32,
    /// 只有一半装得下：HF 上游的 fp16 连续打包格式，加载期整文件上转。
    Fp16Packed,
};

/// 按"图声明区间 vs 文件大小"判定外部权重文件的存储格式。
///
/// HF 上游的 fp16 格式：图把外部张量声明为 FLOAT、offset/length 以 fp32
/// 字节为单位；`*_fp16.bin` 是同一批张量转成 fp16 后连续打包，文件大小
/// 恰为一半，张量位于 offset/2（本项目六模型的实测布局，见
/// docs/architecture.md §2.13）。两种解释都装不下时报错，不做歧义回退。
///
/// @throws genie::Error ModelLoadError —— 两种解释都装不下；判定为 fp16
///         打包但存在 fp16_blocker；fp16 打包文件的字节数为奇数
[[nodiscard]] ExternalWeightLayout classify_external_weights(const ExternalWeightFile& file,
                                                             std::uint64_t file_size);

/// fp16 打包权重文件整体上转后的 FP32 缓冲。
struct Fp32WeightBuffer {
    /// 上转结果（未零初始化后再覆盖 —— GB 级缓冲省一遍 memset）。
    std::unique_ptr<float[]> data;
    /// float 个数（= fp16 文件字节数 / 2）。
    std::size_t count = 0;

    [[nodiscard]] std::size_t size_bytes() const noexcept { return count * sizeof(float); }
};

/// 读取 fp16 打包的权重文件并整体上转为 FP32：第 i 个 half → 第 i 个 float。
///
/// 打包格式下张量在 fp16 文件里的位置恰为 fp32 offset 的一半，因此上转
/// 结果就是对应 fp32 bin 的字节布局 —— 可原样交给推理后端按图声明的
/// offset/length 寻址，**不需要改写图**。分块读取、边读边转：峰值内存
/// ≈ 输出缓冲 + 一个固定大小的读块。转换对全部 65536 个 half 值逐位
/// 精确（含次正规、±0、±Inf、NaN 载荷）。
///
/// @throws genie::Error IoError —— 打开或读取失败；ModelLoadError —— 字节数为奇数
[[nodiscard]] Fp32WeightBuffer read_fp16_weights_as_fp32(const std::string& path);

}  // namespace genie

#endif  // GENIE_ONNX_GRAPH_HPP
