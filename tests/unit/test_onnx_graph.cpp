// tests/unit/test_onnx_graph.cpp
//
// ONNX protobuf 读取器的单元测试。
//
// 这个读取器存在的意义：上游发布的 ONNX 模型是"图壳"（权重在外部 .bin 里），
// 无法被 ONNX Runtime 加载，但图结构完整 —— I/O 契约必须能被**可执行地**锁定。

#include <gtest/gtest.h>

#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

#include "genie/errors.hpp"
#include "genie/onnx_graph.hpp"
#include "support/model_paths.hpp"

using namespace genie;
using namespace genie::test;

TEST(OnnxGraph, ReadsSyntheticModelSignature) {
    const auto path = synthetic_model();
    if (!std::filesystem::exists(path)) {
        GTEST_SKIP() << "synthetic model not available at " << path.string();
    }

    const auto spec = read_onnx_graph(path.string());

    ASSERT_EQ(spec.inputs.size(), 3U);
    EXPECT_EQ(spec.inputs[0].name, "alpha");
    EXPECT_EQ(spec.inputs[1].name, "beta");
    EXPECT_EQ(spec.inputs[2].name, "gamma");
    EXPECT_EQ(spec.inputs[0].type, OnnxElementType::Int64);
    EXPECT_EQ(spec.inputs[1].type, OnnxElementType::Float);
    EXPECT_EQ(spec.inputs[2].type, OnnxElementType::Float);

    ASSERT_EQ(spec.outputs.size(), 3U);
    EXPECT_EQ(spec.outputs[0].name, "out_i");
    EXPECT_EQ(spec.outputs[1].name, "out_f");
    EXPECT_EQ(spec.outputs[2].name, "out_scalar");

    // rank-0 bool 标量
    EXPECT_EQ(spec.outputs[2].type, OnnxElementType::Bool);
    EXPECT_TRUE(spec.outputs[2].shape.empty());

    // beta 的第二维是静态的 1024
    EXPECT_EQ(spec.inputs[1].shape.size(), 2U);
    EXPECT_EQ(spec.inputs[1].shape[1], 1024);

    // 自包含模型不使用外部数据
    EXPECT_FALSE(spec.uses_external_data);
}

TEST(OnnxGraph, RejectsNonOnnxData) {
    const std::vector<std::byte> garbage(64, std::byte{0x7F});
    EXPECT_THROW(read_onnx_graph_from_memory(garbage), Error);
}

TEST(OnnxGraph, MissingFileThrowsIoError) {
    EXPECT_THROW(read_onnx_graph("/definitely/not/here/model.onnx"), Error);
}

TEST(OnnxGraph, DetectsExternalDataInUpstreamModels) {
    if (!model_exists("v2/t2s_encoder_fp32.onnx")) {
        GTEST_SKIP() << "test models not available at " << model_root().string();
    }
    const auto spec = read_onnx_graph(v2_model("t2s_encoder_fp32.onnx").string());
    EXPECT_TRUE(spec.uses_external_data);
    EXPECT_FALSE(spec.external_data_files.empty());
}

TEST(OnnxGraph, ElementTypeNames) {
    EXPECT_STREQ(to_string(OnnxElementType::Float), "float32");
    EXPECT_STREQ(to_string(OnnxElementType::Int64), "int64");
    EXPECT_STREQ(to_string(OnnxElementType::Bool), "bool");
    EXPECT_STREQ(to_string(OnnxElementType::Undefined), "undefined");
}

// ------------------------------------------------------------------ 外部权重：扫描 / 判定 / 上转
//
// 手工拼装最小 wire-format 写进临时文件：造数据比提交二进制资产更能表达
// "扫描器从字节里读出了什么"。

namespace {

void append_varint(std::vector<std::byte>& out, std::uint64_t value) {
    while (value >= 0x80U) {
        out.push_back(static_cast<std::byte>(static_cast<std::uint8_t>(value) | 0x80U));
        value >>= 7U;
    }
    out.push_back(static_cast<std::byte>(static_cast<std::uint8_t>(value)));
}

void append_tag(std::vector<std::byte>& out, const std::uint32_t field, const std::uint32_t wire) {
    append_varint(out, (static_cast<std::uint64_t>(field) << 3U) | wire);
}

void append_len_field(std::vector<std::byte>& out, const std::uint32_t field, const void* data,
                      const std::size_t size) {
    append_tag(out, field, 2);
    append_varint(out, size);
    const auto* bytes = static_cast<const std::byte*>(data);
    out.insert(out.end(), bytes, bytes + size);
}

void append_len_field(std::vector<std::byte>& out, const std::uint32_t field,
                      const std::string& payload) {
    append_len_field(out, field, payload.data(), payload.size());
}

void append_len_field(std::vector<std::byte>& out, const std::uint32_t field,
                      const std::vector<std::byte>& payload) {
    append_len_field(out, field, payload.data(), payload.size());
}

void append_varint_field(std::vector<std::byte>& out, const std::uint32_t field,
                         const std::uint64_t value) {
    append_tag(out, field, 0);
    append_varint(out, value);
}

/// 外部初始化器的参数。默认值即 HF fp16 格式的一个合法张量：
/// data_type=FLOAT（图声明 fp32），offset/length 以 fp32 字节为单位。
struct ExternalTensorSpec {
    std::string name = "w";
    std::string location = "w.bin";
    std::string offset = "0";
    std::string length = "16";
    bool with_length = true;
    std::uint64_t data_type = 1;
};

std::vector<std::byte> make_external_tensor(const ExternalTensorSpec& spec = {}) {
    std::vector<std::byte> t;
    append_varint_field(t, 1, 4);  // dims[0]（repeated int64 非打包形态）
    append_varint_field(t, 2, spec.data_type);
    append_len_field(t, 8, spec.name);
    const auto entry = [&t](const std::string& key, const std::string& value) {
        std::vector<std::byte> e;
        append_len_field(e, 1, key);
        append_len_field(e, 2, value);
        append_len_field(t, 13, e);
    };
    entry("location", spec.location);
    entry("offset", spec.offset);
    if (spec.with_length) {
        entry("length", spec.length);
    }
    append_varint_field(t, 14, 1);  // data_location = EXTERNAL
    return t;
}

/// 自包含初始化器：权重以 raw_data 内联 —— 扫描器必须跳过而不是读进来。
std::vector<std::byte> make_inline_tensor(const std::size_t raw_bytes) {
    std::vector<std::byte> t;
    append_varint_field(t, 2, 1);
    append_len_field(t, 8, std::string("inline"));
    append_len_field(t, 9, std::vector<std::byte>(raw_bytes, std::byte{0x5A}));
    return t;
}

/// GraphProto：node（field 1）在前、initializer（field 5）在后。
std::vector<std::byte> make_graph(const std::vector<std::vector<std::byte>>& initializers,
                                  const std::vector<std::vector<std::byte>>& nodes = {}) {
    std::vector<std::byte> graph;
    for (const auto& node : nodes) {
        append_len_field(graph, 1, node);
    }
    append_len_field(graph, 2, std::string("g"));
    for (const auto& initializer : initializers) {
        append_len_field(graph, 5, initializer);
    }
    return graph;
}

/// 带一个子图属性（AttributeProto.g）的 If 节点。
std::vector<std::byte> make_node_with_subgraph(const std::vector<std::byte>& subgraph) {
    std::vector<std::byte> attribute;
    append_len_field(attribute, 1, std::string("then_branch"));
    append_len_field(attribute, 6, subgraph);
    append_varint_field(attribute, 20, 5);  // type = GRAPH
    std::vector<std::byte> node;
    append_len_field(node, 4, std::string("If"));
    append_len_field(node, 5, attribute);
    return node;
}

std::vector<std::byte> make_model(const std::vector<std::byte>& graph) {
    std::vector<std::byte> model;
    append_varint_field(model, 1, 8);                 // ir_version
    append_len_field(model, 2, std::string("test"));  // producer_name
    append_len_field(model, 7, graph);
    return model;
}

/// 临时文件（析构即删）。
class TempFile {
public:
    explicit TempFile(const std::vector<std::byte>& bytes, const std::string& suffix = ".onnx") {
        static int counter = 0;
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = std::filesystem::temp_directory_path() /
                ("genie_ext_" + std::to_string(stamp) + "_" + std::to_string(counter++) + suffix);
        std::ofstream out(path_, std::ios::binary);
        out.write(reinterpret_cast<const char*>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()));
    }

    ~TempFile() {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }

    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;

    [[nodiscard]] std::string path() const { return path_.string(); }

private:
    std::filesystem::path path_;
};

/// 期望 fn 抛出指定错误码的 genie::Error。
template <typename Fn>
void expect_error(Fn&& fn, const ErrorCode code) {
    try {
        fn();
        ADD_FAILURE() << "expected genie::Error";
    } catch (const Error& e) {
        EXPECT_EQ(e.code(), code) << e.what();
    }
}

/// half → float 的参考实现：逐分支写法（库里是无分支版本，两者必须对
/// 全部 65536 个输入逐位一致）。
std::uint32_t reference_half_to_float_bits(const std::uint16_t half) {
    const std::uint32_t sign = static_cast<std::uint32_t>(half & 0x8000U) << 16U;
    const std::uint32_t exponent = (half >> 10U) & 0x1FU;
    const std::uint32_t mantissa = half & 0x3FFU;
    if (exponent == 0) {
        if (mantissa == 0) {
            return sign;
        }
        std::uint32_t m = mantissa;
        std::uint32_t e = 113U;
        while ((m & 0x400U) == 0) {
            m <<= 1U;
            --e;
        }
        return sign | (e << 23U) | ((m & 0x3FFU) << 13U);
    }
    if (exponent == 0x1FU) {
        return sign | 0x7F800000U | (mantissa << 13U);
    }
    return sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
}

}  // namespace

TEST(ExternalWeights, ScanSummarizesReferencesPerFile) {
    const TempFile file(make_model(make_graph({
        make_external_tensor({.name = "a", .location = "w.bin", .offset = "0", .length = "16"}),
        make_external_tensor({.name = "b", .location = "v.bin", .offset = "0", .length = "8"}),
        make_external_tensor({.name = "c", .location = "w.bin", .offset = "16", .length = "8"}),
    })));

    const auto files = scan_external_weights(file.path());

    ASSERT_EQ(files.size(), 2U);
    EXPECT_EQ(files[0].location, "w.bin");  // 首次出现顺序
    EXPECT_EQ(files[0].tensor_count, 2U);
    EXPECT_EQ(files[0].required_bytes, 24U);
    EXPECT_TRUE(files[0].fp16_blocker.empty()) << files[0].fp16_blocker;
    EXPECT_EQ(files[1].location, "v.bin");
    EXPECT_EQ(files[1].tensor_count, 1U);
    EXPECT_EQ(files[1].required_bytes, 8U);
}

TEST(ExternalWeights, ScanSkipsInlineRawData) {
    // raw_data 必须被跳过而不是读进来（200 KB 越过 seek 阈值，两条跳过路径都走到）。
    const TempFile self_contained(make_model(make_graph({make_inline_tensor(200U * 1024U)})));
    EXPECT_TRUE(scan_external_weights(self_contained.path()).empty());

    const TempFile mixed(make_model(make_graph(
        {make_inline_tensor(200U * 1024U), make_external_tensor(), make_inline_tensor(100)})));
    const auto files = scan_external_weights(mixed.path());
    ASSERT_EQ(files.size(), 1U);
    EXPECT_EQ(files[0].tensor_count, 1U);
    EXPECT_EQ(files[0].required_bytes, 16U);
}

TEST(ExternalWeights, ScanFlagsReferencesOutsideTheMainGraph) {
    // 子图里的外部张量：推理后端的内存替换覆盖不到 —— 必须标记为不可上转，
    // 否则后端会按 fp32 偏移去读磁盘上的 fp16 文件。
    const auto subgraph = make_graph({make_external_tensor({.name = "inner"})});
    const TempFile file(make_model(make_graph({make_external_tensor({.name = "outer"})},
                                              {make_node_with_subgraph(subgraph)})));

    const auto files = scan_external_weights(file.path());

    ASSERT_EQ(files.size(), 1U);
    EXPECT_EQ(files[0].tensor_count, 2U);
    EXPECT_NE(files[0].fp16_blocker.find("'inner'"), std::string::npos) << files[0].fp16_blocker;
    EXPECT_NE(files[0].fp16_blocker.find("subgraph"), std::string::npos) << files[0].fp16_blocker;
}

TEST(ExternalWeights, ScanFlagsTensorsThatCannotBeUpcast) {
    const auto blocker_of = [](const ExternalTensorSpec& spec) {
        const TempFile file(make_model(make_graph({make_external_tensor(spec)})));
        const auto files = scan_external_weights(file.path());
        return files.empty() ? std::string("<no files>") : files[0].fp16_blocker;
    };
    EXPECT_TRUE(blocker_of({}).empty());
    EXPECT_NE(blocker_of({.data_type = 7}).find("not FLOAT"), std::string::npos);
    EXPECT_NE(blocker_of({.offset = "2"}).find("aligned"), std::string::npos);
    EXPECT_NE(blocker_of({.with_length = false}).find("no external_data length"),
              std::string::npos);
}

TEST(ExternalWeights, ScanRejectsMalformedModels) {
    auto bytes = make_model(make_graph({make_external_tensor()}));
    bytes.resize(bytes.size() - 5);  // 截断：graph 字段的长度前缀越过文件尾
    const TempFile truncated(bytes);
    expect_error([&truncated] { (void)scan_external_weights(truncated.path()); },
                 ErrorCode::ModelLoadError);

    const TempFile bad_offset(make_model(make_graph({make_external_tensor({.offset = "12x"})})));
    expect_error([&bad_offset] { (void)scan_external_weights(bad_offset.path()); },
                 ErrorCode::ModelLoadError);

    expect_error([] { (void)scan_external_weights("/definitely/not/here/model.onnx"); },
                 ErrorCode::IoError);
}

TEST(ExternalWeights, ScanMatchesUpstreamGraphShells) {
    // 仓库内置的上游图壳：权重在（不随仓库发布的）外部 bin 里 —— 恰好是
    // 扫描器的真实输入形态。数字与 HF fp32 bin 的大小一致。
    if (!model_exists("v2/t2s_stage_decoder_fp32.onnx")) {
        GTEST_SKIP() << "test models not available at " << model_root().string();
    }
    const auto encoder = scan_external_weights(v2_model("t2s_encoder_fp32.onnx").string());
    ASSERT_EQ(encoder.size(), 1U);
    EXPECT_EQ(encoder[0].location, "t2s_encoder_fp32.bin");
    EXPECT_EQ(encoder[0].tensor_count, 7U);
    EXPECT_EQ(encoder[0].required_bytes, 11465732U);
    EXPECT_TRUE(encoder[0].fp16_blocker.empty()) << encoder[0].fp16_blocker;

    const auto stage = scan_external_weights(v2_model("t2s_stage_decoder_fp32.onnx").string());
    ASSERT_EQ(stage.size(), 1U);
    EXPECT_EQ(stage[0].tensor_count, 291U);
    EXPECT_EQ(stage[0].required_bytes, 306827268U);
    EXPECT_TRUE(stage[0].fp16_blocker.empty()) << stage[0].fp16_blocker;
    // HF 的 fp16 bin 恰为一半 → 判定为打包格式。
    EXPECT_EQ(classify_external_weights(stage[0], 153413634U), ExternalWeightLayout::Fp16Packed);
    EXPECT_EQ(classify_external_weights(stage[0], 306827268U), ExternalWeightLayout::Fp32);
}

TEST(ExternalWeights, ClassifiesByDeclaredRangeVersusFileSize) {
    ExternalWeightFile file;
    file.location = "w.bin";
    file.required_bytes = 16;
    EXPECT_EQ(classify_external_weights(file, 16), ExternalWeightLayout::Fp32);
    EXPECT_EQ(classify_external_weights(file, 64), ExternalWeightLayout::Fp32);
    EXPECT_EQ(classify_external_weights(file, 8), ExternalWeightLayout::Fp16Packed);
    // 两种解释都装不下：报错，不做歧义回退。
    expect_error([&file] { (void)classify_external_weights(file, 7); }, ErrorCode::ModelLoadError);
}

TEST(ExternalWeights, RefusesToUpcastWhenBlocked) {
    ExternalWeightFile file;
    file.location = "w.bin";
    file.required_bytes = 16;
    file.fp16_blocker = "tensor 'x' is not FLOAT (data_type 7)";
    // fp32 判定不受影响：原样交给后端。
    EXPECT_EQ(classify_external_weights(file, 16), ExternalWeightLayout::Fp32);
    expect_error([&file] { (void)classify_external_weights(file, 8); }, ErrorCode::ModelLoadError);

    file.fp16_blocker.clear();
    file.required_bytes = 18;
    expect_error([&file] { (void)classify_external_weights(file, 9); },  // 奇数字节
                 ErrorCode::ModelLoadError);
}

TEST(ExternalWeights, UpcastIsBitExactAcrossAllHalfValuesAndChunks) {
    // 全部 65536 个 half 值重复 16 遍再多一截：既是穷举（次正规 / ±0 /
    // ±Inf / NaN 载荷），也跨过 1 Mi 元素的读块边界。
    constexpr std::size_t kCount = (std::size_t{1} << 20U) + 12345U;
    std::vector<std::byte> bytes(kCount * 2U);
    for (std::size_t i = 0; i < kCount; ++i) {
        const auto half = static_cast<std::uint16_t>(i & 0xFFFFU);
        bytes[2U * i] = static_cast<std::byte>(half & 0xFFU);
        bytes[2U * i + 1U] = static_cast<std::byte>(half >> 8U);
    }
    const TempFile file(bytes, ".bin");

    const auto buffer = read_fp16_weights_as_fp32(file.path());

    ASSERT_EQ(buffer.count, kCount);
    ASSERT_EQ(buffer.size_bytes(), kCount * sizeof(float));
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < kCount; ++i) {
        const auto half = static_cast<std::uint16_t>(i & 0xFFFFU);
        const auto bits = std::bit_cast<std::uint32_t>(buffer.data[i]);
        if (bits != reference_half_to_float_bits(half) && mismatches++ < 5) {
            ADD_FAILURE() << std::hex << "half 0x" << half << " -> 0x" << bits << ", expected 0x"
                          << reference_half_to_float_bits(half);
        }
    }
    EXPECT_EQ(mismatches, 0U);

    // 几个有名有姓的值：防参考实现与被测实现一起错。
    EXPECT_EQ(buffer.data[0x3C00], 1.0F);
    EXPECT_EQ(buffer.data[0xC000], -2.0F);
    EXPECT_EQ(buffer.data[0x0001], std::ldexp(1.0F, -24));
    EXPECT_TRUE(std::isinf(buffer.data[0x7C00]) && buffer.data[0x7C00] > 0.0F);
    EXPECT_TRUE(std::signbit(buffer.data[0x8000]) && buffer.data[0x8000] == 0.0F);
    EXPECT_EQ(std::bit_cast<std::uint32_t>(buffer.data[0x7E01]), 0x7FC02000U);  // NaN 载荷保留
}

TEST(ExternalWeights, UpcastRejectsOddSizeAndMissingFiles) {
    const TempFile odd(std::vector<std::byte>(5, std::byte{0}), ".bin");
    expect_error([&odd] { (void)read_fp16_weights_as_fp32(odd.path()); },
                 ErrorCode::ModelLoadError);
    expect_error([] { (void)read_fp16_weights_as_fp32("/definitely/not/here/w.bin"); },
                 ErrorCode::IoError);

    const TempFile empty(std::vector<std::byte>{}, ".bin");
    EXPECT_EQ(read_fp16_weights_as_fp32(empty.path()).count, 0U);
}

TEST(ExternalWeights, SyntheticFp16AssetIsDetectedAsPacked) {
    // tools/make_synthetic_model.py --external-fp16 的产物：w=[1024] 全 1.0。
    const auto graph = synthetic_model("echo_fp16.onnx");
    const auto data = synthetic_model("echo_fp16_data.bin");
    if (!std::filesystem::exists(graph) || !std::filesystem::exists(data)) {
        GTEST_SKIP() << "echo_fp16 assets not available at " << graph.string();
    }

    const auto files = scan_external_weights(graph.string());
    ASSERT_EQ(files.size(), 1U);
    EXPECT_EQ(files[0].location, "echo_fp16_data.bin");
    EXPECT_EQ(files[0].required_bytes, 4096U);
    EXPECT_EQ(classify_external_weights(files[0], std::filesystem::file_size(data)),
              ExternalWeightLayout::Fp16Packed);

    const auto buffer = read_fp16_weights_as_fp32(data.string());
    ASSERT_EQ(buffer.count, 1024U);
    for (std::size_t i = 0; i < buffer.count; ++i) {
        ASSERT_EQ(buffer.data[i], 1.0F) << "element " << i;
    }
}
