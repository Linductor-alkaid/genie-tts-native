// tests/unit/test_onnx_graph.cpp
//
// ONNX protobuf 读取器的单元测试。
//
// 这个读取器存在的意义：上游发布的 ONNX 模型是"图壳"（权重在外部 .bin 里），
// 无法被 ONNX Runtime 加载，但图结构完整 —— I/O 契约必须能被**可执行地**锁定。

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
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

// ------------------------------------------------------------------ 权重内联转码器
//
// 手工拼装最小 wire-format：ModelProto{ graph{ initializer{ external fp16 } } }。
// 造数据比提交二进制资产更能表达"转码器对字节做什么"。

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

void append_varint_field(std::vector<std::byte>& out, const std::uint32_t field,
                         const std::uint64_t value) {
    append_tag(out, field, 0);
    append_varint(out, value);
}

/// HF fp16 格式的外部初始化器：dims=[2]，data_type=FLOAT（图声明 fp32），
/// offset/length 以 fp32 字节为单位，name="w"，location="w.bin"。
std::vector<std::byte> make_external_tensor(const std::string& offset = "0",
                                            const std::string& length = "16",
                                            const bool with_length = true) {
    std::vector<std::byte> t;
    append_varint_field(t, 1, 2);  // dims[0] = 2（repeated int64 非打包形态）
    append_varint_field(t, 2, 1);  // data_type = FLOAT
    append_len_field(t, 8, std::string("w"));
    const auto entry = [&t](const std::string& k, const std::string& v) {
        std::vector<std::byte> e;
        append_len_field(e, 1, k);
        append_len_field(e, 2, v);
        append_len_field(t, 13, e.data(), e.size());
    };
    entry("location", "w.bin");
    entry("offset", offset);
    if (with_length) {
        entry("length", length);
    }
    append_varint_field(t, 14, 1);  // data_location = EXTERNAL
    return t;
}

std::vector<std::byte> make_model(const std::vector<std::byte>& initializer) {
    std::vector<std::byte> graph;
    append_len_field(graph, 5, initializer.data(), initializer.size());
    std::vector<std::byte> model;
    append_varint_field(model, 1, 8);  // ir_version（顺便测透传）
    append_len_field(model, 7, graph.data(), graph.size());
    append_len_field(model, 2, std::string("test"));  // producer_name（透传样本）
    return model;
}

/// 从转码输出里抠出 initializer 的字节（model → graph → field 5）。
///
/// 返回值非 void，不能用 ASSERT_*（展开成 return;）—— 失败统一
/// ADD_FAILURE + 返回空向量。
std::vector<std::byte> extract_initializer(const std::vector<std::byte>& model) {
    const auto read_varint = [](const std::vector<std::byte>& buf, std::size_t& pos,
                                std::uint64_t& value) {
        value = 0;
        int shift = 0;
        while (pos < buf.size()) {
            const auto byte = static_cast<std::uint8_t>(buf[pos++]);
            value |= static_cast<std::uint64_t>(byte & 0x7FU) << shift;
            if ((byte & 0x80U) == 0) {
                return true;
            }
            shift += 7;
        }
        return false;
    };
    const auto read_len = [&](const std::vector<std::byte>& buf, std::size_t& pos,
                              std::uint64_t& length) {
        if (!read_varint(buf, pos, length)) {
            ADD_FAILURE() << "bad varint";
            return false;
        }
        if (length > buf.size() - pos) {
            ADD_FAILURE() << "length out of range: length=" << length << " pos=" << pos
                          << " buf.size()=" << buf.size();
            return false;
        }
        return true;
    };

    // 顶层：找 field 7（graph），其余字段跳过。
    std::vector<std::byte> graph;
    std::size_t i = 0;
    while (i < model.size()) {
        std::uint64_t key = 0;
        if (!read_varint(model, i, key)) {
            ADD_FAILURE() << "bad key";
            return {};
        }
        const auto field = static_cast<std::uint32_t>(key >> 3U);
        const auto wire = static_cast<std::uint32_t>(key & 0x7U);
        if (wire != 2) {
            std::uint64_t ignored = 0;
            if (wire == 0) {
                if (!read_varint(model, i, ignored)) {
                    return {};
                }
            } else if (wire == 1) {
                i += 8;
            } else if (wire == 5) {
                i += 4;
            } else {
                ADD_FAILURE() << "unexpected wire type";
                return {};
            }
            continue;
        }
        std::uint64_t length = 0;
        if (!read_len(model, i, length)) {
            return {};
        }
        if (field == 7) {
            graph.assign(model.begin() + static_cast<std::ptrdiff_t>(i),
                         model.begin() + static_cast<std::ptrdiff_t>(i + length));
        }
        i += static_cast<std::size_t>(length);
    }

    // graph 里找 field 5（initializer）。
    i = 0;
    while (i < graph.size()) {
        std::uint64_t key = 0;
        if (!read_varint(graph, i, key)) {
            ADD_FAILURE() << "bad key in graph";
            return {};
        }
        const auto field = static_cast<std::uint32_t>(key >> 3U);
        std::uint64_t length = 0;
        if (!read_len(graph, i, length)) {
            return {};
        }
        if (field == 5) {
            return std::vector<std::byte>(graph.begin() + static_cast<std::ptrdiff_t>(i),
                                          graph.begin() + static_cast<std::ptrdiff_t>(i + length));
        }
        i += static_cast<std::size_t>(length);
    }
    return {};
}

/// fp16 测试数据：{+0, 1.0, -2.0, 半精度最小次正规（2^-24）}。
std::vector<std::byte> fp16_samples() {
    const std::array<std::uint16_t, 4> halves = {0x0000, 0x3C00, 0xC000, 0x0001};
    std::vector<std::byte> bytes;
    for (const auto h : halves) {
        bytes.push_back(static_cast<std::byte>(h & 0xFFU));
        bytes.push_back(static_cast<std::byte>(h >> 8U));
    }
    return bytes;
}

}  // namespace

TEST(InlineWeights, UpcastsFp16AndDropsExternalData) {
    // fp16 打包判定：图声明 fp32 区间 [0,16)，文件只有 8 字节（一半）→
    // 按 HF 格式解释：8 字节 = 4 个 fp16 → 上转 16 字节 fp32。
    auto model = make_model(make_external_tensor());
    const std::vector<std::byte> bin = fp16_samples();
    const ExternalWeightSource source = [&](const std::string&) { return &bin; };

    const auto out = inline_external_fp16_weights(model, source);
    ASSERT_FALSE(out.empty());

    const auto tensor = extract_initializer(out);
    ASSERT_FALSE(tensor.empty());
    // 抽字段：external_data / data_location 必须消失，raw_data 必须出现。
    bool has_raw = false;
    bool has_external = false;
    bool has_location_flag = false;
    std::vector<float> values;
    std::size_t i = 0;
    const auto read_varint = [&](std::uint64_t& value) {
        value = 0;
        int shift = 0;
        while (i < tensor.size()) {
            const auto byte = static_cast<std::uint8_t>(tensor[i++]);
            value |= static_cast<std::uint64_t>(byte & 0x7FU) << shift;
            if ((byte & 0x80U) == 0) {
                return true;
            }
            shift += 7;
        }
        return false;
    };
    while (i < tensor.size()) {
        std::uint64_t key = 0;
        ASSERT_TRUE(read_varint(key));
        const auto field = static_cast<std::uint32_t>(key >> 3U);
        const auto wire = static_cast<std::uint32_t>(key & 0x7U);
        if (field == 13) {
            has_external = true;
        }
        if (field == 14) {
            has_location_flag = true;
        }
        if (field == 9 && wire == 2) {
            std::uint64_t length = 0;
            ASSERT_TRUE(read_varint(length));
            ASSERT_EQ(length, 16U);
            for (std::size_t k = 0; k < 4; ++k) {
                float value = 0.0F;
                std::memcpy(&value, tensor.data() + i + k * 4U, sizeof(value));
                values.push_back(value);
            }
            has_raw = true;
            i += static_cast<std::size_t>(length);
            continue;
        }
        if (wire == 0) {
            std::uint64_t ignored = 0;
            ASSERT_TRUE(read_varint(ignored));
        } else if (wire == 2) {
            std::uint64_t length = 0;
            ASSERT_TRUE(read_varint(length));
            i += static_cast<std::size_t>(length);
        } else {
            FAIL() << "unexpected wire type";
        }
    }
    EXPECT_FALSE(has_external);
    EXPECT_FALSE(has_location_flag);
    ASSERT_TRUE(has_raw);
    ASSERT_EQ(values.size(), 4U);
    EXPECT_EQ(values[0], 0.0F);
    EXPECT_EQ(values[1], 1.0F);
    EXPECT_EQ(values[2], -2.0F);
    EXPECT_FLOAT_EQ(values[3], std::pow(2.0F, -24.0F));  // 0x0001 = 半精度最小次正规
}

TEST(InlineWeights, PassesThroughModelWithoutExternalData) {
    // 自包含初始化器（带 raw_data，无 external_data）→ 整段字节原样返回。
    std::vector<std::byte> tensor;
    append_varint_field(tensor, 1, 2);  // dims[0] = 2
    append_varint_field(tensor, 2, 1);  // data_type = FLOAT
    append_len_field(tensor, 8, std::string("w"));
    const float raw[2] = {1.5F, -3.25F};
    append_len_field(tensor, 9, raw, sizeof(raw));
    const auto model = make_model(tensor);
    const ExternalWeightSource source = [](const std::string&) { return nullptr; };
    const auto out = inline_external_fp16_weights(model, source);
    EXPECT_EQ(out, model);
}

TEST(InlineWeights, PreservesUnrelatedModelFields) {
    auto model = make_model(make_external_tensor());
    append_len_field(model, 6, std::string("doc"));
    const std::vector<std::byte> bin = fp16_samples();
    const ExternalWeightSource source = [&](const std::string&) { return &bin; };
    const auto out = inline_external_fp16_weights(model, source);
    ASSERT_FALSE(out.empty());
    // doc_string（field 6，LEN）的 wire 字节 = tag(0x32) len(0x03) 'd' 'o' 'c'，
    // 应原样出现在输出尾部。
    const std::vector<std::byte> pattern = {
        static_cast<std::byte>(0x32), static_cast<std::byte>(0x03), static_cast<std::byte>('d'),
        static_cast<std::byte>('o'), static_cast<std::byte>('c')};
    ASSERT_GE(out.size(), pattern.size());
    const auto it = std::search(out.begin(), out.end(), pattern.begin(), pattern.end());
    EXPECT_NE(it, out.end());
}

TEST(InlineWeights, VerbatimWhenFp32FileFits) {
    // fp32 文件（16 字节 = 图声明区间）→ 字节原样内联，不做数值转换。
    auto model = make_model(make_external_tensor());
    const float raw[4] = {1.5F, -3.25F, 7.0F, -1.0F};
    const std::vector<std::byte> bin(reinterpret_cast<const std::byte*>(raw),
                                     reinterpret_cast<const std::byte*>(raw) + sizeof(raw));
    const ExternalWeightSource source = [&](const std::string&) { return &bin; };
    const auto out = inline_external_fp16_weights(model, source);
    const auto out_tensor = extract_initializer(out);
    ASSERT_FALSE(out_tensor.empty());
    const auto it = std::search(out_tensor.begin(), out_tensor.end(), bin.begin(), bin.end());
    EXPECT_NE(it, out_tensor.end());
}

TEST(InlineWeights, ThrowsWhenProviderLacksFile) {
    auto model = make_model(make_external_tensor());
    const ExternalWeightSource source = [](const std::string&) { return nullptr; };
    try {
        (void)inline_external_fp16_weights(model, source);
        FAIL() << "expected throw";
    } catch (const Error& error) {
        EXPECT_EQ(error.code(), ErrorCode::ModelLoadError);
    }
}

TEST(InlineWeights, ThrowsOnMissingLength) {
    auto model = make_model(make_external_tensor("0", "8", /*with_length=*/false));
    const std::vector<std::byte> bin = fp16_samples();
    const ExternalWeightSource source = [&](const std::string&) { return &bin; };
    EXPECT_THROW((void)inline_external_fp16_weights(model, source), Error);
}

TEST(InlineWeights, ThrowsWhenNeitherInterpretationFits) {
    // 4 字节文件：fp32 区间 [0,16) 装不下；一半区间 [0,8) 也装不下 → 报错。
    auto model = make_model(make_external_tensor());
    const std::vector<std::byte> bin(4, std::byte{0});
    const ExternalWeightSource source = [&](const std::string&) { return &bin; };
    try {
        (void)inline_external_fp16_weights(model, source);
        FAIL() << "expected throw";
    } catch (const Error& error) {
        EXPECT_EQ(error.code(), ErrorCode::ModelLoadError);
    }
}

TEST(InlineWeights, ThrowsOnUnalignedFp16PackedTensor) {
    // 1 字节文件：fp32 装不下；fp16 打包恰好装下但 length=2 非 4 对齐 → 报错。
    auto model = make_model(make_external_tensor("0", "2"));
    const std::vector<std::byte> bin(1, std::byte{0});
    const ExternalWeightSource source = [&](const std::string&) { return &bin; };
    try {
        (void)inline_external_fp16_weights(model, source);
        FAIL() << "expected throw";
    } catch (const Error& error) {
        EXPECT_EQ(error.code(), ErrorCode::ModelLoadError);
    }
}

TEST(InlineWeights, InlinesNonFloatExternalVerbatim) {
    // int64 外部张量（data_type=7）：字节原样内联，不做数值转换。
    std::vector<std::byte> tensor;
    append_varint_field(tensor, 1, 1);  // dims[0] = 1
    append_varint_field(tensor, 2, 7);  // data_type = INT64
    append_len_field(tensor, 8, std::string("ids"));
    const auto entry = [&tensor](const std::string& k, const std::string& v) {
        std::vector<std::byte> e;
        append_len_field(e, 1, k);
        append_len_field(e, 2, v);
        append_len_field(tensor, 13, e.data(), e.size());
    };
    entry("location", "ids.bin");
    entry("offset", "0");
    entry("length", "8");
    append_varint_field(tensor, 14, 1);
    const auto model = make_model(tensor);
    const std::vector<std::byte> bin = {std::byte{0x2A}, std::byte{0}, std::byte{0}, std::byte{0},
                                        std::byte{0},    std::byte{0}, std::byte{0}, std::byte{0}};
    const ExternalWeightSource source = [&](const std::string&) { return &bin; };
    const auto out = inline_external_fp16_weights(model, source);
    const auto out_tensor = extract_initializer(out);
    // raw_data 应与源字节完全一致。
    const auto it = std::search(out_tensor.begin(), out_tensor.end(), bin.begin(), bin.end());
    EXPECT_NE(it, out_tensor.end());
}
