// src/runtime/onnx_graph.cpp — 最小 ONNX protobuf 读取器
//
// protobuf wire format 回顾：
//   每个字段 = varint key + value，key = (field_number << 3) | wire_type
//   wire_type: 0 = varint, 1 = 64-bit, 2 = length-delimited, 5 = 32-bit
//
// 本文件用到的 ONNX 字段编号（来自 onnx.proto）：
//   ModelProto.graph            = 7   (LEN)
//   ModelProto.ir_version       = 1   (varint)
//   GraphProto.input            = 11  (LEN)
//   GraphProto.output           = 12  (LEN)
//   GraphProto.initializer      = 5   (LEN)
//   TensorProto.external_data   = 13  (LEN, repeated StringStringEntryProto)
//   StringStringEntryProto.key   = 1  (LEN string)
//   StringStringEntryProto.value = 2  (LEN string)
//   ValueInfoProto.name         = 1   (LEN string)
//   ValueInfoProto.type         = 2   (LEN)
//   TypeProto.tensor_type       = 1   (LEN)
//   TypeProto.Tensor.elem_type  = 1   (varint int32)
//   TypeProto.Tensor.shape      = 2   (LEN)
//   TensorShapeProto.dim        = 1   (LEN, repeated)
//   Dimension.dim_value         = 1   (varint int64)
//   Dimension.dim_param         = 2   (LEN string)

#include "genie/onnx_graph.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string_view>

#include "genie/errors.hpp"

namespace genie {
namespace {

// ------------------------------------------------------------------ protobuf 原语

class Reader {
public:
    Reader(const std::byte* data, std::size_t size) : data_(data), size_(size) {}

    [[nodiscard]] bool done() const noexcept { return offset_ >= size_; }

    /// 读一个 varint。
    [[nodiscard]] bool read_varint(std::uint64_t& out) {
        std::uint64_t result = 0;
        int shift = 0;
        while (offset_ < size_) {
            const auto byte = static_cast<std::uint8_t>(data_[offset_++]);
            result |= static_cast<std::uint64_t>(byte & 0x7FU) << shift;
            if ((byte & 0x80U) == 0) {
                out = result;
                return true;
            }
            shift += 7;
            if (shift > 63) {
                return false;
            }
        }
        return false;
    }

    /// 读一个 length-delimited 字段，返回其内容视图。
    [[nodiscard]] bool read_bytes(const std::byte*& out, std::size_t& out_size) {
        std::uint64_t length = 0;
        if (!read_varint(length)) {
            return false;
        }
        if (length > size_ - offset_) {
            return false;
        }
        out = data_ + offset_;
        out_size = static_cast<std::size_t>(length);
        offset_ += out_size;
        return true;
    }

    /// 跳过当前字段的值。
    [[nodiscard]] bool skip(std::uint32_t wire_type) {
        switch (wire_type) {
            case 0: {
                std::uint64_t ignored = 0;
                return read_varint(ignored);
            }
            case 1:
                return advance(8);
            case 2: {
                const std::byte* ignored = nullptr;
                std::size_t ignored_size = 0;
                return read_bytes(ignored, ignored_size);
            }
            case 5:
                return advance(4);
            default:
                return false;
        }
    }

    [[nodiscard]] bool advance(std::size_t count) {
        if (count > size_ - offset_) {
            return false;
        }
        offset_ += count;
        return true;
    }

    /// 当前读取位置（字节偏移）—— 转码器靠它截取"原样透传"的字段区间。
    [[nodiscard]] std::size_t pos() const noexcept { return offset_; }

private:
    const std::byte* data_;
    std::size_t size_;
    std::size_t offset_ = 0;
};

/// 把 length-delimited 内容当作字符串读取。
std::string_view as_string(const std::byte* data, std::size_t size) {
    return std::string_view(reinterpret_cast<const char*>(data), size);
}

// ------------------------------------------------------------------ ONNX 结构

// 这几个解析函数是互相递归的（ValueInfo → Type → TensorType → Shape → Dimension），
// 因此先做前向声明。
bool parse_type(const std::byte* data, std::size_t size, OnnxTensorInfo& out);
bool parse_tensor_type(const std::byte* data, std::size_t size, OnnxTensorInfo& out);
bool parse_shape(const std::byte* data, std::size_t size, OnnxTensorInfo& out);
bool parse_dimension(const std::byte* data, std::size_t size, OnnxTensorInfo& out);

/// 解析 ValueInfoProto。
bool parse_value_info(const std::byte* data, std::size_t size, OnnxTensorInfo& out) {
    Reader reader(data, size);
    while (!reader.done()) {
        std::uint64_t key = 0;
        if (!reader.read_varint(key)) {
            return false;
        }
        const auto field = static_cast<std::uint32_t>(key >> 3);
        const auto wire = static_cast<std::uint32_t>(key & 0x7U);
        if (field == 1 && wire == 2) {  // name
            const std::byte* bytes = nullptr;
            std::size_t bytes_size = 0;
            if (!reader.read_bytes(bytes, bytes_size)) {
                return false;
            }
            out.name = std::string(as_string(bytes, bytes_size));
        } else if (field == 2 && wire == 2) {  // type
            const std::byte* bytes = nullptr;
            std::size_t bytes_size = 0;
            if (!reader.read_bytes(bytes, bytes_size)) {
                return false;
            }
            if (!parse_type(bytes, bytes_size, out)) {
                return false;
            }
        } else if (!reader.skip(wire)) {
            return false;
        }
    }
    return true;
}

/// 解析 TypeProto（只关心 tensor_type）。
bool parse_type(const std::byte* data, std::size_t size, OnnxTensorInfo& out) {
    Reader reader(data, size);
    while (!reader.done()) {
        std::uint64_t key = 0;
        if (!reader.read_varint(key)) {
            return false;
        }
        const auto field = static_cast<std::uint32_t>(key >> 3);
        const auto wire = static_cast<std::uint32_t>(key & 0x7U);
        if (field == 1 && wire == 2) {  // tensor_type
            const std::byte* bytes = nullptr;
            std::size_t bytes_size = 0;
            if (!reader.read_bytes(bytes, bytes_size)) {
                return false;
            }
            return parse_tensor_type(bytes, bytes_size, out);
        }
        if (!reader.skip(wire)) {
            return false;
        }
    }
    return true;
}

/// 解析 TypeProto.Tensor。
bool parse_tensor_type(const std::byte* data, std::size_t size, OnnxTensorInfo& out) {
    Reader reader(data, size);
    while (!reader.done()) {
        std::uint64_t key = 0;
        if (!reader.read_varint(key)) {
            return false;
        }
        const auto field = static_cast<std::uint32_t>(key >> 3);
        const auto wire = static_cast<std::uint32_t>(key & 0x7U);
        if (field == 1 && wire == 0) {  // elem_type
            std::uint64_t value = 0;
            if (!reader.read_varint(value)) {
                return false;
            }
            out.type = static_cast<OnnxElementType>(static_cast<int>(value));
        } else if (field == 2 && wire == 2) {  // shape
            const std::byte* bytes = nullptr;
            std::size_t bytes_size = 0;
            if (!reader.read_bytes(bytes, bytes_size)) {
                return false;
            }
            if (!parse_shape(bytes, bytes_size, out)) {
                return false;
            }
        } else if (!reader.skip(wire)) {
            return false;
        }
    }
    return true;
}

/// 解析 TensorShapeProto。
bool parse_shape(const std::byte* data, std::size_t size, OnnxTensorInfo& out) {
    Reader reader(data, size);
    while (!reader.done()) {
        std::uint64_t key = 0;
        if (!reader.read_varint(key)) {
            return false;
        }
        const auto field = static_cast<std::uint32_t>(key >> 3);
        const auto wire = static_cast<std::uint32_t>(key & 0x7U);
        if (field == 1 && wire == 2) {  // dim
            const std::byte* bytes = nullptr;
            std::size_t bytes_size = 0;
            if (!reader.read_bytes(bytes, bytes_size)) {
                return false;
            }
            if (!parse_dimension(bytes, bytes_size, out)) {
                return false;
            }
        } else if (!reader.skip(wire)) {
            return false;
        }
    }
    return true;
}

/// 解析 Dimension。
bool parse_dimension(const std::byte* data, std::size_t size, OnnxTensorInfo& out) {
    Reader reader(data, size);
    std::int64_t dim_value = -1;
    bool symbolic = false;

    while (!reader.done()) {
        std::uint64_t key = 0;
        if (!reader.read_varint(key)) {
            return false;
        }
        const auto field = static_cast<std::uint32_t>(key >> 3);
        const auto wire = static_cast<std::uint32_t>(key & 0x7U);
        if (field == 1 && wire == 0) {  // dim_value
            std::uint64_t value = 0;
            if (!reader.read_varint(value)) {
                return false;
            }
            dim_value = static_cast<std::int64_t>(value);
        } else if (field == 2 && wire == 2) {  // dim_param
            const std::byte* bytes = nullptr;
            std::size_t bytes_size = 0;
            if (!reader.read_bytes(bytes, bytes_size)) {
                return false;
            }
            symbolic = !as_string(bytes, bytes_size).empty();
        } else if (!reader.skip(wire)) {
            return false;
        }
    }

    if (symbolic) {
        out.shape.push_back(-1);
        out.symbolic_dims.push_back(true);
    } else {
        out.shape.push_back(dim_value);
        out.symbolic_dims.push_back(false);
    }
    return true;
}

/// 解析 TensorProto 的 external_data 条目，收集外部文件名。
void parse_external_data(const std::byte* data, std::size_t size, std::vector<std::string>& files) {
    Reader reader(data, size);
    std::string location;
    while (!reader.done()) {
        std::uint64_t key = 0;
        if (!reader.read_varint(key)) {
            return;
        }
        const auto field = static_cast<std::uint32_t>(key >> 3);
        const auto wire = static_cast<std::uint32_t>(key & 0x7U);
        if (field == 1 && wire == 2) {  // key
            const std::byte* bytes = nullptr;
            std::size_t bytes_size = 0;
            if (!reader.read_bytes(bytes, bytes_size)) {
                return;
            }
            if (as_string(bytes, bytes_size) == "location") {
                // 下一个 value 就是文件名
                std::uint64_t next_key = 0;
                if (!reader.read_varint(next_key)) {
                    return;
                }
                const auto next_wire = static_cast<std::uint32_t>(next_key & 0x7U);
                const auto next_field = static_cast<std::uint32_t>(next_key >> 3);
                if (next_field == 2 && next_wire == 2) {
                    const std::byte* value_bytes = nullptr;
                    std::size_t value_size = 0;
                    if (reader.read_bytes(value_bytes, value_size)) {
                        location = std::string(as_string(value_bytes, value_size));
                    }
                }
            }
        } else if (!reader.skip(wire)) {
            return;
        }
    }
    if (!location.empty()) {
        // 同一个 .bin 会被多个 initializer 引用，去重。
        if (std::find(files.begin(), files.end(), location) == files.end()) {
            files.push_back(location);
        }
    }
}

/// 解析 GraphProto。
bool parse_graph(const std::byte* data, std::size_t size, OnnxGraphSpec& out) {
    Reader reader(data, size);
    while (!reader.done()) {
        std::uint64_t key = 0;
        if (!reader.read_varint(key)) {
            return false;
        }
        const auto field = static_cast<std::uint32_t>(key >> 3);
        const auto wire = static_cast<std::uint32_t>(key & 0x7U);
        if ((field == 11 || field == 12) && wire == 2) {  // input / output
            const std::byte* bytes = nullptr;
            std::size_t bytes_size = 0;
            if (!reader.read_bytes(bytes, bytes_size)) {
                return false;
            }
            OnnxTensorInfo info;
            if (!parse_value_info(bytes, bytes_size, info)) {
                return false;
            }
            if (field == 11) {
                out.inputs.push_back(std::move(info));
            } else {
                out.outputs.push_back(std::move(info));
            }
        } else if (field == 5 && wire == 2) {  // initializer
            const std::byte* bytes = nullptr;
            std::size_t bytes_size = 0;
            if (!reader.read_bytes(bytes, bytes_size)) {
                return false;
            }
            // TensorProto.data_location = 2 (EXTERNAL)，字段号 14
            Reader init_reader(bytes, bytes_size);
            while (!init_reader.done()) {
                std::uint64_t init_key = 0;
                if (!init_reader.read_varint(init_key)) {
                    break;
                }
                const auto init_field = static_cast<std::uint32_t>(init_key >> 3);
                const auto init_wire = static_cast<std::uint32_t>(init_key & 0x7U);
                if (init_field == 14 && init_wire == 0) {  // data_location
                    std::uint64_t value = 0;
                    if (!init_reader.read_varint(value)) {
                        break;
                    }
                    // onnx.proto: enum DataLocation { DEFAULT = 0; EXTERNAL = 1; }
                    if (value == 1) {
                        out.uses_external_data = true;
                    }
                } else if (init_field == 13 && init_wire == 2) {  // external_data
                    const std::byte* entry = nullptr;
                    std::size_t entry_size = 0;
                    if (!init_reader.read_bytes(entry, entry_size)) {
                        break;
                    }
                    parse_external_data(entry, entry_size, out.external_data_files);
                } else if (!init_reader.skip(init_wire)) {
                    break;
                }
            }
        } else if (!reader.skip(wire)) {
            return false;
        }
    }
    return true;
}

}  // namespace

const char* to_string(OnnxElementType type) noexcept {
    switch (type) {
        case OnnxElementType::Float:
            return "float32";
        case OnnxElementType::Uint8:
            return "uint8";
        case OnnxElementType::Int8:
            return "int8";
        case OnnxElementType::Uint16:
            return "uint16";
        case OnnxElementType::Int16:
            return "int16";
        case OnnxElementType::Int32:
            return "int32";
        case OnnxElementType::Int64:
            return "int64";
        case OnnxElementType::String:
            return "string";
        case OnnxElementType::Bool:
            return "bool";
        case OnnxElementType::Float16:
            return "float16";
        case OnnxElementType::Double:
            return "float64";
        case OnnxElementType::Uint32:
            return "uint32";
        case OnnxElementType::Uint64:
            return "uint64";
        case OnnxElementType::Undefined:
            break;
    }
    return "undefined";
}

std::string OnnxTensorInfo::shape_string() const {
    std::ostringstream out;
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            out << 'x';
        }
        out << shape[i];
    }
    return out.str();
}

OnnxGraphSpec read_onnx_graph_from_memory(const std::vector<std::byte>& data) {
    OnnxGraphSpec spec;
    bool found_graph = false;

    Reader reader(data.data(), data.size());
    while (!reader.done()) {
        std::uint64_t key = 0;
        if (!reader.read_varint(key)) {
            break;
        }
        const auto field = static_cast<std::uint32_t>(key >> 3);
        const auto wire = static_cast<std::uint32_t>(key & 0x7U);
        if (field == 7 && wire == 2) {  // ModelProto.graph
            const std::byte* bytes = nullptr;
            std::size_t bytes_size = 0;
            if (!reader.read_bytes(bytes, bytes_size)) {
                break;
            }
            if (!parse_graph(bytes, bytes_size, spec)) {
                throw_error(ErrorCode::IoError, "malformed ONNX graph");
            }
            found_graph = true;
        } else if (field == 1 && wire == 0) {  // ModelProto.ir_version
            std::uint64_t value = 0;
            if (!reader.read_varint(value)) {
                break;
            }
            spec.ir_version = static_cast<std::int64_t>(value);
        } else if (!reader.skip(wire)) {
            break;
        }
    }

    if (!found_graph) {
        throw_error(ErrorCode::IoError, "not a valid ONNX model (no graph found)");
    }
    return spec;
}

OnnxGraphSpec read_onnx_graph(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw_error(ErrorCode::IoError, "cannot open ONNX model: " + path);
    }
    file.seekg(0, std::ios::end);
    const auto size = static_cast<std::size_t>(file.tellg());
    file.seekg(0, std::ios::beg);
    std::vector<std::byte> data(size);
    if (size > 0) {
        file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size));
        if (!file) {
            throw_error(ErrorCode::IoError, "failed to read ONNX model: " + path);
        }
    }
    return read_onnx_graph_from_memory(data);
}

// ------------------------------------------------------------------ 权重内联转码器
//
// 本节是"包里存 fp16 权重、加载期上转 FP32"（借鉴上游 Genie-TTS 的分发
// 格式）的库侧地基。策略是 wire-format 级转码：逐字段遍历，只对
// external-data 初始化器做改写（external_data/data_location → raw_data），
// 其余字段（node 属性等）按原始字节透传 —— 不依赖 schema 的其余部分，
// 序列化结果 ORT 必然可读（任意合法 wire format 均可）。
//
// **HF fp16 格式**：图把外部张量声明为 FLOAT（data_type=1），offset/length
// 以 **fp32 字节**为单位（审计实测：span 总长恰等于 fp32 bin 的大小）；
// 对应的 *_fp16.bin 是同一批张量转成 fp16 后**连续打包**，文件大小恰为
// 一半，tensor 内容在文件里的位置 = fp32 offset / 2。转码器对每个外部
// 文件自动判定精度：文件装得下 fp32 区间 → 逐字节原样内联；否则若
// "一半区间"装得下且 4 字节对齐 → 按 fp16 打包上转内联；两者都不行
// 则报错 —— 不存在歧义回退。
//
// 本节用到的额外字段编号（onnx.proto）：
//   TensorProto.dims            = 1  (varint, repeated / packed)
//   TensorProto.data_type       = 2  (varint)
//   TensorProto.raw_data        = 9  (LEN)
//   TensorProto.name            = 8  (LEN string)
//   GraphProto.sparse_initializer = 15 (LEN —— 出现即拒绝，本项目无此需求)

namespace {

/// protobuf 写原语。
class Writer {
public:
    void varint(std::uint64_t value) {
        while (value >= 0x80U) {
            out_.push_back(static_cast<std::byte>(static_cast<std::uint8_t>(value) | 0x80U));
            value >>= 7U;
        }
        out_.push_back(static_cast<std::byte>(static_cast<std::uint8_t>(value)));
    }

    void tag(const std::uint32_t field, const std::uint32_t wire) {
        varint((static_cast<std::uint64_t>(field) << 3U) | wire);
    }

    void len_field(const std::uint32_t field, const std::byte* data, const std::size_t size) {
        tag(field, 2);
        varint(size);
        out_.insert(out_.end(), data, data + size);
    }

    void raw(const std::byte* data, const std::size_t size) {
        out_.insert(out_.end(), data, data + size);
    }

    [[nodiscard]] std::vector<std::byte> take() { return std::move(out_); }

private:
    std::vector<std::byte> out_;
};

/// IEEE half → IEEE single。denormal / Inf / NaN 全部分支处理；
/// 权重文件是 little-endian，宿主（x86 / ARM64）同为 little-endian，
/// 按 uint16 直接 memcpy 读取。
float half_to_float(const std::uint16_t half) {
    const std::uint32_t sign = static_cast<std::uint32_t>(half & 0x8000U) << 16U;
    const std::uint32_t exponent = (half >> 10U) & 0x1FU;
    const std::uint32_t mantissa = half & 0x3FFU;
    std::uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;  // ±0
        } else {
            // 次正规数：规格化进单精度正规范围。半精度次正规 = man × 2^-24，
            // man 的最高位（bit k）左移 (10-k) 次到隐藏位 —— 指数相应从
            // 103 + k 起算（2^(103-127) = 2^-24）。
            std::uint32_t m = mantissa;
            std::uint32_t e = 113U;
            while ((m & 0x400U) == 0) {
                m <<= 1U;
                --e;
            }
            m &= 0x3FFU;  // 隐藏位已进位，从尾数里去掉
            bits = sign | (e << 23U) | (m << 13U);
        }
    } else if (exponent == 0x1FU) {
        bits = sign | 0x7F800000U | (mantissa << 13U);  // Inf / NaN（载荷保留）
    } else {
        bits = sign | ((exponent + 112U) << 23U) | (mantissa << 13U);  // 112 = 127 - 15
    }
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

std::vector<std::byte> upcast_fp16(const std::byte* data, const std::size_t size) {
    std::vector<std::byte> out(size * 2U);
    auto* out_floats = reinterpret_cast<float*>(out.data());
    for (std::size_t i = 0; i < size / 2U; ++i) {
        std::uint16_t half = 0;
        std::memcpy(&half, data + i * 2U, sizeof(half));
        out_floats[i] = half_to_float(half);
    }
    return out;
}

/// external_data 条目（StringStringEntryProto）集合。
using ExternalEntries = std::vector<std::pair<std::string, std::string>>;

[[nodiscard]] const std::string* find_entry(const ExternalEntries& entries,
                                            const std::string& key) {
    for (const auto& [k, v] : entries) {
        if (k == key) {
            return &v;
        }
    }
    return nullptr;
}

/// 一个外部权重文件的解析结果：字节 + 精度判定。
struct ResolvedExternalFile {
    const std::vector<std::byte>* bytes = nullptr;
    bool fp16_packed = false;  ///< true = HF 的 fp16 连续打包格式
};

/// 转码单个 TensorProto：external 初始化器内联为 raw_data，其余原样重放。
std::vector<std::byte> transcode_initializer(
    const std::byte* data, const std::size_t size,
    const std::map<std::string, ResolvedExternalFile>& resolved) {
    Reader reader(data, size);
    std::vector<std::pair<std::size_t, std::size_t>> kept_spans;
    bool is_external = false;
    bool has_raw_data = false;
    std::uint64_t data_type = 1;  // FLOAT 是 onnx 默认值
    std::string name;
    std::string location;
    std::string offset_text;
    bool has_length = false;
    std::string length_text;

    while (!reader.done()) {
        const std::size_t start = reader.pos();
        std::uint64_t key = 0;
        if (!reader.read_varint(key)) {
            throw_error(ErrorCode::ModelLoadError, "malformed initializer (bad wire format)");
        }
        const auto field = static_cast<std::uint32_t>(key >> 3U);
        const auto wire = static_cast<std::uint32_t>(key & 0x7U);
        if (field == 13 && wire == 2) {  // external_data 条目
            const std::byte* entry_bytes = nullptr;
            std::size_t entry_size = 0;
            if (!reader.read_bytes(entry_bytes, entry_size)) {
                throw_error(ErrorCode::ModelLoadError, "malformed external_data entry");
            }
            Reader entry_reader(entry_bytes, entry_size);
            std::string entry_key;
            std::string entry_value;
            while (!entry_reader.done()) {
                std::uint64_t entry_tag = 0;
                if (!entry_reader.read_varint(entry_tag)) {
                    break;
                }
                const auto e_field = static_cast<std::uint32_t>(entry_tag >> 3U);
                const auto e_wire = static_cast<std::uint32_t>(entry_tag & 0x7U);
                if (e_wire != 2 || (e_field != 1 && e_field != 2)) {
                    if (!entry_reader.skip(e_wire)) {
                        break;
                    }
                    continue;
                }
                const std::byte* value_bytes = nullptr;
                std::size_t value_size = 0;
                if (!entry_reader.read_bytes(value_bytes, value_size)) {
                    break;
                }
                (e_field == 1 ? entry_key : entry_value) =
                    std::string(as_string(value_bytes, value_size));
            }
            if (entry_key == "location") {
                location = entry_value;
            } else if (entry_key == "offset") {
                offset_text = entry_value;
            } else if (entry_key == "length") {
                length_text = entry_value;
                has_length = true;
            }
            // field 13 **不保留** —— ONNX 规范禁止 raw_data 与
            // data_location=EXTERNAL 并存，ORT 见到 EXTERNAL 会忽略
            // raw_data 走外部文件路径，内联就失效了。
        } else if (field == 14 && wire == 0) {  // data_location；EXTERNAL = 1
            std::uint64_t flag = 0;
            if (!reader.read_varint(flag)) {
                throw_error(ErrorCode::ModelLoadError, "malformed data_location");
            }
            if (flag == 1) {
                is_external = true;
            }
            // field 14 同上：内联后必须清掉。
        } else if (field == 2 && wire == 0) {  // data_type
            if (!reader.read_varint(data_type)) {
                throw_error(ErrorCode::ModelLoadError, "malformed data_type");
            }
            kept_spans.emplace_back(start, reader.pos());
        } else if (field == 8 && wire == 2) {  // name（只取来报错用）
            const std::byte* name_bytes = nullptr;
            std::size_t name_size = 0;
            if (!reader.read_bytes(name_bytes, name_size)) {
                throw_error(ErrorCode::ModelLoadError, "malformed initializer name");
            }
            name = std::string(as_string(name_bytes, name_size));
            kept_spans.emplace_back(start, reader.pos());
        } else if (field == 9 && wire == 2) {  // raw_data
            has_raw_data = true;
            if (!reader.skip(wire)) {
                throw_error(ErrorCode::ModelLoadError, "malformed raw_data");
            }
            kept_spans.emplace_back(start, reader.pos());
        } else {
            if (!reader.skip(wire)) {
                throw_error(ErrorCode::ModelLoadError, "malformed initializer field");
            }
            kept_spans.emplace_back(start, reader.pos());
        }
    }

    // 非 external、或已内联（带 raw_data）的初始化器：整段原样返回。
    if (!is_external || location.empty() || has_raw_data) {
        return std::vector<std::byte>(data, data + size);
    }
    if (!has_length) {
        throw_error(ErrorCode::ModelLoadError, "external initializer '" + name + "' has no length");
    }

    const auto file = resolved.find(location);
    if (file == resolved.end() || file->second.bytes == nullptr) {
        throw_error(ErrorCode::ModelLoadError, "external weight file not provided: '" + location +
                                                   "' (initializer '" + name + "')");
    }
    const std::uint64_t offset = std::strtoull(offset_text.c_str(), nullptr, 10);
    const std::uint64_t length = std::strtoull(length_text.c_str(), nullptr, 10);
    const std::byte* region = file->second.bytes->data();
    const std::size_t file_size = file->second.bytes->size();

    Writer writer;
    for (const auto& [begin, end] : kept_spans) {
        writer.raw(data + begin, end - begin);
    }
    if (file->second.fp16_packed) {
        // HF 的 fp16 打包：图声明 FLOAT、offset/length 为 fp32 字节单位，
        // 文件里同一张量在 offset/2、占 length/2 字节。
        if (data_type != 1) {
            throw_error(ErrorCode::Unsupported,
                        "fp16-packed weight file '" + location + "' backs non-float initializer '" +
                            name + "' (data_type " + std::to_string(data_type) + ")");
        }
        if (offset % 4 != 0 || length % 4 != 0) {
            throw_error(ErrorCode::ModelLoadError, "external initializer '" + name +
                                                       "' offset/length are not 4-byte aligned (" +
                                                       std::to_string(offset) + ", " +
                                                       std::to_string(length) + ")");
        }
        const std::uint64_t half_offset = offset / 2U;
        const std::uint64_t half_length = length / 2U;
        if (half_offset > file_size || half_length > file_size - half_offset) {
            throw_error(ErrorCode::ModelLoadError,
                        "external initializer '" + name + "' fp16 range [" +
                            std::to_string(half_offset) + ", " +
                            std::to_string(half_offset + half_length) + ") is out of bounds for '" +
                            location + "' (" + std::to_string(file_size) + " bytes)");
        }
        auto upcast = upcast_fp16(region + half_offset, static_cast<std::size_t>(half_length));
        writer.len_field(9, upcast.data(), upcast.size());
    } else {
        // fp32 文件：字节原样内联（含 fp32 / 整数等一切声明类型）。
        if (offset > file_size || length > file_size - offset) {
            throw_error(ErrorCode::ModelLoadError,
                        "external initializer '" + name + "' range [" + std::to_string(offset) +
                            ", " + std::to_string(offset + length) + ") is out of bounds for '" +
                            location + "' (" + std::to_string(file_size) + " bytes)");
        }
        writer.len_field(9, region + offset, static_cast<std::size_t>(length));
    }
    return writer.take();
}

/// 转码 GraphProto：先按外部文件解析精度（fp32 直读 / fp16 打包），
/// 再逐个内联 initializer（field 5），其余字段透传。
std::vector<std::byte> transcode_graph(const std::byte* data, const std::size_t size,
                                       const ExternalWeightSource& source) {
    // 预扫描：收集每个外部文件的 fp32 区间上界（max_end）。
    Reader scan(data, size);
    std::map<std::string, std::uint64_t> max_end;
    while (!scan.done()) {
        std::uint64_t key = 0;
        if (!scan.read_varint(key)) {
            throw_error(ErrorCode::ModelLoadError, "malformed graph (bad wire format)");
        }
        const auto field = static_cast<std::uint32_t>(key >> 3U);
        const auto wire = static_cast<std::uint32_t>(key & 0x7U);
        if (field != 5 || wire != 2) {
            if (!scan.skip(wire)) {
                throw_error(ErrorCode::ModelLoadError, "malformed graph field");
            }
            continue;
        }
        const std::byte* tensor_bytes = nullptr;
        std::size_t tensor_size = 0;
        if (!scan.read_bytes(tensor_bytes, tensor_size)) {
            throw_error(ErrorCode::ModelLoadError, "malformed initializer");
        }
        Reader tensor(tensor_bytes, tensor_size);
        std::string location;
        std::string offset_text;
        std::string length_text;
        bool is_external = false;
        while (!tensor.done()) {
            std::uint64_t tkey = 0;
            if (!tensor.read_varint(tkey)) {
                break;
            }
            const auto t_field = static_cast<std::uint32_t>(tkey >> 3U);
            const auto t_wire = static_cast<std::uint32_t>(tkey & 0x7U);
            if (t_field == 13 && t_wire == 2) {
                const std::byte* entry_bytes = nullptr;
                std::size_t entry_size = 0;
                if (!tensor.read_bytes(entry_bytes, entry_size)) {
                    break;
                }
                Reader entry(entry_bytes, entry_size);
                std::string e_key;
                std::string e_value;
                while (!entry.done()) {
                    std::uint64_t e_tag = 0;
                    if (!entry.read_varint(e_tag)) {
                        break;
                    }
                    const auto e_field = static_cast<std::uint32_t>(e_tag >> 3U);
                    const auto e_wire = static_cast<std::uint32_t>(e_tag & 0x7U);
                    if (e_wire != 2 || (e_field != 1 && e_field != 2)) {
                        if (!entry.skip(e_wire)) {
                            break;
                        }
                        continue;
                    }
                    const std::byte* v = nullptr;
                    std::size_t v_size = 0;
                    if (!entry.read_bytes(v, v_size)) {
                        break;
                    }
                    (e_field == 1 ? e_key : e_value) = std::string(as_string(v, v_size));
                }
                if (e_key == "location") {
                    location = e_value;
                } else if (e_key == "offset") {
                    offset_text = e_value;
                } else if (e_key == "length") {
                    length_text = e_value;
                }
            } else if (t_field == 14 && t_wire == 0) {
                std::uint64_t flag = 0;
                if (!tensor.read_varint(flag)) {
                    break;
                }
                if (flag == 1) {
                    is_external = true;
                }
            } else if (!tensor.skip(t_wire)) {
                break;
            }
        }
        if (is_external && !location.empty() && !length_text.empty()) {
            const std::uint64_t offset = std::strtoull(offset_text.c_str(), nullptr, 10);
            const std::uint64_t length = std::strtoull(length_text.c_str(), nullptr, 10);
            max_end[location] = std::max(max_end[location], offset + length);
        }
    }

    // 解析每个外部文件：fp32 装得下 → 原样；否则一半装得下 → fp16 打包；
    // 都不行 → 报错（不做歧义回退）。
    std::map<std::string, ResolvedExternalFile> resolved;
    for (const auto& [location, end] : max_end) {
        const auto* bytes = source(location);
        if (bytes == nullptr) {
            throw_error(ErrorCode::ModelLoadError,
                        "external weight file not provided: '" + location + "'");
        }
        ResolvedExternalFile file;
        file.bytes = bytes;
        if (bytes->size() < end) {
            if (end > bytes->size() * 2U) {
                throw_error(
                    ErrorCode::ModelLoadError,
                    "external weight file '" + location + "' (" + std::to_string(bytes->size()) +
                        " bytes) is too small for the graph (expects up to " + std::to_string(end) +
                        " bytes) under either fp32 or fp16 interpretation");
            }
            file.fp16_packed = true;
        }
        resolved.emplace(location, file);
    }

    Reader reader(data, size);
    Writer writer;
    while (!reader.done()) {
        const std::size_t start = reader.pos();
        std::uint64_t key = 0;
        if (!reader.read_varint(key)) {
            throw_error(ErrorCode::ModelLoadError, "malformed graph (bad wire format)");
        }
        const auto field = static_cast<std::uint32_t>(key >> 3U);
        const auto wire = static_cast<std::uint32_t>(key & 0x7U);
        if (field == 5 && wire == 2) {  // initializer
            const std::byte* tensor_bytes = nullptr;
            std::size_t tensor_size = 0;
            if (!reader.read_bytes(tensor_bytes, tensor_size)) {
                throw_error(ErrorCode::ModelLoadError, "malformed initializer");
            }
            auto inlined = transcode_initializer(tensor_bytes, tensor_size, resolved);
            writer.len_field(5, inlined.data(), inlined.size());
        } else {
            if (field == 15) {
                throw_error(ErrorCode::Unsupported,
                            "sparse_initializer is not supported by the weight upcast path");
            }
            if (!reader.skip(wire)) {
                throw_error(ErrorCode::ModelLoadError, "malformed graph field");
            }
            writer.raw(data + start, reader.pos() - start);
        }
    }
    return writer.take();
}

}  // namespace

std::vector<std::byte> inline_external_fp16_weights(const std::vector<std::byte>& model,
                                                    const ExternalWeightSource& source) {
    Reader reader(model.data(), model.size());
    Writer writer;
    while (!reader.done()) {
        const std::size_t start = reader.pos();
        std::uint64_t key = 0;
        if (!reader.read_varint(key)) {
            throw_error(ErrorCode::ModelLoadError, "malformed model (bad wire format)");
        }
        const auto field = static_cast<std::uint32_t>(key >> 3U);
        const auto wire = static_cast<std::uint32_t>(key & 0x7U);
        if (field == 7 && wire == 2) {  // graph
            const std::byte* graph_bytes = nullptr;
            std::size_t graph_size = 0;
            if (!reader.read_bytes(graph_bytes, graph_size)) {
                throw_error(ErrorCode::ModelLoadError, "malformed graph field");
            }
            auto transcoded = transcode_graph(graph_bytes, graph_size, source);
            writer.len_field(7, transcoded.data(), transcoded.size());
        } else {
            if (!reader.skip(wire)) {
                throw_error(ErrorCode::ModelLoadError, "malformed model field");
            }
            writer.raw(model.data() + start, reader.pos() - start);
        }
    }
    return writer.take();
}

}  // namespace genie
