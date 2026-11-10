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
#include <bit>
#include <charconv>
#include <fstream>
#include <istream>
#include <iterator>
#include <limits>
#include <memory>
#include <sstream>
#include <string_view>
#include <system_error>

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

// ------------------------------------------------------------------ 外部权重：流式扫描与 fp16 上转
//
// 本节服务"包里存 fp16 权重、加载期上转 FP32"（借鉴上游 Genie-TTS 的分发
// 格式，决策见 docs/architecture.md §2.13）。
//
// **HF fp16 格式**：图把外部张量声明为 FLOAT（data_type=1），offset/length
// 以 **fp32 字节**为单位；对应的 *_fp16.bin 是同一批张量转成 fp16 后
// **连续打包**，文件大小恰为一半，张量位于 offset/2。由此**整文件逐元素
// 上转**（第 i 个 half → 第 i 个 float）得到的缓冲恰好就是 fp32 bin 的
// 布局，每个张量自动落回图声明的 offset —— 不需要逐张量改写图。此前的
// wire-format 内联转码器因此退役：它把整张图连同权重重新序列化，ORT
// 再把这份大 protobuf 解析一遍，一个 300 MB 的模型要整体搬运五六遍
// （实测对比见 docs/performance.md）。
//
// 扫描是**流式**的：只顺着嵌套结构读结构字段，raw_data 等大字段一律
// 跳过 —— 1.2 GB 的自包含模型也只读几 KB。
//
// 本节用到的额外字段编号（onnx.proto）：
//   GraphProto.node                  = 1  (LEN, NodeProto)
//   GraphProto.sparse_initializer    = 15 (LEN, SparseTensorProto)
//   NodeProto.attribute              = 5  (LEN, AttributeProto)
//   AttributeProto.t / g             = 5 / 6   (LEN, TensorProto / GraphProto)
//   AttributeProto.tensors / graphs  = 10 / 11 (LEN, repeated)
//   AttributeProto.sparse_tensor(s)  = 22 / 23 (LEN, SparseTensorProto)
//   SparseTensorProto.values/indices = 1 / 2   (LEN, TensorProto)
//   TensorProto.data_type            = 2  (varint)
//   TensorProto.name                 = 8  (LEN string)
//   TensorProto.raw_data             = 9  (LEN —— 跳过，不读)
//   TensorProto.data_location        = 14 (varint；EXTERNAL = 1)

namespace {

static_assert(std::endian::native == std::endian::little,
              "weight files are little-endian; the fp16 upcast reads them as host uint16");

/// 流式 protobuf 读取器：在 istream 上顺着嵌套结构往下走。
///
/// 每层消息由调用方给出的"结束偏移"界定（不像内存版 Reader 那样持有
/// 整段字节）。跳过时小段直接读掉（命中流缓冲），大段才 seek ——
/// filebuf 的 seek 会丢弃缓冲，把成片的小跳过都换成 seek 反而更慢。
class StreamReader {
public:
    StreamReader(std::istream& in, const std::uint64_t size) : in_(in), size_(size) {}

    [[nodiscard]] std::uint64_t pos() const noexcept { return pos_; }
    [[nodiscard]] std::uint64_t size() const noexcept { return size_; }

    [[nodiscard]] bool read_varint(const std::uint64_t end, std::uint64_t& out) {
        std::uint64_t result = 0;
        int shift = 0;
        while (pos_ < end) {
            const auto c = in_.get();
            if (c == std::istream::traits_type::eof()) {
                return false;
            }
            ++pos_;
            const auto byte = static_cast<std::uint8_t>(c);
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

    /// 读 length 前缀，换算成该字段内容的结束偏移；越过 end 视为损坏。
    [[nodiscard]] bool read_length(const std::uint64_t end, std::uint64_t& field_end) {
        std::uint64_t length = 0;
        if (!read_varint(end, length) || length > end - pos_) {
            return false;
        }
        field_end = pos_ + length;
        return true;
    }

    /// 读一段**小**字段的全部字节（名字、external_data 条目）。
    ///
    /// 上限防的是损坏文件：一个被写坏的长度不该把我们骗去分配 GB 级内存。
    [[nodiscard]] bool read_small(const std::uint64_t field_end, std::string& out) {
        const std::uint64_t length = field_end - pos_;
        if (length > kMaxSmallField) {
            return false;
        }
        out.resize(static_cast<std::size_t>(length));
        if (length > 0 && !in_.read(out.data(), static_cast<std::streamsize>(length))) {
            return false;
        }
        pos_ = field_end;
        return true;
    }

    /// 前进到 target（不早于当前位置、不越过文件尾）。
    [[nodiscard]] bool skip_to(const std::uint64_t target) {
        if (target < pos_ || target > size_) {
            return false;
        }
        const std::uint64_t gap = target - pos_;
        if (gap <= kSeekThreshold) {
            in_.ignore(static_cast<std::streamsize>(gap));
        } else {
            in_.seekg(static_cast<std::streamoff>(target), std::ios::beg);
        }
        if (!in_) {
            return false;
        }
        pos_ = target;
        return true;
    }

    /// 跳过当前字段的值。
    [[nodiscard]] bool skip_field(const std::uint64_t end, const std::uint32_t wire) {
        switch (wire) {
            case 0: {
                std::uint64_t ignored = 0;
                return read_varint(end, ignored);
            }
            case 1:
                return end - pos_ >= 8 && skip_to(pos_ + 8);
            case 2: {
                std::uint64_t field_end = 0;
                return read_length(end, field_end) && skip_to(field_end);
            }
            case 5:
                return end - pos_ >= 4 && skip_to(pos_ + 4);
            default:
                return false;
        }
    }

private:
    static constexpr std::uint64_t kSeekThreshold = 64U * 1024U;
    static constexpr std::uint64_t kMaxSmallField = 1U << 20U;

    std::istream& in_;
    std::uint64_t size_;
    std::uint64_t pos_ = 0;
};

/// 一个外部张量引用（扫描的中间结果）。
struct ExternalRef {
    std::string name;
    std::string location;
    std::uint64_t offset = 0;
    std::uint64_t length = 0;
    std::uint64_t data_type = 1;  // FLOAT 是 onnx 的默认值
    bool has_length = false;
    bool nested = false;  ///< 子图 / 节点属性 / 稀疏张量里的引用
};

/// 解析十进制无符号整数（external_data 的 offset / length 以字符串存储）。
[[nodiscard]] bool parse_decimal(const std::string& text, std::uint64_t& out) {
    const char* first = text.data();
    const char* last = first + text.size();
    const auto [ptr, ec] = std::from_chars(first, last, out);
    return !text.empty() && ec == std::errc{} && ptr == last;
}

/// 解析一条 external_data 条目（StringStringEntryProto{key=1, value=2}）写进 ref。
[[nodiscard]] bool apply_external_entry(const std::string& entry, ExternalRef& ref) {
    Reader reader(reinterpret_cast<const std::byte*>(entry.data()), entry.size());
    std::string key;
    std::string value;
    while (!reader.done()) {
        std::uint64_t tag = 0;
        if (!reader.read_varint(tag)) {
            return false;
        }
        const auto field = static_cast<std::uint32_t>(tag >> 3U);
        const auto wire = static_cast<std::uint32_t>(tag & 0x7U);
        if ((field == 1 || field == 2) && wire == 2) {
            const std::byte* bytes = nullptr;
            std::size_t size = 0;
            if (!reader.read_bytes(bytes, size)) {
                return false;
            }
            (field == 1 ? key : value) = std::string(as_string(bytes, size));
        } else if (!reader.skip(wire)) {
            return false;
        }
    }
    if (key == "location") {
        ref.location = std::move(value);
    } else if (key == "offset") {
        return parse_decimal(value, ref.offset);
    } else if (key == "length") {
        ref.has_length = true;
        return parse_decimal(value, ref.length);
    }
    return true;  // checksum 等其余键与布局无关
}

/// 递归扫描器：ModelProto → GraphProto（含子图）→ TensorProto。
class ExternalRefScanner {
public:
    explicit ExternalRefScanner(StreamReader& reader) : reader_(reader) {}

    [[nodiscard]] bool scan_model() {
        return for_each_len_field(
            reader_.size(), [this](const std::uint32_t field, const std::uint64_t field_end) {
                return field != 7 || scan_graph(field_end, /*nested=*/false, 0);
            });
    }

    std::vector<ExternalRef> refs;

private:
    /// 子图嵌套层数上限：合法模型是个位数，防的是构造出来的病态输入。
    static constexpr int kMaxDepth = 64;

    /// 对消息里每个 LEN 字段调用 visit(field, field_end)，其余字段跳过；
    /// visit 没消费完的部分（或整个不关心的字段）由这里跳到字段末尾。
    template <typename Visit>
    [[nodiscard]] bool for_each_len_field(const std::uint64_t end, Visit&& visit) {
        while (reader_.pos() < end) {
            std::uint64_t key = 0;
            if (!reader_.read_varint(end, key)) {
                return false;
            }
            const auto field = static_cast<std::uint32_t>(key >> 3U);
            const auto wire = static_cast<std::uint32_t>(key & 0x7U);
            if (wire != 2) {
                if (!reader_.skip_field(end, wire)) {
                    return false;
                }
                continue;
            }
            std::uint64_t field_end = 0;
            if (!reader_.read_length(end, field_end) || !visit(field, field_end) ||
                !reader_.skip_to(field_end)) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] bool scan_graph(const std::uint64_t end, const bool nested, const int depth) {
        if (depth > kMaxDepth) {
            return false;
        }
        return for_each_len_field(
            end, [this, nested, depth](const std::uint32_t field, const std::uint64_t field_end) {
                switch (field) {
                    case 5:  // initializer
                        return scan_tensor(field_end, nested);
                    case 1:  // node
                        return scan_node(field_end, depth);
                    case 15:  // sparse_initializer
                        return scan_sparse(field_end);
                    default:
                        return true;
                }
            });
    }

    [[nodiscard]] bool scan_node(const std::uint64_t end, const int depth) {
        return for_each_len_field(
            end, [this, depth](const std::uint32_t field, const std::uint64_t field_end) {
                return field != 5 || scan_attribute(field_end, depth);
            });
    }

    [[nodiscard]] bool scan_attribute(const std::uint64_t end, const int depth) {
        return for_each_len_field(
            end, [this, depth](const std::uint32_t field, const std::uint64_t field_end) {
                switch (field) {
                    case 5:   // t
                    case 10:  // tensors
                        return scan_tensor(field_end, /*nested=*/true);
                    case 6:   // g
                    case 11:  // graphs
                        return scan_graph(field_end, /*nested=*/true, depth + 1);
                    case 22:  // sparse_tensor
                    case 23:  // sparse_tensors
                        return scan_sparse(field_end);
                    default:
                        return true;
                }
            });
    }

    [[nodiscard]] bool scan_sparse(const std::uint64_t end) {
        return for_each_len_field(
            end, [this](const std::uint32_t field, const std::uint64_t field_end) {
                // values / indices
                return (field != 1 && field != 2) || scan_tensor(field_end, /*nested=*/true);
            });
    }

    [[nodiscard]] bool scan_tensor(const std::uint64_t end, const bool nested) {
        ExternalRef ref;
        ref.nested = nested;
        bool external = false;
        while (reader_.pos() < end) {
            std::uint64_t key = 0;
            if (!reader_.read_varint(end, key)) {
                return false;
            }
            const auto field = static_cast<std::uint32_t>(key >> 3U);
            const auto wire = static_cast<std::uint32_t>(key & 0x7U);
            if (field == 2 && wire == 0) {
                if (!reader_.read_varint(end, ref.data_type)) {
                    return false;
                }
            } else if (field == 14 && wire == 0) {
                std::uint64_t location = 0;
                if (!reader_.read_varint(end, location)) {
                    return false;
                }
                external = location == 1;
            } else if ((field == 8 || field == 13) && wire == 2) {
                std::uint64_t field_end = 0;
                std::string bytes;
                if (!reader_.read_length(end, field_end) || !reader_.read_small(field_end, bytes)) {
                    return false;
                }
                if (field == 8) {
                    ref.name = std::move(bytes);
                } else if (!apply_external_entry(bytes, ref)) {
                    return false;
                }
            } else if (!reader_.skip_field(end, wire)) {
                return false;
            }
        }
        if (external && !ref.location.empty()) {
            refs.push_back(std::move(ref));
        }
        return true;
    }

    StreamReader& reader_;
};

/// 把一个引用并进它所在文件的汇总。
void accumulate(const ExternalRef& ref, ExternalWeightFile& file) {
    ++file.tensor_count;
    if (ref.has_length) {
        const std::uint64_t end =
            ref.length > std::numeric_limits<std::uint64_t>::max() - ref.offset
                ? std::numeric_limits<std::uint64_t>::max()
                : ref.offset + ref.length;
        file.required_bytes = std::max(file.required_bytes, end);
    }
    if (!file.fp16_blocker.empty()) {
        return;
    }
    const std::string tensor = "tensor '" + ref.name + "'";
    if (ref.nested) {
        file.fp16_blocker = tensor +
                            " lives in a subgraph / node attribute / sparse tensor "
                            "(in-memory replacement only covers main-graph initializers)";
    } else if (!ref.has_length) {
        file.fp16_blocker = tensor + " has no external_data length";
    } else if (ref.data_type != 1) {
        file.fp16_blocker =
            tensor + " is not FLOAT (data_type " + std::to_string(ref.data_type) + ")";
    } else if (ref.offset % 4 != 0 || ref.length % 4 != 0) {
        file.fp16_blocker = tensor + " offset/length are not 4-byte aligned (" +
                            std::to_string(ref.offset) + ", " + std::to_string(ref.length) + ")";
    }
}

/// IEEE half → IEEE single 的位模式，对全部 65536 个输入逐位精确。
///
/// 无分支写法（让编译器向量化）：正规数只需把指数偏置从 15 挪到 127；
/// Inf/NaN 再挪一次使指数全 1（尾数 = NaN 载荷原样保留）；零与次正规先
/// 按"指数 = 1"的正规数解释，再减去 2^-14 —— 被减数落在 [2^-14, 2^-13)，
/// 减法精确（Sterbenz），结果 m×2^-24 是单精度正规数，不受 FTZ/DAZ 影响。
[[nodiscard]] std::uint32_t half_to_float_bits(const std::uint16_t half) {
    constexpr std::uint32_t kExponentMask = 0x0F800000U;  // half 的 5 位指数，左移 13 位后
    constexpr std::uint32_t kRebias = (127U - 15U) << 23U;
    const std::uint32_t sign = static_cast<std::uint32_t>(half & 0x8000U) << 16U;
    const std::uint32_t magnitude = static_cast<std::uint32_t>(half & 0x7FFFU) << 13U;
    const std::uint32_t exponent = magnitude & kExponentMask;
    const std::uint32_t normal = magnitude + kRebias;
    const std::uint32_t special = normal + kRebias;
    const float renormalized = std::bit_cast<float>(normal + (1U << 23U)) - 0x1p-14F;
    std::uint32_t bits = exponent == kExponentMask ? special : normal;
    bits = exponent == 0 ? std::bit_cast<std::uint32_t>(renormalized) : bits;
    return bits | sign;
}

/// 打开文件并取字节数（seek 到尾 —— 与 read_onnx_graph 同款）。
std::uint64_t open_with_size(std::ifstream& file, const std::string& path, const char* what) {
    file.open(path, std::ios::binary);
    if (!file) {
        throw_error(ErrorCode::IoError, std::string("cannot open ") + what + ": " + path);
    }
    file.seekg(0, std::ios::end);
    const auto end = file.tellg();
    file.seekg(0, std::ios::beg);
    if (end < 0 || !file) {
        throw_error(ErrorCode::IoError,
                    std::string("cannot determine size of ") + what + ": " + path);
    }
    return static_cast<std::uint64_t>(end);
}

}  // namespace

std::vector<ExternalWeightFile> scan_external_weights(const std::string& model_path) {
    std::ifstream file;
    const std::uint64_t size = open_with_size(file, model_path, "ONNX model");

    StreamReader reader(file, size);
    ExternalRefScanner scanner(reader);
    if (!scanner.scan_model()) {
        throw_error(ErrorCode::ModelLoadError,
                    "malformed ONNX model (external weight scan failed): " + model_path);
    }

    std::vector<ExternalWeightFile> files;
    for (const auto& ref : scanner.refs) {
        auto it = std::find_if(files.begin(), files.end(), [&ref](const ExternalWeightFile& f) {
            return f.location == ref.location;
        });
        if (it == files.end()) {
            ExternalWeightFile entry;
            entry.location = ref.location;
            files.push_back(std::move(entry));
            it = std::prev(files.end());
        }
        accumulate(ref, *it);
    }
    return files;
}

ExternalWeightLayout classify_external_weights(const ExternalWeightFile& file,
                                               const std::uint64_t file_size) {
    if (file.required_bytes <= file_size) {
        return ExternalWeightLayout::Fp32;
    }
    // required ≤ 2 × size，不溢出的写法。
    if (file.required_bytes / 2U + file.required_bytes % 2U <= file_size) {
        if (!file.fp16_blocker.empty()) {
            throw_error(ErrorCode::ModelLoadError,
                        "external weight file '" + file.location + "' (" +
                            std::to_string(file_size) + " bytes, graph expects " +
                            std::to_string(file.required_bytes) +
                            ") looks fp16-packed but cannot be upcast: " + file.fp16_blocker);
        }
        if (file_size % 2U != 0) {
            throw_error(ErrorCode::ModelLoadError, "fp16-packed weight file '" + file.location +
                                                       "' has an odd size (" +
                                                       std::to_string(file_size) + " bytes)");
        }
        return ExternalWeightLayout::Fp16Packed;
    }
    throw_error(ErrorCode::ModelLoadError,
                "external weight file '" + file.location + "' (" + std::to_string(file_size) +
                    " bytes) is too small for the graph (expects up to " +
                    std::to_string(file.required_bytes) +
                    " bytes) under either fp32 or fp16 interpretation");
}

Fp32WeightBuffer read_fp16_weights_as_fp32(const std::string& path) {
    std::ifstream file;
    const std::uint64_t size = open_with_size(file, path, "weight file");
    if (size % 2U != 0) {
        throw_error(ErrorCode::ModelLoadError, "fp16 weight file has an odd size (" +
                                                   std::to_string(size) + " bytes): " + path);
    }

    Fp32WeightBuffer buffer;
    buffer.count = static_cast<std::size_t>(size / 2U);
    buffer.data = std::make_unique_for_overwrite<float[]>(buffer.count);

    // 一次读 1 Mi 个 half（2 MB）：块足够大摊薄系统调用，又小到不必在意峰值。
    constexpr std::size_t kChunk = std::size_t{1} << 20U;
    std::vector<std::uint16_t> chunk(std::min(kChunk, buffer.count));
    for (std::size_t done = 0; done < buffer.count;) {
        const std::size_t n = std::min(chunk.size(), buffer.count - done);
        if (!file.read(reinterpret_cast<char*>(chunk.data()),
                       static_cast<std::streamsize>(n * sizeof(std::uint16_t)))) {
            throw_error(ErrorCode::IoError, "failed to read weight file: " + path);
        }
        float* out = buffer.data.get() + done;
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = std::bit_cast<float>(half_to_float_bits(chunk[i]));
        }
        done += n;
    }
    return buffer;
}

}  // namespace genie
