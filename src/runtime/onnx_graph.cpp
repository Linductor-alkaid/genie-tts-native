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
#include <cstring>
#include <fstream>
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

}  // namespace genie
