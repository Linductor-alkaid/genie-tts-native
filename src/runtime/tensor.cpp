// src/runtime/tensor.cpp

#include "genie/tensor.hpp"

#include <algorithm>
#include <sstream>
#include <utility>

#include "genie/errors.hpp"

namespace genie {
namespace {

std::size_t product_of(const Tensor::Shape& shape) {
    std::size_t total = 1;
    for (const auto d : shape) {
        if (d < 0) {
            return 0;
        }
        total *= static_cast<std::size_t>(d);
    }
    return total;
}

template <typename T>
std::span<T> reinterpret_span(std::vector<std::byte>& storage) {
    return {reinterpret_cast<T*>(storage.data()), storage.size() / sizeof(T)};
}

template <typename T>
std::span<const T> reinterpret_span(const std::vector<std::byte>& storage) {
    return {reinterpret_cast<const T*>(storage.data()), storage.size() / sizeof(T)};
}

template <typename T>
void store_bytes(std::vector<std::byte>& storage, const std::vector<T>& src) {
    storage.resize(src.size() * sizeof(T));
    if (!src.empty()) {
        std::memcpy(storage.data(), src.data(), storage.size());
    }
}

}  // namespace

const char* to_string(ElementType type) noexcept {
    switch (type) {
        case ElementType::Float32:
            return "float32";
        case ElementType::Int64:
            return "int64";
        case ElementType::Bool:
            return "bool";
    }
    return "unknown";
}

std::size_t element_size(ElementType type) noexcept {
    switch (type) {
        case ElementType::Float32:
            return sizeof(float);
        case ElementType::Int64:
            return sizeof(std::int64_t);
        case ElementType::Bool:
            return sizeof(bool);
    }
    return 0;
}

Tensor Tensor::zeros(ElementType type, Shape shape) {
    Tensor tensor;
    tensor.type_ = type;
    tensor.shape_ = std::move(shape);
    tensor.element_count_ = product_of(tensor.shape_);
    tensor.storage_.assign(tensor.element_count_ * element_size(type), std::byte{0});
    return tensor;
}

Tensor Tensor::from_float32(Shape shape, std::vector<float> data) {
    Tensor tensor;
    tensor.type_ = ElementType::Float32;
    tensor.shape_ = std::move(shape);
    tensor.element_count_ = product_of(tensor.shape_);
    if (data.size() != tensor.element_count_) {
        throw_error(ErrorCode::InvalidArgument,
                    "tensor shape implies " + std::to_string(tensor.element_count_) +
                        " elements but " + std::to_string(data.size()) + " were supplied");
    }
    store_bytes(tensor.storage_, data);
    return tensor;
}

Tensor Tensor::from_int64(Shape shape, std::vector<std::int64_t> data) {
    Tensor tensor;
    tensor.type_ = ElementType::Int64;
    tensor.shape_ = std::move(shape);
    tensor.element_count_ = product_of(tensor.shape_);
    if (data.size() != tensor.element_count_) {
        throw_error(ErrorCode::InvalidArgument,
                    "tensor shape implies " + std::to_string(tensor.element_count_) +
                        " elements but " + std::to_string(data.size()) + " were supplied");
    }
    store_bytes(tensor.storage_, data);
    return tensor;
}

Tensor Tensor::from_bool(Shape shape, std::vector<bool> data) {
    Tensor tensor;
    tensor.type_ = ElementType::Bool;
    tensor.shape_ = std::move(shape);
    tensor.element_count_ = product_of(tensor.shape_);
    if (data.size() != tensor.element_count_) {
        throw_error(ErrorCode::InvalidArgument,
                    "tensor shape implies " + std::to_string(tensor.element_count_) +
                        " elements but " + std::to_string(data.size()) + " were supplied");
    }
    // std::vector<bool> 是位压缩的，必须逐位展开。
    tensor.storage_.assign(tensor.element_count_ * sizeof(bool), std::byte{0});
    for (std::size_t i = 0; i < data.size(); ++i) {
        const auto value = static_cast<std::byte>(data[i] ? 1 : 0);
        std::memcpy(tensor.storage_.data() + i * sizeof(bool), &value, sizeof(bool));
    }
    return tensor;
}

std::int64_t Tensor::dim(std::size_t dim) const noexcept {
    return dim < shape_.size() ? shape_[dim] : 0;
}

std::span<float> Tensor::floats() noexcept {
    if (type_ != ElementType::Float32) {
        return {};
    }
    return reinterpret_span<float>(storage_);
}

std::span<const float> Tensor::floats() const noexcept {
    if (type_ != ElementType::Float32) {
        return {};
    }
    return reinterpret_span<const float>(storage_);
}

std::span<std::int64_t> Tensor::ints() noexcept {
    if (type_ != ElementType::Int64) {
        return {};
    }
    return reinterpret_span<std::int64_t>(storage_);
}

std::span<const std::int64_t> Tensor::ints() const noexcept {
    if (type_ != ElementType::Int64) {
        return {};
    }
    return reinterpret_span<const std::int64_t>(storage_);
}

std::span<bool> Tensor::bools() noexcept {
    if (type_ != ElementType::Bool) {
        return {};
    }
    return reinterpret_span<bool>(storage_);
}

std::span<const bool> Tensor::bools() const noexcept {
    if (type_ != ElementType::Bool) {
        return {};
    }
    return reinterpret_span<const bool>(storage_);
}

std::optional<Scalar> Tensor::scalar() const noexcept {
    if (element_count_ != 1) {
        return std::nullopt;
    }
    switch (type_) {
        case ElementType::Float32: {
            float value = 0.0F;
            std::memcpy(&value, storage_.data(), sizeof(value));
            return Scalar{value};
        }
        case ElementType::Int64: {
            std::int64_t value = 0;
            std::memcpy(&value, storage_.data(), sizeof(value));
            return Scalar{value};
        }
        case ElementType::Bool: {
            bool value = false;
            std::memcpy(&value, storage_.data(), sizeof(value));
            return Scalar{value};
        }
    }
    return std::nullopt;
}

bool Tensor::matches(const Shape& expected) const noexcept {
    if (expected.size() != shape_.size()) {
        return false;
    }
    for (std::size_t i = 0; i < shape_.size(); ++i) {
        if (expected[i] >= 0 && expected[i] != shape_[i]) {
            return false;
        }
    }
    return true;
}

std::string Tensor::shape_string() const {
    std::ostringstream out;
    for (std::size_t i = 0; i < shape_.size(); ++i) {
        if (i != 0) {
            out << 'x';
        }
        out << shape_[i];
    }
    return out.str();
}

}  // namespace genie
