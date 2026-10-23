// genie/tensor.hpp — 与推理后端无关的轻量张量
//
// 目的：把"张量"这个概念从 ONNX Runtime 里剥离出来。上层（gpt / acoustic / pipeline）
// 只依赖本文件的 Tensor，不直接依赖 Ort::Value，这样后端可替换、测试可脱离 ORT。
//
// 约束：
//   - 只支持 float32 / int64 / bool 三种元素类型 —— 这是上游六个模型实际用到的全部类型。
//   - 行主序连续存储，形状即维度序列。
//   - 单一字节缓冲区 + memcpy 视图访问，避免 strict-aliasing 问题与多份存储。

#ifndef GENIE_TENSOR_HPP
#define GENIE_TENSOR_HPP

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>

namespace genie {

/// 元素类型。
enum class ElementType {
    Float32,
    Int64,
    Bool,
};

[[nodiscard]] const char* to_string(ElementType type) noexcept;

/// 每元素字节数。
[[nodiscard]] std::size_t element_size(ElementType type) noexcept;

/// 标量值 —— 用于 bool 停止条件这类标量输出。
using Scalar = std::variant<float, std::int64_t, bool>;

/// 连续张量。
class Tensor {
public:
    using Shape = std::vector<std::int64_t>;

    Tensor() = default;

    /// 按形状分配张量，内容清零。
    [[nodiscard]] static Tensor zeros(ElementType type, Shape shape);

    /// 由已有数据构造（拷贝）。元素数与形状不匹配时抛 genie::Error。
    [[nodiscard]] static Tensor from_float32(Shape shape, std::vector<float> data);
    [[nodiscard]] static Tensor from_int64(Shape shape, std::vector<std::int64_t> data);
    [[nodiscard]] static Tensor from_bool(Shape shape, std::vector<bool> data);

    [[nodiscard]] ElementType type() const noexcept { return type_; }
    [[nodiscard]] const Shape& shape() const noexcept { return shape_; }
    [[nodiscard]] std::size_t rank() const noexcept { return shape_.size(); }
    [[nodiscard]] std::size_t element_count() const noexcept { return element_count_; }
    [[nodiscard]] std::size_t byte_size() const noexcept { return storage_.size(); }

    /// 第 dim 维的长度。越界返回 0。
    [[nodiscard]] std::int64_t dim(std::size_t dim) const noexcept;

    /// 按 float32 视图访问。类型不匹配时返回空 span。
    [[nodiscard]] std::span<float> floats() noexcept;
    [[nodiscard]] std::span<const float> floats() const noexcept;

    /// 按 int64 视图访问。
    [[nodiscard]] std::span<std::int64_t> ints() noexcept;
    [[nodiscard]] std::span<const std::int64_t> ints() const noexcept;

    /// 按 bool 视图访问。
    [[nodiscard]] std::span<bool> bools() noexcept;
    [[nodiscard]] std::span<const bool> bools() const noexcept;

    /// 标量视图。类型或元素数不匹配时返回 nullopt。
    [[nodiscard]] std::optional<Scalar> scalar() const noexcept;

    /// 原始字节视图。
    [[nodiscard]] std::span<std::byte> bytes() noexcept {
        return {storage_.data(), storage_.size()};
    }
    [[nodiscard]] std::span<const std::byte> bytes() const noexcept {
        return {storage_.data(), storage_.size()};
    }

    /// 形状是否与预期一致（逐维比较，动态维以 -1 表示通配）。
    [[nodiscard]] bool matches(const Shape& expected) const noexcept;

    /// 人类可读的形状描述，如 "1x512"。
    [[nodiscard]] std::string shape_string() const;

private:
    ElementType type_ = ElementType::Float32;
    Shape shape_{};
    std::size_t element_count_ = 0;
    std::vector<std::byte> storage_{};
};

}  // namespace genie

#endif  // GENIE_TENSOR_HPP
