// genie/json.hpp — 最小 JSON 解析器
//
// 为什么自带而不引第三方：
//   模型包 manifest 是本项目**唯一**需要解析 JSON 的地方（音色包描述）。
//   为此引入 nlohmann/json 这样的重量级依赖不划算 —— 端侧部署的依赖越少越好。
//   本文件实现 RFC 8259 所需的最小子集，输出一个可读写的值树。
//
// 特性：
//   - 完整支持对象 / 数组 / 字符串 / 数字 / 布尔 / null
//   - 字符串转义含 \uXXXX（含代理对 → UTF-8）
//   - 数字区分整数与浮点
//   - 有嵌套深度上限，避免恶意输入把栈压爆
//   - 解析失败抛出 genie::Error（带行列信息）

#ifndef GENIE_JSON_HPP
#define GENIE_JSON_HPP

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace genie::json {

class Value;

// 命名说明：类型别名不叫 Array / Object，是因为 Value::Type 的枚举成员
// Array / Object 会遮蔽它们（-Wshadow 会报）。
using ObjectType = std::map<std::string, Value>;
using ArrayType = std::vector<Value>;

/// JSON 值。
class Value {
public:
    enum class Type { Null, Bool, Int, Double, String, Array, Object };

    Value() = default;
    explicit Value(bool value) : type_(Type::Bool), bool_(value) {}
    explicit Value(std::int64_t value) : type_(Type::Int), int_(value) {}
    explicit Value(double value) : type_(Type::Double), double_(value) {}
    explicit Value(std::string value) : type_(Type::String), string_(std::move(value)) {}
    explicit Value(ArrayType array_value) : type_(Type::Array), array_(std::move(array_value)) {}
    explicit Value(ObjectType object_value)
        : type_(Type::Object), object_(std::move(object_value)) {}

    [[nodiscard]] Type type() const noexcept { return type_; }
    [[nodiscard]] bool is_null() const noexcept { return type_ == Type::Null; }
    [[nodiscard]] bool is_bool() const noexcept { return type_ == Type::Bool; }
    [[nodiscard]] bool is_int() const noexcept { return type_ == Type::Int; }
    [[nodiscard]] bool is_double() const noexcept { return type_ == Type::Double; }
    [[nodiscard]] bool is_number() const noexcept { return is_int() || is_double(); }
    [[nodiscard]] bool is_string() const noexcept { return type_ == Type::String; }
    [[nodiscard]] bool is_array() const noexcept { return type_ == Type::Array; }
    [[nodiscard]] bool is_object() const noexcept { return type_ == Type::Object; }

    [[nodiscard]] bool as_bool(bool fallback = false) const noexcept {
        return type_ == Type::Bool ? bool_ : fallback;
    }
    [[nodiscard]] std::int64_t as_int(std::int64_t fallback = 0) const noexcept;
    [[nodiscard]] double as_double(double fallback = 0.0) const noexcept;
    [[nodiscard]] const std::string& as_string() const noexcept { return string_; }
    [[nodiscard]] const ArrayType& as_array() const noexcept { return array_; }
    [[nodiscard]] const ObjectType& as_object() const noexcept { return object_; }
    [[nodiscard]] ArrayType& as_array() noexcept { return array_; }
    [[nodiscard]] ObjectType& as_object() noexcept { return object_; }

    /// 对象成员查找。不存在时返回 nullptr。
    [[nodiscard]] const Value* find(std::string_view key) const;

    /// 对象成员查找，带默认值。
    [[nodiscard]] std::int64_t get_int(std::string_view key, std::int64_t fallback) const;
    [[nodiscard]] bool get_bool(std::string_view key, bool fallback) const;
    [[nodiscard]] std::string get_string(std::string_view key, std::string fallback = {}) const;

    /// 序列化为紧凑 JSON 文本。
    [[nodiscard]] std::string dump() const;

private:
    Type type_ = Type::Null;
    bool bool_ = false;
    std::int64_t int_ = 0;
    double double_ = 0.0;
    std::string string_;
    ArrayType array_;
    ObjectType object_;
};

/// 解析 JSON 文本。
///
/// @throws genie::Error InvalidArgument —— 语法错误（含行列信息）
[[nodiscard]] Value parse(std::string_view text);

/// 解析 JSON 文件。
///
/// @throws genie::Error IoError —— 文件不存在或不可读
/// @throws genie::Error InvalidArgument —— 语法错误
[[nodiscard]] Value parse_file(const std::string& path);

}  // namespace genie::json

#endif  // GENIE_JSON_HPP
