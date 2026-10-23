// genie/errors.hpp — 统一错误模型
//
// 设计原则：
//   1. 错误码是稳定的枚举值，C ABI 与 C++ SDK 共用同一套，避免两套语义漂移。
//   2. Error 继承 std::runtime_error，携带 code + 可选的上下文字段。
//   3. 跨 C ABI 边界一律转成 genie_status，不允许异常穿过 ABI。

#ifndef GENIE_ERRORS_HPP
#define GENIE_ERRORS_HPP

#include <stdexcept>
#include <string>
#include <string_view>

namespace genie {

/// 错误码。数值即 C ABI 的返回值，禁止重排已发布值。
enum class ErrorCode : int {
    Ok = 0,
    InvalidArgument = 1,
    NotFound = 2,
    IoError = 3,
    ModelLoadError = 4,
    InferenceError = 5,
    Unsupported = 6,
    Cancelled = 7,
    Timeout = 8,
    BackendUnavailable = 9,
    Internal = 10,
};

/// 稳定的错误码文本，用于日志与 C ABI。
[[nodiscard]] const char* to_string(ErrorCode code) noexcept;

/// 由错误码反查（未知值返回 "Unknown"）。
[[nodiscard]] ErrorCode error_code_from_string(std::string_view text) noexcept;

/// 运行时异常。抛出即表示一次可预期的失败，不是 bug。
class Error : public std::runtime_error {
public:
    Error(ErrorCode code, std::string message)
        : std::runtime_error(std::move(message)), code_(code) {}

    [[nodiscard]] ErrorCode code() const noexcept { return code_; }

private:
    ErrorCode code_;
};

/// 便捷构造。
[[noreturn]] void throw_error(ErrorCode code, std::string message);

}  // namespace genie

#endif  // GENIE_ERRORS_HPP
