// src/runtime/errors.cpp

#include "genie/errors.hpp"

#include <array>
#include <cstring>
#include <string_view>
#include <utility>

namespace genie {
namespace {

struct ErrorCodeEntry {
    ErrorCode code;
    std::string_view text;
};

constexpr std::array<ErrorCodeEntry, 11> kErrorCodes{{
    {ErrorCode::Ok, "ok"},
    {ErrorCode::InvalidArgument, "invalid_argument"},
    {ErrorCode::NotFound, "not_found"},
    {ErrorCode::IoError, "io_error"},
    {ErrorCode::ModelLoadError, "model_load_error"},
    {ErrorCode::InferenceError, "inference_error"},
    {ErrorCode::Unsupported, "unsupported"},
    {ErrorCode::Cancelled, "cancelled"},
    {ErrorCode::Timeout, "timeout"},
    {ErrorCode::BackendUnavailable, "backend_unavailable"},
    {ErrorCode::Internal, "internal"},
}};

}  // namespace

const char* to_string(ErrorCode code) noexcept {
    for (const auto& entry : kErrorCodes) {
        if (entry.code == code) {
            return entry.text.data();
        }
    }
    return "unknown";
}

ErrorCode error_code_from_string(std::string_view text) noexcept {
    for (const auto& entry : kErrorCodes) {
        if (entry.text == text) {
            return entry.code;
        }
    }
    return ErrorCode::Internal;
}

void throw_error(ErrorCode code, std::string message) {
    throw Error(code, std::move(message));
}

}  // namespace genie
