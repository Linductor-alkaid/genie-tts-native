// src/runtime/session_factory.cpp — 无 ONNX Runtime 时的降级实现
//
// 当 GENIE_ENABLE_ORT=OFF 时，本文件提供 create_session 的桩实现：
// 抛 BackendUnavailable，保证上层代码在两种构建下都能链接并编译。

#include "genie/session.hpp"

#include "genie/errors.hpp"

#if !defined(GENIE_HAVE_ONNXRUNTIME)

namespace genie {

std::unique_ptr<ModelSession> create_session(const std::string& model_path,
                                              const SessionConfig& config) {
    (void)model_path;
    (void)config;
    throw_error(ErrorCode::BackendUnavailable,
                "this build was compiled without ONNX Runtime — reconfigure with "
                "-DGENIE_ENABLE_ORT=ON and provide an ONNX Runtime package");
}

bool has_onnxruntime_backend() noexcept { return false; }

}  // namespace genie

#endif  // GENIE_HAVE_ONNXRUNTIME
