// src/features.cpp — 构建期能力查询
//
// 这一层刻意放在聚合库 genie 里，而不是 genie_core：
// build_features() 要同时知道 ONNX Runtime 与 kairo 是否编入，
// 而 genie_core 不依赖这两者（它必须在两者都缺席时依然可构建）。

#include "genie/version.hpp"

namespace genie {

BuildFeatures build_features() noexcept {
    BuildFeatures features{};
#if defined(GENIE_HAVE_ONNXRUNTIME)
    features.onnxruntime = true;
#else
    features.onnxruntime = false;
#endif
#if defined(GENIE_HAVE_KAIRO)
    features.kairo = true;
#else
    features.kairo = false;
#endif
#if defined(GENIE_SANITIZE_ADDRESS) || defined(GENIE_SANITIZE_UNDEFINED) || \
    defined(GENIE_SANITIZE_THREAD)
    features.sanitizers = true;
#else
    features.sanitizers = false;
#endif
    return features;
}

}  // namespace genie
