// tests/support/model_paths.hpp —— 测试用的模型定位
//
// 模型目录优先级：
//   1. 环境变量 GENIE_TEST_MODEL_DIR
//   2. 编译期注入的默认路径（CMake 传入）
//
// 模型缺失时测试应当 **跳过** 而不是失败 —— 纯逻辑测试不应该因为没有模型权重而红。

#ifndef GENIE_TESTS_SUPPORT_MODEL_PATHS_HPP
#define GENIE_TESTS_SUPPORT_MODEL_PATHS_HPP

#include <cstdlib>
#include <filesystem>
#include <string>

namespace genie::test {

/// 测试模型根目录。
inline std::filesystem::path model_root() {
    if (const char* env = std::getenv("GENIE_TEST_MODEL_DIR"); env != nullptr && env[0] != '\0') {
        return std::filesystem::path(env);
    }
#ifdef GENIE_TEST_MODEL_DIR_DEFAULT
    return std::filesystem::path(GENIE_TEST_MODEL_DIR_DEFAULT);
#else
    return std::filesystem::path("third_party/genie-tts-models");
#endif
}

/// 某个模型文件是否存在。
inline bool model_exists(const std::string& relative) {
    std::error_code ec;
    return std::filesystem::exists(model_root() / relative, ec);
}

/// V2 模型路径。
inline std::filesystem::path v2_model(const std::string& name) {
    return model_root() / "v2" / name;
}

/// V2ProPlus 模型路径。
inline std::filesystem::path v2proplus_model(const std::string& name) {
    return model_root() / "v2ProPlus" / name;
}

/// 合成测试模型路径。
///
/// 上游发布的 ONNX 模型是"图壳"（权重在外部 .bin 里，不在仓库中），
/// 无法被 ONNX Runtime 加载；真正跑运行时路径用的是这个自包含的合成模型。
inline std::filesystem::path synthetic_model(const std::string& name = "echo.onnx") {
#ifdef GENIE_TEST_SOURCE_DIR_DEFAULT
    return std::filesystem::path(GENIE_TEST_SOURCE_DIR_DEFAULT) / "tests" / "models" / "synthetic" /
           name;
#else
    return model_root().parent_path().parent_path() / "tests" / "models" / "synthetic" / name;
#endif
}

}  // namespace genie::test

#endif  // GENIE_TESTS_SUPPORT_MODEL_PATHS_HPP
