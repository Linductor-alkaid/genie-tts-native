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

}  // namespace genie::test

#endif  // GENIE_TESTS_SUPPORT_MODEL_PATHS_HPP
