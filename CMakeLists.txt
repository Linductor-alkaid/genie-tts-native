cmake_minimum_required(VERSION 3.20)

# C 语言是为了 C ABI 的测试：那个测试用 **C 编译器**编译，
# 头文件里混进 C++ 语法的话会直接编译失败。
project(genie_tts_native
    VERSION 0.1.0
    DESCRIPTION "Native C++20 inference engine for Genie-TTS (GPT-SoVITS V2 / V2ProPlus)"
    LANGUAGES CXX C
)

# --------------------------------------------------------------------------------------
# 全局设置
# --------------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(GENIE_BUILD_TESTS    "Build the test suite"              ON)
option(GENIE_BUILD_CLI      "Build the genie-tts CLI"            ON)
option(GENIE_BUILD_SHARED   "Build shared libraries"             ON)
option(GENIE_ENABLE_ORT     "Enable the ONNX Runtime backend"    ON)
option(GENIE_ENABLE_KAIRO   "Enable the kairo scheduling layer"  ON)
option(GENIE_ENABLE_SANITIZERS "Enable address/undefined sanitizers" OFF)
# OpenJTalk（日语音素转换）：内置的第三方 C++ 代码 ✓。
#
# MSVC 上默认**关闭** ✓ —— 内置的 mecab 是 POSIX 系的代码
# （mmap / dirent / strdup 那一套 ✗），在 MSVC 上能不能过没有验证过 ✗。
# 与其让 Windows CI 红着 ✓，不如默认关掉 ✓；
# 确认能编过的人可以显式 -DGENIE_ENABLE_OPENJTALK=ON ✓（会被尊重 ✓）。
if(DEFINED GENIE_ENABLE_OPENJTALK)
    # 用户显式指定了 —— 尊重用户的决定 ✓
else()
    set(GENIE_ENABLE_OPENJTALK_DEFAULT ON)
    if(MSVC)
        set(GENIE_ENABLE_OPENJTALK_DEFAULT OFF)
    endif()
    option(GENIE_ENABLE_OPENJTALK "Enable the OpenJTalk backend (Japanese G2P)"
           ${GENIE_ENABLE_OPENJTALK_DEFAULT})
endif()

# 刻意不向 CMAKE_MODULE_PATH 追加本项目的 cmake/ 目录。
# 依赖（kairo 等）会用 include(CompilerWarnings) 这类方式查找同名模块，
# 共享的 module path 会让本项目的模块遮蔽依赖的模块，导致依赖构建失败。
# 本项目的模块一律用显式路径引入。
include(GNUInstallDirs)
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/CompilerWarnings.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/Sanitizers.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/Dependencies.cmake")

# --------------------------------------------------------------------------------------
# 目标
# --------------------------------------------------------------------------------------
# OpenJTalk（日语音素转换）：**内置**的第三方 C++ 代码（BSD ✓），
# 源码在 third_party/openjtalk/ —— 词典不在仓库里（103 MB ✓），
# 运行时由音色包给出路径 ✓。
if(GENIE_ENABLE_OPENJTALK)
    add_subdirectory(third_party/openjtalk)
endif()

add_subdirectory(src)
add_subdirectory(tools)

if(GENIE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# --------------------------------------------------------------------------------------
# 安装与打包
# --------------------------------------------------------------------------------------
install(EXPORT genieTargets
    FILE genieTargets.cmake
    NAMESPACE genie::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/genie"
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/genieConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion
)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/genieConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/genieConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/genie"
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/genieConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/genieConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/genie"
)

message(STATUS "")
message(STATUS "genie-tts-native ${PROJECT_VERSION}")
message(STATUS "  build type ......... ${CMAKE_BUILD_TYPE}")
message(STATUS "  ONNX Runtime ....... ${GENIE_ORT_STATUS}")
message(STATUS "  kairo scheduler .... ${GENIE_KAIRO_STATUS}")
message(STATUS "  tests .............. ${GENIE_BUILD_TESTS}")
message(STATUS "")
