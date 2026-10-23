# Dependencies.cmake — 第三方依赖解析
#
# 统一策略：每个依赖都遵循「系统包 → FetchContent 拉取 → 关闭对应特性」三级降级，
# 并把最终结果写入 GENIE_<DEP>_STATUS，供顶层 CMakeLists 打印。
#
# 可用 CMake 变量覆盖：
#   FETCHCONTENT_SOURCE_DIR_<UPPERCASE_NAME>  指向本地已有源码树（离线/沙箱构建）
#   GENIE_ONNXRUNTIME_ROOT                     指定 ONNX Runtime 预编译包根目录
#   GENIE_KAIRO_TAG                            指定 kairo 的 git ref（默认固定 commit）

include(FetchContent)

# --------------------------------------------------------------------------------------
# GoogleTest —— 测试框架
# --------------------------------------------------------------------------------------
set(GENIE_GTEST_TAG "v1.15.2" CACHE STRING "GoogleTest 版本")

find_package(GTest CONFIG QUIET)
if(GTest_FOUND)
    set(GENIE_GTEST_STATUS "system (${GTest_VERSION})")
    set(GENIE_GTEST_TARGET GTest::gtest_main)
else()
    message(STATUS "GTest not found in system — fetching ${GENIE_GTEST_TAG}")
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        ${GENIE_GTEST_TAG}
        GIT_SHALLOW    TRUE
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    set(GENIE_GTEST_STATUS "fetched (${GENIE_GTEST_TAG})")
    set(GENIE_GTEST_TARGET GTest::gtest_main)
endif()

# --------------------------------------------------------------------------------------
# ONNX Runtime —— CPU 推理后端（可选）
#
# 优先使用官方预编译包自带的 CMake config（1.17+ 提供 onnxruntimeConfig.cmake）。
# 未找到时退化为手工 find_path / find_library。
# 两者都失败时：关闭 ORT 后端，核心库与测试仍然可用（使用可插拔的桩后端）。
# --------------------------------------------------------------------------------------
set(GENIE_ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime 预编译包根目录")

find_package(onnxruntime CONFIG QUIET)

if(NOT onnxruntime_FOUND AND GENIE_ONNXRUNTIME_ROOT)
    # 官方预编译包布局：
    #   <root>/include/onnxruntime_cxx_api.h        （部分版本在 include/onnxruntime/ 下）
    #   <root>/lib/libonnxruntime.so.1.22.1          （另有同名软链 libonnxruntime.so）
    find_path(ORT_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        HINTS "${GENIE_ONNXRUNTIME_ROOT}/include"
        PATH_SUFFIXES "" "onnxruntime" "onnxruntime/core/session"
        NO_DEFAULT_PATH)
    find_library(ORT_LIBRARY
        NAMES onnxruntime
        HINTS "${GENIE_ONNXRUNTIME_ROOT}/lib" "${GENIE_ONNXRUNTIME_ROOT}/lib64"
        NO_DEFAULT_PATH)
    if(ORT_INCLUDE_DIR AND ORT_LIBRARY)
        get_filename_component(ORT_LIBDIR "${ORT_LIBRARY}" DIRECTORY)
        add_library(onnxruntime::onnxruntime SHARED IMPORTED)
        set_target_properties(onnxruntime::onnxruntime PROPERTIES
            IMPORTED_LOCATION "${ORT_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE_DIR}")
        # 运行时需要能找到版本化的 .so —— 预编译包里的 libonnxruntime.so 通常是
        # 指向 libonnxruntime.so.<ver> 的相对软链，rpath 指到 lib 目录即可。
        set_property(TARGET onnxruntime::onnxruntime APPEND PROPERTY
            INTERFACE_LINK_OPTIONS
                "LINKER:-rpath,${ORT_LIBDIR}"
                "LINKER:-rpath-link,${ORT_LIBDIR}")
        set(onnxruntime_FOUND TRUE)
        message(STATUS "ONNX Runtime found at ${GENIE_ONNXRUNTIME_ROOT}")
    endif()
endif()

if(onnxruntime_FOUND)
    set(GENIE_ORT_STATUS "ON")
    set(GENIE_ENABLE_ORT ON CACHE BOOL "Enable the ONNX Runtime backend" FORCE)
else()
    if(GENIE_ENABLE_ORT)
        message(WARNING
            "ONNX Runtime not found — the ONNX backend is disabled. "
            "Point -DGENIE_ONNXRUNTIME_ROOT=<prefix> at a prebuilt package to enable it.")
    endif()
    set(GENIE_ORT_STATUS "OFF (not found)")
    set(GENIE_ENABLE_ORT OFF CACHE BOOL "Enable the ONNX Runtime backend" FORCE)
endif()

# --------------------------------------------------------------------------------------
# kairo —— 调度与并发基础设施（可选）
#
# kairo 是本项目自有的并发库，提供 Executor / Channel / StopToken / 优先级调度。
# 本项目用它承担：多会话并发、流水线化合成、流式 PCM 分发、协作式取消。
#
# 固定到具体 commit 而非 tag —— 上游 kairo 尚未发布 tag，commit 才能保证可复现。
# 注意：kairo 仓库没有 tag，且浅克隆无法取到任意 commit，
# 因此这里必须用完整 SHA 且不做浅克隆。
# --------------------------------------------------------------------------------------
set(GENIE_KAIRO_TAG "e8976652019b66f6b18a3a8eba29e3b492b8726f"
    CACHE STRING "kairo git ref (完整 commit SHA)")

find_package(kairo CONFIG QUIET)

if(kairo_FOUND)
    set(GENIE_KAIRO_STATUS "system")
    set(GENIE_KAIRO_TARGET kairo::kairo)
else()
    message(STATUS "kairo not found in system — fetching ${GENIE_KAIRO_TAG}")
    set(KAIRO_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    set(KAIRO_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(KAIRO_ENABLE_GPU     OFF CACHE BOOL "" FORCE)
    set(KAIRO_ENABLE_CUDA    OFF CACHE BOOL "" FORCE)
    set(KAIRO_ENABLE_OPENCL  OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        kairo
        GIT_REPOSITORY https://github.com/Linductor-alkaid/kairo.git
        GIT_TAG        ${GENIE_KAIRO_TAG}
        GIT_SHALLOW    FALSE
    )
    FetchContent_MakeAvailable(kairo)
    set(GENIE_KAIRO_STATUS "fetched (${GENIE_KAIRO_TAG})")
    set(GENIE_KAIRO_TARGET kairo)
endif()
