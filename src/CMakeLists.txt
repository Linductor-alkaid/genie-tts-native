# src/CMakeLists.txt — 库目标定义
#
# 目标分层：
#   genie_core   —— 纯 C++ 核心，不依赖任何外部推理后端，始终可构建
#   genie_ort    —— ONNX Runtime 后端（可选）
#   genie_sched  —— kairo 调度层（可选）
#   genie        —— 聚合上面三者的最终库

find_package(Threads REQUIRED)

# 版本头由 project() 版本号单一来源生成，避免两处硬编码漂移。
configure_file(
    "${CMAKE_SOURCE_DIR}/include/genie/version.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/genie/version.hpp"
    @ONLY
)

# --------------------------------------------------------------------------------------
# genie_core
# --------------------------------------------------------------------------------------
set(GENIE_CORE_SOURCES
    runtime/errors.cpp
    runtime/version.cpp
    runtime/config.cpp
    runtime/tensor.cpp
    runtime/session_factory.cpp
    audio/audio.cpp
)

add_library(genie_core STATIC ${GENIE_CORE_SOURCES})

target_include_directories(genie_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(genie_core PUBLIC cxx_std_20)

# 告警与 sanitizer 只作用于本项目自身代码，且不应泄漏到安装后的导出集，
# 因此用 $<BUILD_INTERFACE:> 限定为构建期依赖。
target_link_libraries(genie_core
    PUBLIC
        $<BUILD_INTERFACE:genie::warnings>
    PRIVATE
        $<BUILD_INTERFACE:Threads::Threads>
)

if(GENIE_ENABLE_SANITIZERS)
    target_link_libraries(genie_core PUBLIC $<BUILD_INTERFACE:genie::sanitizers>)
endif()

add_library(genie::core ALIAS genie_core)

# --------------------------------------------------------------------------------------
# genie_ort —— ONNX Runtime 后端
# --------------------------------------------------------------------------------------
if(GENIE_ENABLE_ORT)
    add_library(genie_ort STATIC
        runtime/ort_backend.cpp
    )
    target_link_libraries(genie_ort
        PUBLIC
            genie_core
        PRIVATE
            onnxruntime::onnxruntime
    )
    target_compile_definitions(genie_ort PUBLIC GENIE_HAVE_ONNXRUNTIME=1)
    add_library(genie::ort ALIAS genie_ort)
endif()

# --------------------------------------------------------------------------------------
# genie_sched —— kairo 调度层
# --------------------------------------------------------------------------------------
if(GENIE_ENABLE_KAIRO)
    add_library(genie_sched STATIC
        scheduler/scheduler.cpp
    )
    target_link_libraries(genie_sched
        PUBLIC
            genie_core
        PRIVATE
            ${GENIE_KAIRO_TARGET}
    )
    target_compile_definitions(genie_sched PUBLIC GENIE_HAVE_KAIRO=1)
    add_library(genie::sched ALIAS genie_sched)
endif()

# --------------------------------------------------------------------------------------
# genie —— 聚合库
#
# 是一个真实的静态库而不是 INTERFACE 库：build_features() 需要同时知道
# ONNX Runtime 与 kairo 是否编入，而 genie_core 不依赖这两者。
# --------------------------------------------------------------------------------------
add_library(genie STATIC
    features.cpp
)
target_link_libraries(genie
    PUBLIC
        genie_core
    PRIVATE
        $<BUILD_INTERFACE:Threads::Threads>
)
if(GENIE_ENABLE_ORT)
    target_link_libraries(genie PRIVATE genie_ort)
    target_compile_definitions(genie PRIVATE GENIE_HAVE_ONNXRUNTIME=1)
endif()
if(GENIE_ENABLE_KAIRO)
    target_link_libraries(genie PRIVATE genie_sched)
    target_compile_definitions(genie PRIVATE GENIE_HAVE_KAIRO=1)
endif()
if(GENIE_ENABLE_SANITIZERS)
    target_link_libraries(genie PRIVATE genie::sanitizers)
endif()
add_library(genie::genie ALIAS genie)

# --------------------------------------------------------------------------------------
# 安装
# --------------------------------------------------------------------------------------
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/genie
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h")

set(GENIE_INSTALL_TARGETS genie_core genie)
if(GENIE_ENABLE_ORT)
    list(APPEND GENIE_INSTALL_TARGETS genie_ort)
endif()
if(GENIE_ENABLE_KAIRO)
    list(APPEND GENIE_INSTALL_TARGETS genie_sched)
endif()

install(TARGETS ${GENIE_INSTALL_TARGETS}
        EXPORT genieTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
