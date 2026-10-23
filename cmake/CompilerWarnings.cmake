# CompilerWarnings.cmake — 统一编译告警策略
#
# 原则：警告即错误只对项目自有代码生效（PRIVATE），第三方依赖不受影响。

add_library(genie_warnings INTERFACE)
add_library(genie::warnings ALIAS genie_warnings)

if(MSVC)
    target_compile_options(genie_warnings INTERFACE
        /W4
        /permissive-
        /utf-8
        /Zc:__cplusplus
        /wd4324   # 对齐相关，结构体布局警告由 C ABI 层统一处理
    )
    target_compile_definitions(genie_warnings INTERFACE
        NOMINMAX
        WIN32_LEAN_AND_MEAN
        _CRT_SECURE_NO_WARNINGS
    )
else()
    target_compile_options(genie_warnings INTERFACE
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -Wconversion
        -Wsign-conversion
        -Wnon-virtual-dtor
        -Wold-style-cast
        -Wcast-align
        -Wunused
        -Woverloaded-virtual
        -Wnull-dereference
        -Wdouble-promotion
    )
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(genie_warnings INTERFACE -Werror=return-type)
    endif()
endif()
