# Sanitizers.cmake — ASan / UBSan / TSan 开关
#
# 独立的顶层开关，不依赖 CMAKE_BUILD_TYPE —— CI 的 Release+TSan job 依赖此路径。

option(GENIE_SANITIZE_ADDRESS   "Enable AddressSanitizer"  OFF)
option(GENIE_SANITIZE_UNDEFINED "Enable UndefinedBehaviorSanitizer" OFF)
option(GENIE_SANITIZE_THREAD    "Enable ThreadSanitizer"   OFF)

add_library(genie_sanitizers INTERFACE)
add_library(genie::sanitizers ALIAS genie_sanitizers)

set(_genie_san_flags "")

if(GENIE_SANITIZE_ADDRESS)
    list(APPEND _genie_san_flags address)
endif()
if(GENIE_SANITIZE_UNDEFINED)
    list(APPEND _genie_san_flags undefined)
endif()
if(GENIE_SANITIZE_THREAD)
    list(APPEND _genie_san_flags thread)
endif()

if(_genie_san_flags)
    if(MSVC)
        message(WARNING "Sanitizers are not supported on MSVC; ignoring.")
    else()
        list(JOIN _genie_san_flags "," _genie_san_list)
        target_compile_options(genie_sanitizers INTERFACE
            -fsanitize=${_genie_san_list} -fno-omit-frame-pointer -g)
        target_link_options(genie_sanitizers INTERFACE
            -fsanitize=${_genie_san_list})
        message(STATUS "Sanitizers enabled: ${_genie_san_list}")
    endif()
endif()
