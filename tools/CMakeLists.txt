# tools/CMakeLists.txt —— 命令行工具

if(GENIE_BUILD_CLI)
    add_executable(genie-tts cli/main.cpp)
    target_link_libraries(genie-tts PRIVATE genie::genie)
    target_include_directories(genie-tts PRIVATE ${CMAKE_SOURCE_DIR}/include)
    set_target_properties(genie-tts PROPERTIES OUTPUT_NAME genie-tts)

    install(TARGETS genie-tts RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

# genie-bench：分阶段性能基准（roadmap M0.3 / M7 的基准报告数据源）。
# 只依赖公开 API（SynthesisResult.timing / Engine::load_timing），
# 独立于 CLI 开关 —— 测基线不该要求装 CLI。
if(GENIE_BUILD_BENCH)
    add_executable(genie-bench bench/main.cpp)
    target_link_libraries(genie-bench PRIVATE genie::genie)
    target_include_directories(genie-bench PRIVATE ${CMAKE_SOURCE_DIR}/include)
    set_target_properties(genie-bench PROPERTIES OUTPUT_NAME genie-bench)
    if(WIN32)
        # 峰值 RSS 走 GetProcessMemoryInfo（psapi）。
        target_link_libraries(genie-bench PRIVATE psapi)
    endif()

    install(TARGETS genie-bench RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
