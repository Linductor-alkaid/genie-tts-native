# tools/CMakeLists.txt —— 命令行工具

if(GENIE_BUILD_CLI)
    add_executable(genie-tts cli/main.cpp)
    target_link_libraries(genie-tts PRIVATE genie::genie)
    target_include_directories(genie-tts PRIVATE ${CMAKE_SOURCE_DIR}/include)
    set_target_properties(genie-tts PROPERTIES OUTPUT_NAME genie-tts)

    install(TARGETS genie-tts RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
