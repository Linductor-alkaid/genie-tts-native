# tests/CMakeLists.txt —— 测试套件
#
# 分层（对应里程碑 M4.1）：
#   unit        —— 纯逻辑：文本、音符、张量、采样、音频
#   component   —— 各 ONNX 子模型的输入输出契约
#   integration —— ONNX Runtime 运行时路径
#   regression  —— 多语言、音色、异常输入（M4 起）
#
# 模型说明：
#   - 上游发布的六个 ONNX 模型是"图壳"（权重在外部 .bin 里，合计约 1.1 GB，
#     不在仓库中），因此**不能**被 ONNX Runtime 加载。
#   - 契约测试因此直接解析 ONNX protobuf（genie::read_onnx_graph），
#     不需要权重、不需要 ORT —— 在任何构建下都能跑。
#   - 运行时路径测试用自包含的合成模型 tests/models/synthetic/echo.onnx。

set(GENIE_TEST_MODEL_DIR_DEFAULT "${CMAKE_SOURCE_DIR}/third_party/genie-tts-models"
    CACHE PATH "默认测试模型目录")

add_library(genie_test_support INTERFACE)
target_include_directories(genie_test_support INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(genie_test_support INTERFACE
    "GENIE_TEST_MODEL_DIR_DEFAULT=\"${GENIE_TEST_MODEL_DIR_DEFAULT}\""
    "GENIE_TEST_SOURCE_DIR_DEFAULT=\"${CMAKE_SOURCE_DIR}\"")
target_link_libraries(genie_test_support INTERFACE genie::genie ${GENIE_GTEST_TARGET})

# --------------------------------------------------------------------------------------
# unit
# --------------------------------------------------------------------------------------
add_executable(genie_unit_tests
    unit/test_errors.cpp
    unit/test_config.cpp
    unit/test_tensor.cpp
    unit/test_audio.cpp
    unit/test_scheduler.cpp
    unit/test_onnx_graph.cpp
    unit/test_json.cpp
    unit/test_sha256.cpp
    unit/test_voice_package.cpp
    unit/test_symbols.cpp
    unit/test_pinyin.cpp
    unit/test_text.cpp
    unit/test_t2s.cpp
    unit/test_engine.cpp
    # 必须排在最后：它要覆盖"调度器就绪状态下进程退出"这条路径。
    unit/test_scheduler_lifetime.cpp
)
target_link_libraries(genie_unit_tests PRIVATE genie_test_support)
add_test(NAME genie_unit COMMAND genie_unit_tests)
# 每个测试都有超时：挂死时 ctest 会杀掉它并报出是哪个，
# 而不是让 CI job 一直耗到 6 小时上限。
set_tests_properties(genie_unit PROPERTIES TIMEOUT 120)

# --------------------------------------------------------------------------------------
# c —— C ABI
#
# 用 **C 编译器**编译：头文件里混进 C++ 语法的话这里会直接编译失败。
# ABI 的使用者（Go / Rust / Python ctypes）拿到的就是纯 C 头文件，
# 这个约束必须由测试来守。
# --------------------------------------------------------------------------------------
add_executable(genie_c_api_test
    c/test_c_api.c
)
target_link_libraries(genie_c_api_test PRIVATE genie::c)
target_include_directories(genie_c_api_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME genie_c_api COMMAND genie_c_api_test)
set_tests_properties(genie_c_api PROPERTIES TIMEOUT 60)

# --------------------------------------------------------------------------------------
# component —— 模型 I/O 契约（不需要 ONNX Runtime，也不需要权重）
# --------------------------------------------------------------------------------------
add_executable(genie_component_tests
    component/test_model_contract.cpp
)
target_link_libraries(genie_component_tests PRIVATE genie_test_support)
add_test(NAME genie_component COMMAND genie_component_tests)

# --------------------------------------------------------------------------------------
# integration —— ONNX Runtime 运行时路径
# --------------------------------------------------------------------------------------
if(GENIE_ENABLE_ORT)
    add_executable(genie_integration_tests
        integration/test_synthesis.cpp
    )
    target_link_libraries(genie_integration_tests PRIVATE genie_test_support)
    add_test(NAME genie_integration COMMAND genie_integration_tests)
endif()
