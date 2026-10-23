# tests/CMakeLists.txt —— 测试套件
#
# 分层（对应里程碑 M4.1）：
#   unit        —— 纯逻辑：文本、音符、张量、采样、音频
#   component  —— 各 ONNX 子模型的输入输出契约
#   integration —— 完整 TTS 合成链路
#   regression —— 多语言、音色、异常输入
#
# component / integration 需要真实模型。仓库内置了上游 Genie-TTS 的六个 ONNX
# 模型（MIT，见 third_party/genie-tts-models/LICENSE），因此这些测试开箱即跑。
# 若模型缺失，对应测试会自动跳过而不是失败。

set(GENIE_TEST_MODEL_DIR_DEFAULT "${CMAKE_SOURCE_DIR}/third_party/genie-tts-models"
    CACHE PATH "默认测试模型目录")

add_library(genie_test_support INTERFACE)
target_include_directories(genie_test_support INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(genie_test_support INTERFACE
    "GENIE_TEST_MODEL_DIR_DEFAULT=\"${GENIE_TEST_MODEL_DIR_DEFAULT}\"")
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
)
target_link_libraries(genie_unit_tests PRIVATE genie_test_support)
add_test(NAME genie_unit COMMAND genie_unit_tests)

# --------------------------------------------------------------------------------------
# component —— 模型 I/O 契约（需要 ONNX Runtime）
# --------------------------------------------------------------------------------------
if(GENIE_ENABLE_ORT)
    add_executable(genie_component_tests
        component/test_model_contract.cpp
    )
    target_link_libraries(genie_component_tests PRIVATE genie_test_support)
    add_test(NAME genie_component COMMAND genie_component_tests)
endif()

# --------------------------------------------------------------------------------------
# integration —— 端到端推理链路
# --------------------------------------------------------------------------------------
if(GENIE_ENABLE_ORT)
    add_executable(genie_integration_tests
        integration/test_synthesis.cpp
    )
    target_link_libraries(genie_integration_tests PRIVATE genie_test_support)
    add_test(NAME genie_integration COMMAND genie_integration_tests)
endif()
