# 编码与协作规范

规范类内容的**最高约束**是 [`AGENTS.md`](../AGENTS.md)。本文是速查与展开。

---

## 命名

| 类别 | 约定 | 示例 |
|---|---|---|
| 文件 | `snake_case` | `model_loader.cpp` |
| 公开头 | `snake_case.hpp` | `include/genie/session.hpp` |
| 命名空间 | `genie` | `genie::Engine` |
| 类型 | `PascalCase` | `genie::SynthesisOptions` |
| 函数 | `snake_case` | `detect_language()` |
| 变量 | `snake_case` | `sample_rate` |
| 成员变量 | `trailing_underscore` | `model_path_` |
| 局部常量 | `kPascalCase` | `kMaxDecodeSteps` |
| 宏 | `GENIE_XXX` | `GENIE_VERSION_STRING` |
| 枚举值 | `PascalCase` | `ErrorCode::ModelLoadError` |

C ABI 用全小写 + 前缀，避免污染 C 命名空间：

```c
genie_status genie_engine_create(const char* model_dir, genie_engine** out);
```

---

## 头文件

```cpp
#ifndef GENIE_SESSION_HPP     // 不用 #pragma once
#define GENIE_SESSION_HPP
...
#endif
```

- 公开头**不得**包含 ORT / kairo / 第三方头
- 公开头必须自包含（单独 include 可编译）
- 只放声明，不放定义（模板与内联除外）
- 头文件里的类型要带 Doxygen 注释

---

## 格式

`.clang-format`（基于 Google，4 空格，行宽 100）。提交前：

```bash
find include src tools tests -name '*.hpp' -o -name '*.cpp' | xargs clang-format -i
```

CI 的 `format` 任务会做 `clang-format --dry-run --Werror`，不通过就不合并。

---

## 告警

`genie::warnings` 对本项目代码开启：

```
-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion
-Wnon-virtual-dtor -Wold-style-cast -Wcast-align -Wunused
-Woverloaded-virtual -Wnull-dereference -Wdouble-promotion
```

新增告警必须处理。不要用 `#pragma GCC diagnostic` 掩盖。

---

## 错误处理

```cpp
// 领域层
throw genie::Error(genie::ErrorCode::NotFound, "voice package not found: " + path);

// 或用便捷函数
genie::throw_error(genie::ErrorCode::IoError, "cannot open WAV file: " + path);

// C ABI 层：异常绝不越过边界
extern "C" int genie_do_something(...) {
    try {
        ...
        return GENIE_OK;
    } catch (const genie::Error& e) {
        return static_cast<int>(e.code());
    } catch (const std::exception&) {
        return GENIE_ERROR_INTERNAL;
    }
}
```

---

## 注释

- 解释**为什么**，不解释**是什么**
- 引用上游行为时注明出处，格式：`upstream src/genie_tts/Core/Inference.py t2s_cpu()`
- 与上游不一致的地方，必须写明**原因**和**影响评估**

```cpp
// 上游在文本前加一个句号，防止第一句被漏掉。
// 见 upstream src/genie_tts/Core/Inference.py tts()
// 去掉它会让首音素分布与上游不同，导致 Golden Dataset 逐级比对失败。
constexpr char kSentencePrefix[] = "。";
```

---

## 提交信息

```
<type>(<scope>): <中文摘要>

<正文>
```

`type`：`feat` `fix` `refactor` `perf` `test` `docs` `build` `ci` `chore`
`scope`：`runtime` `text` `audio` `gpt` `acoustic` `scheduler` `cli` `abi`

---

## 测试写法

```cpp
TEST_F(V2ModelContract, StageDecoderUsesIyNamesAndPastKv) {
    auto session = create_session(path, config_);
    const auto& inputs = session->inputs();

    ASSERT_EQ(inputs.size(), 50U);
    EXPECT_EQ(inputs[0].name, "iy");
    EXPECT_EQ(inputs[1].name, "iy_emb");
}
```

- `ASSERT_*` 用于**后续代码依赖**该条件的场景
- `EXPECT_*` 用于继续验证其他条件的场景
- **返回值的函数里不能用 `ASSERT_*`**（宏展开成 `return;`）——
  改用抛异常，或改写成带输出参数的形式
- 模型缺失时用 `GTEST_SKIP()`，不要用 `FAIL()`

---

## CMake

- 依赖统一在 `cmake/Dependencies.cmake` 里解析
- **不要**把本项目的 `cmake/` 目录加进 `CMAKE_MODULE_PATH` ——
  会遮蔽依赖的同名模块（kairo 就因此构建失败过）。用显式路径 `include()`
- 公开头的包含路径用 `BUILD_INTERFACE` / `INSTALL_INTERFACE` 生成器表达式
- 目标名带 `genie::` 别名（`genie::core` / `genie::ort` / `genie::sched` / `genie::genie`）
