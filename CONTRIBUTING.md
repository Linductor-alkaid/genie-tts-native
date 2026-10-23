# 贡献指南

欢迎参与 genie-tts-native。

**先读 [`AGENTS.md`](AGENTS.md)** —— 那里是本项目的管理规范，本文件只讲具体操作。

## 环境要求

| 依赖 | 版本 | 说明 |
|---|---|---|
| CMake | ≥ 3.20 | |
| C++ 编译器 | 支持 C++20 | GCC 12+ / Clang 15+ / MSVC 19.3+ |
| ONNX Runtime | 1.22.1（CPU） | 可选，但真实推理需要 |
| Ninja | 任意 | 构建生成器 |

ONNX Runtime 用官方预编译包即可：

```bash
curl -fsSL -o ort.tgz \
  https://github.com/microsoft/onnxruntime/releases/download/v1.22.1/onnxruntime-linux-x64-1.22.1.tgz
tar -xzf ort.tgz -C /opt/onnxruntime --strip-components=1
```

## 构建

```bash
# 默认（kairo 调度层自动从 GitHub 拉取）
cmake --preset debug
cmake --build --preset debug
ctest --preset debug --output-on-failure
```

可用的 preset：`debug` / `release` / `asan` / `ubsan` / `tsan` / `warnings`。

指定本地 ONNX Runtime：

```bash
cmake --preset debug -DGENIE_ONNXRUNTIME_ROOT=/opt/onnxruntime
```

离线构建（已有 kairo 源码树）：

```bash
cmake --preset debug -DFETCHCONTENT_SOURCE_DIR_KAIRO=/path/to/kairo
```

不启用任何外部后端（最小闭环）：

```bash
cmake --preset debug -DGENIE_ENABLE_ORT=OFF -DGENIE_ENABLE_KAIRO=OFF
```

## 提交一个改动

```bash
git switch -c feat/my-change
# ... 改动 ...
cmake --preset debug && cmake --build --preset debug && ctest --preset debug
git push -u origin feat/my-change
# 在 GitHub 上开 PR，等 CI 全绿，再合并
```

## 代码风格

- 4 空格缩进，行宽 100 —— 用仓库里的 `.clang-format`
- 提交前跑一次 `clang-format -i`：

```bash
find include src tools tests -name '*.hpp' -o -name '*.cpp' | xargs clang-format -i
```

## 测试

```bash
ctest --preset debug --output-on-failure
```

模型相关的测试需要仓库内置的模型（`third_party/genie-tts-models/`），开箱即用。

## 报告问题

开 issue 时请附上：

- 操作系统与 CPU 架构
- `genie-tts version` 的输出
- 复现步骤
- 完整错误信息

## 许可证

贡献的代码以 MIT 协议发布，与本项目保持一致。详见 [`LICENSE`](LICENSE)。
