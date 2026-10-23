# AGENTS.md — 项目管理规范

> 本文件是 genie-tts-native 的**最高约束**。任何人（人类或 AI）在本仓库里干活，
> 都以本文件为准。规则冲突时，优先级顺序是：
> **本文件 > `docs/architecture.md` > `CONTRIBUTING.md` > 单个 PR 的描述。**

---

## 1. 项目是什么

**genie-tts-native** 是 [Genie-TTS](https://github.com/High-Logic/Genie-TTS) 的
**纯 C++ 原生推理引擎**：把上游的 GPT-SoVITS V2 / V2ProPlus ONNX 推理链路
完整迁移到 C++20，**运行时不依赖 Python / PyTorch / 任何推理服务**。

上游参考：<https://github.com/High-Logic/Genie-TTS>（MIT）

### 1.1 交付形态

| 形态 | 说明 |
|---|---|
| C++ SDK | `include/genie/` 下的头文件 + `genie::` 命名空间 |
| C ABI | `include/genie/c/genie_tts.h`，供 Go / Rust / ctypes 调用 |
| CLI | `genie-tts` 可执行文件 |

### 1.2 明确不做的事

- **不做**通用推理后端抽象（首期只支持 ONNX Runtime CPU EP）
- **不做** ncnn / TensorRT / CUDA 移植（等 ONNX 路径稳定并拿到性能数据后单独立项）
- **不做**服务端 / HTTP API（那是上游 Python 侧的事，本项目是嵌入式引擎）
- **不做** GUI

### 1.3 依赖边界

| 关注点 | 承担者 | 说明 |
|---|---|---|
| 推理后端 | ONNX Runtime CPU EP | 唯一的模型执行路径 |
| **调度与并发** | **kairo**（自有库） | 线程池、任务优先级、有界通道、协作式取消 |
| 测试 | GoogleTest + CTest | |
| 音频 IO | 自带 RIFF/PCM 编解码 | 不引入第三方音频库 |
| 文本前端 | 自带 + 词典 | 详见 §5.3 |

> **调度与并发一律走 kairo，不自己造线程池。**
> kairo 仓库：<https://github.com/Linductor-alkaid/kairo>（固定到 commit，见 `cmake/Dependencies.cmake`）
> 需要线程池、任务优先级、有界队列、取消、定时器、串行执行上下文时，**先查 kairo 有没有**，
> 只有确实没有才在本项目里补。

---

## 2. 里程碑驱动的工作方式

### 2.1 里程碑

工作按 `docs/roadmap.md` 的 M0–M7 推进。**一次 PR 只做一件事**，
并且这件事必须是某个里程碑的一个可验证切片。

| 里程碑 | 名称 | 出口条件 |
|---|---|---|
| M0 | 上游审计与基线 | 推理链路、依赖清单、Golden Dataset、性能基线 |
| M1 | C++ Runtime 基础设施 | CMake、ONNX 封装、模型加载、张量管理 |
| M2 | 文本与音频前处理 | 规范化、G2P、参考音频特征 |
| M3 | 完整推理迁移 | 语义生成、声学合成、端到端输出 |
| M4 | 正确性与原生 SDK | 一致性测试、C++ API、CLI、跨平台构建 → **v0.1.0** |
| M5 | CPU 性能优化 | 解码优化、线程调优、内存复用、量化 → **v0.2.0** |
| M6 | 低延迟流式 TTS | 增量生成、异步流水线、取消 → |
| M7 | 发布与工程化 | C ABI、安装包、文档、CI、性能报告 → **v0.3.0** |

### 2.2 分支模型

```
master  ──●──────────●──────────●──────────●──▶
         ▲          ▲          ▲          ▲
         │          │          │          │
      (PR 合入)   (PR 合入)   (PR 合入)   (PR 合入)
         ▲          ▲          ▲          ▲
         │          │          │          │
    feat/xxx    feat/yyy    feat/zzz    fix/www
```

**规则**

1. `master` 始终可构建、可测试。**不接受直接向 master 推送。**
2. 每个 PR 对应**一个分支**，分支名 `<type>/<简短描述>`，例如：
   - `feat/runtime-ort-backend`
   - `feat/scheduler-kairo-channel`
   - `fix/resample-edge-case`
   - `docs/upstream-audit-update`
3. 分支从 `master` 切出，**不与其他开发分支互相合并**（避免交叉污染）。
4. 一个 PR 完成后合并，**分支随即删除**。

### 2.3 合并流程（强制）

```
切分支 → 改动 → 本地验证 → 推送 → 开 PR → CI 全绿 → 合并 → 删分支
```

**CI 全绿是合并的硬前置条件。** 具体检查项：

- [ ] `core` 任务通过（核心库 + 单元测试）
- [ ] `kairo` 任务通过（含调度层）
- [ ] `ort` 任务通过（含真实模型契约与端到端推理）
- [ ] `sanitize` 任务通过（ASan + UBSan 无告警）
- [ ] `thread` 任务通过（TSan 无数据竞争）
- [ ] `format` 任务通过（clang-format 无差异）
- [ ] `windows` 任务通过

> 改了调度/并发相关代码 → `thread` 任务**必须**是绿的，不接受"CI 偶发失败，重跑一次就好"。

### 2.4 PR 规模

- 单个 PR 的改动量以**能在一小时内被完整 review 完**为准
- 超过 ~800 行有效改动，或跨 3 个以上模块的 PR，**先开 issue 对齐方案再动手**
- 纯格式化改动（clang-format 全量应用）单独一个 PR，不与功能改动混在一起

### 2.5 Definition of Done

一个 PR 算完成，必须同时满足：

1. 代码通过 §2.3 的全部 CI 检查
2. 新增行为有对应测试；改动既有行为时同步改了测试
3. 公开 API（`include/genie/`）有文档注释，说明**为什么**而不只是**是什么**
4. `CHANGELOG.md` 的 Unreleased 段有对应条目
5. 涉及模型契约的改动，同步更新了 `tests/component/test_model_contract.cpp`
6. 涉及推理链路的改动，在 `docs/upstream-audit.md` 里留下了对应的审计记录

---

## 3. 架构约束

### 3.1 分层

```
        ┌──────────────────────────────┐
        │  tools/cli        include/genie/c  │   ← 入口层
        ├──────────────────────────────┤
        │  src/pipeline     src/gpt     │   ← 编排层
        │  src/acoustic     src/text    │   ← 领域层
        ├──────────────────────────────┤
        │  src/runtime      src/audio   │   ← 基础设施层
        │  src/scheduler                │
        └──────────────────────────────┘
```

**依赖只能向下。** 领域层不得直接依赖 `src/runtime/ort_backend.cpp` 里的 ORT 类型。

### 3.2 张量抽象

- 领域层只认 `genie::Tensor`（`include/genie/tensor.hpp`）
- 任何时候都**不允许**在领域层出现 `Ort::Value`
- 只支持 `float32` / `int64` / `bool` 三种元素类型 —— 这是上游六个模型的**全部**类型
  （见 `docs/upstream-audit.md`）。新增类型需要先改审计文档。

### 3.3 错误处理

- 领域层用 `genie::Error`（带 `ErrorCode`）抛错
- **C ABI 边界一律转成 `genie_status`，异常不允许穿过 ABI**
- 错误码数值是**已发布契约**，只增不改

### 3.4 调度

- 需要并发/异步时用 `genie::Scheduler`（内部是 kairo）
- 需要流式数据分发时用 `genie::AudioChunkChannel`（内部是 kairo 的 `MpscChannel`）
- 公开 API **不暴露 kairo 类型**，保证 kairo 可替换
- 通道是**有界**的：满了必须由调用方决定阻塞、丢弃还是降级，不允许无界堆积

---

## 4. 编码规范

### 4.1 基本

| 项 | 约定 |
|---|---|
| 语言标准 | C++20（`CMAKE_CXX_STANDARD 20`） |
| 命名空间 | `genie`（公开）、`genie::test`（测试） |
| 文件命名 | `snake_case.cpp` / `snake_case.hpp` |
| 类型命名 | `PascalCase` |
| 函数/变量 | `snake_case` |
| 常量 | `kPascalCase`（见 `src/` 内的用法） |
| 成员变量 | `trailing_underscore` |
| 缩进 | 4 空格，行宽 100（`.clang-format`） |
| 头文件守卫 | `#ifndef GENIE_..._HPP` |
| 头文件 | **只声明**，`#pragma once` 不用，统一 include guard |

### 4.2 头文件

- 公开头 `include/genie/*.hpp` 里**不得**包含 ORT / kairo / 第三方头
  （第三方类型必须藏在 `.cpp` 或 PIMPL 后面）
- 公开头必须自包含：单独 include 就能编译通过
- 头文件里的函数要有 Doxygen 风格注释

### 4.3 编译告警

`genie::warnings` 接口库对**本项目自有代码**开启 `-Wall -Wextra -Wpedantic -Wshadow
-Wconversion -Wsign-conversion -Wold-style-cast ...`。
新增告警必须处理，不许用 `#pragma GCC diagnostic` 掩盖。

### 4.4 注释

- 解释**为什么**，不解释**是什么**（what the code says 已经够了）
- 引用上游行为时，注明上游文件路径与函数名，便于对照
- 涉及"与上游不一致"的地方，必须写明**差异原因**与**影响评估**

---

## 5. 测试规范

### 5.1 分层

| 层级 | 位置 | 内容 |
|---|---|---|
| Unit | `tests/unit/` | 纯逻辑，无模型依赖 |
| Component | `tests/component/` | 各 ONNX 子模型的输入输出契约 |
| Integration | `tests/integration/` | 端到端推理链路 |
| Regression | `tests/regression/` | 多语言、音色、异常输入（M4 起） |

### 5.2 要求

- 新增功能**必须**带测试
- 修 bug **必须**先写能复现的测试
- 改了模型契约 → `tests/component/test_model_contract.cpp` 必须同步
- 测试**不允许**依赖网络；模型用仓库内置的 `third_party/genie-tts-models/`
- 模型缺失时测试应当 **skip** 而不是 fail

### 5.3 Golden Dataset

Golden Dataset（M0.2）用于验证与 Python 上游的**逐级一致性**：

```
文本 → 规范化文本 → 音素序列 → BERT 特征 → 语义 token → PCM
```

每一级都要有 Python 参考实现产出的基准文件，C++ 侧逐级比对。
**容差按级定义**：文本/音素必须完全一致；浮点张量用绝对+相对容差；
最终 PCM 不要求逐样本相同（采样与声学模型本身有非确定性），
改用分阶段数值误差 + 语音质量指标。

---

## 6. 文档规范

| 文档 | 内容 | 更新时机 |
|---|---|---|
| `README.md` | 定位、快速开始、当前状态 | 功能/用法变化 |
| `docs/architecture.md` | 分层、关键设计决策及其理由 | 架构变化 |
| `docs/upstream-audit.md` | 上游推理链路的审计结论 | 审计结论变化 |
| `docs/roadmap.md` | 里程碑进度 | 每个里程碑完成 |
| `docs/conventions.md` | 命名、格式、提交规范 | 规范变化 |
| `CHANGELOG.md` | 版本变更 | 每个 PR |

**关键设计决策必须写进 `docs/architecture.md`**，包含：决策、备选方案、选择理由、影响。
半年后回看时，这一节比代码更能说明"当初为什么这么做"。

---

## 7. 模型与资产

- 模型权重**不入库**（除 `third_party/genie-tts-models/` 里上游已随仓库发布的六个模型）
- 用户自己的 voice package 在本地，路径由 `manifest.json` 指定
- 第三方资产的授权边界记录在 `NOTICE` 与 `third_party/genie-tts-models/LICENSE`

---

## 8. 提交与沟通

- 提交信息格式见 `.github/PULL_REQUEST_TEMPLATE.md`
- **一个 PR 一个主题**。混在一起的改动会被要求拆开
- 拿不准的设计先开 issue 讨论，不要在 PR 里现改方案
- 发现上游（Genie-TTS）的 bug 或行为反直觉，**记进 `docs/upstream-audit.md`**，
  不要在本项目里"顺手修正" —— 保持与上游行为一致是本项目的首要目标
