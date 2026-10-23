# 架构与关键设计决策

> 本文记录**为什么**这么设计。半年后回看时，这一节比代码更能说明当初的取舍。
> 规范类内容在 [`AGENTS.md`](../AGENTS.md)，本文只讲架构。

---

## 1. 分层

```
┌──────────────────────────────────────────────────────────┐
│  tools/cli            include/genie/c        ← 入口层    │
├──────────────────────────────────────────────────────────┤
│  src/pipeline         src/gpt                ← 编排层    │
│  src/acoustic         src/text                           │
├──────────────────────────────────────────────────────────┤
│  src/runtime          src/audio              ← 基础设施  │
│  src/scheduler                                   层       │
└──────────────────────────────────────────────────────────┘
```

**依赖只能向下。** 领域层（gpt / acoustic / text）不认识 `Ort::Value`，
只认识 `genie::Tensor`。

### 目标划分

| 库 | 依赖 | 何时编入 |
|---|---|---|
| `genie_core` | 仅标准库 | 永远 |
| `genie_ort` | ONNX Runtime | `GENIE_ENABLE_ORT=ON` 且能找到 ORT |
| `genie_sched` | kairo | `GENIE_ENABLE_KAIRO=ON` |
| `genie` | 聚合 | 永远（接口库） |

拆分的理由：让**纯逻辑**（文本、音频、张量、调度语义）在没有 ORT 的环境里也能编译和测试。
CI 的 `core` 任务就是这条路径的守卫。

---

## 2. 关键决策

### 2.1 张量抽象：为什么不让上层直接用 `Ort::Value`

**决策**：领域层只依赖 `genie::Tensor`。

**理由**

1. **可测**：文本/音频/调度逻辑不需要 ORT 就能写单测。
2. **可换**：将来要支持 ncnn，换掉的是 `genie_ort`，领域层不动。
3. **可审计**：模型的输入输出契约在 `TensorSpec` 里是**显式**的，
   而 `Ort::Value` 的实际形状要到运行时才知道。

**代价**：多一层拷贝。缓解手段是 `Tensor` 用单一 `std::vector<std::byte>` 存储 +
`memcpy` 视图访问，避免 strict-aliasing 问题。

**为什么只支持三种元素类型**：上游六个模型的全部类型就是 float32 / int64 / bool。
多支持一种就要多一套视图与转换，而目前没有需求。

### 2.2 按名字绑定模型输入输出

**决策**：ONNX Runtime 后端一律按**名字**喂输入，不按位置。

**理由**：上游 `t2s_cpu()` 用 `zip(input_names, [y, y_emb, *present_key_values])`
按位置传参，变量名 `y`/`y_emb` 与模型真实输入名 `iy`/`iy_emb` **不一致**。
位置绑定在输入顺序变化时会静默错位 —— 不报错，只输出垃圾。

按名字绑定把这条隐式约定变成了显式契约，并已固化为
`tests/component/test_model_contract.cpp` 里的断言。

### 2.3 采样：主机侧没有采样器

**决策**：不实现主机侧 top-k / top-p / temperature / 重复惩罚。

**理由**：审计确认上游的采样烘焙在 ONNX 图内（`first_stage_decoder` 与
`stage_decoder` 各含一个 `TopK` 算子），Python 侧没有任何采样代码。
主机侧实现一套采样器是**死代码**。

唯一可控的随机性是 ONNX Runtime 的全局随机种子。
要暴露采样参数，必须重新导出模型（独立工作项）。

详见 [`upstream-audit.md`](upstream-audit.md) §5。

### 2.4 调度与并发：全部交给 kairo

**决策**：线程池、任务优先级、有界通道、协作式取消，一律用
[kairo](https://github.com/Linductor-alkaid/kairo)。本项目不造线程池。

**理由**

1. **不重复造轮子**：kairo 已经有可工作窃取的线程池、优先级调度、
   预分配无锁通道、StopToken —— 重新实现一遍只会得到一个更差的版本。
2. **一致性**：kairo 是本项目姊妹项目（同一作者），两边共享同一套并发语义。
3. **可替换**：公开 API 不暴露 kairo 类型（`genie::Scheduler` / `genie::AudioChunkChannel`
   都是自有类型），将来换掉不影响上层。

**kairo 集成中确认的语义**（踩过的坑，记在这里）：

- kairo 的默认异步执行器是**进程级一次性**资源：`shutdown()` 之后不能再 `initialize()`。
  因此 `genie::Scheduler` 把「停止接活」（`shutdown()`，可恢复）与
  「销毁执行器」（`release_executor()`，不可恢复）拆成两个动作。
- `MpscChannel` 是**单消费者**语义：多个线程可以 `try_send`，但只有一个线程应当消费。
- `CommResult` 用 `explicit operator bool()`，不是 `.ok()`。

### 2.5 音频：不引入第三方音频库

**决策**：自带 RIFF/WAV 编解码 + 自带重采样。

**理由**

1. 需求只覆盖 **PCM WAV（单声道/多声道）+ 32 kHz↔16 kHz 重采样**，
   用 `dr_wav` / `miniaudio` / `libsndfile` 都要多一个依赖，
   而端侧部署的依赖越少越好。
2. 上游用 `soundfile`（libsndfile）+ `soxr`，两个都是重量级依赖。

**代价与缓解**

| 差异 | 影响 | 缓解 |
|---|---|---|
| `soxr 'hq'` vs 窗函数法带限 sinc | 语音频段（< 8 kHz）差异远低于听阈 | M0 阶段做量化对照（待办） |
| 多声道降采样用均值 | 与上游 `np.mean(wav, axis=1)` **一致** | 已有单元测试锁定 |

### 2.6 错误模型：异常 + 稳定错误码

**决策**：C++ 侧用 `genie::Error`（携带 `ErrorCode`）抛异常；C ABI 侧转成 `genie_status`。

**理由**：异常在 C++ 侧最自然，且能携带上下文；
但异常**不允许穿过 C ABI**（会跨越语言边界 UB），所以 ABI 边界必须转换。

错误码数值是**已发布契约**，只增不改 —— C ABI 的用户会硬编码这些值。

---

## 3. 推理链路与上游的对齐点

以下行为**必须**与上游一致，任何偏离都要在 `upstream-audit.md` 记录原因：

| 行为 | 上游位置 |
|---|---|
| 文本前置 `'。'` | `Core/Inference.py` `tts()` |
| 语义 token ≥ 1024 视为 EOS 并截断 | `Core/Inference.py` `tts()` |
| 解码循环上限 500 | `Core/Inference.py` `t2s_cpu()` |
| `pred_semantic` 保留 `[1, 1, L]` 的 singleton 维 | `Core/Inference.py` `tts()` |
| 多声道取均值转单声道 | `Audio/Audio.py` `load_audio()` |
| 中文 + ASCII 字母 → 混读模式 | `GetPhonesAndBert.py` `get_phones_and_bert()` |
| V2ProPlus 的 vits 不接收 `ref_audio` | `Core/Inference.py` `tts()` |

---

## 4. 线程与并发模型

```
调用线程
   │  genie::Engine::synthesize()
   ▼
genie::Scheduler::submit()  ──►  kairo Executor（线程池）
   │                              │
   │                              ├─ 文本前端（CPU 密集）
   │                              ├─ T2S 自回归解码（串行、逐 token）
   │                              └─ 声学合成（CPU 密集）
   ▼
genie::AudioChunkChannel  ──►  消费者（播放 / 落盘）
   （有界，满了由调用方决定阻塞或丢弃）
```

**约束**

- 同一 `ModelSession` **不是**线程安全的：ORT 的 `Session::Run` 并发调用需要
  外部同步。本项目通过**每个并发请求一个 Session** 或显式互斥来保证。
- 模型内存是**只读共享**的：多个请求共享已加载的模型，解码状态与可变缓存必须隔离。
  （对应计划书 M5.4「模型共享」）

---

## 5. 模型包（voice package）格式

```text
voice-package/
├── manifest.json
├── models/
│   ├── t2s_encoder.onnx
│   ├── t2s_first_stage_decoder.onnx
│   ├── t2s_stage_decoder.onnx
│   ├── vits.onnx
│   ├── prompt_encoder.onnx     # V2ProPlus
│   ├── chinese-hubert-base.onnx
│   └── RoBERTa.onnx            # 可选
├── tokenizer/
└── reference/
```

`manifest.json` 记录：模型版本、语言、采样率、模型文件清单、校验值、
所需资源版本、可选能力（V2ProPlus / RoBERTa）。

**校验值**是必须的：模型文件被替换是最常见的部署事故，
而 ONNX 加载失败时的报错信息通常看不出根因。

> 格式细节随 M1 的 model loader 落地，本节届时更新。

---

## 6. 模型授权

- `third_party/genie-tts-models/` 下的六个模型来自上游 Genie-TTS 仓库（MIT），随源码发布。
- **用户自己的音色模型不入库**，其授权由相应作者决定。
- `chinese-hubert-base.onnx` / `RoBERTa.onnx` / 说话人验证模型**不由本项目分发**，
  由用户在模型包中自行提供。
