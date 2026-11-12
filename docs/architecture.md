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
| `genie_sched` | kairo（可降级为标准库实现） | 永远编入；kairo 决定用哪套实现 |
| `genie` | 聚合（静态库） | 永远 |

**零第三方依赖的自带实现**（端侧部署的依赖越少越好）：

| 能力 | 自带实现 | 为什么不引第三方 |
|---|---|---|
| JSON 解析 | `genie::json` | 只需要解析音色包清单，不值得引入 nlohmann/json |
| SHA-256 | `genie::sha256` | 只为一个摘要算法引入 OpenSSL 不划算 |
| ONNX 图元信息 | `genie::read_onnx_graph` | 需要在没有权重、没有 ORT 的情况下锁定 I/O 契约 |
| WAV 编解码 | `genie::WavCodec` | 需求只覆盖 PCM WAV + 32k↔16k 重采样 |
| 重采样 | `genie::resample` | 见 §2.5 |

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

### 2.7 文本前端：符号表必须是"重新实现"而不是"抄"

模型输入的 `text_seq` 是**符号 ID 序列**，不是文字。符号表由上游
`G2P/SymbolsV2.py` 的 `create_master_symbol_list()` 生成 —— 规则是确定的
（若干个集合求并 → 排序 → 追加方括号 → 追加韩文 → 追加粤文）。

本项目**重新实现这条规则**，而不是把上游的 732 个符号抄成一张表。
理由：抄来的表没有"为什么是这样"的线索，改一个符号就悄悄错位；
重新实现规则则任何顺序变动都会与上游对不上，而测试会立刻发现。

```
genie::symbols::master_v2()   →  732 个符号，顺序即 ID
```

符号 ID 错一位**不会报错**，只会合成出乱码 —— 所以这一层用**全表摘要**
（SHA-256）硬锁，而不是抽样。

音素 ID 的推导分两步，每一步都有全量比对：

```
汉字 → tone3 拼音 → (声母, 带调韵母) → 音素符号 → 符号 ID
         │              │                │
    内置词典      与 pypinyin 等价    等价于 opencpop 词表
```

拼音切分规则已对 **41,923 个汉字**与 pypinyin 逐条比对（切分与音素两层），
比对脚本见 `tools/g2p/verify_pinyin_rules.py`。

多音字、变调、儿化与上游有偏差，逐一记录在 `docs/upstream-audit.md` §9.4。

### 2.8 引擎：合成是"多阶段 + 可打断"的

一次合成天然分阶段：

```
文本 → 音素 ID ─┐
                 ├─► t2s_encoder ─► first_stage ─► stage ×N ─► vits ─► PCM
参考音频条件 ───┘
```

其中**自回归解码是串行的**（必须逐 token），但**文本前端与声码器是 CPU 密集**
的，可以与其他请求重叠。

所以分工是：

- `genie::T2SDecoder`：纯逻辑，只依赖 `ModelSession` 抽象 —— 可单测，
  不依赖任何后端；
- `genie::Engine`：编排。流式合成时把整条流水线提交给 kairo 执行器，
  调用线程只从 `AudioChunkChannel` 取音频块。

**公开 API 不暴露任何 kairo 类型** —— 换掉调度后端不会破坏上层。

**生命周期约定**（踩过坑）：

- 一个 `Engine` 同时只跑一个流式合成（模型会话不是线程安全的）；
- 引擎析构时会**取消并等待**在途任务 —— 后台任务捕获了引擎指针，
  引擎先死就是 use-after-free；
- 音频通道由任务与调用方**共同持有**（`shared_ptr`）——
  调用方先放手，通道不能在任务还在往里写的时候析构。

**退出顺序的坑**：kairo 的 `Executor` 是函数级静态，在 `initialize()` 里
才被首次引用 → **比 `Scheduler` 单例更晚构造、更早析构**。所以
`Scheduler` 的析构函数**不能**再碰 kairo（那是 use-after-free）。
症状不是崩溃而是挂住：kairo 的 shutdown 最多等在途任务 300 秒。
析构只做 genie 侧的收尾（置取消标志 + 等任务收尾）。

> `tests/unit/test_scheduler_lifetime.cpp` 专门覆盖这条路径 ——
> 它让进程**带着就绪的调度器退出**。别的调度器测试都在 TearDown 里
> `shutdown()`，反而绕开了它。

### 2.9 C ABI：单独编成共享库

`libgenie_tts.so` / `genie_tts.dll` 是**独立的共享库**，不是静态库。
理由：Go / Rust / Python ctypes 的调用方通常不链接 C++ 静态库，
而是 dlopen 一个共享库 —— 让他们链接 C++ 运行时是个不必要的负担。

- 头文件 `include/genie/c/genie_tts.h` 是**纯 C**，不含任何 C++ 语法；
- 编译符号可见性设为 hidden，只导出 `genie_*`；
- 错误码与 C++ 侧 `genie::ErrorCode` 是**同一套枚举值**，
  编译期用 `static_assert` 钉死 —— 两套语义漂移是最难查的一类 bug；
- **异常绝不穿过 ABI 边界**：所有入口 try/catch，C++ 异常转成 `genie_status`；
- `tests/c/test_c_api.c` **用 C 编译器编译** —— 头文件里混进 C++ 语法会直接失败。

### 2.10 性能计时：常开采集，随结果返回

M0.3 要性能基线、M5 要子模型级耗时占比 —— 数据从哪来？决策是
**计时埋点常开、没有开关**，报告作为数据随结果返回（`SynthesisResult.timing`、
`Engine::load_timing()`），而不是走日志。

- **为什么常开**：每个阶段只做两次 `steady_clock` 采样（纳秒级），相对秒级的
  合成开销可以忽略。一个永远开着、没人需要关的开关，比一个要沿
  `synthesize → decode → decode_loop` 整条链传递的布尔参数便宜 —— 后者
  污染每一层签名，还制造"开了计时结果会不会不一样"的心理负担。
- **为什么随结果返回而不是打日志**：库不引入日志依赖；怎么呈现由调用方决定
  —— CLI 打表格（`synth --timing`）、`genie-bench` 算统计写基线
  （`docs/performance.md`）、SDK 使用者直接编程消费。同一份数据三种用途。
- **阶段边界选在编排层**（`synthesize_impl`）+ T2S 内部再拆一层（encoder /
  first_stage / stage_decoder×N）：前者是架构分层的自然缝，后者是成本大头
  且计数（迭代次数）只有 decode_loop 自己知道。不做到逐 ORT-Run 级别 ——
  那是 ORT 内建 profiler 的活（M5.1）。
- **流式路径挂在流尾（M6）**：分段合成逐段复用 `synthesize_one`，各段
  报告按阶段名并入一份 `StreamFooter.timing`（同名累加、count 为段数之和），
  随通道 footer 交给消费者；失败时已完成段的耗时也尽力保留。
- `T2SDecoder::decode/decode_loop` 的计时走可选出参（`TimingReport*`，
  默认 nullptr）而不是全局单例：解码器声明"只持有配置、状态在栈上"，
  全局可变状态会破坏这个承诺。

### 2.11 解码热路径的宿主拷贝纪律（M5.2）

**背景**：M0.3 基线（`docs/performance.md`）显示 `t2s/stage_decoder` 占合成
耗时 62%–83%，且每 token 耗时随序列长度上升（15.3 → 24.3 ms）。审计发现
每步解码把 KV cache（48 个张量，长句末段约 34 MB）在宿主侧完整拷贝约
五遍：`read_output` 拷入 state → `build_stage_feed` 按值拷入 feed →
`to_ort_value` 分配 + memcpy 进 ORT 张量 → 输出拷回临时 `vector` →
再 memcpy 进 Tensor 存储。这些拷贝量与序列长度成正比，是 ms/token
随长度上升的主要嫌疑之一。

**决策**（三层，语义全部不变）：

1. **move 语义贯穿解码循环**（`src/engine/t2s.cpp`）：`read_output` 从
   输出里 move 出 y_emb / KV cache；`build_stage_feed` 改为可变引用，
   把这两类成员 move 进 feed。安全性依赖一个循环不变式：state 的
   张量成员每轮 run 之后都被新输出**整体替换**，中间态无人再读；
   `state.tokens` 留在原地（循环结束后的 EOS 归零与切片还要用）。
2. **输入零拷贝包装**（`src/runtime/ort_backend.cpp`）：`to_ort_value`
   改为 `CreateTensor` 直接引用 Tensor 的存储缓冲，不再"分配 + memcpy"。
   契约：buffer 生命周期覆盖 Run 调用（feed 以 const 引用贯穿）、
   ORT CPU EP 不改写输入。这推翻了初版"避免 const_cast"的取舍 ——
   cast 掉的 const 只来自 `run(const TensorFeed&)` 的接口签名，底层
   存储本来就可写；别名问题与 Tensor 既有的 `reinterpret_span` 视图同款。
   若未来 ORT 某算子需要就地改写输入，此契约失效，必须回退拷贝路径
   （在 `to_ort_value` 的注释里有同样的警示）。
3. **输出单次拷贝**：新增 `Tensor::from_raw(type, shape, const void*)`
   （恰好一次 memcpy），`from_ort_value` 改走它 —— 此前是"拷进临时
   vector 再 memcpy 进存储"两遍。

**备选与不选的原因**：IOBinding / 预分配输出缓冲可以做到输出也零拷贝，
但需要为 50 个 KV 张量维护跨步复用的缓冲池并管理 ORT arena 交互，
复杂度显著更高；先落这批"语义不变、无新状态"的改动，用实测数据决定
是否值得再上 IOBinding。**后续裁决（M5 第二轮，§2.18）**：上了——
以"会话内自学习形状增长 + 预测失准自动回退"的形态，零接口变更；
实测长文本 ms/token -12%，见 docs/performance.md §IOBinding。

**验证**：模型 I/O 契约测试、回归 IoTrace、真实模型端到端全绿；
`genie-bench` 前后对比数据记入 `docs/performance.md`。

### 2.12 会话线程与自旋的默认值（M5.3）

**决策**：`intra_op_threads` 默认 8、`allow_spinning` 默认**关**
（`session.intra_op.allow_spinning=0`）。数据（i9-13900KF，2026-11-06
扫描，见 `docs/performance.md`）：

- 线程 12/16 在长句上劣化 11–12%（6 个会话各开更大的池会把工作摊到
  P/E 两类核上），收益为负 → 线程保持 8。
- 自旋关之后延迟持平或更优（短句 12.5 vs 14.4 ms/token），进程核占用
  从 9–19 降到 4–5 —— 引擎一次合成只有**一个**会话在干活，闲置会话
  的自旋线程纯属内耗，对嵌入式部署与多请求并发都是净损失。

**稳定性依据**：高线程数 × 自旋是实测机器的已知不稳定区（采集期间
设备死机一次；v0.2.0 也记录过同机高负载下 ORT 内部段错误/浮点异常）。
默认值的选择因此偏向"少而稳"：延迟不吃亏、资源占用低、踩雷面小。

**备选与不选的原因**：按会话分配不同线程预算（t2s 会话多、前端会话
少）理论上更精细，但会话是一次合成内**串行**使用的，单会话池的大小
才是唯一有效变量，扫过之后 8 已经够用；CPU 亲和性 / 核排除留待
M5 后续（个别核不稳时是直接的止血手段）。


### 2.13 fp16 权重分发：包体积换加载时间（M5，借鉴上游）

**背景**：HF 上游发布的模型带 `*_fp16.bin` —— 图把张量声明为 FLOAT、
offset/length 以 fp32 字节为单位，文件是同一批张量转 fp16 后**连续打包**
（大小恰为一半；`t2s_shared` 的 582 个外部张量全部 4 字节对齐、span 总长
恰等于 fp32 bin 大小，审计实测）。上游 Python 在 ModelManager 里做同样的
"读 fp16 → 内存上转 → 注入 ONNX proto"。

**决策（第二版，2026-11-10）**：`manifest.weight_format = "fp16"` 的包，
引擎加载期（`src/engine/model_loader.cpp`）：

1. **流式扫描**每个模型的外部张量（`scan_external_weights`：只读结构
   字段，raw_data 一律跳过 —— 8 个模型含 1.2 GB 的 RoBERTa 合计 ~9 ms），
   按"图声明区间 vs 文件大小"判定每个外部文件是 fp32 还是 fp16 打包；
2. fp16 打包文件**整文件逐元素上转**（第 i 个 half → 第 i 个 float）——
   打包布局下张量位于 offset/2，上转结果恰好就是 fp32 bin 的布局，
   **不需要改写图**；多个模型共用的文件（t2s 两个解码器的 `t2s_shared`）
   只转一次；
3. 用 ONNX Runtime 的 `AddExternalInitializersFromFilesInMemory`（≥ 1.18）
   以内存缓冲替换这些文件、照常按路径建会话；ORT 建会话时拷贝，
   缓冲在全部会话建完后释放。

自包含模型与 fp32 外部文件照常走路径加载（含 SessionFactory 注入缝）。
fp32 包不扫描、不上转，默认不变。

**第一版为什么退役**：第一版（#41/#42）在 wire-format 层把外部张量内联
进 protobuf、再从内存建会话。实测（信任模式）加载 9.47 s，比 fp32 多
~5.5 s，开销来自两处：① 自包含模型（RoBERTa 1.2 GB、SV 184 MB）也被
整份读进内存再整份重写，纯属白干（RoBERTa 0.96 → 3.39 s）；② 有外部
权重的模型，权重被整体搬运五六遍（读 bin、上转、张量 / 图 / 模型三层
写回、ORT 再解析一遍大 protobuf）。第二版：**9.47 → 4.40 s**，比 fp32
（3.84 s）只多 0.56 s，其中上转 0.40 s；峰值 RSS 5.4 → 4.1 GB
（docs/performance.md）。

**备选与不选**：

- 打包期离线转 fp32（`convert_fp16_weights.py`）：无运行时改动，但用户
  要付双倍磁盘（存 fp16 下载、再转 fp32 落盘）；
- 逐张量改写图（第一版）：见上；
- ORT 的 `AddExternalInitializers`（逐张量 OrtValue）：要逐张量建
  OrtValue 并知道形状；整文件替换只需一个缓冲 —— 它利用了打包布局
  "位置 = offset/2"的性质，更简单；
- 零拷贝模式（`session.use_external_initializer_file_buffers_directly`）：
  能省 ORT 那份拷贝，但缓冲须与会话同寿，且 CI 的 ORT 1.22 是否支持
  未验证 —— 留作后续。

**约束与风险**：

- ORT 的内存替换只覆盖**主图**初始化器。子图 / 节点属性 / 稀疏张量里
  的外部引用若指向 fp16 文件，后端会按 fp32 偏移去读磁盘上的 fp16
  文件 —— 扫描器把这类引用记为不可上转，加载即报错（本项目六模型
  实测没有这种引用）。
- "ORT 建会话时拷贝"由集成测试锁定：建完会话立刻释放缓冲，再推理并
  逐元素校验；"不注入就加载失败"同样有用例，说明替换确实生效。
- 上转对全部 65536 个 half 值逐位精确（穷举单测，含次正规 / Inf /
  NaN 载荷）；fp16 精度损失与上游发布的 fp16 权重一致（不是额外降
  精度）；音频听感回归待人工确认；回退点 = manifest 改回 fp32。


### 2.14 加载期并行与 kairo 的绑核问题（M5）

**背景**：冷启动（信任模式）里建会话占 ~85%，各会话互不依赖 —— 并行
后这一段从 3.35 s 压到 ~1.16 s（~2.9×）。

**绑核问题（kairo < 0.8 的阻塞，2026-11-10 探针实测）**：

- 旧版 kairo 默认线程池把 worker i **绑到单核** i % hw（空
  `cpu_affinity` 自动分配 [0..hw-1]，且配置里没有"不绑核"选项）；
- ORT 会话的 intra-op 线程在**建会话时**创建，继承创建线程的亲和性
  —— 在 kairo worker 上建出的会话，推理线程全被钉在 worker 的那个核
  上（探针实测：worker allowed=1，其上建出的 ORT 线程 allowed=1），
  冷启动快了、推理却会成倍变慢。

**解除（kairo 升至 53ee306，其 PR #227）**：默认池空 `cpu_affinity`
改为 **OS 自由调度**（与 BlockingIoConfig 原有语义对齐；显式设值仍按
worker_id % size 确定性绑核）。复测：kairo worker 与其上建出的 ORT
线程均 allowed=0-31。附带收益：流式合成的 `Run` 调用线程不再可能被
钉在 E 核上。

**实现（genie 侧）**：调度层新增 `detail::parallel_for`（fork-join；
kairo 没有 parallel_for / wait-all、`Scheduler::submit` 吞异常 ——
AGENTS.md §1.3 "kairo 没有的才补"）。调用线程自己也是一个 worker
（原子领活，辅助任务排不上队也能独自完成 —— 从执行器任务内部调用
不会死锁）；异常全部收齐后重抛下标最小的。`load_model_sessions`
的 fp16 上转（按文件）与建会话（按模型）各以 `concurrency =
intra_op_threads`（默认 8）路并行；TimingReport 非线程安全 —— 各模型
耗时记槽位、阶段结束按请求顺序提交。

**瞬时负载形态**：峰值 ≈ 并行路数 × 会话线程数 = 8 × 8 —— 与一次
合成的线程形态同量级，且 kairo 自旋关、ORT 线程不绑核，OS 自由调度
摊负载；本机（高负载多核敏感）实测连续多轮无异常。

**备选记录（绑核时代否决的方案，留档）**：调度层临时放宽 worker
亲和性再还原（改写执行器自己线程的状态，Linux/Windows 各一套）；
引擎自起线程（违反 §1.3）。

### 2.15 动态 INT8 与混合精度包（M5）

**背景**：fp16 分发（§2.13）省的是包体积与磁盘，加载期上转后**运行时
仍是 fp32**，常驻内存不动（fp32 包 ~2.6 GB 权重，其中 RoBERTa 1.2 GB
单文件是最大头）。要缩常驻，只能动"运行时权重表示"。

**选型：动态 INT8（`quantize_dynamic`，逐模型混合精度）**：

- 本机（AVX2 + AVX_VNNI，无 AVX512-FP16）上 MLAS 有 u8s8 内核直配；
  fp16 常驻会被 CPU EP 逐算子 cast 回 fp32，省内存但速度倒退，不取。
- 动态量化不需要校准集；静态 INT8（捕获的真实输入可当校准集）与
  混合精度逐算子搜索留作后续档位（roadmap M5 阶梯）。
- "图简化"单独做不省内存（省的只有图优化后的临时表示），并入 ORT
  Format 议题一并缓行。

**精度裁决交给门控，不靠拍脑袋**：每个模型独立量化、独立过
`tools/gate/` 的精度门控（token 类输出按 fp32 自身重放的噪声地板相对
裁决；特征类 rel-L2 + 余弦；vits 音频 advisory + 听测）。没过的角色
保持 fp32 —— 这就是**逐模型 `weight_format` 覆盖**存在的原因。

**manifest 语义**：包级 `weight_format` 描述默认值（"int8" 表示这是
量化包）；真正的加载行为按逐模型覆盖裁决。格式版本仍是 1 ——
覆盖字段是可选的加法，老引擎读新包不受影响（忽略覆盖时按包级加载，
最多在混合包里把 int8 模型也照 fp32 读……不会发生：int8 模型图里
声明的就是 int8 张量，老引擎按 fp32 读会在校验期报错，不会静默错）。

**加载路径**：int8 外部数据/内联权重由 ORT 原样直读，与 fp32 同一条
路径 —— `ModelLoadRequest::upcast_fp16` 因此从包级开关改成**逐请求**
标志（混合包里 fp16 与 int8/fp32 共存时，上转只该发生在真正 fp16
打包的请求上）。张量抽象不动：int8 只存在于权重层，领域层永远只见
float32 / int64 / bool（§2.1 的约束不破）。

**影响**：权重常驻理论上限 -75%（4 字节/元素 → 1 字节/元素）；实测
见 `docs/performance.md` 量化一节。量化工具在 `tools/deploy/quantize_package.py`
（镜像整包 → 逐角色量化 → 重写 manifest 与 SHA-256 → ORT 复核可加载）。

### 2.16 静态 INT8 档与量化 bin 去重（M5 第二档）

**静态档（`--static`，QDQ）**：`quantize_static` 在图内插
QuantizeLinear/DequantizeLinear，**Conv 从此可量化** —— 这是相对动态档
（只认 MatMul/Gemm 常量 B 侧）的唯一增量。激活 scale 离线校准，校准集
直接复用精度门控的捕获目录（真实流水线输入，零额外采集工具）。
备选方案里被否的：量化器默认算子集（可能含 ConvTranspose，CPU EP 无
int8 内核，会话初始化 NOT_IMPLEMENTED —— `--ops` 显式列出
`Conv,MatMul,Gemm` 把风险挡在量化期）；并发量化（校准期建 ORT 会话 +
shape 推理，实测过一次 SIGSEGV，静态档强制串行）。

**实测裁决（2026-11-11，见 performance.md §静态 INT8）**：激活 int8
误差逐层累积，解码器 KV / roberta / hubert 全部退档；**Conv 静态化只对
vits 有净收益**（-47 MB 常驻），且代价是 QDQ 图加载期优化变慢
（+0.9 s 冷加载）与 vocoder 略慢（DequantizeLinear 开销吃掉 u8s8 收益）
—— "静态档更快"的假设在本机不成立，第二档的价值最终是**内存**而非
速度，是否入包由听测裁决。层内混合精度（attention 保持 fp32）是
规避激活误差的后续路径；QDQ 冷加载成本顺带指到 ORT Format 离线转换。

**量化 bin 去重**：动态档实测发现 t2s 双解码器共享权重量化后产出
**逐字节相同**的 int8 bin（量化是权重的确定性函数）却各存一份
（306 → 576 MB 的包体回归）。工具在量化后按内容哈希去重：留一份，
其余 .onnx 的 `external_data.location` 改指向它 —— 恢复 fp32 包
"一份 bin 双 role 引用"的既有约定（manifest 本就以 role→file 映射，
无引擎改动）。**常驻不降**（每个会话仍各自拷贝外部数据，无 mmap 页
共享），纯包体收益（实测 -288 MB）；QDQ 图因激活 scale 内联位置不同，
bin 不再逐字节相同，去重是机会性特性。

### 2.17 ORT Format 包（M5 冷启动收尾）

**决策**：提供 `convert_package_to_ort.py`（离线图优化 → 自包含 .ort），
**引擎零改动** —— .onnx / .ort 在加载链路上的唯一差异是文件内容本身，
manifest 的 file 字段无扩展名假设，ORT 按 `.ort` 后缀嗅探格式（不依赖
`session.load_model_format` 配置键，规避 ORT 版本差）。fp16 上转扫描
只作用于 fp16 标签的包，ORT Format 包不做 fp16 分发（工具拒绝），两条
路径天然不相交。

**备选与取舍**：备选是"运行期图优化缓存"（SessionOptions 的
optimized_model_filepath 在首次运行时写缓存）——被否：缓存失效管理、
多进程竞争与首跑惩罚都比离线转换复杂，且 ORT Format 本来就是官方
形态。**代价**：(1) .ort 与转换侧 ORT 版本 / 优化级别 / 硬件类绑定
（level-ALL 的 NCHWc 变换是硬件相关的），升级 ORT 必须重转 —— 接受，
因为运行时本来就钉在部署机的一套 ORT 上；(2) .ort 自包含，t2s 双解码器
的"共享 bin 双引用"约定不复存在，包体 +13% —— 接受，换冷启动 -23%
（vits QDQ 会话 -598 ms）；离线优化附带剪掉推理不用的权重（vits
-110 MB），部分对冲。**裁决**：两形态并存，本地磁盘 / 启动敏感的部署
用 .ort，跨版本分发 / 磁盘受限用 .onnx（实测见 performance.md
§ORT Format）。

### 2.18 IOBinding 输出缓冲池（M5 解码热路径第二轮）

§2.11 的"是否值得再上 IOBinding"的答案：**上，且以最小形态**。
难点不在绑定本身，在形状：`BindOutput` 的 Ort::Value 必须与图输出逐维
一致，而解码循环的 KV / token 序列每步都在长。方案是**会话内自学习**——
前两次 run 走普通路径观测输出形状，从相邻差里学"每个输出哪个维度 +
增多少"；此后预测形状、绑定自管缓冲，ORT 直接写入，`Tensor::from_buffer`
接管缓冲（vector 指针换手）—— 宿主侧每步 51 份输出 memcpy 消失。
**零接口变更**：`ModelSession::run` 签名不动，FakeSession、门控捕获
装饰器、域层全部无感。

三条护栏：(1) 预测失准（会话跨合成复用时形状跳变）时 ORT 在内核执行
**之前**拒绝绑定，回退普通路径重学，无半成品状态；(2) 连续两次失准且
从未成功的会话（单次调用的 first_stage_decoder 等）永久禁用绑定——
否则它们会陷入"每步学习→每步预测失败"的永久双跑；(3) 多维同时变的
输出定型为不可预测，整个会话放弃绑定。等价性不依赖开关：两条路径
逐字节一致由集成测试锁定（`IoBindingPoolKeepsResultsIdentical`）。

**备选与不选**：给 `run()` 加输出形状提示参数（域层知道每步的形状）——
被否：接口变更波及假会话 / 门控装饰器 / 域层三处，只为省两次观测 run，
自学习把复杂度封在会话内部。**裁决**（i9-13900KF，ORT 1.31，
performance.md §IOBinding）：长文本 ms/token **20.24 → 17.81（-12%）**、
短文本 -2%，RSS 持平——≥5% 的 kill criterion 达标，**默认开**
（`SessionConfig::enable_io_binding`，单次调用会话的失败成本可忽略）。

### 2.19 分段流式合成（M6：分句 + 真分块输出）

**背景**：M6 之前的 `synthesize_streaming()` 是"伪流式"——整篇文本一次
解码、一次声码，最后把**整段音频**作为一个块推进通道。首块延迟 == 端到端
延迟，流式接口没有兑现任何低延迟承诺。

**约束**：VITS 声码器是单次非增量 ONNX 图，token 级的句内流式需要重新导出
模型（归 M6 评估，见 performance.md 的 KV 重导出一节）——不改模型的前提下，
延迟的天花板就是"一段话的合成时间"。

**决策**：**分句合成**。整段文本切成句段（`genie::text::split_for_synthesis`，
规则见其头注释），逐段走完 前端 → T2S → VITS 后**立刻**推一段音频；
除末段外每段尾部追加可配置的句间静音（默认 150 ms，模拟自然停顿）。
同步 `synthesize()` **不分句** —— 上游一致性优先（§24.1：上游长文本
本来就要求调用方自己分句喂）；分句只属于流式路径，且单段口径与
"调用方分句逐句喂"完全一致（§24.3 已确认韵律等价）。

配套决策：

- **单段合成核心共享**：`synthesize_one()` 是同步与流式两条路径共用的
  阶段编排（前端 → BERT → 解码 → 声码）。同步路径抽出来必须逐字节不变
  —— golden / 回归测试锁定。语言检测是"整条请求一次"的开销，不进单段。
- **流尾元数据（`StreamFooter`）挂在通道上**：音频块只传波形，整条流的
  分阶段耗时与失败原因收在 footer（生产者 close 前设置，消费者排空后读）。
  契约：**通道关闭 ⇒ footer 可读**。协作式取消不算错误（调用方自己发起的，
  通道关闭本身就是信号），只有真失败才填 `footer.error`。
- **有界通道的背压**：通道满时生产者短轮询（`try_push` + 10 ms 小睡），
  消费者不腾位就等 —— 这就是背压；轮询间隙随时响应取消。不用
  `push(timeout)`：它按值收参、超时返回时实参已被移走，重试反而要多一份
  底稿拷贝。
- **守护任务兜底关通道**：取消有一个"永远漏收口"的洞 —— 任务还在队列里
  没开跑就被取消时，kairo 走 `RequestedBeforeStart`，任务体**根本不执行**，
  没有人去 close 通道，消费者的"读到关闭为止"循环会永远等下去。每条流
  挂一个守护任务：等合成任务落定（挂在 future 上，不烧 CPU）后，若通道
  未关则补 footer + close。正常路径下它是空转。
- **`SchedulerConfig::stream_capacity` 转正**：这个字段从写下那天起就没被
  消费过；流式通道容量现在从它取（`Scheduler::stream_capacity()`）。

**备选与不选**：
- *token 级句内流式*（增量 vocoder / 流式声码图）—— 需要改模型导出，
  与"不改模型资产"的 M6 范围冲突，评估结论留档在 performance.md；
- *固定时长切块*（按字符数硬切句段）—— 不看标点的切法会把句子拦腰斩断，
  韵律明显劣化；本实现只在超长且无标点时才硬切；
- *通道传富对象*（`AudioChunk{samples, is_last, timing…}`）—— 消费侧要为
  每个块解包，而耗时/错误本质是"整条流一次"的信息，塞进 footer 更贴合
  单消费者的读取模式。

### 2.19.1 增量会话（M6：StreamingSynthesis）

一次性 `synthesize_streaming(text)` 与增量 `start_streaming()` **共用同一个
worker**（`Engine::run_stream_session`）——一次性封装只是
`start + pin_language + feed + finish` 的语法糖。一份逻辑意味着取消收口、
footer 契约、背压策略只需要修对一次。

- **增量抽取与整段切分同源**：`feed()` 反复调用
  `genie::text::complete_prefix`（与 `split_for_synthesis` 同一套状态机、
  只走到第一刀）——逐字节喂入与一次性喂入产出**相同段序列**，由
  单测锁定。已知差异两处（增量场景的本质）：段边缘空白不修剪（前端
  规范化吃掉，音频等价）；文本末尾的残余不交付，留给 `finish()` 冲刷。
- **有界段队列**（容量 8）：合成跟不上时 `feed()` 阻塞 —— 背压从音频
  通道延伸到文本入口；阻塞期间取消立刻放弃。
- **单流记账持公开对象**：踩过的坑 —— 一次性封装里会话是局部变量，
  封装返回就析构；析构即取消的话，worker 还在队列里没开跑就被扼杀
  （表现为"没有音频、通道秒关"）。`Engine::Impl` 因此持
  `shared_ptr<StreamingSynthesis>`（公开对象）保活；活跃会话未终结时
  再开流抛 `Busy`（新错误码，C ABI 同步 `GENIE_ERROR_BUSY = 11`）。
- **语言口径**：一次性封装对整段检测后 `pin_language`（与同步路径
  同口径）；纯增量喂入不预设时以**第一个句段**检测、全流固定 ——
  增量场景拿不到全文，这是必要妥协（文档写明）。

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


**校验与信任模式**：manifest 登记的 SHA-256 是完整性契约的落地
（`VoicePackage::load(root, verify_hash)`）。校验是逐文件全量读盘，
成本与包体积线性 —— 默认开启；`EngineConfig::verify_voice_package`
可以显式关掉（CLI / bench 的 `--no-verify`），用于包由受信流程产出、
冷启动敏感的场景。跳过校验只跳过**哈希计算**，文件存在性检查仍然生效：
缺文件与坏 manifest 的报错路径不变。

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
所需资源版本、可选能力（V2ProPlus / RoBERTa）、权重精度
（`weight_format`，缺省 fp32）。

**校验值**是必须的：模型文件被替换是最常见的部署事故，
而 ONNX 加载失败时的报错信息通常看不出根因。

**weight_format（fp16 权重分发）**：manifest 声明
`"weight_format": "fp16"` 时，外部权重用 HF 上游的 fp16 打包布局
（图声明 fp32、文件存一半大小的 fp16），引擎加载期在内存里上转 FP32
再建会话（见 §2.13）。缺省 fp32；自包含模型（RoBERTa 1.2 GB 单文件）
不受影响。

> 格式细节随 M1 的 model loader 落地，本节届时更新。

---

## 6. 模型授权

- `third_party/genie-tts-models/` 下的六个模型来自上游 Genie-TTS 仓库（MIT），随源码发布。
- **用户自己的音色模型不入库**，其授权由相应作者决定。
- `chinese-hubert-base.onnx` / `RoBERTa.onnx` / 说话人验证模型**不由本项目分发**，
  由用户在模型包中自行提供。
