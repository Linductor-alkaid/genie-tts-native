# 路线图

状态标记：⏳ 未开始 ｜ 🚧 进行中 ｜ ✅ 完成 ｜ ⛔ 明确不做

---

## 里程碑总览

| 阶段 | 名称 | 核心交付 | 优先级 | 状态 |
|---|---|---|---|---|
| M0 | 上游审计与基线建立 | 模型结构、依赖清单、参考数据、性能基线 | P0 | 🚧 |
| M1 | C++ Runtime 基础设施 | CMake、ONNX 封装、模型加载、张量管理 | P0 | ✅ |
| M2 | 文本与音频前处理迁移 | 文本规范化、G2P、参考音频特征处理 | P0 | ✅ |
| M3 | GPT-SoVITS 完整推理迁移 | 语义生成、声学合成、Vocoder、端到端输出 | P0 | ✅ |
| M4 | 正确性验证与原生 SDK | 一致性测试、C++ API、CLI、跨平台构建 | P0 | 🚧 |
| M5 | CPU 推理性能优化 | 解码优化、线程调优、内存复用、量化 | P1 | ✅ |
| M6 | 低延迟流式 TTS | 增量生成、异步流水线、取消机制 | P1 | 🚧 |
| M7 | 发布与工程化 | C ABI、安装包、文档、CI、性能报告 | P1 | 🚧 |

**阶段边界**：M4 完成即形成第一个可用版本；M5–M6 完成后形成面向实时交互的优化版本。

---

## 各阶段进度

### M0 上游审计与性能基线 — 🚧

**M0.1 源码与依赖审计**

- ✅ 推理调用图与全部 ONNX Session
- ✅ 各模型的输入、输出、数据类型、动态维度（[upstream-audit.md §3](upstream-audit.md)）
- ✅ 自回归解码、采样、停止条件、缓存实现的确认
- ✅ 中英日韩前处理依赖清单
- ✅ 参考音频编码、重采样、裁剪与缓存机制
- ✅ 运行时依赖 vs 离线转换依赖的区分
- ✅ MIT 许可证与第三方授权边界
- ✅ ONNX Runtime 版本约束的记录（作为待复测信息，不永久钉死）
- ✅ 锁定上游 commit（基线 `d347fd0f`，见 upstream-audit 各节）

**M0.2 Golden Dataset**

- 🚧 语料构造：中文短句 / 长句、中英混读、英文、日文、韩文、特殊文本、边界输入
  —— 文本层的语料与黄金语料已就位（normalizer / g2pM / jieba / sandhi /
  english / korean / japanese / language_split，共 8 套 `tests/golden/*.json`）；
  音频层有了**精度门控捕获**（`tools/gate/`：真实流水线逐模型 I/O 落盘 +
  分级容差裁决，见 `docs/regression.md` 精度门控一节），与 Python 上游的
  PCM 级黄金比对待权重可达
- 🚧 逐级参考输出：规范化文本 → 音素 → BERT 特征 → 语义 token → PCM
  —— 前四级已逐级锁定；PCM 级由精度门控接管（fp32 自基线 +
  噪声地板相对裁决），与 Python 上游的比对待权重可达
- ✅ 非确定性环节用分阶段容差而非逐样本比对（AGENTS §5.3 落地为
  `tools/gate/tolerances.json`：token 逐位一致按噪声地板相对、连续张量
  rel-L2 + 余弦、音频 advisory）—— 2026-11-11 实测：采样在图内、种子在
  会话内固定，同进程同配置**逐 run 序号可复现**（fp32 自比 WAV 逐字节
  相同），跨进程/跨 ORT 版本的近平局 token 翻转是稳定的构建间偏差
  （地板 ≈0.991）；`SynthesisOptions::seed` 仍是预留字段（upstream-audit §5）

**M0.3 性能基线**

- ✅ 模型加载时间、冷启动延迟 —— `Engine::load_timing()`（PR #32 采集端），
  实测见 `docs/performance.md` 冷启动表（v2ProPlus 包 7.6 s，其中音色包
  SHA-256 校验占一半）
- ✅ 完整合成时间、RTF —— `genie-bench`（三档默认文本 × N 次统计 +
  warmup 消抖）；首个可播放音频块时间 = 端到端（当前流式整段推送，
  M6 真分块后单独测）
- ✅ 各 ONNX 子模型耗时 —— 合成阶段表（`t2s/encoder`、`t2s/first_stage`、
  `t2s/stage_decoder`×N、`vocoder`、`text_bert`）；基线实测 stage_decoder
  占合成 65%–84%，每 token 18–26 ms 且随序列长度上升（KV cache 逐轮
  变长、宿主侧每步全量拷贝 —— M5 第一轮优化后降至 13–18 ms/token，
  见 `docs/performance.md`）
- ✅ 峰值 RSS、CPU 利用率（进程级均值）、线程数 —— genie-bench 输出；
  ⏳ 内存分配次数（需 allocator hook，M5 前补）、CPU 利用率时间线采样

**出口条件**：推理调用链、依赖清单、Golden Dataset、性能报告全部完成。

### M1 C++ Runtime 基础设施 — ✅

- ✅ 项目结构与 CMake 体系
- ✅ ONNX Runtime 封装（RAII、SessionOptions、CPU EP、异常转换）
- ✅ ONNX 模型元数据检查、输入输出张量类型与形状验证
- ✅ 张量抽象（`genie::Tensor`）
- ✅ 模型包（voice package）加载与 manifest 校验
- ✅ 多 Session 生命周期管理
- ✅ 采样率 / 校验值校验
- ✅ `find_package()` 支持与安装规则（`cmake/genieConfig.cmake.in`）
- ✅ 静态库 / 共享库双形态（`genie_core` / `genie` / `genie_c`）

### M2 文本与音频前处理迁移 — ✅

- ✅ 音频：WAV 编解码、重采样
- ✅ 音素符号表（732 个符号，与上游逐字一致）
- ✅ Unicode 规范化、标点处理
- ✅ 数字展开（阿拉伯数字 → 中文读法）
- ✅ 拼音与音素转换（41,923 字与 pypinyin 全量比对一致）
- ✅ 音素 ID 映射
- ✅ 日期 / 时间 / 电话 / 量词展开（对齐上游 `TextNormalizer`，见 upstream-audit §15）
- ✅ 繁简转换
- ✅ 变调（ToneSandhi）与儿化（Erhua）（2026-10-30 接上 jieba 分词；
  jieba 路径韵母序列 42/44 与上游一致，见 upstream-audit §9.6）
- ✅ 中文分词（jieba 直译，oracle 103/103，见 upstream-audit §9.8）
- ✅ 多音字 g2pM（2026-10-28：C++ 前向复刻，oracle 47/47，见 upstream-audit §9.7）
- ✅ 词级读音修正（上游 `polyphonic.pickle` 4.5 万词，随 g2pm 容器携带）
- ✅ 英文 G2P（2026-10-29：规范化 + TweetTokenizer + CMU 词典 + POS 标注 + GRU 兜底，
  oracle 94 句的三项逐条一致，见 upstream-audit §17）
- ✅ 日文 G2P（2026-10-29：OpenJTalk 运行时内置 + 全上下文标签解析，
  oracle 124/124 逐音素一致；词典 103 MB 不入仓库、由音色包给出，
  见 upstream-audit §19）
- ✅ 韩文 G2P（2026-10-29：g2pk2 全链路移植，oracle 53/53 逐音素一致，
  mecab-ko 形态标注未移植，见 upstream-audit §18）
- ✅ 中英混合文本切分（2026-10-28：split_language 移植 + 逐块 TextNormalizer + 按块 RoBERTa，见 upstream-audit §6.3；英文块已接入英文 G2P，2026-10-29）
- ✅ 中文 RoBERTa 特征（C++ tokenizer + 特征提取 + 对齐，见 upstream-audit §14）
- ✅ 参考音频特征（HuBERT `ssl_content`，见 upstream-audit §14.4）

> **本阶段是整个项目最大的工作量来源。** 计划书估计 1～3 周，
> 实际很可能更久 —— 中文 `g2pM` 的多音字模型与韩文 `g2pk2` 都没有成熟的 C++ 等价物。
>
> 当前状态：四种语言的中文/英文/日文/韩文 G2P 全部落地，
> 偏差逐条记录在 `docs/upstream-audit.md` §9.4。
> 剩下的已知缺口是 **jieba 的 posseg HMM（POS 标注）** —— 它会决定最后
> 一类变调差异（§9.6 的"叠词轻声"分支）。

### M3 GPT-SoVITS 完整推理迁移 — ✅

- ✅ T2S 链路跑通（`tests/integration/`）
- ✅ 自回归 token 生成、KV cache 递推（按名字绑定 + present→past 映射）
- ✅ EOS 检测与最大长度限制（逐行对齐上游语义）
- ✅ 声学合成（VITS）与端到端流水线
- ✅ 目标 API：`genie::Engine::synthesize()` / `synthesize_streaming()`
- ✅ 参考语义条件构建（参考音频的 phoneme / bert / ssl 特征）
- ✅ 文本与音素条件编码（text_bert，见 upstream-audit §14）
- ✅ V2ProPlus 适配（`prompt_encoder` / `ge` / `ge_advanced`，见 upstream-audit §12）
- ✅ 采样器归属确认：采样在 ONNX 图内，主机侧没有采样器
  （upstream-audit §5）；`SynthesisOptions::seed` 是预留字段，
  确定性测试要等图内行为确认后才有意义

### M4 正确性验证与原生 SDK — 🚧

- ✅ 四级测试体系（unit / component / integration / c）
- ✅ C ABI 共享库（纯 C 头文件，C 编译器编译的测试守着）
- ✅ `genie-tts` CLI（version / info / g2p / normalize / inspect / synth）
- ✅ regression 层（Golden Dataset：模型 I/O 契约快照，见 `docs/regression.md`）
  - 锁住每个模型收到的全部输入（名字 / dtype / 形状 / 秩）
  - 立刻抓出了 `text_seq` / `ref_bert` / `text_bert` 三处**秩错**
- ✅ ASan / UBSan / TSan 常态化（CI 两个 job：`asan + ubsan` 与 `tsan`；
  TSan 只在 CI 跑 —— 本地同时开 asan + tsan 会把 3 GB 的沙箱 OOM 掉）
- ✅ 跨平台构建：Linux x86-64 ✅ / Windows x86-64 ✅（MSVC 默认不编 OpenJTalk）
- ⏳ 跨平台构建：Linux ARM64（交叉编译 CI）
- ⏳ 内存泄漏、模型重复加载、异常恢复、长文本稳定性、连续推理检查
- ✅ **v0.1.0 / v0.2.0 / v0.3.0 / v0.3.1 发布**

### M5 CPU 端侧性能优化 — ✅

> **收官注（2026-11-11）**：M5 按"本机硬件范围"结题 —— 动作项全部
> 落地或有实测裁决的 ⛔ 记录；唯一挂账是 **CPU 亲和性**（等坏核排查，
> 硬件问题不属于本里程碑）。跨硬件的剩余空间（AVX-512 系、NEON）
> 随目标硬件另启。

- ✅ 子模型级性能分析（各阶段耗时占比、算子耗时）
  —— ✅ 阶段占比与 ms/token 归一（genie-bench 输出 iterations，
  `docs/performance.md`）；✅ ORT 内建 profiler 算子级耗时（2026-11-11：
  `SessionConfig::profile_output_dir` + `genie-bench --profile` +
  `summarize_profile.py`；stage_decoder 算子分布见 performance.md
  §算子级耗时 —— KV 图内管理 Transpose+Concat ≈21% 是 MatMul 外最大块，
  宿主侧拷贝在 profiler 不可见）；Cache Miss / SIMD 利用率未做（无
  轻量采集口子，明确挂起）
- ✅ KV Cache 复用、预分配解码缓冲区、减少 Host 内存拷贝 ——
  ✅ 第一轮：解码循环 move 语义、输入零拷贝包装、输出单次拷贝
  （ms/token -23%~-31%，`docs/performance.md`）；
  ✅ 第二轮：IOBinding 输出缓冲池（2026-11-11，会话内自学习形状增长，
  宿主侧每步 51 份输出拷贝消失，长文本 ms/token -12% 默认开，
  `docs/performance.md` §IOBinding）—— 宿主侧可摘的拷贝摘完，图内
  KV concat/Transpose（≈21%）需改模型导出，归 M6 流式一并评估
- ✅ ONNX Runtime 参数搜索（线程数、图优化、执行模式、内存 arena/pattern、
  CPU 亲和性、ORT Format）—— ✅ 线程 × 自旋（默认 8 线程、自旋关）、
  图优化级别 / arena / 内存模式（默认全开，扫描无稳健更优组合，
  `docs/performance.md`）；
  ✅ ORT Format（2026-11-11：`convert_package_to_ort.py`，冷加载
  -23% / QDQ 会话 -598 ms，包体 +13%，`docs/performance.md` §ORT
  Format）；执行模式 Parallel 明确不采（与串行解码形态不匹配）；
  ⛔ CPU 亲和性（等坏核排查 —— 硬件问题，M5 唯一挂账）
- ✅ 冷启动 —— ✅ 音色包信任模式（跳过 SHA-256，冷启动 -51%）；
  ✅ fp16 权重分发（包体积 -21%；加载期整文件上转 + ORT 内存替换，
  冷启动只比 fp32 多 ~0.7 s，`docs/performance.md`）；✅ 会话内并行预打包
  （opt-in：-8% 冷启动换 +0.5 GB 常驻内存）；✅ 跨会话并行加载 —— kairo 升至 53ee306（默认池 OS 自由调度）后落地
  （parallel_for fork-join，8 路；会话段 ~2.9×，冷启动 3.77 → ~1.8 s），
  见 architecture.md §2.14
- ✅ 内存占用 —— ✅ 加载期会话释放 + 还页（常驻 -1.2 GB，
  `docs/performance.md` §内存）；✅ 动态 INT8 混合精度包（常驻再 -0.93 GB
  至 1.25 GB，见 §量化）；⛔ t2s 双解码器共享权重（ORT 探针实测
  省不出来；包体层面的去重已落，常驻不变）；✅ 输出缓冲预分配
  （IOBinding 轮，实测 RSS 持平 —— KV 本体由 ORT arena 管理，无进一步
  空间）
- ✅ 模型常驻、参考音频缓存、文本特征缓存、缓冲区复用
  —— 参考音频特征已在加载期算好常驻（engine 构造期）；模型常驻由
  量化档兑现（权重 4 → 1 字节/元素）；缓冲区复用由 IOBinding 轮 +
  ORT arena 覆盖；⛔ 文本特征缓存（特征逐请求变化，无复用价值）
- ✅ 量化（FP32 → 图简化 → 权重量化 → 动态 INT8 → 静态 INT8 → 混合精度）
  —— ✅ **动态 INT8 第一档落地**（2026-11-11）：逐模型门控 + 混合精度包
  （`WeightFormat::Int8` + 逐模型覆盖，`tools/deploy/quantize_package.py`）。
  实测：常驻 **2.18 → 1.25 GB（-43%）**、冷加载 -26%、短文本 RTF -35%
  （u8s8/AVX_VNNI），`docs/performance.md` §动态 INT8 量化。
  roberta/hubert int8 的 A/B 听测 **2026-11-11 通过**（音色/清晰度/自然度
  无差异，容差按实测上界放宽并留档，tolerances.json）；
  ✅ **静态 INT8 第二档结案**（2026-11-11）：QDQ + Conv 量化，校准集直接吃
  genie-gate 捕获（`--static --calibration-dir`）。裁决 —— **激活 int8 量化
  全面退档**（解码器 KV rel 0.33 / roberta rel 0.378 / hubert rel 0.736，
  均远超包络，`docs/performance.md` §静态 INT8）；Conv 静态化唯一幸存者
  **vits** 听测通过入包（-47 MB 常驻，冷加载 +0.9 s，e2e token 一致率 1.0，
  2026-11-11 定版：aggressive + vits 静态 + bin 去重 ≈ 1.34 GB 包）；
  量化 bin 自动去重（t2s 双解码器共享权重，包体再 -288 MB，常驻不变）；
  ✅ ORT Format 摊薄 QDQ 冷加载（2026-11-11：pkg-final-ort 冷加载
  2477→1906 ms，vits 会话 -598 ms，见 performance.md §ORT Format）；
  ⛔ roberta 层内混合精度（动态档已过听测定版、静态档被拒 —— 精度余量
  已够，再切没有收益面）；⛔ speaker_verification 禁量化（cos 0.264，
  音色崩坏，实测记录在案）
- ✅ CPU 微架构优化 —— ✅ AVX2/AVX_VNNI 路径已由 ORT/MLAS 内核与量化档
  收益兑现（u8s8）；⛔ NEON（ARM 侧归 M4 的 ARM64 交叉编译，本机 x86
  无对象）、⛔ NUMA（单插槽台式机无对象）、⛔ AVX-512 系（依赖带
  AVX512-FP16/VNNI 的下一代硬件，本机只有 AVX2+AVX_VNNI —— 随目标
  硬件另启）；内存对齐由 ORT arena 管理，无自管缓冲

> ⚠️ **采样器优化不在此列** —— 采样在 ONNX 图内，主机侧无可优化对象。
> 详见 [upstream-audit.md §5](upstream-audit.md)。

### M6 低延迟流式 TTS — 🚧

- ✅ 异步合成流水线（`Engine::synthesize_streaming()`：任务跑在 kairo
  调度器上，调用线程只消费音频通道）
- ✅ 取消与抢占（`Engine::cancel_streaming()` + CancellationToken）
- ⏳ 增量文本输入与分段
- ⏳ 音频分块输出（当前是整段音频一次性推入通道）
- ⏳ 流式性能测试（TTFA、缓冲区欠载、分句静音、爆音、取消响应时间、并发延迟）

> 调度与并发基础设施由 kairo 提供，见 [architecture.md §2.4](architecture.md)。

### M7 发布与工程化 — 🚧

- ✅ 稳定 C++ SDK（`genie::core` / `genie::genie`）
- ✅ C ABI（`genie_c` 共享库，纯 C 头文件）
- ✅ `find_package()` 支持
- ✅ 静态库 / 动态库构建
- ✅ Linux / Windows 构建（CI 各一个 job）
- ✅ ASan / UBSan / TSan 常态化
- ✅ 模型转换与模型包校验工具（`tools/deploy/`）
- ✅ 部署与 API 文档（`docs/deployment.md`、`docs/usage.md`）
- ⏳ 故障排查文档、性能优化文档
- ✅ CPU Benchmark 工具与首份报告 —— `tools/bench/`（genie-bench）+
  `docs/performance.md`（M0.3 基线；换硬件 / 换 ORT 重测后更新该文档）
- 🚧 **v0.2.0 发布** ✅ → **v0.3.0 发布**（文档同步 + jieba 生产路径）✅ →
  **v0.4.0 发布**（性能优化专版：M0.3 基线 + M5 三轮优化；CHANGELOG /
  路线图 / 发布检查单同步）

---

## 明确不纳入首轮

- GPT-SoVITS **V3 / V4** 或其他架构（会破坏性能优化的稳定基线）
- **ncnn** 移植（等 ONNX C++ Runtime 稳定并获得完整性能数据后单独立项）
- GPU 后端（CUDA / TensorRT）
