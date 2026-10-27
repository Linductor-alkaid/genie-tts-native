# 路线图

状态标记：⏳ 未开始 ｜ 🚧 进行中 ｜ ✅ 完成 ｜ ⛔ 明确不做

---

## 里程碑总览

| 阶段 | 名称 | 核心交付 | 优先级 | 状态 |
|---|---|---|---|---|
| M0 | 上游审计与基线建立 | 模型结构、依赖清单、参考数据、性能基线 | P0 | 🚧 |
| M1 | C++ Runtime 基础设施 | CMake、ONNX 封装、模型加载、张量管理 | P0 | 🚧 |
| M2 | 文本与音频前处理迁移 | 文本规范化、G2P、参考音频特征处理 | P0 | 🚧 |
| M3 | GPT-SoVITS 完整推理迁移 | 语义生成、声学合成、Vocoder、端到端输出 | P0 | ⏳ |
| M4 | 正确性验证与原生 SDK | 一致性测试、C++ API、CLI、跨平台构建 | P0 | ⏳ |
| M5 | CPU 推理性能优化 | 解码优化、线程调优、内存复用、量化 | P1 | ⏳ |
| M6 | 低延迟流式 TTS | 增量生成、异步流水线、取消机制 | P1 | ⏳ |
| M7 | 发布与工程化 | C ABI、安装包、文档、CI、性能报告 | P1 | ⏳ |

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
- ⏳ 锁定上游 commit（作为可复现基线）

**M0.2 Golden Dataset**

- ⏳ 语料构造：中文短句 / 长句、英文、中英混读、日文、韩文、特殊文本、边界输入
- ⏳ 逐级参考输出：规范化文本 → 音素 → BERT 特征 → 语义 token → PCM
- ⏳ 固定随机种子；非确定性环节用分阶段容差而非逐样本比对

**M0.3 性能基线**

- ⏳ 模型加载时间、冷/热启动延迟
- ⏳ 首个可播放音频块生成时间、完整合成时间、RTF
- ⏳ 各 ONNX 子模型耗时
- ⏳ 峰值 RSS、CPU 利用率、线程数、内存分配次数

**出口条件**：推理调用链、依赖清单、Golden Dataset、性能报告全部完成。

### M1 C++ Runtime 基础设施 — 🚧

- ✅ 项目结构与 CMake 体系
- ✅ ONNX Runtime 封装（RAII、SessionOptions、CPU EP、异常转换）
- ✅ ONNX 模型元数据检查、输入输出张量类型与形状验证
- ✅ 张量抽象（`genie::Tensor`）
- ⏳ 模型包（voice package）加载与 manifest 校验
- ⏳ 多 Session 生命周期管理
- ⏳ 采样率 / 校验值校验

### M2 文本与音频前处理迁移 — 🚧

- ✅ 音频：WAV 编解码、重采样
- ✅ 音素符号表（732 个符号，与上游逐字一致）
- ✅ Unicode 规范化、标点处理
- ✅ 数字展开（阿拉伯数字 → 中文读法）
- ✅ 拼音与音素转换（41,923 字与 pypinyin 全量比对一致）
- ✅ 音素 ID 映射
- ✅ 日期 / 时间 / 电话 / 量词展开（对齐上游 `TextNormalizer`，见 upstream-audit §15）
- ✅ 繁简转换
- ⏳ 变调（ToneSandhi）与儿化（Erhua）
- ⏳ 中文分词（`word2ph` 目前按单字统计）
- ⏳ 英文 G2P（留成可插拔接口）
- ⏳ 日文 G2P（留成可插拔接口）
- ⏳ 韩文 G2P（留成可插拔接口）
- ⏳ 中英混合文本切分
- ⏳ 中文 RoBERTa 特征（C++ tokenizer + 特征提取 + 对齐）
- ⏳ 参考音频特征（HuBERT `ssl_content`）

> **本阶段是整个项目最大的工作量来源。** 计划书估计 1～3 周，
> 实际很可能更久 —— 中文 `g2pM` 的多音字模型与韩文 `g2pk2` 都没有成熟的 C++ 等价物。
>
> 当前进度：中文主链路（符号表 → 规范化 → 拼音 → 音素 ID）已经打通，
> 并用 `genie-tts g2p` 单独可查。剩下的偏差见 `docs/upstream-audit.md` §9.4。

### M3 GPT-SoVITS 完整推理迁移 — 🚧

- ✅ T2S 链路跑通（`tests/integration/`）
- ✅ 自回归 token 生成、KV cache 递推（按名字绑定 + present→past 映射）
- ✅ EOS 检测与最大长度限制（逐行对齐上游语义）
- ✅ 声学合成（VITS）与端到端流水线
- ✅ 目标 API：`genie::Engine::synthesize()` / `synthesize_streaming()`
- ✅ 参考语义条件构建（参考音频的 phoneme / bert / ssl 特征）
- ⏳ 文本与音素条件编码（text_bert）
- ⏳ 随机数生成与确定性测试
- ⏳ V2ProPlus 适配（`prompt_encoder` / `ge` / `ge_advanced`）

### M4 正确性验证与原生 SDK — 🚧

- ✅ 四级测试体系（unit / component / integration / c）
- ✅ C ABI 共享库（纯 C 头文件，C 编译器编译的测试守着）
- ✅ `genie-tts` CLI（version / info / g2p / inspect / synth）
- ⏳ 内存泄漏、模型重复加载、异常恢复、长文本稳定性、连续推理检查
- ✅ regression 层（Golden Dataset：模型 I/O 契约快照，见 `docs/regression.md`）
  - 锁住每个模型收到的全部输入（名字 / dtype / 形状 / 秩）
  - 立刻抓出了 `text_seq` / `ref_bert` / `text_bert` 三处**秩错**
- ⏳ 跨平台构建（Linux x86-64 → Windows x86-64 → Linux ARM64）
- ⏳ **v0.1.0 发布**

### M5 CPU 端侧性能优化 — ⏳

- ⏳ 子模型级性能分析（各阶段耗时占比、算子耗时、Cache Miss、SIMD 利用率）
- ⏳ KV Cache 复用、预分配解码缓冲区、减少 Host 内存拷贝
- ⏳ ONNX Runtime 参数搜索（线程数、图优化、执行模式、内存 arena/pattern、CPU 亲和性、ORT Format）
- ⏳ 模型常驻、参考音频缓存、文本特征缓存、缓冲区复用、模型共享
- ⏳ 量化（FP32 → 图简化 → 权重量化 → 动态 INT8 → 静态 INT8 → 混合精度）
- ⏳ CPU 微架构优化（AVX2/AVX-512、NEON、内存对齐、NUMA）

> ⚠️ **采样器优化不在此列** —— 采样在 ONNX 图内，主机侧无可优化对象。
> 详见 [upstream-audit.md §5](upstream-audit.md)。

### M6 低延迟流式 TTS — ⏳

- ⏳ 增量文本输入与分段
- ⏳ 异步合成流水线
- ⏳ 音频分块输出（PCM chunk 回调 / 阻塞读取）
- ⏳ 取消与抢占（StopToken、请求 ID、取消状态）
- ⏳ 流式性能测试（TTFA、缓冲区欠载、分句静音、爆音、取消响应时间、并发延迟）

> 调度与并发基础设施由 kairo 提供，见 [architecture.md §2.4](architecture.md)。

### M7 发布与工程化 — ⏳

- ⏳ 稳定 C++ SDK
- ⏳ C ABI
- ⏳ `find_package()` 支持
- ⏳ 静态库 / 动态库构建
- ⏳ Linux / Windows 构建完善
- ⏳ ASan / UBSan / TSan 常态化
- ⏳ 模型转换与模型包校验工具
- ⏳ 部署、API、性能优化、故障排查文档
- ⏳ CPU Benchmark 报告
- ⏳ **v0.2.0 / v0.3.0 发布**

---

## 明确不纳入首轮

- GPT-SoVITS **V3 / V4** 或其他架构（会破坏性能优化的稳定基线）
- **ncnn** 移植（等 ONNX C++ Runtime 稳定并获得完整性能数据后单独立项）
- GPU 后端（CUDA / TensorRT）
