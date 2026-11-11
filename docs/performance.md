# 性能基线与优化记录（M0.3 / M5）

本文记录整条推理链路的**分阶段耗时**：M0.3 的首份基线，以及 M5 第一轮
优化（宿主拷贝消除 + 会话线程调优）的前后对比与参数扫描。计时采集端是
`SynthesisResult.timing` / `Engine::load_timing()`（见 `docs/architecture.md`
§2.10），数字由 `genie-bench` 采集；JSON 原始记录在
[`benchmarks/`](benchmarks/)（基线 + 本轮全部扫描点，每条含环境、逐 run
样本与解码步数）。

> 数字**只在同代硬件 + 同 ORT 版本上可比**。换机器、换 ORT 版本、
> 换音色包都要重测，重测后替换本文表格并把旧表挪到文末"历史"。

## 测量环境

| 项 | 优化前（基线） | 优化后（M5 第一轮） |
|---|---|---|
| 引擎 | v0.3.0（master a905acc）+ 分阶段计时（#32） | + 宿主拷贝消除 + 会话线程调优 |
| 日期 | 2026-11-06 | 2026-11-06 |
| CPU | 13th Gen Intel Core i9-13900KF（24 核 32 线程，P+E 混合） | 同左 |
| OS | Linux 6.8.0-124-generic x64 | 同左 |
| 构建 | `cmake --preset release`（-O3 -DNDEBUG） | 同左 |
| 推理后端 | ONNX Runtime 1.31.0 CPU EP | 同左 |
| 会话线程 | `intra_op_threads = 8`，自旋 = ORT 默认（开） | `intra_op_threads = 8`，**自旋关**（M5.3 扫描后定） |
| 音色包 | feibi，`model_version: v2ProPlus`，32 kHz，2.4 GB | 同左 |
| 方法 | 每条文本 warmup 1 次 + 测量 5 次；ms/token 按 `t2s/stage_decoder` 的 count 归一 | 同左 |

## 复现

```bash
cmake --preset release -DGENIE_ONNXRUNTIME_ROOT=<ort 包> && cmake --build --preset release
./build/release/tools/genie-bench \
    --model <音色包目录> --runs 5 --warmup 1 --threads 8 \
    --json baseline.json     # 默认 spinning=0；旧默认用 --spinning 1
```

## 基线（2026-11-06，优化前 master，JSON：`benchmarks/baseline-2026-11-06.json`）

### 冷启动（Engine 构造，单次，v2ProPlus 包）

| 阶段 | ms | 占比 | 说明 |
|---|---|---|---|
| `load/voice_package` | 3837 | 50.9% | **manifest 校验 + 全部模型文件的 SHA-256**（2.4 GB） |
| `load/roberta`（含分词器） | 936 | 12.4% | 其中 ORT 会话 936 ms，分词器词表 <1 ms |
| `load/vits` | 857 | 11.4% | |
| `load/t2s_first_stage_decoder` | 423 | 5.6% | |
| `load/t2s_stage_decoder` | 397 | 5.3% | |
| `load/speaker_verification` | 286 | 3.8% | V2ProPlus |
| `load/global_embeddings` | 256 | 3.4% | SV → prompt_encoder，上游留到首推这里提前到加载期 |
| `load/chinese_hubert` | 165 | 2.2% | |
| `load/jieba` | 61 | 0.8% | |
| `load/reference` | 98 | 1.3% | 参考音频加载 + 双采样率重采样 |
| `load/reference_ssl` | 74 | 1.0% | HuBERT 参考特征 |
| `load/reference_bert` | 46 | 0.6% | RoBERTa 参考特征 |
| `load/t2s_encoder` | 21 | 0.3% | |
| `load/english_frontend` | 25 | 0.3% | CMU 词典等数据文件 |
| `load/prompt_encoder` | 45 | 0.6% | |
| `load/g2pm` | 9 | 0.1% | |
| **合计** | **7536** | | 未单独计时的装配开销 ≈ 各阶段之和与 total 的差 |

音色包完整性校验（SHA-256 扫 2.4 GB）占冷启动一半。**信任模式已落地**
（`EngineConfig::verify_voice_package` / `--no-verify`，默认仍校验）：
实测冷启动 7406 → 3589 ms（-51.5%），`load/voice_package` 3789 → 0.2 ms
—— 上表数字为默认（校验开启）口径。

**fp16 权重包**（`weight_format: fp16`，包体积 2.4 → 1.9 GB）。同机
2026-11-10，信任模式，原始记录 `benchmarks/2026-11-10-load/`：

| 包 / 加载路径 | 冷启动 total | 上转 | 峰值 RSS |
|---|---|---|---|
| fp32 | 3722 ms | — | 3.4 GB |
| fp16 第一版（protobuf 内联转码，#41/#42） | 9467 ms | — | 5.3 GB |
| **fp16 第二版**（流式扫描 + 整文件上转 + ORT 内存替换） | **4486 ms** | 394 ms（4 个文件） | 4.1 GB |

第一版比 fp32 多 ~5.5 s：约一半是**自包含模型白走了转码**（RoBERTa
0.96 → 3.39 s、SV 0.30 → 0.62 s —— 此前记录的"RoBERTa 不受影响"有误），
另一半是有外部权重的模型被整体搬运五六遍。第二版比 fp32 只多
~0.6–0.8 s：上转 0.39 s（`t2s_shared` 被两个解码器共用、只转一次），
扫描 8 个模型（含 1.2 GB 的 RoBERTa）合计 6 ms。合成阶段无差异（运行
时仍是 FP32）。fp16 不再是"拿冷启动换体积"—— 分发体积敏感时可以
放心选它（决策见 architecture.md §2.13）。

### 合成（5 次 mean，单位 ms）

| 文本 | 语义 token | 音频 (s) | total wall | RTF mean | stage_decoder | first_stage | vocoder | text_bert |
|---|---|---|---|---|---|---|---|---|
| 短 | 49 | 1.93 | 1386 | **0.718** | 898（64.8%） | 77 | 378（27.3%） | 29 |
| 中 | 137 | 5.45 | 3413 | **0.628** | 2551（74.7%） | 65 | 741（21.7%） | 52 |
| 长 | 351 | 14.02 | 10910 | **0.778** | 9171（84.1%） | 85 | 1548（14.2%） | 101 |

每 token（stage_decoder 总耗时 ÷ 步数）：**18.26 / 18.59 / 26.10 ms** ——
随序列长度明显上升（KV cache 逐轮变长、宿主侧每步全量拷贝约五遍 +
图内注意力随长度变贵）。进程平均核占用 17.1 / 11.6 / 9.1（自旋线程
常驻空转的典型形态）；峰值 RSS 3.96 GB。

## M5 第一轮优化（2026-11-06）

### 改动

1. **宿主拷贝消除**（M5.2，`perf/t2s-host-copy-elimination`，语义不变）：
   解码循环 move 语义（KV cache / y_emb 每步从拷贝 5 遍降为
   ORT 输出 → Tensor 存储 1 次拷贝 + 输入侧零拷贝包装 +
   输出侧单次 memcpy）；细节见 architecture.md §2.11、
   upstream-audit.md §21。
2. **会话线程调优**（M5.3，`perf/ort-session-tuning`）：
   `SessionConfig::allow_spinning`（`session.intra_op.allow_spinning`）
   默认改**关**；线程数默认保持 8。依据见下文扫描。

### 优化后数据（threads=8，spinning=0，5 次 mean，单位 ms）

| 文本 | 语义 token | 音频 (s) | total wall | RTF mean (min–max) | stage_decoder | vocoder | text_bert |
|---|---|---|---|---|---|---|---|
| 短 | 47 | 1.84 | 861 | **0.468**（0.456–0.497） | 589（68.4%） | 202 | 25 |
| 中 | 142 | 5.64 | 2686 | **0.476**（0.463–0.488） | 2026（75.4%） | 578 | 32 |
| 长 | 344 | 13.73 | 7748 | **0.564**（0.551–0.574） | 6209（80.1%） | 1411 | 55 |

每 token：**12.53 / 14.27 / 18.04 ms**；进程平均核占用 **4.8 / 4.6 / 4.2**；
峰值 RSS 3.96 GB（基本持平）。

### 前后对比

| 指标 | 基线 | 优化后 | 变化 |
|---|---|---|---|
| ms/token（短 / 中 / 长） | 18.26 / 18.59 / 26.10 | 12.53 / 14.27 / 18.04 | **-31.4% / -23.3% / -30.9%** |
| RTF mean（短 / 中 / 长） | 0.718 / 0.628 / 0.778 | 0.468 / 0.476 / 0.564 | 最差 run 0.870 → 0.574 |
| 端到端 wall（长句） | 10.91 s | 7.75 s | -29.0% |
| 进程平均核占用（短） | 17.1 | 4.8 | **≈ 1/3.6** |
| 峰值 RSS | 3.96 GB | 3.96 GB | 持平 |

- ms/token 随长度的上升被大幅抹平（18.3→26.1 变成 12.5→18.0）：
  与序列长度成正比的宿主拷贝正是"越编越慢"的主因之一，剩下的
  长度效应主要来自图内注意力本身。
- vocoder 也提速（短句 378 → 202 ms）：它同样受益于输入零拷贝包装
  与自旋线程退场后的调度让利。两项改动对 vocoder 的贡献占比未单独
  归因（不值得为归因再多跑机器）。
- 核占用从 9–19 降到 4–5：自旋线程退场后，合成期间真正干活的
  就是当前会话的那 8 个 intra-op 线程。

### 参数扫描（threads × spinning，同机同时段）

| 配置 | ms/token 短 / 中 / 长 | RTF 长 | 核占用 长 | 备注 |
|---|---|---|---|---|
| 8 × spin=1（旧默认） | 14.40 / 13.47 / 17.70 | 0.567 | 9.5 | 参考点 |
| **8 × spin=0（新默认）** | **12.53 / 14.27 / 18.04** | 0.564 | 4.2 | 延迟持平或更优，核占用 1/3–1/4 |
| 12 × spin=0 | 12.92 / 14.39 / 20.07 | 0.643 | 5.2 | 长句劣化 11% |
| 16 × spin=0 | 13.37 / 14.87 / 20.13 | 0.650 | 6.2 | 长句劣化 12% |

- **线程不是越多越好**：6 个会话各开更大线程池会把工作摊到 P/E
  两类核上，长句反而劣化 11–12% → 默认保持 8。
- **自旋关**：自旋是为"下一个算子马上就来"准备的忙等；本引擎一次
  合成只有一个会话在干活，其余会话的自旋线程纯属内耗。
- ⚠️ **稳定性警示**：12/16 线程 × spin=1 是本机已知的不稳定区，
  本次扫描**刻意跳过** —— 采集期间设备在进入该区间时死机一次；
  此前 v0.2.0 也记录过同机高负载下 ORT 内部段错误/浮点异常
  （见 `tools/cli/main.cpp` 的注释）。该组合对延迟也无收益
  （spin=1 只在 ≤8 线程时与 spin=0 打平），不值得冒稳定性风险。
  若怀疑个别核不稳，CPU 亲和性/核排除是 M5 后续的候选（roadmap M5.3）。

### 参数扫描第二轮：图优化 / arena / 内存模式（M5.3 收口，2026-11-09）

安全区（threads=8 × spin=0）下补扫 SessionConfig 其余三个维度，每组
3 文本 ×5 run：

| 配置 | ms/token 短 / 中 / 长 | RTF 长 | 冷启动 total |
|---|---|---|---|
| **graph_opt=3 + arena + pattern（默认）** | 12.93 / 14.39 / 20.12 | 0.621 | 7550 |
| graph_opt=2 | 11.93 / 13.99 / 19.45 | 0.598 | 7170 |
| graph_opt=1 | 12.71 / 14.11 / 18.87 | 0.593 | 7469 |
| arena=0 | 12.47 / 14.25 / 18.90 | 0.615 | 7518 |
| mem_pattern=0 | 12.83 / 14.63 / 19.56 | 0.610 | 7522 |

**结论：默认配置保留**。各组合差异（长句 1–6%）小于同配置先后两次
测量的 run 间漂移（默认配置长句本轮 20.12 vs 第一轮 18.04，差 11%）
—— 没有稳健优于默认的组合。图内注意力的算子成本主导了剩余差异，
ORB 图优化级别在 1→3 之间对本图的收益已被 LayerNorm/Attention 融合
饱和（g1 与 g3 在噪声带内打平）。执行模式（Parallel）不扫：它面向
单会话内多流并发，与自回归串行解码的形态不匹配；CPU 亲和性因本机
疑似坏核待排查暂缓；ORT Format 需要离线转换工具链，与量化一并排期。

### 加载期并行（冷启动，2026-11-10）

冷启动（信任模式）里建会话占 ~85%，8 个模型串行建。两条并行化路径的结论：

**① 跨会话并行（按模型同时建多个会话）—— 已采用。** 这条路最初被否决：探针
实测 kairo 默认线程池把每个 worker **绑到单个核心**（`cpu_affinity` 为空时
自动分配 [0..hw-1]），而 ORT 会话的 intra-op 线程在建会话时创建、**继承创建
线程的亲和性** —— 在 kairo worker 上建出的会话，推理线程全被钉在同一个核上，
冷启动的收益抵不过推理变慢的代价。

**解除**：kairo 升至 53ee306（其 PR #227）—— 默认池空 `cpu_affinity` 改为
**OS 自由调度**（显式设值仍按 worker_id % size 确定性绑核）。复测：kairo
worker 与其上建出的 ORT 线程均 allowed=0-31，推理不再被钉核。此后建会话按
模型并行（8 路），调度层新增 `detail::parallel_for`（fork-join；调用线程
自带一份活、异常重抛最小下标、无 kairo 时串行）：

| 包 / 路径 | 冷启动（3 轮范围） | 会话段 |
|---|---|---|
| fp32 串行（此前） | 3766 ms | 3.35 s |
| **fp32 并行** | **1738–1846 ms** | **~1.16 s（~2.9×）** |
| fp16 串行 | 4387 ms（上转 394 ms） | — |
| **fp16 并行** | **2039 ms** | 上转 152 ms |

合成延迟不变（ms/token 14.15–14.77，基线带内）；连续多轮无稳定性异常
（瞬时形态 8 路 × 8 线程与一次合成同量级）。计时新增 `load/sessions`
（建会话墙钟）；并行时各模型自己的条目相互重叠，之和会超过它。
决策与绑核问题的来龙去脉见 architecture.md §2.14。

**② 会话内并行预打包（`SessionConfig::parallel_prepack`）—— opt-in，默认关。**
预打包摊到本会话自己的 intra-op 线程池（由调用线程创建，亲和性不受影响）。
ORT 1.31 才有这个键，更早的版本（含上游钉的 1.22）忽略它：

| 配置（ORT 1.31，信任模式） | 冷启动 fp32（三轮均值） | 冷启动 fp16 | 常驻 RSS |
|---|---|---|---|
| 关（默认） | 3766 ms | 4387 ms | ~3.4 GB |
| 开 | 3461 ms（-8.1%） | 4111 ms（-6.3%） | ~3.9 GB（**+0.5 GB**） |

RSS 时间线（100 ms 采样）显示多出的 ~0.5 GB 在加载后不回落 —— 是常驻
成本，不是瞬时峰值。用 0.5 GB 内存换 ~0.3 s 启动不划算，默认关；内存
宽裕、冷启动敏感的部署可以打开。复现：`genie-bench --parallel-prepack 1`，
原始记录 `benchmarks/2026-11-10-prepack/`。

### 内存占用（2026-11-10）

fp32 包（信任模式，ORT 1.31，threads=8）。逐会话常驻增量（探针实测）：
RoBERTa 1148 MB、HuBERT 328 MB、t2s_first_stage 298 MB、t2s_stage 297 MB、
vits 224 MB、SV 182 MB、prompt_encoder 84 MB、t2s_encoder 14 MB —— 合计
~2.6 GB，全部是进程私有内存（RssAnon）；arena / mem_pattern / glibc
malloc arena 上限对它没有影响（差异在测量噪声内）。

**加载期会话释放（已落地）**：HuBERT / 说话人验证 / prompt_encoder 只在
加载期算参考特征与全局嵌入，结果缓存后即释放会话并 `malloc_trim` 还页：

| 变体 | 加载后常驻 | 峰值 | 退出前 |
|---|---|---|---|
| 不释放（旧） | 3351 MB | 3903 MB | 3903 MB |
| 只释放不还页 | 2931 MB | 3461 MB | 3461 MB |
| **释放 + malloc_trim（当前默认）** | **2075 MB** | **3291 MB** | 3200 MB |

常驻 **-1.2 GB**（其中释放会话本身 -0.6 GB，glibc 还页再 -0.4 GB）；
释放一步实测 37 ms（计时 `load/release_load_only_models`）；合成延迟与
不释放打平（三轮 ms/token 在噪声内）。回归测试
`Engine.ReleasesLoadOnlySessionsAfterConstruction`。

**t2s 双解码器共享权重（调研结论：不采用）**：两图的 291 个初始化器
同名同内容（同一份 `t2s_shared_fp32.bin`），理论可省 ~0.3 GB。ORT 的
`AddInitializer` + `PrepackedWeightsContainer` 探针实测：**省不出来** ——

- `AddInitializer` 注入共享缓冲：第二个会话只省 23 MB（305 → 274），
  图优化把权重重排进自己的存储，共享缓冲只在注入时省了一份"反序列化"，
  重排产物仍各存一份；
- `PrepackedWeightsContainer` 共享预打包缓冲：第二会话不减反增
  （+569 MB vs 分开加载 +297 MB）；
- 关掉预打包也不行（848 vs 555 MB），关图优化更不行（推理 33 ms/token，
  慢 2.6 倍）。

266 MB 的权重喂给 Transpose 算子 —— 它们的重排（prepack / 优化后的
常量折叠副本）正是重复存储的来源。替代路径是 **fp16 分发 + 运行时
fp16 推理**（权重不重排 fp32 副本）或 int8 量化（M0.2 之后）。

原始记录：`benchmarks/2026-11-10-memory/`（bench 的 JSON 新增
`rss_after_load_kb` / `rss_end_kb` —— 常驻量比峰值更能反映长期占用）。

### 动态 INT8 量化（2026-11-11，M5 量化第一档）

按 M5 阶梯落地**动态 INT8**（`quantize_dynamic`，per-channel，
MatMulConstBOnly）。工具 `tools/deploy/quantize_package.py`；精度裁决
`tools/gate/`（捕获 → 变体重放 → 分级容差，见 regression.md）。
权重在 HuggingFace 不可达，所有基线为 **C++ fp32 自基线**。

**逐模型门控（fp32 基线 × 2 份地板对照，4 条文本）**：

| 角色 | 权重体积 fp32→int8 | 门控 | 最差指标（int8 vs fp32） | 备注 |
|---|---|---|---|---|
| roberta | **1197→368 MB（-69%）** | ❌ 超容差 | rel-L2 0.138，cos 0.991（容差 0.02 / 0.995） | 常驻收益的唯一大头 |
| chinese_hubert | 377→122 MB（-68%） | ❌ 超容差 | rel-L2 0.162，cos 0.988 | 加载期释放会话，**不省常驻**，只省包体与加载 |
| speaker_verification | 185→185 MB（0） | ❌ **崩坏** | rel-L2 1.21，**cos 0.264** | Conv 为主几乎没量化也照样崩 —— **禁量化**，音色直接换人 |
| t2s stage / first | 307→289 MB（-6%） | ✅ | y 采样一致率 == 地板；连续输出 rel ≤0.059 | conformer 以 Conv 为主，MatMul 只占 6% |
| t2s_encoder | 11.5→11.5 MB（0） | ✅ | 全零差 | 图内无常量 B MatMul，量化器没动它 |
| vits | 249→251 MB（+1%） | advisory | rel 0.91 / cos 0.58 == fp32 重放地板 | 图内 z 噪声，逐样本比较本就无意义；Conv 为主没收益 |
| prompt_encoder | 88→26 MB（-71%） | ✅ | ge rel-L2 0.018，cos 0.9998 | 加载期释放会话，只省包体与加载 |

**采样确定性与 e2e 的边界（重要实测结论）**：

- 同进程、同配置：**逐 run 序号完全可复现** —— fp32 双引擎自比的
  语义 token、音频 WAV 逐字节一致（地板 mel 距离 0.0）。
- 跨进程：fp32 **自己都不可复现**（同一文本两次进程 token 数 58 vs 339）——
  采样种子随进程变。因此 **e2e token 一致率不能当质量指标**：
  int8 条件化会让采样路径从首个 token 就走上另一条同样有效的韵律路径。
  听觉终审只能是 WAV A/B 听测（`/home/ybt/genie-assets/quant-ab/`，
  fp32 × {aggressive / roberta / hubert} 各 4 条文本）。
- 跨构建（C++ ORT 1.31 vs Python ORT 1.22 重放）：近平局采样 token 有
  **稳定的构建间偏差**（地板一致率 ≈0.9914，两次独立重放翻的是同一个
  token）—— token 类门控因此按地板相对裁决。

**混合精度包实测（`pkg-aggressive` = roberta + t2s 三件套 + prompt_encoder
int8，SV / vits / hubert 保持 fp32；包 2550→1672 MB，-34%）**：

| 指标 | fp32 包 | int8 混合包 | Δ |
|---|---|---|---|
| 冷加载（信任模式） | 2298 ms | **1692 ms** | -26% |
| **加载后常驻** | **2.176 GB** | **1.252 GB** | **-43%（-0.93 GB）** |
| 峰值 RSS | 4.20 GB | 2.31 GB | -45% |
| RTF（短/中/长） | 0.882 / 0.799 / 0.740 | **0.572 / 0.586 / 0.723** | 短文本 **-35%** |
| ms/token（长文本） | 29.5 | 28.8 | -2% |

常驻收益几乎全部来自 **roberta int8（约 -0.9 GB）**；短文本 RTF 的
-35% 是 MLAS u8s8（AVX_VNNI）对 RoBERTa 前向的加速。

**听测裁决（2026-11-11，已定版）**：roberta / hubert 的 int8 特征误差
（rel ≤0.162，方向余弦 ≥0.988）超出最初的严格容差。A/B 听测
（fp32 × aggressive / roberta-only / hubert-only，各 4 文本：
中文短/长、中英混、数字电话号码）**无差异** —— 音色 / 清晰度 /
自然度均不受影响。`tools/gate/tolerances.json` 据此放宽这两类
条件化特征角色（rel 0.20 / cos 0.985，留了实测上界的一档余量）
并写明依据；`pkg-aggressive` 即为定版的混合精度包形态。
原始记录 `benchmarks/2026-11-11-quant/`。

**下一档（M5 阶梯继续）**：静态 INT8（捕获数据可当校准集）、
roberta 层内混合精度（attention 保持 fp32）、Conv 系（vits / t2s）的
int8 评估 —— Conv 不量化是 t2s/vits 收益趋零的根因。

### 静态 INT8（2026-11-11，M5 量化第二档）

第二档补上动态档的盲区：`quantize_static` + QDQ（图内插
QuantizeLinear/DequantizeLinear）让 **Conv 参与量化**，激活 scale 来自
校准集 —— 校准集直接用 genie-gate 捕获的真实流水线输入
（`--stage-steps 8` 补采，stage_decoder 前 8 步全量，367 MB）。
工具同 `tools/deploy/quantize_package.py --static`（静态档强制串行：
并发 `quantize_static` 校准实测过一次 SIGSEGV）。u8s8 + per-channel。

**逐模型门控（静态 vs fp32，capture-s8 口径，地板相对）**：

| 角色 | 静态权重体积 | 门控 | 最差指标（静态 vs fp32） | 动态档对照 | 裁决 |
|---|---|---|---|---|---|
| t2s 双解码器 | 288 MB/个（≈动态） | ❌ | **KV cache rel 0.33 / cos 0.944**（激活误差 24 层逐层累积；y token 仍在地板上） | KV rel ≤0.059 | **退档**，保持动态 |
| roberta | 367 MB（=动态，无常量 Conv） | ❌ | rel 0.378 / cos 0.933 | rel 0.138 / cos 0.991 | **退档**，超听测校准包络（0.20/0.985）近一倍 |
| chinese_hubert | 95 MB（动态 122，Conv 前端再省 27） | ❌ | **rel 0.736 / cos 0.749** | rel 0.162 / cos 0.988 | **退档**，Conv 前端量化毁掉 ssl 特征 |
| t2s_encoder | 7.9 MB（-31%，Conv 静态化生效） | ❌ | **prompts exact 0.748，fp32 地板 = 1.0** | 全零差（动态没动它） | **退档**：encoder 的 prompts 选择是确定性的（地板 1.0，不同于解码器的 0.9914），25% 初始 prompt 改写是系统性漂移；收益仅 3.6 MB |
| vits | 200 MB（-19%；ConvTranspose 无 int8 内核保持 fp32） | ✅ advisory | 音频 rel 0.86–1.02 == fp32 重放地板量级 | advisory | **唯一幸存者**，听测通过 |

**结论：激活 int8 量化全面被质量门槛拒绝** —— 动态档的"只量化权重"
恰好避开了激活误差；静态档为 Conv 付出的激活代价对这三个特征/解码
模型都不可接受。Conv 静态化的净收益收窄到 **vits 一个角色**。

**候选包 `pkg-static-vits`（= 定版混合包 aggressive + vits 静态；
包 1672→1625 MB）同会话背靠背 bench（loadavg ≈155，另一项目训练在跑，
绝对值仅供同表内配对比较）**：

| 指标 | fp32 | aggressive（现役） | static-vits |
|---|---|---|---|
| 加载后常驻 | 2.13 GB | 1.17 GB | **1.13 GB（-47 MB）** |
| 峰值 RSS | 4.08 GB | 2.24 GB | 2.20 GB |
| 冷加载（信任模式） | 1818 ms | **1547 ms** | 2487 ms（**+940 ms**） |
| RTF（短/中/长） | 0.518 / 0.537 / 0.666 | 0.497 / 0.523 / 0.651 | 0.488 / 0.527 / 0.650 |
| vocoder 分段（短/长，ms） | 278 / 1902 | 245 / 1890 | 258 / 1975（**不快反慢**） |

- **常驻 -47 MB** 是唯一净收益（vits 是最后一个 fp32 常驻大块）；
- **冷加载 +0.9 s**：QDQ 图在加载期的图优化变慢（vits 会话 985 → 1884 ms），
  摊薄手段是 ORT Format 离线转换（M5 挂账项）；
- **合成不快反慢**：QDQ 路径的 DequantizeLinear 开销吃掉 u8s8 Conv 收益
  —— 静态档的"更快"假设在本机（AVX2 + AVX_VNNI，无 AVX512）不成立；
- e2e（aggressive × static-vits，同进程配对）：4 文本 **token 前缀一致率
  全 1.0** —— vits 在 t2s 下游，语义路径完全不受影响，A/B 只差 vocoder
  渲染层（WAV：`/home/ybt/genie-assets/quant-ab/staticvits_*`）。

**bin 去重（工具回归修复的端到端验证）**：用合入去重的工具重建
aggressive 策略包 —— t2s 双解码器的逐字节相同 int8 bin 自动归一
（`deduped 1 identical weight bin(s), saved 288 MB`），**包 2550→1384 MB
（-45.7%）**，manifest 回到"一份 bin 双 role 引用"的 fp32 约定，ORT 复核
全绿。同会话 bench：常驻 1.256 GB ≈ 1.230 GB（**不变** —— 会话各自拷贝
外部数据，无 mmap 页共享），加载 / RTF 持平。纯包体收益，权重与现役包
逐字节相同。注意去重只在**动态档**自动生效：静态 QDQ 的初始化器布局
两解码器不同（激活 scale 内联位置不同），bin 不再逐字节相同，属机会性
特性，不误伤。

**听测裁决（2026-11-11，已定版）**：A/B 听测（aggressive × static-vits，
4 文本）**通过** —— vits 静态入包，定版形态 = aggressive 策略 + vits 静态
+ bin 去重（`pkg-final`，1384 MB 基础上再 -47 MB ≈ **1.34 GB 包**）。
-47 MB 常驻换 +0.9 s 冷加载的交换获用户认可；vocoder 渲染层的
静态量化伪影在听阈之下（与 e2e token 一致率 1.0 的隔离结论互证）。

原始记录 `benchmarks/2026-11-11-quant/`（gate-static / gate-static-vits /
gate-aggressive-control / floor-s8-1,2 / bench-static-* / staticvits.json）。

**下一档（M5 阶梯继续）**：ORT Format 离线转换（顺带摊薄 QDQ 冷加载
成本）；roberta 层内混合精度已 ⛔（见 roadmap：动态档已定版，精度余量
足够，再切没有收益面）。

### 算子级耗时（2026-11-11，M5 性能分析收尾）

ORT 内建 profiler 的采集端落地：`SessionConfig::profile_output_dir` +
`genie-bench --profile <dir>`（`tools/bench/summarize_profile.py` 聚合成
算子表）。每次 Run 都追加记录、trace 文件可观（stage_decoder 一次采集
1.24 GB、被 ORT 内部 100 万事件上限截断 —— 分布仍代表性成立，步间是
同质的），**开 profiler 的计时数字不作基线**。采集口径：pkg-final、
默认 3 文本 × warmup 1 + runs 3。

**stage_decoder（解码热路径，2 100 步样本）**：

| 算子 | 总耗时 (ms) | 占比 | 说明 |
|---|---|---|---|
| MatMul | 5 644 | **49.6%** | attention/FFN 主计算，48 层 × 51 处/步 |
| Transpose | 1 389 | 12.2% | attention 的 KV 转置 |
| Concat | 1 040 | **9.1%** | **KV cache 每步 past+new 拼接**（图结构固有） |
| Mul / Add / Div / Sqrt | 1 262 | 11.0% | 逐元素与 RMS 归一化 |
| DynamicQuantizeMatMul | 366 | 3.2% | **动态 int8 的运行期激活量化**（量化档的图内代价） |
| Reshape / Unsqueeze / Slice / Shape | 1 059 | 9.4% | 形状整理 |

两个直接结论：**KV cache 的图内管理（Transpose + Concat ≈ 21%）是
MatMul 之外最大的开销块** —— 它是上游图结构决定的，宿主侧优化动不了，
流式（M6）若要继续压 ms/token 得从图改起（KV 切片替代整段 concat，
需要动模型导出）；宿主侧**输出拷贝在 profiler 里不可见**（它不是图内
kernel），IOBinding 原型（下一节）的收益上限只能靠对比实验测定。

**vits（静态 QDQ）**：QLinearConv **67.2%**、DequantizeLinear 8.0%、
ConvTranspose 6.0%（均值 10.8 ms/次，无 int8 内核保持 fp32）—— 静态
档的执行期形态与第二档实测一致。

其余角色表见采集目录聚合输出（roberta 以 MatMul+SkipLayerNormalization
为主，hubert 以 Conv 系为主），不逐一体积。

## 已知问题（影响基准覆盖面）

- ~~混读长文本会失败~~：已在 #34 修复（英文 BERT 块以全零特征占位）。
  混读工况可用 `genie-bench --language hybrid-chinese-english --text ...`
  单独测量；**混读基线尚未采集**，待补。默认样例仍保持纯中文，
  以便与既有基线可比。
- **本机高负载多核可能死机**：疑似个别 CPU 核在高负载下不稳。
  基准/压测时避免 6 会话 × 高线程 × 自旋的组合；默认配置
  （8 线程 × 自旋关）连续采集多轮无异常。

## 未覆盖指标（M0.3 遗留）

| 指标 | 状态 | 途径 |
|---|---|---|
| 内存分配次数 | ⏳ | 需要 allocator hook（operator new 统计或 malloc 计数），M5 前补 |
| CPU 利用率时间线 | ⏳ | 目前只有进程级均值；逐阶段利用率需要采样线程，M5 评估 |
| 首个可播放音频块时间（TTFA） | ⏳ | 当前流式是**整段推送**（合成完才 close 通道），TTFA == 端到端；M6 真分块后单独测 |
| 算子级耗时 | ⏳ | ORT 内建 profiler（`SessionOptions::enable_profiling`），M5.1 引入 |

## 历史

### 首份基线（2026-11-05，v0.3.0 + PR #32，ORT 1.31.0，threads=8 自旋开）

当时的合成表与上面"基线"一节同源（同一台机器连续两日采集，数字在
采样随机性内一致），主要差异是当时没有 ms/token 归一 —— token 数
来自另一次运行、与耗时表不自洽（报告过 15.3/17.7/24.3 ms/token，
事后核对无法复现，2026-11-06 起由 bench 直接输出 iterations 归一）。
当时的解读结论（t2s 自回归占大头、每 token 耗时随长度上升、
自旋线程推高核占用与方差）由本轮数据证实并量化。
