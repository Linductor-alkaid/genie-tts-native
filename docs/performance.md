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
