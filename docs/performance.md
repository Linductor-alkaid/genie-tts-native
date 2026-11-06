# 性能基线（M0.3）

本文记录整条推理链路的**分阶段耗时基线**。计时采集端是
`SynthesisResult.timing` / `Engine::load_timing()`（见 `docs/architecture.md`
§2.10），本文的数字由 `genie-bench` 采集。

> 基线数字**只在同代硬件上可比**。换机器、换 ORT 版本、换音色包都要重测，
> 重测后直接替换本文表格（保留一张历史表归档在文末也行）。

## 测量环境

| 项 | 值 |
|---|---|
| 引擎 | genie-tts-native v0.3.0 + 分阶段计时（PR #32） |
| 日期 | 2026-11-05 |
| CPU | 13th Gen Intel Core i9-13900KF（24 核 32 线程，P+E 混合） |
| OS | Linux 6.8.0-124-generic x64 |
| 构建 | `cmake --preset release`（-O3 -DNDEBUG） |
| 推理后端 | ONNX Runtime 1.31.0 CPU EP |
| 线程 | 每会话 `intra_op_threads = 8`（CLI / bench 默认） |
| 音色包 | feibi，`model_version: v2ProPlus`，32 kHz，2.4 GB（含 RoBERTa / HuBERT / g2pM / jieba / 英文词典） |
| 方法 | 每条文本 warmup 1 次 + 测量 5 次取 mean / min / median / max |

## 复现

```bash
cmake --preset release && cmake --build --preset release
./build/release/tools/genie-bench \
    --model <音色包目录> --runs 5 --warmup 1 --threads 8 \
    --json baseline.json
```

## 冷启动（Engine 构造，单次，v2ProPlus 包）

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

冷启动的一半时间在**音色包完整性校验**（SHA-256 扫 2.4 GB）。常驻进程场景
这笔钱只付一次；对"每次进程冷起"的场景（CLI 一次性合成）它就是主成本 ——
"信任模式跳过校验"是 M5 的候选优化项（需要显式开关，默认仍校验）。

## 合成（5 次 mean，单位 ms）

| 文本 | 字数 | 音素 | 语义 token | 音频 (s) | total wall | RTF mean (min–max) | t2s/stage_decoder | t2s/first_stage | vocoder | text_bert | frontend |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 短（"今天天气真不错。"） | 8 | 15 | 54 | 1.92 | 1320 | **0.688** (0.652–0.752) | 826 | 98 | 364 | 29 | 0.06 |
| 中（"今天天气真不错，我们出去散步吧，顺便去公园里看看花。"） | 25 | 37 | 137 | 5.42 | 3279 | **0.605** (0.592–0.610) | 2427 | 77 | 721 | 51 | 0.35 |
| 长（"二零二五年秋天，……自然的语音。"） | 63 | 129 | 354 | 13.94 | 10379 | **0.745** (0.731–0.751) | 8593 | 95 | 1582 | 104 | 0.81 |

其他指标：`language_detect` < 0.01 ms；`t2s/encoder` ≈ 2 ms；
峰值 RSS **3.96 GB**；进程平均核占用（CPU 时间 / 墙钟）短句 17.5、中句 11.8、
长句 9.2。

## 解读

1. **t2s 自回归循环是绝对大头**（占合成 74%–86%）。折算每 token：
   短句 15.3 ms、中句 17.7 ms、长句 **24.3 ms** —— 随序列长度上升
   （KV cache 线性增长、无复用），这正是 M5 第一刀该砍的地方：
   KV cache 复用、量化、降线程竞争。
2. **vocoder 第二**（12%–28%，与音频长度线性）。当前与解码串行 ——
   它只依赖完整语义 token，没有流水线化的必要（分块流式 M6 除外）。
3. **文本前端可忽略**（< 1 ms）；`text_bert`（RoBERTa）随文本长度线性但
   占比 < 1.5%。前端优化对端到端没有意义，除非做流式首响应。
4. **方差不可忽视**：本轮短句 `t2s/first_stage` 52–177 ms（3 倍跨度）、
   长句 `stage_decoder` 8295–9024 ms；上一轮中句 `stage_decoder` 曾出现
   1274–2860 ms 的更大摆动。13900KF 的 P/E 核混合调度 + ORT 自旋线程是
   主要嫌疑。RTF 报 min/median/max 三档就是为了暴露它；对外承诺延迟用 max。
5. **RTF 全部 < 1**：当前配置下实时合成成立（长句最差 0.752）。
6. **峰值 RSS 3.96 GB**：多会话 × 8 线程的 ORT arena。内存受限部署
   （嵌入式）要降线程数或拆会话 —— 与 roadmap M5.3 对应。

## 已知问题（影响基准覆盖面）

- ~~混读长文本会失败~~：已在 #34 修复（英文 BERT 块以全零特征占位）。
  混读工况可用 `genie-bench --language hybrid-chinese-english --text ...`
  单独测量；**混读基线尚未采集**，待补。默认样例仍保持纯中文，
  以便与既有基线可比。

## 未覆盖指标（M0.3 遗留）

| 指标 | 状态 | 途径 |
|---|---|---|
| 内存分配次数 | ⏳ | 需要 allocator hook（operator new 统计或 malloc 计数），M5 前补 |
| CPU 利用率时间线 | ⏳ | 目前只有进程级均值；逐阶段利用率需要采样线程，M5 评估 |
| 首个可播放音频块时间（TTFA） | ⏳ | 当前流式是**整段推送**（合成完才 close 通道），TTFA == 端到端；M6 真分块后单独测 |
| 算子级耗时 | ⏳ | ORT 内建 profiler（`SessionOptions::enable_profiling`），M5.1 引入 |

## 历史

（首次基线，无历史版本。换硬件 / 换 ORT / 换音色包重测后，把旧表挪到这里并注明环境。）
