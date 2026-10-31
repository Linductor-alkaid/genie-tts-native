# 上游审计：Genie-TTS 推理链路

> 审计对象：<https://github.com/High-Logic/Genie-TTS>（MIT，Copyright (c) 2025 High_Logic）
> 审计基线：仓库默认分支，审计日期 2026-10-23
> 审计目的：为 C++ 移植确定**权威的行为契约**，避免"照着印象写"。

本文是 M0 里程碑的核心产出。所有结论都注明了在上游的位置，便于复核。

---

## 1. 结论摘要

| 问题 | 结论 |
|---|---|
| 推理链路有几段？ | V2 是 **4 段**（encoder → first_stage_decoder → stage_decoder 循环 → vits）；V2ProPlus 额外插入 sv_model + prompt_encoder |
| 采样在哪？ | **在 ONNX 图内**。`first_stage_decoder` 与 `stage_decoder` 各含一个 `TopK` 算子，主机侧不采样 |
| 随机性怎么控？ | 只有 ONNX Runtime 的全局随机种子 |
| 张量类型有几种？ | **float32 / int64 / bool** 三种，没有别的 |
| BERT 特征维度？ | **1024** |
| HuBERT 特征维度？ | **768** |
| 采样率？ | 主链路 **32000 Hz**，HuBERT / 说话人向量用 **16000 Hz** |
| 解码循环上限？ | **500** 步；语义 token 上限 **1000**（`MAX_T2S_LEN`） |
| EOS 判定？ | 语义 token **≥ 1024** 即视为非法/EOS，截断到第一个之前 |

---

## 2. 模型清单

上游仓库**随源码发布了全部六个 ONNX 模型**（`src/genie_tts/Data/`），
本项目已按 MIT 协议收录到 `third_party/genie-tts-models/`。

> 注意：这些文件是**图壳**，权重在外部 .bin 里（见 §2.1）。

| 模型 | 文件 | 大小 |
|---|---|---|
| T2S encoder | `v2/t2s_encoder_fp32.onnx` | 14 KB |
| T2S first stage decoder | `v2/t2s_first_stage_decoder_fp32.onnx` | 413 KB |
| T2S stage decoder | `v2/t2s_stage_decoder_fp32.onnx` | 413 KB |
| VITS 声码器（V2） | `v2/vits_fp32.onnx` | 1.6 MB |
| prompt encoder | `v2ProPlus/prompt_encoder_fp32.onnx` | 43 KB |
| VITS 声码器（V2ProPlus） | `v2ProPlus/vits_fp32.onnx` | 1.6 MB |

### 2.1 ⚠️ 关键发现：这些模型是「图壳」，权重不在仓库里

上游仓库随源码发布的六个 ONNX 模型**不包含权重**。它们是「relink」之后的图：
每个 initializer 都指向一个**外部 .bin 文件**（`data_location = EXTERNAL`）。

| 模型 | 外部权重文件 | 权重总量 |
|---|---|---|
| `t2s_encoder_fp32.onnx` | `t2s_encoder_fp32.bin` | 11.5 MB |
| `t2s_first_stage_decoder_fp32.onnx` | `t2s_first_stage_decoder_fp32_weights.bin` | 306.8 MB |
| `t2s_stage_decoder_fp32.onnx` | `t2s_stage_decoder_fp32_weights.bin` | 306.8 MB |
| `vits_fp32.onnx` | `vits_fp32.bin` | 161.7 MB |
| `prompt_encoder_fp32.onnx` | `prompt_encoder_weights.bin` | 88.5 MB |
| `v2ProPlus/vits_fp32.onnx` | `vits_weights.bin` | 248.7 MB |
| **合计** | | **约 1.12 GB** |

这些 .bin 由上游的**模型转换流程**从原始 GPT-SoVITS 模型生成
（见 `src/genie_tts/Converter/v2/T2SConverter.py` 的
`step1_create_fp16_bin_with_key_mapping` → `step2_relink_onnx_for_fp32`），
**不在 Git 仓库中**。

**这意味着**：

1. 这些 ONNX 文件**无法**被 ONNX Runtime 直接加载 ——
   会报 `filesystem error: cannot get file size: No such file or directory`。
2. 但**图结构是完整的**：输入输出的名字、类型、维度全都在。
   因此 I/O 契约可以被**可执行地**锁定，而不需要 1.1 GB 的权重。
3. 本项目的做法：
   - 契约测试直接解析 ONNX protobuf（`genie::read_onnx_graph`），
     **不需要权重、不需要 ONNX Runtime**，因此在任何构建下都能跑；
   - ONNX Runtime 的运行时路径（加载 → 按名字绑定 → 推理 → 解析输出）
     用自包含的**合成模型**验证（`tests/models/synthetic/echo.onnx`）；
   - 真实权重的端到端推理需要用户自行跑一遍上游转换流程产出 .bin，
     属于部署侧的事，不在本仓库范围内。

> 对应代码：`include/genie/onnx_graph.hpp`（最小 protobuf 读取器）、
> `tests/component/test_model_contract.cpp`（契约断言）、
> `tools/make_synthetic_model.py`（合成模型生成器）。

**仓库里没有**的两个模型（上游从用户数据目录加载）：

| 模型 | 用途 | 来源 |
|---|---|---|
| `chinese-hubert-base.onnx` | 从 16 kHz 参考音频提取 `ssl_content` | 上游 GUI 下载 |
| `RoBERTa.onnx` | 中文韵律特征（可选） | 上游 GUI 下载 |
| 说话人验证模型 | 产出 `sv_emb`（V2ProPlus） | 上游 GUI 下载 |

> 移植含义：这三个模型需要**可选加载**，缺失时降级（`ssl_content` 退化为零张量、
> BERT 特征退化为零张量、跳过 V2ProPlus 分支），而不是让引擎起不来。

---

## 3. 权威 I/O 契约

以下签名由 `onnx` 1.23 的 shape inference 从模型文件直接提取，
并已固化为可执行测试：`tests/component/test_model_contract.cpp`。

### 3.1 `t2s_encoder`

**输入**

| 名称 | 类型 | 形状 |
|---|---|---|
| `ref_seq` | int64 | `[1, ref_length]` |
| `text_seq` | int64 | `[1, text_length]` |
| `ref_bert` | float32 | `[ref_length, 1024]` |
| `text_bert` | float32 | `[text_length, 1024]` |
| `ssl_content` | float32 | `[1, 768, ssl_length]` |

**输出**

| 名称 | 类型 | 形状 |
|---|---|---|
| `x` | float32 | `[1, ?, 512]` |
| `prompts` | int64 | `[1, ?]` |

### 3.2 `t2s_first_stage_decoder`

**输入**：`x` float32 `[1, x_length, 512]`、`prompts` int64 `[1, prompts_length]`

**输出**（50 个）：`y` int64 `[1, ?]`、`y_emb` float32 `[1, prompts_length, 512]`、
`present_k_layer_{0..23}` / `present_v_layer_{0..23}` float32 `[past_sequence_len, 1, 512]`

### 3.3 `t2s_stage_decoder`

**输入**（50 个）：

| 名称 | 类型 | 形状 |
|---|---|---|
| `iy` | int64 | `[1, iy_length]` |
| `iy_emb` | float32 | `[1, iy_emb_length, 512]` |
| `past_k_layer_{0..23}` | float32 | `[past_sequence_len, 1, 512]` |
| `past_v_layer_{0..23}` | float32 | `[past_sequence_len, 1, 512]` |

**输出**（51 个）：`y` int64 `[1, ?]`、`y_emb` float32 `[1, ?, 512]`、
`stop_condition_tensor` **bool [标量]**、`present_k_layer_{0..23}` / `present_v_layer_{0..23}`

> ⚠️ **这是最容易踩的坑。**
> 上游 `src/genie_tts/Core/Inference.py` 的 `t2s_cpu()` 是这样传参的：
> ```python
> input_feed = {name: data for name, data in zip(input_names, [y, y_emb, *present_key_values])}
> ```
> 即**按位置**对应，变量名叫 `y` / `y_emb`。
> 但模型真实的输入名是 **`iy` / `iy_emb`**。
> 上游这样写能跑通纯粹是因为 `get_inputs()` 的顺序恰好是 `iy, iy_emb, past_k_*...`。
>
> **本移植一律按名字绑定。** 位置绑定一旦上游调整输入顺序就会静默错位 ——
> 不会报错，只会输出垃圾。

### 3.4 `vits`（V2 声码器）

| 方向 | 名称 | 类型 | 形状 |
|---|---|---|---|
| 输入 | `text_seq` | int64 | `[1, text_length]` |
| 输入 | `pred_semantic` | int64 | `[1, 1, pred_length]` |
| 输入 | `ref_audio` | float32 | `[1, audio_length]` |
| 输出 | `audio` | float32 | `[?]` |

> `pred_semantic` 是**三维**且中间维为 1。上游用
> `np.expand_dims(y[:, -idx:], axis=0)` 得到 `[1, 1, idx]` —— 与契约一致。
> 本移植构造时也必须保留这个 singleton 维。

### 3.5 `prompt_encoder`（V2ProPlus）

| 方向 | 名称 | 类型 | 形状 |
|---|---|---|---|
| 输入 | `ref_audio` | float32 | `[1, audio_length]` |
| 输入 | `sv_emb` | float32 | `[1, 20480]` |
| 输出 | `ge` | float32 | `[?, 1024, 1]` |
| 输出 | `ge_advanced` | float32 | `[?, 512, 1]` |

> `sv_emb` 的 20480 = 16 kHz 音频的说话人向量（展平）。缺失说话人验证模型时，
> V2ProPlus 分支不可用。

### 3.6 `vits`（V2ProPlus 声码器）

| 方向 | 名称 | 类型 | 形状 |
|---|---|---|---|
| 输入 | `text_seq` | int64 | `[1, text_length]` |
| 输入 | `pred_semantic` | int64 | `[1, 1, pred_length]` |
| 输入 | `ge` | float32 | `[1, 1024, 1]` |
| 输入 | `ge_advanced` | float32 | `[1, 512, 1]` |
| 输出 | `audio` | float32 | `[?]` |

---

## 4. 推理链路

### 4.1 V2

```
文本
 └→ get_phones_and_bert(text, language)      # G2P
     ├→ text_seq  : int64  [1, N]
     └→ text_bert : float32 [N, 1024]

参考音频
 └→ ReferenceAudio
     ├→ audio_32k  : float32 [1, M]           # 32 kHz
     ├→ audio_16k  : float32 [1, M/2]         # 16 kHz
     └→ ssl_content: float32 [1, 768, T]      # cn_hubert(audio_16k)

encoder(ref_seq, text_seq, ref_bert, text_bert, ssl_content)
 └→ x, prompts

first_stage_decoder(x, prompts)
 └→ y, y_emb, present_k/v_*

loop（最多 500 次）:
    stage_decoder(iy=y, iy_emb=y_emb, past_k/v_* = present_k/v_*)
     └→ y, y_emb, stop_condition_tensor, present_k/v_*
    stop_condition_tensor 为真 → 跳出

语义 token 清洗：剔除 ≥ 1024 的元素，保留第一个之前的部分

vits(text_seq, pred_semantic=[1,1,L], ref_audio=audio_32k)
 └→ audio : float32
```

### 4.2 V2ProPlus

与 V2 的差别只在声学段：

```
sv_model(audio_16k)                 → sv_emb
prompt_encoder(ref_audio, sv_emb)   → ge, ge_advanced
vits(text_seq, pred_semantic, ge, ge_advanced) → audio
```

> V2ProPlus **不传 `ref_audio` 给 vits**，改传两个全局 embedding。

### 4.3 上游的文本处理

`src/genie_tts/Core/Inference.py` 的 `tts()` 第一行是：

```python
text = '。' + text  # 防止漏第一句。
```

**前置一个句号**。这是为了让 T2S 的第一个音素落在稳定位置。
本移植必须照做，否则首音素分布与上游不同。

---

## 5. 采样在图内 —— 重要结论

对 `t2s_first_stage_decoder` 与 `t2s_stage_decoder` 做算子统计：

```
first_stage_decoder : {'Constant': 454, 'Gather': 31, 'Greater': 1, 'Where': 3, 'Less': 2, 'TopK': 1}
stage_decoder       : {'Constant': 461, 'Gather': 29, 'Less': 2, 'Where': 2, 'TopK': 1, 'Greater': 1}
```

两个模型各含**一个 `TopK`**，并配有 `Greater` / `Less` / `Where` 做阈值比较。
上游 Python 侧**没有任何采样代码** —— 采样、温度、重复惩罚全部烘焙在 ONNX 图里。

**移植含义**

- 计划书 M5.2.4 提到的"采样算法原生化（Top-K、Top-P、Temperature、重复惩罚）"
  **不适用于本项目** —— 主机侧没有采样器可优化。
- 主机侧可调的只有 **ONNX Runtime 全局随机种子**
  （`Ort::SetGlobalRandomSeed`，影响图内 `RandomNormal` 类算子）。
- 若未来要暴露采样参数，必须**重新导出模型**，把采样参数做成图输入。
  这是一个独立的模型工作项，不是本项目的代码改动。

---

## 6. 前处理行为

### 6.1 参考音频（`src/genie_tts/Audio/Audio.py` + `ReferenceAudio.py`）

```python
wav, original_sr = sf.read(audio_path, dtype='float32')
if original_sr != target_sampling_rate:
    wav = soxr.resample(wav, original_sr, target_sampling_rate, quality='hq')
```

- 多声道 → 单声道：`np.mean(wav, axis=1)`（**各声道算术平均**）
- 目标采样率 32000；再从 32000 重采样到 16000 给 HuBERT 用
- 两次重采样都用 `soxr` 的 `'hq'` 档

**移植含义**：本项目用自带的窗函数法带限 sinc 重采样，**不是** `soxr`。
量化差异见 `docs/architecture.md`「重采样一致性」。

### 6.2 多语言 G2P 依赖（`requirements.txt`）

| 语言 | 依赖 |
|---|---|
| 中文 | `pypinyin`、`g2pM`、`jieba_fast` |
| 英文 | `nltk` |
| 日文 | `pyopenjtalk-plus` |
| 韩文 | `jamo`、`ko_pron`、`g2pk2`、`eunjeon` |

**移植含义**：这是整个项目**最大的工作量来源**，也超出 C++ 生态的现成覆盖范围。
M2 阶段的现实策略是：中文用内置拼音词典 + 自建规则，英文用规则 + 词典，
日韩在 M2 后期单独处理。

### 6.3 中英混读

`src/genie_tts/GetPhonesAndBert.py`：

```python
if language.lower() == 'chinese' and re.search(r'[a-zA-Z]', prompt_text):
    language = 'hybrid-chinese-english'
```

中文 G2P 的 `pattern_filter` 会**删掉所有英文字母**；混读文本若走纯中文路径，
会得到空音素序列并产生噪声。因此显式指定中文 + 含 ASCII 字母时必须切到混读模式。

混读模式的处理是**按语言切块**（`split_language`）：
`re.split(r'[a-zA-Z]+')` 与 `re.findall` 交错拼装 —— 字母串归英文块，
其余（汉字、数字、标点、空白）全部留在中文块，纯空白的中文块被丢弃。
每个中文块各自走一遍完整的 `chinese_to_phones`（内含 TextNormalizer），
英文块走英文 G2P，BERT 特征对英文块是全零行，最后逐块拼接：

```python
phones_seq = np.concatenate(list_phones, axis=1)
text_bert = np.concatenate(list_berts, axis=0)
```

**块边界的语义副作用**：变调/儿化不跨块。上游逐块调 G2P，
块与块之间不做三声连读 —— `"你好Mira很高兴见到你"` 切成
`[你好][Mira][很高兴见到你]`，「你好」块内闭环（你→2）；
若把英文删掉整段处理（`你好很高兴见到你`），变调会跨词段作用（好→2）。
两者在**上游**就不同，移植必须保持逐块行为。

本移植（2026-10-28）：`genie::detect_language()` 实现了语言切换规则；
`src/text/language_split.cpp` 逐字移植 `split_language`（与上游正则逻辑
24/24 语料逐块一致，黄金语料 `tests/golden/language_split.json`）；
前端对每个中文块单独走完整 TextNormalizer + G2P，RoBERTa 按块编码再拼接
（`PhonemeSequence::bert_chunks` 携带分块与 word2ph 对齐）。
英文块本实现暂且丢弃 —— 上游产出英文音素（见 §9.4 英文 G2P）。

---

## 7. 上游依赖清单

`requirements.txt` / `pyproject.toml`：

**通用**：`onnx`、`onnxruntime==1.22.1`、`tokenizers`、`numpy`、`soundfile`、`sounddevice`、
`soxr`、`pyyaml`、`pydantic`、`huggingface_hub[hf_xet]`、`fastapi`、`uvicorn[standard]`

**分语言 G2P**：`pyopenjtalk-plus`（日）、`nltk`（英）、`pypinyin` + `g2pM` +
`jieba_fast`（中）、`jamo` + `ko_pron` + `g2pk2` + `eunjeon`（韩）

**运行时必需**：`onnxruntime`、`numpy`、`tokenizers`、`soundfile`、`soxr`
**运行时不需要**：`onnx`（转换期）、`fastapi`/`uvicorn`（服务端）、`sounddevice`（播放）、
`huggingface_hub`（下载）、各语言 G2P 依赖（M2 之后可能变成需要）

### ONNX Runtime 版本

上游固定 `onnxruntime==1.22.1`，注释写明"指定版本号以避免 1.23 的性能倒退"。

**本项目的立场**：CI 默认用 1.22.1 复现上游基线，但这是**可复现性选择，不是永久钉死**。
审计文档里把它记为"需要复测的兼容性信息"。后续 M5 性能阶段应做 1.22 / 1.23+ 的对照测量。

---

## 8. 许可证与授权

上游为 MIT（`LICENSE`，Copyright (c) 2025 High_Logic）。
上游仓库随源码发布了六个 ONNX 模型，本项目按 MIT 收录，归属见 `NOTICE`。

**用户自己的音色模型（voice package）不在本仓库范围内**，其授权由相应作者决定。


---

## 9. 文本前端：已实现的部分与差异

M2 的文本前端分三层落地：**符号表**、**文本规范化**、**G2P**。
这一节记录每一层与上游的关系 —— 哪些是逐字等价的，哪些有偏差，为什么。

### 9.1 符号表：逐字等价

`genie::symbols` **重新实现了**上游 `G2P/SymbolsV2.py` 的 `create_master_symbol_list()`，
而不是抄它的列表。规则是确定的，所以 732 个符号的顺序必然一致。

| 项 | 上游 | 本实现 | 等价性 |
|---|---|---|---|
| 符号总数 | 732 | 732 | ✅ |
| 排序 | `sorted(set(...))`，按码位 | 按 UTF-8 字节序 | ✅ 本表内等价（唯一的多字节字符 "…" 在所有 ASCII 之后，也早于韩文/粤文） |
| 全表 SHA-256 | `0771f6cd…74da7` | 同左 | ✅ |

`tests/unit/test_symbols.cpp` 用**全表摘要**锁死，任何顺序变动都会被抓到。

> 为什么值得这么较真：符号 ID 错一位，模型收到的就是完全不同的音素序列，
> 而且**不会报错** —— 只会合成出乱码。

### 9.2 拼音切分：与 pypinyin 逐字等价（41,923 字全量比对）

上游的中文 G2P 用 `pypinyin` 的 `to_initials()` / `to_finals_tone3()` 切分声母韵母。
本实现用 C++ 复刻了这两套规则，并做了**全量比对**：

```
$ python3 tools/g2p/verify_pinyin_rules.py /tmp/dump_pinyin
比对字符数: 41923 / 41923
纯鼻音特例（无韵母，不产生音素）: 5 个
切分与音素全部一致 ✓
```

比对覆盖两层：切分（声母 / 带调韵母）与音素序列。

**5 个纯鼻音特例**（㕶 呣 嗯 噷 𠮾）没有韵母：上游用带调拼音，`to_initials` 得到空；
本实现用 tone3，得到声母。两边最终都**不产生音素**，行为一致。

### 9.3 音素推导：opencpop 查表（2026-10-27 修正为与上游逐条一致）

上游最后一步是查 opencpop 词表（`opencpop-strict.txt`，429 条，
HuggingFace 资源包 `GenieData/G2P/ChineseG2P/`）。
本实现早期**误用自推规则**替代查表——沙箱当时访问不到 HF——
这是错的：pypinyin 的韵母本身不分翘舌 `ir`，`ve` / `EE` 等
表示在自推集合里也不存在，导致 zh/ch/sh/ri/zi/ci/si 一类音节的
韵母全部错成普通 `i`、`üe`（月）整个丢字、`er`（二）丢 `EE` 声位、
`ü`（语）错成 `u`。听感上就是"咬字不清、读音跑偏"。

现已内嵌词表（`src/text/opencpop_table.cpp`，429 条，由
`tools/g2p/generate_opencpop_table.py` 生成），转换逐条对齐上游
`_pinyin_to_opencpop_phones`：

- 有声母：`uei→ui` / `iou→iu` / `uen→un` 收缩后拼声母查表
- 零声母：`pinyin_rep_map`（`ing→ying` / `i→yi` / `in→yin` / `u→wu`）优先，
  否则按 `single_rep_map` 首字母改写（`v→yu` / `e→e` / `i→y` / `u→w`）
- 查表得到 `(声母符号, 韵母符号 + 声调)`；键不存在等价上游 KeyError → 跳过该字
- **验证**：12,010 字与 Python 参考实现（pypinyin 0.55 + 同一词表）全量
  比对 **0 差异**；`tests/golden/` 无需快照，摘要锁定在
  `Pinyin.TableDigestIsStable`

### 9.4 ⚠️ 有偏差的部分

| 项 | 上游 | 本实现 | 影响 | 原因 |
|---|---|---|---|---|
| 多音字 | `g2pM`（kakaobrain 双向 LSTM，整句上下文判定） | **已实现**（2026-10-28，C++ 前向复刻，oracle 47/47 逐字一致，见 §9.7） | 模型自身的误判在**模型层**被忠实保留（如「行长在银行上班」里给银行读 xíng）；随后由上游同款的词级修正表改回来，最终读音与上游一致 | 模型仅 1.6 MB，`third_party/g2pm/g2pm.bin` 随仓库发布 |
| 变调 | `ToneSandhi.py`（18KB 规则） | **已实现**（2026-10-28；2026-10-30 接上 jieba 分词，见 §9.6） | jieba 路径韵母序列 42/44 一致；差异全部是 POS 依赖分支与 `cut_for_search` 切点（§9.6 逐条列出） | 无 POS 标注（jieba posseg 的 HMM 未移植） |
| 儿化 | `Erhua.py` | **已实现**（2026-10-28，见 §9.6） | 同上 | — |
| 词级读音修正 | `CorrectPronunciation.py`（polyphonic.pickle，4.5 万词） | **已实现**（2026-10-28，随 g2pm 容器携带；词边界走 jieba 分词结果，见 §9.8，见 §9.7） | 缺 jieba 的音色包退回整段最长匹配近似（罕见，且词表条目本身是真词，方向为正向修正） | g2pM 的已知误判靠它兜底（「长大」「分行」类），缺了它模型反而不如单字表 |
| 繁简转换 | `char_convert.py`（65KB 词表） | **已实现**（2026-10-27，10,737 组，见 §15） | 无（已对齐） | — |
| 日期/时间/电话/量词 | `Normalization/` 的专门规则 | **已实现**（2026-10-27，见 §15） | 无（已对齐） | — |
| jieba 分词 | `jieba_fast` + 词性标注 | **已实现**（2026-10-28，算法+词典直译，oracle 103/103，见 §9.8；POS HMM 待接入） | 分词边界与上游一致；POS 依赖的变调分支仍取默认行为（切片 B） | **勘误（2026-10-28）**：此前记载"上游 word2ph 按分词结果统计"有误 —— 上游 `ChineseG2P.g2p` 的 `word2ph` 同样是**逐字**统计（`zip(initials, finals)` 循环内逐字 append），jieba 影响的是 POS 变调分支、儿化词边界与词级读音修正，不影响 word2ph 粒度 |
| 混读分块 | `split_language` 逐块 G2P + 逐块 RoBERTa | **已实现**（2026-10-28，见 §6.3；英文块 2026-10-29 起接入英文 G2P，见 §17） | 中文块获得完整 TextNormalizer；块边界阻断变调与上游一致 | — |
| 英文 G2P | `nltk` + CMU 词典（3.5MB） | **已实现**（2026-10-29，规范化 + TweetTokenizer + CMU 词典 + POS 标注 + GRU 兜底，oracle 94/94，见 §17） | OOV 词缺 wordsegment 词典时直接交给 GRU（音色包可选角色 `english_word_segment`） | 词典与模型共 12 MB，随仓库发布于 `third_party/english/` |
| 日文 G2P | `pyopenjtalk`（OpenJTalk） | **已实现**（2026-10-29，OpenJTalk 运行时内置 + 全上下文标签解析，oracle 124/124，见 §19） | 词典 103 MB 不入仓库，由音色包角色 `japanese_dictionary` 在运行时给出 | OpenJTalk 运行时源码内置（BSD），词典走音色包 |
| 韩文 G2P | `jamo` + `ko_pron` + `g2pk2` | **已实现**（2026-10-29，g2pk2 全链路移植，oracle 53/53，见 §18） | mecab-ko 形态标注未移植（依赖标记的四条音变规则不触发） | 规则与数据全部内置，无需外部资源 |

**可插拔接口**：`genie::text::TextFrontend`。使用者可以自己实现它，
把更准的 G2P（或者把 g2pM 导出的结果）接进来。

### 9.5 拼音词典的来源与范围

- 数据：**pypinyin**（MIT）的单字拼音表
- 字表范围：**jieba 词典**（MIT）覆盖的 12,010 个汉字，覆盖实际文本 99.99% 以上的 token 频次
- 词典之外的字会被跳过（不产生音素），不会让整次合成失败
- 生成脚本：`tools/g2p/generate_pinyin_table.py`（可复现）
- 音素表摘要：`b381f96b…0b83a`，由 `tests/unit/test_pinyin.cpp` 锁死

### 9.6 变调与儿化：ToneSandhi / Erhua 实装（2026-10-28；2026-10-30 按 jieba 实装重写）

上游 `G2P/Chinese/ToneSandhi.py`（源头 PaddleSpeech，Apache-2.0）与
`Erhua.py` 负责中文的"不/一"变调、三声连读、轻声和儿化。
两者都作用在 **jieba 分词结果**（`(word, pos)` 流）上，基线 commit
`d347fd0f`。本移植**已移植 jieba**（§9.8），所以词边界有两条路径：

- **jieba 路径（生产）**：音色包带 `jieba` 角色时，前端按
  `jieba::Tokenizer::cut` 的词边界切分 —— 与上游同源的统计分词。
- **降级路径**：音色包不带 `jieba` 时退回"上游四张词表 ∪ 单字"的
  贪心最长匹配。

另外：

- **词表**：内嵌上游四张词表（`must_neural_tone_words` 420 词、
  `must_not_neural_tone_words` 40 词、erhua 两张 8+44 词），
  由 `tools/g2p/generate_sandhi_tables.py` 生成
  （`src/text/sandhi_table.cpp`）。
- **规则**：`_bu_sandhi` / `_yi_sandhi` / `_neural_sandhi` /
  `_three_sandhi` / `merge_erhua` 全分支照搬；POS 依赖分支按下表取默认。

**POS 缺失的默认行为**（上游用 `jieba_fast.posseg` 的词性标注）：

| 分支 | 上游条件 | 本实现 | 已知误伤 |
|---|---|---|---|
| 叠词轻声 | POS ∈ {n, v, a} | 直接应用（must_not 表仍豁免） | "慢慢"这类副词叠词被误轻声 |
| 了/着/过轻声 | POS ∈ {ul, uz, ug} | 单字单元直接应用 | "过河"的 过 被误轻声 |
| "们子/上下里/的地得"后缀 | POS ∈ {r,n}/{s,l,f}/(无) | 直接应用 | 单字"地"在"地下"类词里被误轻声（jieba 会把"地下"当一个词） |
| "一 V 一 V"合并 | 两侧 POS = v | **不移植** | "想一想"的"一"读四声（上游轻声） |
| 儿化跳过 | POS ∉ {a, j, nr} | 只查词表 | 极少数形容词/人名后缀"儿"被儿化 |
| `_split_word`（三字词切分） | `jieba.cut_for_search` | 最长词表前缀，退化 [1, size-1] | "洗脸水"切 1+2（上游 2+1） |

**oracle 比对**（`tools/verify_sandhi.py`，语料
`tools/g2p/sandhi_corpus.txt` 44 段）：两侧喂**同一份逐字韵母**
（取自本项目拼音表，隔离 g2pM 差异），上游用 jieba 顶替
jieba_fast 跑真分词：

| 路径 | 变调域边界（pre_merge 之后）一致 | 逐字韵母序列一致 |
|---|---|---|
| jieba（生产） | **40/44 = 90.9%** | **42/44 = 95.5%** |
| 降级（无 jieba） | 17/44 = 38.6% | 30/44 = 68.2% |

（"降级路径"那一列是 2026-10-28 首次移植时测的数；当时工具脚本
的边界比对是坏的 —— 它把词面**拼接**后再比，拼接结果恒等于原文，
于是"44/44 一致"这个数其实是恒真的。2026-10-30 修好比对逻辑后
重测，两条路径的数字如上表。）

jieba 路径剩余 4 处边界差异与 2 处韵母差异，成因都已定位：

- **边界差异（4 处）**：`jieba.posseg.lcut` 与 `jieba.lcut` 在少量
  低频/OOV 词上的切分不同 —— posseg 走的是**带 POS 约束的 HMM**，
  会把"里种着"合成一个词、把"那本新"切成"那 / 本新"。本移植的
  jieba 对齐的是 `jieba.lcut`（`tools/g2p/verify_jieba.py` 逐词
  103/103 一致，黄金语料 `tests/golden/jieba.json` 锁进 CI），
  所以差异出在上游用了 posseg 这一侧，不是分词实现的错。
- **韵母差异（2 处）**：
  1. "洗脸水"——上游 `_split_word` 用 `jieba.cut_for_search` 取最短
     子词定切点（2+1），本移植查词表（1+2），"洗"的声调因此不同；
  2. "老板娘慢慢地走进了那家小饭馆儿"——"慢慢"在上游被标成 POS=d，
     叠词轻声的条件是 POS ∈ {n, v, a}，所以上游不转轻声；本移植无
     POS，直接转轻声。

降级路径的 14 处韵母差异里，有 12 处由 jieba 直接消掉（"看不懂"
的整词、"地下/家里/上来/下去"的后缀字等），剩下的 2 处与上面相同。

**潜在缓解**（未做，属于与上游数据的补充，需要时再立项）：移植
jieba 的 posseg HMM（约 5.5 MB 数据）以拿到 POS，消掉上表里最后
一类差异；或往分词词表里补一小批"单字后缀"复合词。g2pM 多音字与
`correct_pronunciation`（polyphonic.pickle 整词修正）见 §9.7。

**黄金语料**：两份，都锁进 CI ——
`tests/golden/sandhi.json`（降级路径，44 段的处理单元与韵母）与
`tests/golden/sandhi_jieba.json`（jieba 生产路径）。
两条路径的行为都要锁：缺 jieba 的音色包与带 jieba 的音色包都得稳定。
分支级单测 36 项。


---

### 9.7 多音字：g2pM 实装（2026-10-28）

上游 `ChineseG2P.g2p` 的每字读音来自 pip 包 `g2pM`（kakaobrain/g2pM 训练、
mozillazg 打包，Apache-2.0）：**单层双向 LSTM + 两层全连接**，权重共约
1.6 MB（不是 BERT，也不是 ONNX —— pip 包以 numpy 权重 `np_ckpt.pkl` 与
三张词典发布，推理在 numpy 里手写）。本移植：

- `tools/deploy/convert_g2pm.py` 把四个 pkl 转成单一容器
  `third_party/g2pm/g2pm.bin`（1.8 MB，回读校验逐字节零失真），随仓库
  发布 —— 黄金语料比对因此进 CI（与六大模型图壳同先例）。
- `src/engine/g2pm.cpp` 逐行复刻 `g2pM.py` 的前向计算（embedding →
  双向 LSTM → 多音字位置的隐状态 → FC(relu) → FC → argmax）。上游
  numpy 以 float64 累积，本实现载入时把 float32 权重升为 double，
  保持一致累积语义。
- 音色包可选角色 `g2pm`；缺失时回退单字拼音表最常用读音（与
  RoBERTa / HuBERT 的可选降级同款模式）。
- oracle：`tools/g2p/verify_g2pm.py` 以 `G2pM()(seg, char_split=True)` 为
  基准，47 句多音字语料**逐字一致**（黄金语料
  `tests/golden/g2pm.json` 锁定）。

**模型层的判定被逐字复刻**：g2pM 自身的误判原样保留在**模型层** ——
例如「行长在银行上班」里模型给「银行」的行读 xíng、「银行分行下午三点关门」
里模型给「分行」读 xíng。这些判定被黄金语料 `tests/golden/g2pm.json`
逐字锁定：修"模型错误"等于偏离上游，不做（§8 同款原则）。

注意**模型层 ≠ 最终输出**：下一段的词级修正表会把这些误判改回来，
所以最终读音与上游一致（上游同样是"模型 → 修正表"两层结构）。

**词级读音修正层（2026-10-28 补齐）**：g2pM 有已知误判 ——
「长大」给 cháng（应为 zhǎng）、「银行分行」给 fēn xíng（应为 fēn háng）；
单字最常用读音表在这两处反而正确。上游的对策是
`CorrectPronunciation.py` + `polyphonic.pickle`（HF
`GenieData/G2P/ChineseG2P/polyphonic.pickle`，45,046 词）：
g2pM 输出之后、变调之前按词表修正（整词命中整体替换，否则逐字兜底）。
本实现把词表打进 g2pm.bin 容器（v2，3.1 MB），`Polyphone` 接口新增
`correct_pronunciation` 虚方法，前端在读音装配前调用 —— 与上游
流水线位置一致。

与上游的差异在**词边界的来源**：

- 上游按 jieba 词传入修正层；
- 本移植在音色包带 `jieba` 角色时同样按 jieba 词传入
  （`word_reading()` 整词命中整体替换，未命中则逐字兜底
  `char_reading()`）—— 与上游同款路径；
- 音色包**不带** jieba 时拿不到词边界，退回**整段最长匹配**近似
  （`correct_pronunciation()`，词表条目本身是真词，属正向修正；
  影响评估记录于 §9.4）。

单测锁定：`G2pmFrontend.UsesContextReadings`（银行/分行均读 háng）、
`G2pmFrontend.CorrectsModelMisjudgments`（长大读 zhǎng）。

与上游的残留差异：POS 依赖的变调分支仍取默认行为（jieba，见 §9.6）。

### 9.8 jieba 分词：DAG/DP/HMM 直译（2026-10-28）

上游 `ChineseG2P.g2p` 用 `jieba_fast.posseg.lcut` 对每个标点段分词；
分词结果（`pre_merge_for_modify` 的输入）决定变调合并的边界、儿化的
词检查与词级读音修正的粒度。本移植逐行复刻 jieba 0.42.1（jieba_fast
与其结果一致、C 加速；算法/数据同源，MIT）：

- `tools/deploy/convert_jieba.py`：dict.txt（349,046 词，含**前缀片段
  freq=0** —— 上游 gen_pfdict 的行为，DAG 扫描靠它们前进）与
  finalseg HMM 概率 → 单一容器 `third_party/jieba/jieba.bin`
  （8.3 MB，回读校验零失真）。
- `src/text/jieba.cpp`：get_DAG（词典前向扫描）→ calc（逆序对数概率
  DP）→ `__cut_DAG`（单字缓存块按词典存在性走 HMM 或逐字产出）→
  finalseg viterbi（BMES）。三处极易走样的语义都以 Python 行为为准：
  calc 平局取更大 x（元组 max）；viterbi PrevStatus 列表顺序（平局取
  最后一个）；缺失概率固化 MIN_FLOAT（-3.14e100）。
- 前端接线（上游顺序）：jieba 分词 → 逐字读音（g2pM/单字表）→
  pre_merge（跨词合并）→ **词级读音修正**（整词查表 + 单字兜底，
  替代 §9.7 的整段最长匹配近似）→ 变调 → 儿化。无 jieba 的音色包
  退回词表贪心匹配 + 整段最长匹配修正（降级路径，行为与 #23 一致）。
- oracle：`tools/g2p/verify_jieba.py` 以 jieba Python（HMM=True 默认
  路径）为基准，103 句语料**逐词一致**（含 DP 走向与 HMM OOV 路径），
  黄金语料 `tests/golden/jieba.json` 随仓库锁定进 CI。

残留差异：jieba_fast 的**词性标注**（posseg 的 HMM POS，约 5.5 MB
数据）未移植 —— 词典词的词性已随容器携带（dict.txt 第三列），OOV 词
的 POS 与依赖 POS 的变调分支（一/量词、 moments 后缀等）留待切片 B。

**posseg 与 cut 不是同一件事**（2026-10-30 实测）：上游调的是
`jieba_fast.posseg.lcut`，posseg 走的是**带 POS 约束的 HMM**，与
普通 `jieba.lcut` 的切分结果在少量低频/OOV 词上不同 —— 例如
"里种着"（posseg 合成一个词 vs cut 切成"里种 / 着"）、"那本新"
（posseg 切"那 / 本新" vs cut 切"那本 / 新"）。本移植对齐的是
`jieba.lcut`（oracle 103/103 逐词一致）。这 4 处边界差异是 §9.6
韵母比对里剩余差异的成因之一；移植 posseg 的 HMM 可以一并消掉。

## 10. T2S 自回归解码：逐行对齐的结果

解码循环是整个推理链路里最容易写错的一段。本节记录**从上游 ONNX 图里
读出来的确切语义**（不是从代码猜的），以及本移植的处理方式。

### 10.1 图里读出来的结构

`third_party/genie-tts-models/v2/t2s_stage_decoder_fp32.onnx` 的关键节点：

```
y    = Concat(iy, /Cast_2_output_0)        ← iy 是**完整**的 token 序列
y_emb= Concat(iy_emb, Gather(word_emb, ...))
stop_condition_tensor = Greater(ReduceSum(Or(Equal(ArgMax, 1024), ...)), 0)
```

`t2s_first_stage_decoder_fp32.onnx`：

```
y     = Concat(prompts, /Cast_5_output_0)   ← prompt token + 首个生成的 token
y_emb = Gather(word_embeddings.weight, prompts)
```

由此确定三件事：

1. **EOS token = 1024**。模型的停止条件就是"新 token == 1024"，
   与上游 `np.where(semantic_tokens >= 1024)` 用的是同一个阈值。
2. **`iy` 要喂完整的 token 序列**，不能喂标量 —— 模型内部是
   `Concat(iy, 新 token)`，喂标量输出长度就不对。
3. **KV cache 的名字会变**：首阶段输出 `present_k_layer_N`，
   下一阶段输入 `past_k_layer_N`。名字不一样，必须显式映射。
   层数不写死，从模型的实际输出名里扫出来。

### 10.2 上游那几行索引运算的语义

```python
for idx in range(0, 500):
    ...
    y, y_emb, stop_condition_tensor, *present_key_values = stage_decoder.run(None, input_feed)
    if stop_condition_tensor: break
y[0, -1] = 0
return np.expand_dims(y[:, -idx:], axis=0)
```

| 上游写法 | 实际语义 | 本移植的处理 |
|---|---|---|
| `for idx in range(0, 500)` | 正常跑完时 `idx == 499`（**不是** 500）；break 时 `idx` 是 break 的那次下标 | 照搬。用 `while` + 显式计数模拟 |
| `y[0, -1] = 0` | 把最后一个 token **置零**（不是截断）。EOS 因此不会命中后续的 `>= 1024` 截断，序列末尾会留一个 0 | 照搬 |
| `y[:, -idx:]` | numpy 里 `-0 == 0`，所以 `idx == 0` 时等价于 `x[:, 0:]`（**全取**），会连 prompt 一起带上 | 照搬 |
| `np.expand_dims(..., axis=0)` | 得到 `[1, 1, L]` —— 这正是 `vits` 的 `pred_semantic` 输入形状 | 照搬 |

> 这几处都是上游的**实际行为**，不是笔误。改了会影响听感，所以逐条照搬并加测试锁住。

### 10.3 本移植的三处刻意偏离

| 项 | 上游 | 本移植 | 原因 |
|---|---|---|---|
| 输入绑定 | `zip(input_names, [y, y_emb, *kv])` —— **按位置** | **按名字** | 位置绑定依赖 `get_inputs()` 的顺序恰好是 `iy, iy_emb, past_k_*...`。上游一旦调整顺序就会静默错位 —— 不报错，只输出垃圾 |
| KV cache 名字 | 直接把 `present_*` 的输出当 `past_*` 的输入传 | 显式做 `present_X` → `past_X` 映射 | 名字确实不同，必须显式转换（上游靠 `zip` 的位置巧合绕过了） |
| 取消 | `stop_event.is_set()` → `return None`（放弃整次合成） | 返回空序列，由上层决定 | 本项目的 API 不返回 `None` |

### 10.4 文本前置与 EOS 截断

```python
text = '。' + text                      # 防止漏第一句
...
eos_indices = np.where(semantic_tokens >= 1024)
if len(eos_indices[0]) > 0:
    first_eos_index = eos_indices[-1][0]
    semantic_tokens = semantic_tokens[..., :first_eos_index]
```

- **文本前置一个句号**：`。` 在符号表里（ID 321），会占一个音素位。
  照搬 —— 它影响首音素分布。
- **EOS 截断**：形状是 `[1, 1, L]`，所以 `eos_indices[0]` 是第 0 维
  （长度 1），`[-1]` 取最后一个元素 —— 也就是"第一个 EOS 的位置"。
  照搬。

### 10.5 句尾挤压：EOS 采样时机决定音频长度（2026-10-28 观测）

**现象**：合成"十月二日的語音測試。"时，某次输出末音节"试"的韵母
没有自然收尾 —— 语音能量一直持续到文件最后一个采样点附近，
听感是"说了一半音频就结束"。

**机制**：vits 的输出帧数由 `pred_semantic` 的长度决定。实测三组合成，
音频时长都**精确等于** `语义 token 数 × 40 ms`
（111 → 4.44 s、125 → 5.00 s、54 → 2.16 s）。因此 AR 解码在第几轮
采样到 EOS（采样在图内完成，见 §5），音频就在哪里结束 —— EOS 来得早，
最后一个音节的元音就被"挤掉"，没有衰减尾巴。

**证据**：同一文本连续生成 5 次，token 数分别为 54 / 56 / 56 / 57 / 58。
按 40 ms 窗做 RMS 包络：54 那次末尾 39 个窗（约 1.56 s）全部带语音能量、
直达文件末尾；其余 4 次结尾都有自然回落（末窗 RMS ≤ 0.012）。

**结论**：不是本移植的缺陷 —— 已逐项排查：引擎全量拷贝 vits 输出、
WAV 帧数完整、音素序列完整、`y[0,-1]=0` 与 EOS 截断逻辑与上游逐条
一致（§10.2 / §10.4）。上游 Python 跑的是同一批 ONNX 图，EOS 时机
同分布，句尾挤压是**上游固有现象**（GPT-SoVITS 社区的常见反馈）。
按 §8 原则不顺手修正。

**可能的缓解**（均属与上游不一致的增强，M6 阶段如需要再单独立项）：
检测"语音能量未收尾即截断"时自动重采一次；或末 token 之后补最少留白
（如 80 ms）。

---

## 11. 参考音频条件

上游的 `tts(text, prompt_audio, ...)` 必须传 `prompt_audio` —— 它是
`Audio/ReferenceAudio.py` 里的 `ReferenceAudio`，装着四样东西：

| 字段 | 喂给 | 来源 | 缺失时 |
|---|---|---|---|
| `phonemes_seq` | `t2s_encoder` 的 `ref_seq` | 参考文本走文本前端 | 空序列 |
| `text_bert` | `t2s_encoder` 的 `ref_bert` | Chinese RoBERTa | **全零特征** |
| `ssl_content` | `t2s_encoder` 的 `ssl_content` | chinese-hubert-base.onnx | **全零特征** |
| `audio_32k` | `vits` 的 `ref_audio` | 参考音频重采样到 32 kHz | 无（必需） |

> **参考音频契约的隐藏要求（2026-10-27 补）**：上游 `load_audio()` 在重采样后
> 会给波形**拼接 0.3 秒静音**（`SILENCE_TO_APPEND_S`，16 kHz 下 4800 采样），
> `audio_32k` / `audio_16k` 乃至 HuBERT 特征都带这个尾巴。静音帧是契约的一部分：
> 它告诉自回归解码器"参考内容到此为止"。本实现曾漏掉拼接，
> 结果真 ssl_content 会让 T2S 在第 1 轮就采样到 EOS（详见 §14.4）。

**降级路径照搬**：RoBERTa 与 HuBERT 都是可选模型，缺了就用全零特征。
上游的 `GetPhonesAndBert.py` 就是这么降级的。

本实现的对应关系：

```
voice-package/reference/audio.wav   → ref_audio（重采样到 32 kHz）
voice-package/reference/text.txt    → 文本前端 → ref_seq
（缺 RoBERTa / HuBERT）             → 全零特征，形状与真特征一致
```

> **形状必须显式给对**。形状错了模型不会报错，只会输出垃圾。
> `ref_bert` 是 `[1, 音素数, 1024]`，`ssl_content` 是 `[1, 768, 帧数]`。
>
> **`ssl_content` 的帧数语义**（2026-10-24 修正）：上游由 chinese-hubert-base
> 对 **16 kHz** 波形推理产出（`ReferenceAudio.__init__` 里
> `cn_hubert.run(None, {'input_values': self.audio_16k})`），HuBERT 步长
> 320 采样 → **50 帧/秒**，帧数 = 16 kHz 采样数 / 320。
> 本实现曾按 32 kHz 帧数除 320（=100 帧/秒），零张量长度多一倍；
> 虽然数值是零，但长度参与 t2s_encoder 对 `ssl_content` 的注意力归一化，
> 零帧数量翻倍会稀释韵律注意力的权重分布。已改为按 16 kHz 波形计算。

---

## 12. V2ProPlus：全局说话人嵌入

上游的 V2ProPlus 比 V2 多一条链路（`ReferenceAudio.update_global_emb`）：

```python
sv_emb = speaker_verification_model.run(None, {'waveform': self.audio_16k})[0]
self.global_emb, self.global_emb_advanced = prompt_encoder.run(None, {
    'ref_audio': self.audio_32k,
    'sv_emb': sv_emb,
})
```

然后 V2ProPlus 的 vits 吃 `ge` / `ge_advanced`（**不是** `ref_audio`）：

| 张量 | 形状 | 喂给 |
|---|---|---|
| `ge` | `[1, 1024, 1]` | vits（V2ProPlus） |
| `ge_advanced` | `[1, 512, 1]` | vits（V2ProPlus） |

### 三条采样率

上游把同一段参考音频喂给三个模型，各要一个采样率：

| 用途 | 采样率 | 张量名 |
|---|---|---|
| HuBERT | 16 kHz | `input_values` |
| 说话人验证 | 16 kHz | `waveform` |
| prompt_encoder / vits | 32 kHz | `ref_audio` |

本实现对参考音频同时准备 32 kHz 与 16 kHz 两个版本。

### 降级与硬依赖

- **HuBERT 缺失** → `ssl_content` 是全零特征（软降级）。
- **说话人验证模型缺失** → **V2ProPlus 分支不可用**（硬依赖）。
  上游在 `update_global_emb` 里判断 `if model_manager.load_sv_model():`，
  算不出来时 `global_emb` 保持 `None`，vits 就会拿到 `ge=None`。
  本实现在**加载期**就报错 —— 早失败比合成出垃圾好。

---

## 13. 张量的**秩**：最容易错、也最难发现的地方

上游的模型契约（`docs/upstream-audit.md` §3）是**逐维**核对的，形状写错一个数字
模型就会拒绝输入。但**秩**（rank）错得更多、更隐蔽 —— 比如把 `[1, L, 1024]`
喂给声明为 `[L, 1024]` 的输入。

本项目在写回归层时，从上游的真实 ONNX 图里挖出了三处**秩错**：

| 张量 | 正确形状 | 曾经的错误 | 上游依据 |
|---|---|---|---|
| `text_seq` | `[1, L]`（秩 2） | `[1, 1, L]`（秩 3） | `phones_seq = np.array([phones])` |
| `ref_bert` | `[L_ref, 1024]`（秩 2） | `[1, L_ref, 1024]`（秩 3） | ONNX 图 `ref_bert: [ref_length, 1024]` |
| `text_bert` | `[L_text, 1024]`（秩 2） | `[1, L_text, 1024]`（秩 3） | ONNX 图 `text_bert: [text_length, 1024]` |

> `pred_semantic` 确实是秩 3（`[1, 1, L]`）—— 上游对它做了
> `np.expand_dims(y[:, -idx:], axis=0)`。**别和 `text_seq` 搞混**，
> 它俩喂给的是同一个张量，形状却差一个维度。

**为什么这些 bug 能活这么久**：假会话之前不校验形状，喂什么它都收。
现在假会话会照着真模型的 spec 校验 dtype / 秩 / 静态维度，
形状错位在**单测阶段**就会炸。

**教训**：模型契约不能只写成"名字对得上"。形状和秩都要锁。

---

## 14. 中文韵律特征：RoBERTa 实装（2026-10-27）

上游中文路径的 `text_bert` / `ref_bert` 由 Chinese RoBERTa 产出
（`GetPhonesAndBert.py` 的 `_get_phones_and_bert_pure_lang`）：
tokenizer 编码规范化文本 → RoBERTa 推理 → 按 `word2ph` 展开。
本实现补齐了这条链路；**没有 RoBERTa 时仍走全零降级**（与上游一致）。

音色包约定：

```
manifest.json  → {"role": "roberta", "file": "models/roberta.onnx"}
models/vocab.txt → BERT 词表（与模型同目录，行号即 ID）
```

驱动加载的是**角色是否存在**（与 prompt_encoder 一致）；
manifest 的 `capabilities.roberta` 只是信息性标注。

### 14.1 分词器

上游用 HuggingFace `tokenizers` 加载 tokenizer.json（chinese-roberta-wwm-ext
标准流水线：BertNormalizer → BertPreTokenizer → WordPiece → [CLS]...[SEP]）。
本实现按同一流水线自带 C++ 版（`src/text/roberta_tokenizer.cpp`），
词表读 vocab.txt。**黄金数据**：`tests/golden/roberta_tokenizer.json`
由 Python `tokenizers==0.23.3` 生成，C++ 逐 ID 对齐（含英文 WordPiece
子词、标点独立、大小写与变音符剥离）。

与上游的已知差异：

- `strip_accents` 不做完整 NFD 分解，只覆盖 Latin-1 / Latin-Extended-A
  常用预组合字符。TTS 中文输入里带变音符的拉丁文极罕见。
- 标点集合是 ASCII `ispunct` + 常用 CJK 标点兜底；经文本规范化后
  CJK 标点大多已映射为 ASCII。

### 14.2 word2ph 展开

复刻上游 `_expand_roberta_output` 的三布局归一化，**判定顺序照抄**：

1. `T == 音素数` → 视为已是音素级，直接用
2. `T == len(word2ph) + 2` → 去 [CLS]/[SEP] 后按 word2ph 重复
3. `T == len(word2ph)` → 直接按 word2ph 重复
4. 其余 → 报错（上游是 ValueError）

> **上游怪癖（照抄）**：判定 1 优先于 2/3。本实现 word2ph 按单字统计
> （见 §6 差异记录），中文二字词恰好满足 `音素数 == 字数 + 2`
> （每字 2 音素），会命中判定 1 把 [CLS]/[SEP] 行一并透传 ——
> 上游用同样顺序判定，行为一致，不改。

RoBERTa 输入三件套：`input_ids` / `attention_mask` / `token_type_ids`
（全零，litagin 导出需要第三个输入）。输出 `[T, 1024]`（或
`[1, T, 1024]`，batch=1 时数据连续，按行数展开即可）。

### 14.3 与上游不一致的**随机性**结论（勘误）

此前文档称"唯一可控的随机性来源是 ORT 的全局随机种子"——经核实
onnxruntime 1.31：图内采样节点未带 `seed` 属性时由 ORT 自行生成，
session config 亦无随机种子键。**合成结果目前不可复现**，
`SynthesisOptions.seed` 与 CLI `--seed` 是预留字段（README 已勘误）。

### 14.4 参考音频内容特征：chinese-hubert-base 实装（2026-10-27）

上游 `ssl_content` 由 chinese-hubert-base 对 16 kHz 波形推理产出
（`ReferenceAudio.__init__`：`cn_hubert.run(None, {'input_values': audio_16k})[0]`），
输出**直接**喂给 t2s_encoder —— 转置已在导出时烘进图
（`last_hidden_state: [1, 768, T']`，T' = 16k 采样数 / 320 = 50 帧/秒）。
本实现补齐该链路：音色包可选 `chinese_hubert` 角色（图 + 同目录外部权重
`chinese-hubert-base_weights.bin`，fp32 字节 = fp16 发布文件 ×2，
部署侧预转换，方法与 §14 的 RoBERTa 相同；已核验图内 209 个外部
张量偏移 0 起点连续无缝，整文件转换与上游逐张量注入等价）。

**移植时踩掉的坑（记录以飨后人）**：接通真特征后 T2S 在第 1 轮就采样到
EOS，`y[:, -idx:]`（idx=0）把整段 prompts 当成输出 —— 合成结果变成复读
参考音频。排查结论：不是权重（外部数据偏移连续）、不是形状（图声明
`[1, 768, ssl_length]` 相符）、不是异常帧（216 帧范数均匀 ~7），
而是 §11 末尾的**静音尾巴**：上游 load_audio() 拼的 0.3 秒静音帧
给解码器提供了"参考内容结束"的边界信号。补上拼接后，
生成 token 数恢复随文本伸缩（短句 ~50、长句 ~100）。

---

## 15. 中文文本规范化：TextNormalizer 实装（2026-10-27）

上游的 `ChineseG2P.normalize_text()` 第一步是 `TextNormalizer.normalize()`，
它做的事比"全角转半角 + 数字转中文"多得多：

| 能力 | 例 | 上游读法 | 此前本实现 |
|---|---|---|---|
| 年份逐位读 | `2024年` | 二零二四年 | 二千零二十四年 |
| 手机号逐位读 | `13812345678` | 幺三八幺二三四五六七八 | 一百三十八亿一千二百三十四万五千六百七十八 |
| 日期 | `2024年3月15日` | 二零二四年三月十五日 | 二千零二十四年三月十五日 |
| 时间 | `15:30:45` | 十五点半四十五秒 | 153045 |
| 电话 | `010-12345678` | 零幺零减幺二三四五六七八 | 一百零一亿… |
| 分数 | `3/4` | 四分之三 | 三/四 |
| 百分比 | `50%` | 百分之五十 | 50% |
| 区间 | `3-5个` | 三个到五个 | 3-5个 |
| 繁简 | `這個時間` | 这个时间 | （不转） |
| 量词 / 单位 | `5cm` | 五厘米 | 5cm |
| 温度 | `-3摄氏度` | 零下三度 | 负三摄氏度 |
| 数学运算 | `1+2=3` | 一加二等于三 | 1+2=3 |
| 版本号 | `1.2.3` | 一点二点三 | 1.2.3 |

少了这一步，合成出来的中文**读法就是错的** —— 年份、手机号、电话尤其明显。

### 移植方式

上游用 Python 正则实现（`G2P/Chinese/Normalization/`，源头 PaddlePaddle /
WeTextProcessing，Apache-2.0）。本实现**手写扫描**，不用 `std::regex`：

- 上游有两个模式带**反向先行断言**（`(?<!\d)`），ECMAScript 不支持；
- MSVC 的 `std::regex` 慢且实现差异多；
- 本项目一贯自己实现（JSON / SHA-256 / UTF-8 / WAV 都是）。

替换顺序**逐行对齐**上游 —— 顺序会互相影响（比如
`RE_ASMD` 先把 `400-123-4567` 里的 `-` 换成"减"，
后面的 `RE_NATIONAL_UNIFORM_NUMBER` 就再也匹配不上了）。

### 照抄的上游怪癖

| 怪癖 | 表现 | 说明 |
|---|---|---|
| `RE_DEFAULT_NUM` 在 `RE_NUMBER` 之前 | `123` → **幺**二三 | 三位以上逐位读 |
| `replace_temperature` 取 group(3) | `-3摄氏度` → 零下三**度** | group(3) 是小数组，单位是 group(4) —— 上游的编号 bug，但行为要对齐 |
| 句末标点表里没有 ASCII 冒号 | `15:30:45` 不被切开 | 上游的 `SENTENCE_SPLITOR` 用的是全角 "：" |
| `_post_replace` 先把 `/` 换成"每" | `每/天` → 每每天 | |

### 验证

`tools/verify_normalizer.py` 把上游的 Python 包当 **oracle**：
63 条语料逐条比对，**全量一致**。
语料固化在 `tests/golden/normalizer.json`，
由 `tests/unit/test_normalizer.cpp` 守着。

### 中英混读路径的偏差

上游的混读模式（`hybrid-chinese-english`）会把文本按语言**切块**，
每个中文块各自走一遍 `TextNormalizer`。

本实现已对齐该结构（2026-10-28，见 §6.3）：`split_by_language` 切块，
每个中文块各自走完整 TextNormalizer，RoBERTa 按块编码再拼接
（`PhonemeSequence::bert_chunks`）。纯中文输入不走分块（与上游
'chinese' 模式一致）；混读输入里中文块的日期 / 电话 / 数字读法与
纯中文路径完全一致（例：`"Mira打13800138000了"` → `幺三八零零幺三八零零零`）。

**残留偏差**：英文块上游产出英文音素（g2p_en），本实现暂且丢弃 ——
详见 §9.4 英文 G2P 行。切分、对齐与 BERT 分块结构已就位，
英文 G2P 落地后只需补上英文块的音素与全零 BERT 行。

### 授权

繁简转换表（10,737 组）来自上游的 `char_convert.py`（Apache-2.0）。
生成脚本 `tools/g2p/generate_char_convert.py` 逐字保留原始版权与许可声明，
授权说明见仓库根目录 `NOTICE`。

---

## 16. 待补充（M0 剩余项）

- [ ] Golden Dataset：文本 / 音素 / 中间张量 / 语义 token / PCM 的 Python 参考输出
  - 音素部分已完成：见 §9.2 的 41,923 字全量比对
- [ ] 性能基线：模型加载、首包延迟、RTF、各子模型耗时、峰值 RSS
- [ ] ONNX Runtime 1.22 vs 1.23+ 的性能对照
- [ ] 重采样与 `soxr 'hq'` 的量化误差

## 17. 英文 G2P 实装（2026-10-29）

### 17.1 上游的英文链路

上游的英文 G2P 全部在 `G2P/English/` 下，链路是：

```
text_normalize(text)                # EnglishG2P.py
  ├─ REP_MAP                         # "[;:：，；]"→","、引号→'、"。"→"."、"！"→"!"、"？"→"?"
  ├─ normalize(text)                 # Normalization.py：20 多条规则（见 17.3）
  └─ replace_consecutive_punctuation(text)
word_tokenize(normalized_text)      # nltk TweetTokenizer
CleanG2p.__call__(tokens)           # EnglishG2P.py
  ├─ 两张词表对齐（原始词 ↔ 规范化后的词）
  ├─ "I" + "the first" 特判
  ├─ 同形异音词按 POS 选读音（read / complex / lead / presents）
  └─ _query_word(word)
       ├─ CMU 发音词典（上游用的是 engdict_cache.pickle + engdict-hot.rep）
       ├─ "'s" 处理
       ├─ 连字符拆分
       ├─ WordSegmenter（ngram Viterbi，wordsegment 包的 unigram/bigram）
       └─ g2p_en 的 seq2seq GRU（OOV 兜底）
g2p(text)                           # 过滤 + 符号表核对
```

数据来源（上游英文链路的全部外部依赖）：

| 数据 | 上游来源 | 本仓库 | 授权 |
|---|---|---|---|
| CMU 发音词典 | `GPT-SoVITS/text/cmudict.rep` → `engdict_cache.pickle`（143,202 词）+ `engdict-hot.rep`（热词表） | `third_party/english/cmudict.bin`（143,205 词） | BSD（CMU） |
| 词性标注权重 | nltk `averaged_perceptron_tagger_eng` | `third_party/english/pos_tagger.bin` | Apache-2.0 |
| seq2seq GRU | `Kyubyong/g2p` 的 `checkpoint20.npz` | `third_party/english/english_g2p.bin` | MIT |
| ngram 分词词典 | pypi 包 `wordsegment` | **不随仓库发布**（见 17.5） | MIT |

词典选择上刻意用了上游的 `engdict_cache.pickle` 而不是标准的
`cmudict.dict`：后者缺 "onest" 这类词，会让 OOV 路径与上游分叉。

### 17.2 本移植的实现

- `src/text/english/normalization.cpp`：手写扫描器逐条对齐上游
  `Normalization.py` 的 20 多条规则（不引入 `std::regex`）。
- `src/text/english/tokenizer.cpp`：nltk TweetTokenizer 的移植。
- `src/text/english/cmu_dict.cpp` / `pos_tagger.cpp` / `seq2seq.cpp` /
  `word_segmenter.cpp`：词典、averaged perceptron 词性标注、GRU 前向、
  ngram Viterbi。
- `src/text/english/g2p.cpp`：主流程（词表对齐、同形异音词、`_query_word`
  的查表顺序、`g2p()` 的过滤规则）。
- `src/text/english_frontend.cpp`：把音素序列转成 `PhonemeSequence`
  （音素 → 符号表 id）。
- 音色包新增四个可选角色：`english_cmu_dict`（**硬依赖**）、
  `english_pos_tagger`、`english_g2p_model`、`english_word_segment`。
- 引擎按语言选前端：纯英文走英文前端；中文 / 混读走中文前端，
  混读路径里的英文块交给英文前端（此前是丢弃，见 §9.4）。

### 17.3 规范化规则的逐条对齐

上游 `normalize()` 的规则顺序（**顺序本身会影响结果**）：

1. NFD 分解 + 丢掉所有 Mn 组合记号
2. `@` → " at "
3. 9 条缩写（`Mr.` / `Mrs.` / `Dr.` / `Prof.` / `St.` / `Co.` / `Ltd.` /
   `e.g.` / `i.e.`，带 IGNORECASE）
4. 金额后缀（`$3 million`）
5. 电话号码
6. 尺寸（`10 x 20`）
7. 罗马数字
8. 年代（`1990s`）
9. 比分（`3-2`）
10. 日期（`3/15/2024`）
11. 时间（`3:30 pm`）
12. 序数号（`1.` **后面必须跟空格**）
13. 去千分位逗号
14. 货币（`$19.99`）
15. 度量（`5cm`）
16. 分数（`1/2`）
17. 小数（`3.14`）
18. 序数后缀（`21st`）
19. 字母数字混排（`A1`）
20. 首字母缩写（`ABC`）
21. 数字（`42`）
22. 小写化
23. `%` → " percent"
24. 域名循环（`x.y` → `x dot y`）
25. 字符白名单 `[^a-z0-9'.,?!:;-]`
26. 空白折叠 + strip

对齐时踩到的几个坑（都是**静默出错**级别）：

- **`\b` 不是"下一个字符不是词字符"**。Python 的 `\b` 是"两侧恰好一侧是
  词字符"。串尾没有字符（非词字符），所以 `"abc"` 末尾是边界、
  `"abc."` 末尾**不是**边界。上游的时间正则 `\b(...)?\b` 在 `"3:30 pm."`
  上会回溯：先试 `"pm."`（末尾没边界）→ 退回 `"pm"` → 句号被保留。
  按"下一个不是词字符"实现会把句号吞掉。
- **`(?<!\w)` 与 `\b` 也不是一回事**。数字规则是 `(?<!\w)-?\d+(?!\w)`：
  `"A-12"` 里负号前面是 `"A"`（词字符）→ 负号不算进数字 → 数字从 `"1"`
  开始 → `"a-twelve"`；只有 `" -12"` 才是 `"minus twelve"`。
- **金额后缀保留原始大小写**。上游 `_expand_currency_suffix` 用的是
  匹配到的**原文**，所以 `"$3 MILLION"` 会留下 `"MILLION"`，随后被
  首字母缩写规则拆成 `"m i l l i o n"`。
- **缩写规则带 IGNORECASE**，`"MR."` 也要认。
- **时间是 IGNORECASE 的**，`"12:00 AM"` 里的 `"AM"` 要被时间规则吃掉
  （而不是留给缩写规则拆成 `"a m"`）。
- **`normalize()` 与 `text_normalize()` 不是一回事**：REP_MAP 与连续标点
  折叠只发生在 `text_normalize()` 里。上游 `normalize("Well... I don't
  know.")` 保留 `"..."`，`text_normalize` 才把它压成 `"."`。

### 17.4 验证

`tools/g2p/verify_english_g2p.py` 把**上游的 Python 实现当 oracle**，
逐条比对三样东西：

- 规范化结果（`normalize()`）
- `text_normalize()` 结果
- 音素序列（逐音素）

语料 `tools/g2p/english_corpus.txt`（94 句）覆盖：缩写、货币、电话、
尺寸、罗马数字、年代、比分、日期、时间、序数、分数、小数、百分比、
域名、邮箱、URL、字母数字混排、缩写词、连字符、撇号、省略号、
全角标点、变音符号、破折号、Unicode 标点。

当前结果：**94/94 三项全部一致**。黄金语料 `tests/golden/english_g2p.json`
由该脚本生成，单测（`tests/unit/test_english.cpp`）逐条比对。

### 17.5 与上游的偏差

1. **ngram 分词词典不随仓库发布**。`wordsegment` 的 unigram/bigram 文本
   共约 12 MB，本仓库只保留词典的**格式**（`wordsegment/unigrams.txt` 等
   三个文本文件），运行时从音色包的 `english_word_segment` 角色读。
   缺失时 OOV 词直接交给 GRU —— 与上游不同（上游一定有）。
   黄金语料里 6 条依赖分词的用例被标记为 `needs_word_segment`，
   单测只比它们的规范化结果；完整比对由 verify 脚本在有词典时完成。
2. **TweetTokenizer 的 emoji 分支未移植**：肤色修饰符（U+1F3FB–U+1F3FF）
   与国旗（区域指示符对）不参与分词。
3. **`namedict` 是上游的死代码**：规范化后文本已全小写，
   `o_word.istitle()` 永不成立，所以那一路不移植。
4. **Unicode 类别是近似**：词字符判定按区间近似（拉丁 / CJK / 假名 / 谚文
   算词字符，标点 / 符号 / emoji 不算）；Mn 组合记号用的是 Python
   `unicodedata` 导出的**精确**区间表
   （`src/text/english/combining_marks.hpp`，由
   `tools/g2p/generate_combining_marks.py` 生成）。
5. **上游的英文路径不产 RoBERTa 特征**（`text_bert` 全零），
   所以英文前端的 `bert_chunks` 恒为空。

## 18. 韩语 G2P 实装（2026-10-29）

### 18.1 上游的韩文链路

上游 `G2P/Korean/KoreanG2P.py` 的链路：

```
g2p(text)
  ├─ _latin_to_hangul(text)     # 逐个拉丁字母 → 韩文读法（"a" → "에이"）
  ├─ _number_to_hangul(text)    # 数字 → 韩文（汉字数词 / 固有数词）
  ├─ _g2p(text)                 # g2pk2 的 G2p()
  │    ├─ idioms.txt            # 惯用词的读音替换（119 条）
  │    ├─ convert_eng(text)     # 英语 → 韩文（上游的 _latin_to_hangul 之后基本是空转）
  │    ├─ annotate(text, mecab) # ★ mecab-ko 形态标注（给 /J /E /P /B 打标记）
  │    ├─ convert_num(text)     # 阿拉伯数字展开（上游那一步之后已无数字）
  │    ├─ h2j(text)             # 组合字 → 组合字母
  │    ├─ 12 条特殊规则         # g2pk2/special.py
  │    ├─ 402 条规则表          # g2pk2/table.csv（"받침 + 起首辅音"）
  │    ├─ 4 组连音规则          # g2pk2/regular.py
  │    └─ compose(text)         # 组合字母 → 组合字
  ├─ _divide_hangul(text)       # j2hcj(h2j(...)) + 复合元音拆分
  ├─ _fix_g2pk2_error(text)     # "ㅇㅡㄹ ㄹ" → "ㅇㅡㄹ ㄴ"
  ├─ re.sub(r"([\u3131-\u3163])$", r"\1.", text)   # 结尾的兼容字母后补 "."
  └─ 逐字符过符号表             # 不在表里的换成 "停"
```

输出是**兼容字母**（U+3131–U+314E）——上游最后把音节拆开了，音素表里存的就是兼容字母。

### 18.2 本移植的实现

- `src/text/korean/jamo.cpp`：jamo（Apache-2.0）的 h2j / j2hcj + g2pk2 的 compose。
- `src/text/korean/korean.cpp`：上游的辅助函数 + g2pk2 的流水线（手写扫描器，不引 `std::regex`）。
- `src/text/korean/tables.hpp`：**生成文件**（119 条惯用语 + 402 条规则表 + 126 条字母映射），
  由 `tools/g2p/generate_korean_tables.py` 从 pip 上的 `g2pk2` / `jamo` 包导出。
- `src/text/korean_frontend.cpp`：音素 → 符号 ID。
- 引擎按检测到的语言选前端：韩文文本走韩文前端（不需要任何外部资源 ✓）。

### 18.3 与上游的偏差：mecab-ko 形态标注没有移植

g2pk2 的第 3 步 `annotate()` 依赖 **mecab-ko**：

- mecab 是 C++ 形态分析器（本仓库不引 C++ 第三方依赖 ✓）；
- mecab-ko 词典约 100 MB（`mecab-ko-dic` 的 sys.dic 是二进制词典 ✓）；
- 上游的 Windows 路径用 eunjeon、Linux 路径用 python-mecab-ko ✓。

本移植**不做**这一步，因此：

1. 文本里没有 `/J` `/E` `/P` `/B` 标记 ✓；
2. 依赖标记的四条音变规则不会触发 ✓：
   `rieulgiyeok`（ᆰ+ᄀ/ᄁ）、`rieulbieub`（ᆲ/ᆴ+ᄀ…）、
   `verb_nieun`（ᆫ/ᆷ/ᆬ/ᆱ+ᄀ…）、`modifying_rieul` 的 `/E` 分支 ✓；
3. 其余规则（惯用语、特殊规则、402 条规则表、连音、compose）**全部照做** ✓。

上游的 `namedict`/`english.py` 之类也不需要：`convert_eng` 在上游自己的
`_latin_to_hangul` 之后已经没有拉丁字母可转了 ✓（本移植同理 ✓）。

**影响面**：动词词干收音的鼻音化/流音化（"읽다" → "익따" 这类 ✓ 在没有
标注时也一致 ✓ —— 因为它走的是规则表 ✓ 而非 /P 规则 ✓），但
"할수록" 这类以 ㄹ 收尾的副词形态变化会与上游不同 ✓。

### 18.4 验证

`tools/g2p/verify_korean_g2p.py` 把上游的 Python 实现当 oracle
（mecab 换成 stub ✓ —— 也就是"不做标注"那条路径 ✓），
逐音素比对。语料 `tools/g2p/korean_corpus.txt`（53 句）覆盖：

- 日常句子 ✓、数字（汉字数词 / 固有数词 / 千分位 ✓）、量词 ✓；
- 拉丁字母混排 ✓、标点 ✓、汉字混排 ✓；
- 典型的音变场景：읽다 / 있다 / 그것 / 밥 / 밖 / 부엌 ✓。

当前结果：**53/53 逐音素一致** ✓。黄金语料 `tests/golden/korean_g2p.json`
由该脚本生成，单测（`tests/unit/test_korean.cpp`）逐条比对 ✓。

## 19. 日语 G2P 实装（2026-10-29）

### 19.1 上游的日文链路

上游 `G2P/Japanese/JapaneseG2P.py` 的链路：

```
g2p(text)
  ├─ if not text.strip(): return []          # Python 的 strip 去 Unicode 空白
  ├─ _text_normalize(text)
  │    ├─ "%" / "％" → "パーセント"           # 两遍 re.sub
  │    ├─ 折叠连续标点 ([,./?!~…・])\1+ → \1
  │    └─ text.lower()                        # Unicode 感知的小写
  ├─ _JAPANESE_MARKS_RE.split(norm_text)      # 按**非**日语字符切段
  ├─ _JAPANESE_MARKS_RE.findall(norm_text)    # 取出那些标点
  ├─ 逐段：pyopenjtalk.make_label(pyopenjtalk.run_frontend(segment))[1:-1]
  │    └─ 解析 OpenJTalk 的全上下文标签：
  │         p3  = re.search(r"-(.*?)\+", lab)          # 第三个音素
  │         a1  = re.search(r"/A:([0-9\-]+)\+", lab)
  │         a2  = re.search(r"\+(\d+)\+", lab)
  │         a3  = re.search(r"\+(\d+)/", lab)
  │         f1  = re.search(r"/F:(\d+)_", lab)
  │         e3  = re.search(r"!(\d+)_", lab)
  │         # 韵律符号：
  │         a3 == 1 and a2_next == 1 and p3 in "aeiouAEIOUNcl" → "#"
  │         a1 == 0 and a2_next == a2 + 1 and a2 != f1     → "]"
  │         a2 == 1 and a2_next == 2                       → "["
  ├─ 把标点插回对应位置（mark.strip() 之后）
  └─ _post_replace_phoneme：全角标点 → 半角（"：" → "," 等）
```

上游的日语路径**不产出 RoBERTa 特征**（`text_bert` 全零 ✓），
音素表用的是 `symbols_v2` ✓（与本项目的符号表一致 ✓ —— 已逐个核对 ✓）。

### 19.2 本移植的实现

- **OpenJTalk 运行时内置**：`third_party/openjtalk/`（BSD 3-Clause ✓，
  Nagoya Institute of Technology ✓）—— 源码取自 `r9y9/open_jtalk` 的
  `1.11` 分支 ✓（与 pyopenjtalk 同源 ✓）。
  只编译运行时那部分 ✓（jpcommon / mecab / mecab2njd / njd / njd2jpcommon /
  njd_set_* / text2mecab ✓），不编译命令行工具 ✓。
- `src/text/japanese/openjtalk.{hpp,cpp}`：`Mecab` + `NJD` + `JPCommon` 的封装 ✓
  （pimpl ✓），`load(dictionary_dir)` / `labels(text)` ✓。
- `src/text/japanese/japanese.{hpp,cpp}`：上游的规范化 / 切段 / 标签解析 ✓
  （手写扫描器 ✓，不引 `std::regex` ✓）。
- `src/text/japanese_frontend.cpp`：音素 → 符号 ID ✓。
- 引擎按检测到的语言选前端 ✓；音色包新增角色 `japanese_dictionary` ✓
  （词典目录 ✓）。
- CLI：`g2p --language japanese --japanese-dict <目录>` ✓。

### 19.3 与上游的偏差与坑

**偏差 1：词典不入仓库**（103 MB ✓ —— 与英文的 wordsegment 词典同理 ✓）。
词典由 `tools/deploy/fetch_openjtalk_dictionary.py` 从 pypi 的 `openjtalk`
wheel 里取出 ✓（附 SHA-256 校验 ✓），运行时由音色包给出路径 ✓
（`Mecab_load(m, dicdir)` 的 dicdir 是运行时参数 ✓ —— 不需要重新编译 ✓）。

**偏差 2：词典在仓库外，所以 CI 跑不到"真链路"**。
单测只跑"纯逻辑"那一层 ✓（文本 → 标签 → 音素 ✓），
标签本身由 oracle 生成后存进黄金语料 ✓。
需要真词典的那条链路由 `tools/g2p/verify_japanese_g2p.py` 在本地跑 ✓。

**坑 1：Python 的 `\d` 是 Unicode 感知的** ✓。
上游的日语字符集写的是 `[A-Za-z\d...]` ✓ —— Python 3 的 `\d` 匹配**所有**
Nd 类数字 ✓，不只是 ASCII ✓。所以全角 '０'（U+FF10）属于日语字符 ✓
（它**不在**上游显式列出的 `\uff11-\uff19` 里 ✓ —— 那个区间从 １ 开始 ✓）。
按显式区间判断的话，'０' 会被当成标点 ✓ → 切段位置变了 ✓ →
音素序列完全不同 ✓（"１００パーセント" 会从 "hyaku pasento" 变成别的 ✓）。

**坑 2：`str.strip()` 去的是 Unicode 空白** ✓。
上游的 `mark.strip()` 会把全角空格 U+3000 整个丢掉 ✓；只去 ASCII 空白的话
U+3000 会留在音素序列里 ✓。

**坑 3：`str.lower()` 也是 Unicode 感知的** ✓。
全角 'Ａ' → 'ａ' ✓、希腊字母 ✓、西里尔字母 ✓ 都会变小写 ✓。

**坑 4：`[1:-1]` 砍的是**音素列表**的首尾元素** ✓，不是"跳过首尾标签" ✓。
末尾标签若不是 sil（例如 "w^a-sil+xx=xx" ✓），它贡献的音素要留下 ✓、
只丢掉最后一个韵律符号 ✓ —— 按"跳过首尾标签"实现就会少一个音素 ✓。

**坑 5：`p3 in "aeiouAEIOUNcl"` 是**子串**判断** ✓（Python 的 `in` 对字符串）✓，
不是"字符集合" ✓ —— "cl" 也在里面 ✓。

**偏差 3：MSVC 上默认不编 OpenJTalk** ✓。
内置的 mecab 是 POSIX 系的代码 ✓（mmap / dirent / strdup 那一套 ✗），
在 MSVC 上能不能过**没有验证过** ✗ —— 与其让 Windows CI 红着 ✓，
不如在 MSVC 上默认关掉 ✓（`GENIE_ENABLE_OPENJTALK` 默认 OFF ✓，
显式 `-DGENIE_ENABLE_OPENJTALK=ON` 会被尊重 ✓）。
Linux / macOS 上默认开启 ✓。

**坑 6：mecab 里藏着一个 `main()`** ✓。
`third_party/openjtalk/mecab/src/mecab-dict-index.cpp` 有全局 `int main()` ✓
（词典索引工具的入口 ✓）。它一旦被链进可执行文件 ✓，
链接器就会把它当成程序入口 ✓ —— 症状是"测试程序莫名其妙地跑起了 mecab" ✓
（启动时去当前目录找 unk.def ✓、编译词典 ✓、然后崩掉 ✓）。
所以内置的源文件列表是**显式写出**的 ✓（不用 `aux_source_directory` ✓）。

**坑 7：标签解析的正则全是"最左匹配"** ✓。
`\+(\d+)\+` 在 "A:-2+3+3/B:..." 上取到的是 "3" ✓（第一个 "+数字+" ✓），
不是 "2" ✓ —— 韵律符号的判定全靠这个 ✓。

### 19.4 验证

`tools/g2p/verify_japanese_g2p.py` 把上游的 Python 实现（pyopenjtalk ✓）
当 oracle ✓，逐音素比对 ✓。语料 `tools/g2p/japanese_corpus.txt`（124 句 ✓）
覆盖：

- 平假名 / 片假名 / 汉字 / 半角片假名 / 小书假名 ✓；
- 长音符 ✓、促音 ✓、拨音 ✓、叠词 ✓；
- 标点：ASCII 与全角 ✓、省略号 ✓、中点 ✓、引号 ✓、括号 ✓；
- 数字：半角 ✓、全角 ✓、百分比（% 与 ％ ✓）；
- 拉丁字母混排 ✓、全角字母 ✓、全角空格 ✓；
- 中英日韩混排 ✓。

当前结果：**124/124 逐音素一致** ✓。黄金语料 `tests/golden/japanese_g2p.json`
由该脚本生成 ✓，单测（`tests/unit/test_japanese.cpp`）逐条比对 ✓。

## 20. RoBERTa 的语言范围（2026-10-29 修）

上游的 `_get_phones_and_bert_pure_lang` 按语言分四条路 ✓：

| 语言 | 音素 | text_bert |
|---|---|---|
| chinese | `chinese_to_phones` | **真跑 RoBERTa** ✓ |
| hybrid-chinese-english | 逐块 | **中文块真跑** ✓，英文块全零 ✓ |
| english | `english_to_phones` | **全零** ✓ |
| korean | `korean_to_phones` | **全零** ✓ |
| japanese（else 分支） | `japanese_to_phones` | **全零** ✓ |

RoBERTa 是**中文**模型 ✓ —— 喂别的语言的音素只会得到行数对不上的布局 ✓，
上游的 `_expand_roberta_output` 遇到对不上是直接 `raise` ✓。

本项目以前不管什么语言都跑 RoBERTa ✓ —— 音色包里带了 RoBERTa 又合成
英文/韩文/日文时 ✗，合成整个失败 ✓。现在只有中文（含中英混读）路径跑 ✓，
其余语言走全零 ✓（见 §20 与 CHANGELOG 的 Fixed 段 ✓）。

同一批修的还有"缺词典时静默回退到中文前端" ✓ —— 以前日文没有词典时 ✗，
日语文本被中文前端过滤成空 ✓，合成出来是静音 ✓；
现在明确报错 ✓，并说清楚缺哪个角色 ✓。
参考文本的前端也改成按**音色包声明的语言**选 ✓
（上游 `ReferenceAudio.set_text(prompt_text, language)` ✓）。

> 日文链路（打包 → 装配 → 合成）自 #28 之后的提交起才真正端到端可用 ✓：
> #28 本身只完成了**报错路径**（缺词典时报错而非静默回退 ✓），
> engine 加载期的日文前端装配与 `japanese_dictionary` 的 manifest 登记
> 是在其后补上的 ✓ —— 只有 #28 的话，包里带了词典也仍是"缺词典"报错 ✗。
> `english_word_segment` 的 manifest 登记同样在此批补上 ✓（此前目录是死重，
> 英文 OOV 静默降级到 GRU ✓，见 §17）。
