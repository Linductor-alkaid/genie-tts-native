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

本移植的 `genie::detect_language()` 已实现该规则，并有单元测试覆盖。

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

### 9.3 音素推导：等价于上游的 opencpop 查表

上游最后一步是查 opencpop 词表（`opencpop-strict.txt`，从 HuggingFace 资源包下载）。
沙箱访问不到 HuggingFace，因此本实现按 opencpop 的划分规则**推导**等价结果：

- 有声母：`(声母, 收缩后的韵母 + 声调)`，收缩规则 `uei→ui` / `iou→iu` / `uen→un`
- 零声母且以 y/w 开头：`(y|w, 剩余部分 + 声调)`
- 零声母且是元音：`(元音, 整个音节 + 声调)`（opencpop 词表就是这么写的，例如 `a a`、`ao ao`）
- `er` 是特例：`(er, er + 声调)`

### 9.4 ⚠️ 有偏差的部分

| 项 | 上游 | 本实现 | 影响 | 原因 |
|---|---|---|---|---|
| 多音字 | `g2pM`（BERT 上下文判定） | 取**最常用**读音 | 同音字在特定词里读音不同 | `g2pM` 需要下载 BERT 权重，沙箱访问不到 HuggingFace |
| 变调 | `ToneSandhi.py`（18KB 规则） | **未实现** | 三声连读等场景声调不对 | 规则依赖 jieba 的词性标注 |
| 儿化 | `Erhua.py` | **未实现** | "事儿" 之类的读音不对 | 同上 |
| 分词 | `jieba_fast` + 词性标注 | **未实现**，按单字统计 `word2ph` | `word2ph` 粒度变粗 | 影响韵律特征，不影响音素本身 |
| 繁简转换 | `char_convert.py`（65KB 词表） | **未实现** | 繁体输入读不出 | 词表过大 |
| 日期/时间/电话/量词 | `Normalization/` 的专门规则 | **未实现** | "2024年" 会读成 "二千零二十四 年" | 依赖上述词表 |
| 英文 G2P | `nltk` + CMU 词典（3.5MB） | **未实现**（留成可插拔接口） | 英文不能合成 | 词典过大，不适合内嵌进端侧库 |
| 日文 G2P | `pyopenjtalk-plus` | **未实现**（留成可插拔接口） | 日文不能合成 | 依赖 OpenJTalk 词典 |
| 韩文 G2P | `jamo` + `ko_pron` | **未实现**（留成可插拔接口） | 韩文不能合成 | 同上 |

**可插拔接口**：`genie::text::TextFrontend`。使用者可以自己实现它，
把更准的 G2P（或者把 g2pM 导出的结果）接进来。

### 9.5 拼音词典的来源与范围

- 数据：**pypinyin**（MIT）的单字拼音表
- 字表范围：**jieba 词典**（MIT）覆盖的 12,010 个汉字，覆盖实际文本 99.99% 以上的 token 频次
- 词典之外的字会被跳过（不产生音素），不会让整次合成失败
- 生成脚本：`tools/g2p/generate_pinyin_table.py`（可复现）
- 音素表摘要：`b381f96b…0b83a`，由 `tests/unit/test_pinyin.cpp` 锁死

---

## 10. 待补充（M0 剩余项）

- [ ] Golden Dataset：文本 / 音素 / 中间张量 / 语义 token / PCM 的 Python 参考输出
  - 音素部分已完成：见 §9.2 的 41,923 字全量比对
- [ ] 性能基线：模型加载、首包延迟、RTF、各子模型耗时、峰值 RSS
- [ ] ONNX Runtime 1.22 vs 1.23+ 的性能对照
- [ ] 重采样与 `soxr 'hq'` 的量化误差
