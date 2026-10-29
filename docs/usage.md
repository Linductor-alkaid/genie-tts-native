# 使用说明

三种用法：命令行、C++ SDK、C ABI。

> 模型权重下载、fp16→fp32 转换与音色包组装（推理的前置准备）见
> [`docs/deployment.md`](deployment.md)。

---

## 1. 命令行

```bash
genie-tts <command> [options]
```

| 命令 | 说明 | 需要模型 |
|---|---|---|
| `version` | 版本与构建特性 | 否 |
| `info` | 构建信息与可用后端 | 否 |
| `g2p` | 文本 → 音素序列 | 否 |
| `inspect <dir>` | 音色包清单与校验结果 | 否 |
| `synth` | 合成语音 | 是 |

### 通用选项

| 选项 | 说明 |
|---|---|
| `--model <dir>` | 音色包目录（含 `manifest.json`） |
| `--text <text>` | 要合成的文本 |
| `--output <file>` | 输出 WAV 路径 |
| `--language <lang>` | `auto` / `chinese` / `english` / `japanese` / `korean` / `hybrid-chinese-english` |
| `--seed <n>` | 预留（见下方说明） |
| `--threads <n>` | 每个模型会话的 ORT intra-op 线程数（默认 8） |

> **`--seed` 目前不能让结果可复现**：采样（TopK / 温度）在 ONNX 图内，
> ONNX Runtime 没有公开图内采样器的种子接口，seed 只是被透传到
> `SynthesisOptions`（与 C ABI 同一套字段）。要可复现需要改导出图或
> 等 ORT 提供钩子，见 `docs/upstream-audit.md`。
>
> **多核机器建议留意 `--threads`**：Engine 会创建多个模型会话，每个会话
> 默认开满全部核心，合计线程数可能过百。CLI 默认每会话 8 线程；
> 追求低延迟且核心充足时可调大。

### 文本 → 音素

排查文本前端的问题时最有用 —— **不需要模型**，直接看到模型会收到的 `text_seq`：

```console
$ genie-tts g2p --text "你好，世界！2024年。"
language: chinese
normalized: 你好,世界!二千零二十四年.
phonemes (23):
  0	n	227
  1	i3	168
  2	h	158
  3	ao3	119
  4	,	1
  ...
```

`normalized` 是规范化之后的文本（数字已展开成中文读法、标点已映射），
后面是音素符号与它们在符号表里的 ID。

英文文本需要额外的 G2P 资源（词典是硬依赖），用 `--english-dir` 指到
资源目录：

```console
$ genie-tts g2p --language english --english-dir third_party/english/ \
      --text "Hello world, this is a test."
language: english
normalized: hello world, this is a test.
phonemes (19):
  0	HH	...
```

目录里应有 `cmudict.bin`（必需）、`pos_tagger.bin`、`english_g2p.bin`
与可选的 `wordsegment/`（ngram 分词词典，缺失时 OOV 词直接交给 GRU）。

日语文本需要 OpenJTalk 的词典（约 103 MB，仓库不带），用
`--japanese-dict` 指到词典目录：

```console
$ genie-tts g2p --language japanese \
      --japanese-dict /tmp/oj-dict/open_jtalk_dic_utf_8-1.11 \
      --text "こんにちは、世界！"
language: japanese
normalized: こんにちは,世界!
phonemes (18):
  0	k	222
  1	o	229
  ...
```

词典可以用 `tools/deploy/fetch_openjtalk_dictionary.py` 取（从 pypi 的
`openjtalk` wheel 里解出来，附 SHA-256 校验）。OpenJTalk 的运行时源码
已经内置在 `third_party/openjtalk/`（BSD），不需要另外编译。

### 合成

```console
$ genie-tts synth --model voice-package/ --text "你好" --output out.wav
wrote out.wav (48000 frames, 32000 Hz, 1.500 s)
  phonemes: 12
  semantic tokens: 34
```

---

## 2. C++ SDK

```cpp
#include <genie/audio.hpp>
#include <genie/engine.hpp>

genie::Engine engine("voice-package");

genie::SynthesisOptions options;
options.language = genie::Language::Chinese;
options.seed = 42;

const auto result = engine.synthesize("你好，我是 Mira。", options);
genie::WavCodec::write_pcm16("output.wav", result.audio);
```

`SynthesisResult` 里有三样东西：

| 字段 | 内容 |
|---|---|
| `audio` | 合成出的波形（float32 单声道） |
| `phoneme_ids` | 音素符号 ID —— 模型输入 |
| `semantic_tokens` | 语义 token —— T2S 输出 |

### 流式

```cpp
auto channel = engine.synthesize_streaming("你好", options);

genie::AudioBuffer chunk;
while (channel->try_pop(chunk)) {
    play(chunk.samples);
}
// 通道关闭 = 合成结束
```

合成任务提交给 kairo 执行器，调用线程只负责消费音频。

取消：

```cpp
engine.cancel_streaming();
```

> **生命周期**：一个 `Engine` 同时只跑一个流式合成（模型会话不是线程安全的）。
> 引擎析构时会自动取消并等待在途任务。

### 只做文本前端

```cpp
#include <genie/text.hpp>

auto frontend = genie::text::make_frontend(genie::Language::Chinese);
const auto sequence = frontend->convert("你好");
for (std::size_t i = 0; i < sequence.symbols.size(); ++i) {
    std::cout << sequence.symbols[i] << " = " << sequence.ids[i] << "\n";
}
```

---

## 3. C ABI

编译成共享库 `libgenie_tts.so`（Linux）/ `genie_tts.dll`（Windows）。
头文件是**纯 C**，没有 C++ 语法。

```c
#include "genie/c/genie_tts.h"

genie_engine* engine = NULL;
genie_status status = genie_engine_create("voice-package", &engine);
if (status != GENIE_OK) {
    fprintf(stderr, "create failed: %s\n", genie_status_string(status));
    return 1;
}

status = genie_engine_synthesize_wav(engine, "你好", GENIE_LANGUAGE_CHINESE, 0, "out.wav");
genie_engine_destroy(engine);
```

### Python（ctypes）

```python
import ctypes

lib = ctypes.CDLL("./libgenie_tts.so")
lib.genie_version.restype = ctypes.c_char_p
print(lib.genie_version().decode())
```

### Go / Rust

直接 dlopen / `libloading` 即可，符号名就是头文件里的那些。

### API 一览

| 函数 | 说明 |
|---|---|
| `genie_version()` | 版本字符串（静态存储，不要释放） |
| `genie_engine_create(dir, out)` | 创建引擎 |
| `genie_engine_destroy(engine)` | 销毁引擎（传 NULL 是无操作） |
| `genie_engine_synthesize(...)` | 合成，返回 float32 缓冲区（用 `genie_free` 释放） |
| `genie_engine_synthesize_wav(...)` | 合成并直接写 WAV |
| `genie_phonemize(text, lang, out_ids, out_count)` | 文本 → 音素 ID（不需要模型） |
| `genie_free(ptr)` | 释放本库分配的缓冲区 |
| `genie_status_string(status)` | 状态码的文本描述 |

### 约定

- 所有函数返回 `genie_status`（`GENIE_OK` = 0），**异常绝不穿过 ABI 边界**；
- 所有返回的指针由本库分配，用 `genie_free` 释放；
- 字符串一律 UTF-8；
- 状态码与 C++ 侧 `genie::ErrorCode` 是同一套枚举值。

---

## 4. 音色包

见 [`examples/voice-package/README.md`](../examples/voice-package/README.md)。

最小结构：

```text
voice-package/
├── manifest.json
├── models/
│   ├── t2s_encoder.onnx
│   ├── t2s_first_stage_decoder.onnx
│   ├── t2s_stage_decoder.onnx
│   ├── vits.onnx
│   ├── roberta.onnx        # 可选：中文韵律特征（RoBERTa）
│   ├── vocab.txt           # 跟随 roberta.onnx，同目录，行号即 ID
│   ├── chinese-hubert-base.onnx      # 可选：参考音频内容特征（HuBERT）
│   └── chinese-hubert-base_weights.bin  # 跟随 hubert 图的外部权重
└── reference/
    ├── audio.wav           # 参考音频（克隆音色的前提）
    └── text.txt            # 参考文本（可选）
```

> **`roberta` 角色是可选的**：有的话中文合成走真韵律特征（推荐，
> 对中文听感影响显著），没有就退化为全零特征（与上游降级一致）。
> 词表读模型同目录下的 `vocab.txt`（BERT 经典格式，一行一个 token）。
> **`chinese_hubert` 角色同样可选**：接通后 `ssl_content` 从全零升级为
> 真参考音频内容特征。V2ProPlus 音色包另需 `prompt_encoder` 与
> `speaker_verification` 两个角色。

`manifest.json` 里的 `sha256` 建议填 —— 模型文件被替换是最常见的部署事故，
而 ONNX 加载失败时的报错通常看不出根因。

```bash
sha256sum models/t2s_encoder.onnx
```

---

## 5. 构建选项

| 选项 | 默认 | 说明 |
|---|---|---|
| `GENIE_ENABLE_ORT` | 自动探测 | ONNX Runtime 后端 |
| `GENIE_ENABLE_KAIRO` | 自动探测 | kairo 调度后端（关掉会退化为标准库实现） |
| `GENIE_BUILD_TESTS` | ON | 测试 |
| `GENIE_BUILD_CLI` | ON | 命令行 |
| `GENIE_SANITIZE_ADDRESS` | OFF | AddressSanitizer |
| `GENIE_SANITIZE_UNDEFINED` | OFF | UndefinedBehaviorSanitizer |
| `GENIE_SANITIZE_THREAD` | OFF | ThreadSanitizer |

预设：`debug` / `release` / `asan` / `ubsan` / `tsan`。
