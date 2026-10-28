# genie-tts-native

**Genie-TTS 的纯 C++ 原生推理引擎** —— GPT-SoVITS V2 / V2ProPlus，纯 ONNX Runtime，
**运行时不依赖 Python / PyTorch / 任何推理服务**。

> 上游参考：<https://github.com/High-Logic/Genie-TTS>（MIT）
> 本项目把上游的 Python 推理链路完整迁移到 C++20，并针对 CPU 端侧部署做系统性优化。

---

## 为什么做这个

上游 Genie-TTS 用 Python + ONNX Runtime 实现了完整的 GPT-SoVITS 推理，但作为**嵌入式引擎**
有几个硬伤：

- **运行时依赖链太长**：Python → onnxruntime → tokenizers → pypinyin / g2pM / nltk /
  pyopenjtalk / jamo / ko_pron / g2pk2 …… 端侧部署时这些都得装上
- **多语言 G2P 无法在 C++ 侧复用**：中文靠 pypinyin + g2pM + jieba_fast，英文靠 nltk，
  日文靠 pyopenjtalk-plus，韩文靠 g2pk2 —— 每一项都是独立的 Python 生态
- **性能不可控**：Python 层的 GIL、对象分配、numpy 拷贝都在热路径上

本项目的目标：把整条链路用 C++ 重写，让部署产物只剩一个二进制 + 一组 ONNX 模型。

---

## 当前状态

| 里程碑 | 状态 |
|---|---|
| M0 上游审计与基线 | ✅ 推理链路、模型 I/O 契约、依赖清单已固化到 [`docs/upstream-audit.md`](docs/upstream-audit.md) |
| M1 C++ Runtime 基础设施 | ✅ CMake 体系、ONNX Runtime 封装、张量抽象、音色包、JSON / SHA-256 / UTF-8 自带实现 |
| M2 文本与音频前处理 | 🚧 音频完成；中文文本前端完成（符号表逐字一致、拼音全量比对一致、TextNormalizer 规范化、变调与儿化）；中文韵律特征（RoBERTa）已实装；英文 / 日文 / 韩文 G2P 留成可插拔接口 |
| M3 完整推理迁移 | ✅ T2S 自回归解码、VITS、端到端引擎、流式流水线、参考音频条件、V2ProPlus |
| M4 原生 SDK | ✅ C++ SDK + C ABI 共享库 + CLI |
| M6 回归层 | ✅ Golden Dataset（模型 I/O 契约快照）|
| M5–M7 | ⏳ 性能优化、发布 |

进度详见 [`docs/roadmap.md`](docs/roadmap.md)。

---

## 快速开始

### 部署：拿权重、装音色包

推理需要一组 ONNX 模型（按音色发布）+ 一个音色包目录。完整手把手教程
—— **权重来源、下载命令、fp16→fp32 转换、音色包组装、首次合成** —— 见
**[`docs/deployment.md`](docs/deployment.md)**。仓库自带两个部署脚本：

```bash
python3 tools/deploy/convert_fp16_weights.py …   # 上游 fp16 外部权重 → fp32（带图校验）
python3 tools/deploy/make_voice_package.py …     # 组装音色包：改名 + SHA-256 + manifest.json
```

模型来源：上游 HF 仓库 [`High-Logic/Genie`](https://hf-mirror.com/High-Logic/Genie)
（`CharacterModels/v2ProPlus/<音色>/` 每音色一套 TTS 模型；
`GenieData/` 公共 HuBERT 与说话人验证；`GenieData(Optional)/RoBERTa/`
中文韵律，可选）。仓库内 `third_party/genie-tts-models/` 只带了几张
**图壳**（跑契约测试用），不含权重。

### 构建

```bash
# 需要 ONNX Runtime 预编译包
cmake --preset debug -DGENIE_ONNXRUNTIME_ROOT=/opt/onnxruntime
cmake --build --preset debug
ctest --preset debug --output-on-failure
```

> 不带 ONNX Runtime 也能构建与测试（`GENIE_ENABLE_ORT=OFF`）——
> 纯逻辑部分（符号表、拼音切分、文本规范化、解码控制流）都不依赖推理后端。

### CLI

```bash
genie-tts version                  # 版本与构建特性
genie-tts info                     # 构建信息与可用后端
genie-tts g2p --text "你好"         # 文本 → 音素（不需要模型）
genie-tts inspect voice-package/   # 音色包清单与校验
genie-tts synth --model voice-package/ --text "你好" --output out.wav
```

### C++ SDK

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

流式合成（后台跑在 kairo 执行器上，调用线程只取音频块）：

```cpp
auto channel = engine.synthesize_streaming("你好", options);
genie::AudioBuffer chunk;
while (channel->try_pop(chunk)) {
    // 播出去
}
```

### C ABI

```c
#include "genie/c/genie_tts.h"

genie_engine* engine = NULL;
if (genie_engine_create("voice-package", &engine) != GENIE_OK) { /* ... */ }

genie_engine_synthesize_wav(engine, "你好", GENIE_LANGUAGE_CHINESE, 0, "out.wav");
genie_engine_destroy(engine);
```

> `libgenie_tts.so` / `genie_tts.dll` 是独立的共享库，
> Go / Rust / Python ctypes 直接 dlopen 即可，不需要链接 C++ 运行时。

---

## 架构

```
genie-tts-native/
├── include/genie/        公开头（C++ SDK + C ABI）
├── src/
│   ├── runtime/          ONNX Runtime 封装、张量、模型加载、音色包、JSON / SHA-256 / UTF-8
│   ├── text/             文本规范化、符号表、拼音与音素转换
│   ├── engine/           T2S 自回归解码、合成引擎
│   ├── audio/            WAV 编解码、重采样
│   ├── c/                C ABI 实现
│   └── scheduler/        基于 kairo 的调度与并发
├── tools/
│   ├── cli/              genie-tts 命令行
│   ├── deploy/           部署脚本（fp16 转换、音色包组装）
│   └── g2p/              词表生成器与文本前端比对工具
├── tests/                unit / component / integration / c
└── third_party/          内置的上游 ONNX 模型（MIT）
```

**调度与并发由 [kairo](https://github.com/Linductor-alkaid/kairo) 承担** —— 线程池、
任务优先级、有界通道、协作式取消。公开 API **不暴露任何 kairo 类型**。
详见 [`AGENTS.md`](AGENTS.md) §1.3。

---

## 与上游的关键差异

**采样在图内**：上游把 T2S 采样（TopK / 温度 / 重复惩罚）实现在 ONNX 图内，
`first_stage_decoder` 与 `stage_decoder` 各含一个 `TopK` 算子。
因此本项目**没有**主机侧的 `top_k` / `top_p` / `temperature` 参数 ——
那些是图内常量，改动它们需要重新导出模型，而不是改配置。
图内采样器的随机种子 ONNX Runtime 没有公开入口（节点未带 `seed` 属性时
由 ORT 自行生成），所以**合成结果目前不可复现**；`SynthesisOptions.seed`
与 CLI `--seed` 只是预留字段，如实标注，待 ORT 提供钩子后接通。

**按名字绑定模型输入输出**：上游按位置传参（`zip(input_names, [...])`），
而真实张量名是 `iy` / `iy_emb` / `past_k_layer_N`。位置绑定在输入顺序变化时会静默错位。

**符号表与拼音规则逐字复刻**：符号表（732 个）重新实现了上游的生成规则，
与上游**顺序完全一致**；拼音切分与 pypinyin 的 `to_initials` / `to_finals_tone3`
对 **41,923 个汉字**全量比对一致。

其余差异（文本规范化的覆盖范围、多音字处理、变调与儿化等）见
[`docs/upstream-audit.md`](docs/upstream-audit.md)。

---

## 文档

| 文档 | 内容 |
|---|---|
| [`AGENTS.md`](AGENTS.md) | **项目管理规范**（最高约束） |
| [`docs/architecture.md`](docs/architecture.md) | 架构与关键设计决策 |
| [`docs/upstream-audit.md`](docs/upstream-audit.md) | 上游推理链路审计结论 |
| [`docs/deployment.md`](docs/deployment.md) | **部署教程**：权重下载、fp16 转换、音色包组装、首次合成 |
| [`docs/usage.md`](docs/usage.md) | CLI / C++ SDK / C ABI 使用说明 |
| [`docs/regression.md`](docs/regression.md) | 回归层：Golden Dataset 与模型 I/O 契约守卫 |
| [`docs/roadmap.md`](docs/roadmap.md) | 里程碑与进度 |
| [`docs/conventions.md`](docs/conventions.md) | 命名、格式、提交规范 |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | 开发环境与提交流程 |

---

## 许可证

MIT，见 [`LICENSE`](LICENSE)。

本项目包含上游 Genie-TTS 发布的 ONNX 模型（MIT），归属说明见
[`third_party/genie-tts-models/LICENSE`](third_party/genie-tts-models/LICENSE) 与
[`NOTICE`](NOTICE)。
