# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### Fixed

- **调度器退出路径的 use-after-free**：kairo 的 `Executor` 是函数级静态，
  在 `Scheduler::initialize()` 里才被首次引用，因此比 `Scheduler` 单例
  **更晚构造、更早析构**。进程退出时 `Executor` 已经没了，而
  `Scheduler` 的析构函数仍会调 `request_task_cancel()` —— 症状不是崩溃，
  而是**挂住**（kairo 的 shutdown 最多等在途任务 300 秒）。
  析构现在只做 genie 侧的收尾（置取消标志 + 等任务收尾），
  不再触碰 kairo。
- **流式合成的生命周期**：后台任务捕获了引擎指针，而引擎可能在任务还在跑的
  时候被析构；音频通道也是裸指针，调用方先放手就会悬垂。现在引擎析构时
  会取消并**等待**任务收尾，通道改由任务与调用方共同持有（shared_ptr）。

### Added

- **C ABI 共享库**（`libgenie_tts.so` / `genie_tts.dll`）：纯 C 头文件，
  Go / Rust / Python ctypes 直接 dlopen 即可，不需要链接 C++ 运行时。
  新增 `genie_engine_synthesize_wav()` 与 `genie_phonemize()`。
  错误码与 C++ 侧 `genie::ErrorCode` 是同一套枚举值，编译期静态断言钉死。
- **合成引擎**（`genie::Engine`）：把音色包、模型会话、文本前端与调度层
  串成一次完整合成。`synthesize()` 同步，`synthesize_streaming()` 走 kairo
  执行器并通过 `AudioChunkChannel` 交付音频块。
- **T2S 自回归解码**（`genie::T2SDecoder`）：逐行对齐上游 `t2s_cpu()`，
  包括 `range(0,500)` 的下标语义、`y[0,-1]=0` 的置零、`y[:,-idx:]` 在
  `idx==0` 时全取这几个**容易写错**的细节。详见 docs/upstream-audit.md §10。
- **会话工厂扩展点**（`genie::SessionFactory`）：可替换推理后端，
  也让测试能注入假会话，在没有权重、没有 ORT 的环境里跑完整流水线。
- **音素符号表**（`genie::symbols`）：重新实现上游 `SymbolsV2.py` 的生成规则，
  732 个符号与上游**逐字一致**（顺序 + 全表摘要双重锁定）。
  符号 ID 错一位不会报错、只会合成出乱码，所以这一层必须硬锁。
- **拼音 → 音素**（`genie::pinyin`）：复刻 pypinyin 的 `to_initials()` /
  `to_finals_tone3()` 规则，已对 **41,923 个汉字**与 pypinyin 全量比对一致。
- **中文文本前端**（`genie::text::TextFrontend`）：汉字 → 音素符号 → 符号 ID。
  多音字取最常用读音；变调 / 儿化 / 分词未实现，差异记录在 docs/upstream-audit.md。
- **文本规范化**（`genie::text::normalize`）：全角→半角、阿拉伯数字→中文读法、
  标点映射与折叠、目标语言字符过滤。
- **内置 UTF-8 工具**（`genie::utf8`）：逐字符解码、码点编码、全半角转换。
- **CLI `g2p` 子命令**：不需要模型就能看到模型会收到的 text_seq。
- **音色包（voice package）**：`genie::VoicePackage` 解析并校验 manifest.json，
  支持按角色定位模型、**SHA-256 完整性校验**、必需角色的加载期检查。
  示例见 `examples/voice-package/`。
- **自带 JSON 解析器**（`genie::json`）：RFC 8259 最小子集，带嵌套深度上限。
- **自带 SHA-256**（`genie::sha256`）：用于模型完整性校验。
- **项目骨架**：CMake 体系（`CMakePresets.json` + `cmake/` 模块）、C++20 编译基线、
  告警策略、sanitizer 开关。
- **核心类型**：`genie::Error` / `genie::ErrorCode`、`genie::Tensor`（float32 / int64 / bool）、
  `genie::TensorSpec`、`genie::ModelSession` 抽象。
- **ONNX Runtime 后端**：按**名字**绑定模型输入输出（上游 Python 依赖位置对应，
  实际张量名不一致，详见 `docs/upstream-audit.md`）。
- **音频**：自带 RIFF/WAV 编解码（PCM 8/16/24/32 + IEEE float）、窗函数法带限 sinc 重采样。
- **调度与并发**：基于自有库 [kairo](https://github.com/Linductor-alkaid/kairo) 的
  `genie::Scheduler`（可取消任务）与 `genie::AudioChunkChannel`（有界流式音频通道）。
- **测试**：unit（46 例）/ component（模型 I/O 契约）/ integration（T2S 端到端链路）。
- **工程化**：CI（core / kairo / ort / asan+ubsan / tsan / windows / format）、
  `AGENTS.md` 管理规范、`CODEOWNERS`、PR 模板。

### 已知差异（与上游 Genie-TTS）

- 上游把 T2S 采样（TopK / 温度 / 重复惩罚）实现在 ONNX 图内，本移植**不提供**主机侧
  `top_k` / `top_p` / `temperature` 参数 —— 这些是图内常量。唯一可控的随机性是
  ONNX Runtime 的全局随机种子。
- 重采样使用自带的窗函数法带限 sinc 实现，非上游的 `soxr 'hq'`。
  量化差异见 `docs/architecture.md`。

[Unreleased]: https://github.com/Linductor-alkaid/genie-tts-native/compare/v0.1.0...HEAD
