// include/genie/engine.hpp
//
// 合成引擎：把音色包、模型会话、文本前端和调度层串成一次完整的合成。
//
// 一次合成的阶段：
//
//   文本 → 音素 ID ─┐
//                   ├─► t2s_encoder ─► t2s_first_stage ─► t2s_stage ×N ─► vits ─► PCM
//   参考音频条件 ───┘
//
// 调度（kairo）在这里的职责：
//   - **流式**合成时，整条流水线跑在 kairo 执行器上，调用线程只负责消费音频；
//   - 文本前端与声码器是 CPU 密集阶段，可以与其他请求重叠；
//   - 自回归解码本身是串行的（必须逐 token），但可以被协作式取消打断。
//
// 设计约束：
//   - `Engine` **不是**线程安全的：模型会话不是线程安全的，多请求并发请用
//     多个 Engine 实例（每个实例持有自己的会话）。
//   - 公开 API 不暴露任何 kairo 类型 —— 换掉调度后端不会破坏上层。

#ifndef GENIE_ENGINE_HPP
#define GENIE_ENGINE_HPP

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "genie/audio.hpp"
#include "genie/config.hpp"
#include "genie/scheduler.hpp"
#include "genie/session.hpp"
#include "genie/t2s.hpp"
#include "genie/text.hpp"
#include "genie/timing.hpp"
#include "genie/voice_package.hpp"

namespace genie {

/// 模型会话工厂。
///
/// 默认用 ONNX Runtime。留这个扩展点有两个原因：
///   - 换一个推理后端时不用改 Engine；
///   - 测试可以注入假会话，在没有权重、没有 ORT 的环境里跑完整流水线。
using SessionFactory = std::function<std::unique_ptr<ModelSession>(const std::string& path,
                                                                   const SessionConfig& config)>;

/// 引擎加载侧配置。
struct EngineConfig {
    /// 是否校验音色包 manifest 登记的 SHA-256（逐文件全量读盘）。
    ///
    /// 校验成本与包体积线性（2.4 GB ≈ 3.8 s，冷启动的一半）—— 常驻进程
    /// 只付一次；"每次进程冷起"的 CLI 场景它是主成本。跳过校验仅建议
    /// 用于"包由本机受信流程产出"的场景；文件存在性检查不受此开关影响，
    /// 缺文件照样报错。默认**开启**校验。
    bool verify_voice_package = true;

    /// 模型会话配置（线程数、图优化、自旋等）。
    SessionConfig session;
};

/// 合成结果。
struct SynthesisResult {
    /// 合成出的音频。
    AudioBuffer audio;
    /// 音素符号 ID 序列（模型输入）。
    std::vector<std::int64_t> phoneme_ids;
    /// 语义 token 序列（T2S 输出）。
    std::vector<std::int64_t> semantic_tokens;
    /// 实际使用的语言。
    Language language = Language::Auto;
    /// 各阶段耗时（M0.3 性能基线）。
    ///
    /// 计时常开：每阶段只做两次 steady_clock 采样，开销可忽略。
    /// CLI / 基准工具据此打印阶段表，SDK 使用者可以直接编程消费。
    TimingReport timing;
};

/// 合成引擎。
///
/// 构造时加载音色包与模型；析构时释放。
class Engine {
public:
    /// 加载音色包并创建模型会话。
    ///
    /// @param root 音色包根目录（含 manifest.json）
    /// @throws genie::Error 音色包不合法、模型缺失或推理后端不可用
    explicit Engine(std::string root);

    /// 加载音色包并创建模型会话（带参数）。
    ///
    /// @param root  音色包根目录
    /// @param config 会话配置（线程数、图优化等）
    Engine(std::string root, const SessionConfig& config);

    /// 加载音色包并创建模型会话（加载侧 + 会话配置）。
    ///
    /// @param root   音色包根目录
    /// @param config 加载与 会话配置
    Engine(std::string root, const EngineConfig& config);

    /// 加载音色包，用自定义的会话工厂创建模型会话。
    ///
    /// @param root    音色包根目录
    /// @param factory 会话工厂
    /// @param config  会话配置
    Engine(std::string root, SessionFactory factory, const SessionConfig& config = {});

    /// 加载音色包，用自定义的会话工厂创建模型会话（加载侧 + 会话配置）。
    ///
    /// @param root    音色包根目录
    /// @param factory 会话工厂
    /// @param config  加载与会话配置
    Engine(std::string root, SessionFactory factory, const EngineConfig& config);

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    Engine(Engine&&) = delete;
    Engine& operator=(Engine&&) = delete;
    ~Engine();

    /// 合成一段语音。
    ///
    /// 同步执行：调用线程会一直等到音频出完。
    ///
    /// @param text    要合成的文本
    /// @param options 合成参数
    /// @throws genie::Error 模型契约不符、输入非法或推理失败
    [[nodiscard]] SynthesisResult synthesize(std::string_view text,
                                             const SynthesisOptions& options = {});

    /// 流式合成（M6 真分块）。
    ///
    /// 合成任务提交给调度器（kairo）执行；调用方从返回的通道里取音频块。
    /// 调用线程完全不参与计算，可以直接拿去播放。
    ///
    /// 文本先按句读分句（句末终止符落刀、超长才在逗号级标点回退切分；
    /// ASCII `.` 不切，小数/时间安全），逐段走完 前端 → T2S → VITS
    /// 后**立刻**把该段音频推入通道
    /// —— 首块延迟是第一句话的合成耗时，不用等整篇。除最后一段外，
    /// 每段音频尾部追加 `stream.sentence_silence_ms` 的静音（自然句间停顿）。
    /// `stream.max_segment_chars == 0` 时退回旧行为：整段一次解码、一个块。
    ///
    /// 语言检测整条流只做一次（与同步路径同口径），全段共用一个前端。
    ///
    /// 通道在合成结束（或出错、被取消）后关闭；阶段耗时与失败原因在
    /// `channel->footer()` 里（错误时 `footer()->error` 非 nullopt，
    /// 协作式取消不算错误）。消费者读到"排空且已关闭"即结束。
    ///
    /// 返回 **shared_ptr** 而不是 unique_ptr：后台任务也持有通道的一份引用，
    /// 这样即使调用方先放手，通道也不会在任务还在往里写的时候被析构。
    ///
    /// @param text    要合成的文本
    /// @param options 合成参数
    /// @param stream  分段参数（分句上限、句间静音）
    /// @return 音频块通道
    [[nodiscard]] std::shared_ptr<AudioChunkChannel> synthesize_streaming(
        std::string_view text, const SynthesisOptions& options = {},
        const StreamOptions& stream = {});

    /// 取消最近一次流式合成。
    ///
    /// @return 是否命中了一个仍在运行的合成任务
    bool cancel_streaming();

    /// 音色包清单。
    [[nodiscard]] const VoiceManifest& manifest() const noexcept;

    /// 采样率。
    [[nodiscard]] int sample_rate() const noexcept;

    /// 引擎构造期（模型加载 / 参考特征 / 全局嵌入）的分阶段耗时。
    ///
    /// 这是 M0.3 里"冷启动"那一半的基线数据：构造一次引擎通常只发生
    /// 一次，其大头（每个 ONNX 会话的初始化）值得单独看。
    [[nodiscard]] const TimingReport& load_timing() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    /// 合成实现。cancelled 非空时会在解码循环里检查取消。
    [[nodiscard]] SynthesisResult synthesize_impl(std::string_view text,
                                                  const SynthesisOptions& options,
                                                  const CancellationToken* cancelled);

    /// 单段合成核心：前端转换 → RoBERTa → T2S 解码 → 声码器。
    ///
    /// 同步路径与流式路径共用 —— 同步 `synthesize()` 的行为必须与
    /// 抽取前逐字节一致（golden 一致性），流式只是把多段分别调它。
    /// 文本必须是已带上游 `"。"` 前缀的最终文本；语言与前端由调用方
    /// 选定（语言检测是"整条请求一次"的开销，不进单段）。
    /// 各阶段耗时记入调用方给的 report（同名阶段累加合并）；
    /// 端到端 total 与 timing 的挂载由调用方负责。
    [[nodiscard]] SynthesisResult synthesize_one(
        const std::string& prepared_text, Language language, const text::TextFrontend& frontend,
        const SynthesisOptions& options, const CancellationToken* cancelled, TimingReport& report);

    /// 流式合成的任务体：分段 → 逐段 synthesize_one → 逐段推通道。
    /// 所有失败都收敛为"footer.error + 关闭通道"，不会向调度器抛异常。
    void synthesize_streamed(const std::string& text, const SynthesisOptions& options,
                             const StreamOptions& stream, const CancellationToken* cancelled,
                             AudioChunkChannel& channel);
};

}  // namespace genie

#endif  // GENIE_ENGINE_HPP
