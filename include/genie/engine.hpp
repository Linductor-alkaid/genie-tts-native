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

/// 增量流式合成会话（M6）。
///
/// 与一次性的 synthesize_streaming() 不同，会话允许调用方把文本
/// **分多次喂进来**（典型场景：ASR 的中间结果持续流入）——
/// 每凑齐一个完整句段就立刻开始合成，音频从 channel() 里流出。
///
/// 生命周期：start_streaming() → feed()×N → finish()（或随时 cancel()）。
///
/// 线程安全：feed / finish / cancel 可从任意线程调用（内部有锁）；
/// 但一个 Engine 同时只有一个**活跃**会话 —— 模型会话不是线程安全的，
/// 活跃期间再次 start_streaming() 抛 Busy。
///
/// 结束语义与一次性接口一致：channel() 在全部句段合成完（或出错、
/// 被取消）后关闭；"排空且已关闭"即结束，流尾元数据在 footer() 里。
class StreamingSynthesis {
public:
    /// 丢弃会话即放弃这条流：析构请求取消并等待合成任务收尾
    /// （worker 持有引擎指针，不能让它看到已析构的引擎）。
    ~StreamingSynthesis();

    StreamingSynthesis(const StreamingSynthesis&) = delete;
    StreamingSynthesis& operator=(const StreamingSynthesis&) = delete;

    /// 音频块通道。会话存活期间恒有效。
    [[nodiscard]] std::shared_ptr<AudioChunkChannel> channel() const noexcept;

    /// 喂入一段文本（线程安全）。
    ///
    /// 文本先累积，凑齐完整句段（句末终止符；超长时软终止符回退/硬切，
    /// 规则与一次性接口同源 —— 增量喂入与一次性喂入产出相同的段序列）
    /// 就投给合成循环。段队列**有界**：合成跟不上时 feed 阻塞（背压），
    /// 阻塞期间会话被取消则抛 Cancelled。finish() 之后调用抛
    /// InvalidArgument；空文本是 no-op。feed 的调用方应自行串行化
    /// （引擎本身不是线程安全的）。
    void feed(std::string_view text);

    /// 输入结束：缓冲里的残余文本作为最后一个句段冲刷（线程安全）。
    ///
    /// 只标记"没有更多输入"，**不等待**合成完成 —— 完成信号仍是通道
    /// 关闭。重复调用是 no-op；之后 feed 抛 InvalidArgument。
    void finish();

    /// 协作式取消：停止合成、关闭通道（线程安全）。重复调用安全。
    ///
    /// 取消**不算错误** —— footer 不带 error，通道关闭本身就是信号；
    /// 已排队未合成的段与缓冲残余一并丢弃。
    void cancel();

    /// 会话是否已终结（合成完成 / 出错 / 被取消）。终结后即可开始
    /// 下一条流 —— Engine 的单流限制以它为准。
    [[nodiscard]] bool finished() const noexcept;

private:
    friend class Engine;
    struct Impl;

    /// Engine 专用构造。
    explicit StreamingSynthesis(std::shared_ptr<Impl> impl);

    std::shared_ptr<Impl> impl_;
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

    /// 开启一条增量流式会话（M6）。
    ///
    /// 文本经 session->feed() 分多次喂入；已存在**未终结**的会话时抛
    /// Busy（模型会话不是线程安全的，一个 Engine 同时只跑一条流）。
    /// Engine 持有会话的一份内部引用（单流记账）；调用方放手会话对象
    /// 即视为放弃这条流（析构请求取消）。
    ///
    /// @param options 合成参数
    /// @param stream  分段参数（分句上限、句间静音）
    /// @throws genie::Error Busy —— 已有活跃会话
    [[nodiscard]] std::shared_ptr<StreamingSynthesis> start_streaming(
        const SynthesisOptions& options = {}, const StreamOptions& stream = {});

    /// 取消当前流式合成（一次性接口或增量会话）。
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

    /// start_streaming 的内部形态：一次性封装以 initial_text / pinned
    /// **构造期注入**（worker 与它们的关系是"提交前写定、此后不可变"
    /// —— 无锁读的合法性来自这里；曾把它们做成提交后加锁写，worker
    /// 无锁读与 wrapper 的写在两线程竞态，Windows 的 optional 断言与
    /// TSan 的错误码漂移都源自此）。
    [[nodiscard]] std::shared_ptr<StreamingSynthesis> start_streaming_internal(
        const SynthesisOptions& options, const StreamOptions& stream,
        std::optional<Language> pinned, std::optional<std::string> initial_text);

    /// 流式会话的任务体：取段 → 逐段 synthesize_one → 逐段推通道 →
    /// eof 残余冲刷 → footer + 关闭通道。所有失败收敛为
    /// "footer.error + 关闭通道"，不向调度器抛异常。
    /// 逻辑放在 Engine（而不是会话类）上：它要碰会话选择器、前端工厂
    /// 与单段核心 —— 全是引擎私有件。
    void run_stream_session(std::shared_ptr<StreamingSynthesis::Impl> session,
                            const CancellationToken& token);
};

}  // namespace genie

#endif  // GENIE_ENGINE_HPP
