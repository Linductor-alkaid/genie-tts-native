// genie/scheduler.hpp — 调度与并发
//
// 本项目把「调度与并发」全部交给自研库 kairo（https://github.com/Linductor-alkaid/kairo），
// 不自己造线程池。kairo 承担：
//   - 线程池与任务优先级（submit / submit_cancellable / submit_after）
//   - 有界无锁通道（MpscChannel），用于流式 PCM 分发
//   - 协作式取消（StopToken）
//
// 本文件暴露的是**与 kairo 无关**的稳定 API：
//   - 换掉 kairo 不会破坏上层（engine / CLI / C ABI）
//   - 未编译 kairo 时仍有可用的降级实现（见 scheduler_stub.cpp）
//
// 为什么需要调度层（对应里程碑 M6 低延迟流式 TTS）：
//   一次合成天然是多阶段的 —— 文本前端 → 语义生成（自回归，逐 token 串行）
//   → 声学合成 → 音频输出。流式场景下这些阶段需要重叠执行，
//   且必须能被打断（用户说了新句子就要丢弃旧结果）。

#ifndef GENIE_SCHEDULER_HPP
#define GENIE_SCHEDULER_HPP

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "genie/audio.hpp"
#include "genie/errors.hpp"
#include "genie/timing.hpp"

namespace genie {

/// 协作式取消令牌。任务应周期性检查 cancelled()。
class CancellationToken {
public:
    CancellationToken() = default;

    [[nodiscard]] bool cancelled() const noexcept { return flag_ != nullptr && flag_->load(); }

    /// 内部使用：绑定到底层标志。
    void bind(std::shared_ptr<std::atomic<bool>> flag) noexcept { flag_ = std::move(flag); }

private:
    std::shared_ptr<std::atomic<bool>> flag_;
};

/// 调度器配置。
struct SchedulerConfig {
    /// 最小工作线程数。0 = 自适应（按硬件并发数，至少 2）。
    std::size_t min_threads = 0;
    /// 最大工作线程数。0 = 自适应（等于硬件并发数）。
    std::size_t max_threads = 0;
    /// 任务队列容量。
    std::size_t queue_capacity = 1024;
    /// 流式音频通道容量（chunk 数）。
    std::size_t stream_capacity = 64;
    /// 是否开启任务监控统计。
    bool enable_monitoring = false;
};

/// 调度器。进程级单例。
///
/// 典型用法：
/// @code
///   auto& sched = genie::Scheduler::instance();
///   sched.initialize({});
///   auto id = sched.submit([](const genie::CancellationToken& token) {
///       // ... 长任务 ...
///   });
///   sched.cancel(id);
///   sched.shutdown();
/// @endcode
class Scheduler {
public:
    using TaskId = std::uint64_t;

    static Scheduler& instance();

    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;
    ~Scheduler();

    /// 初始化。可重复调用（幂等）。
    ///
    /// 注意（kairo 语义）：kairo 的默认异步执行器是**进程级、一次性**的资源 ——
    /// 一旦被 shutdown 就不能再 initialize。本类因此把"停止接活"与"销毁执行器"
    /// 拆成两个动作，见 shutdown() 与 release_executor()。
    void initialize(const SchedulerConfig& config = {});

    /// 停止接活：拒绝新任务，取消在途任务，等待它们收尾。
    ///
    /// **不**销毁底层执行器。进程内可以再次 initialize()。
    void shutdown();

    /// 彻底释放底层执行器（kairo 执行器随之终止）。
    ///
    /// 此操作**不可恢复** —— 之后再调用 initialize() 会抛错。
    /// 进程正常退出时无需显式调用。
    void release_executor();

    [[nodiscard]] bool initialized() const noexcept;

    /// 提交一个可取消任务。返回任务 id。
    ///
    /// 任务抛出的异常会被捕获并转为任务失败状态，不影响调度器本身。
    TaskId submit(std::function<void(const CancellationToken&)> task);

    /// 请求取消。返回是否命中了一个仍在运行的任务。
    bool cancel(TaskId id);

    /// 等待任务结束。超时返回 false。
    bool wait(TaskId id, std::chrono::milliseconds timeout);

    /// 当前在途任务数。
    [[nodiscard]] std::size_t in_flight() const noexcept;

    /// 累计任务失败数（任务抛出异常）。
    [[nodiscard]] std::uint64_t failure_count() const noexcept;

    /// 流式音频通道的容量（chunk 数），取自初始化时的 SchedulerConfig::stream_capacity。
    ///
    /// 未初始化（或未显式配置）时返回默认值 64。流式合成用它创建
    /// AudioChunkChannel —— 这个字段在 M6 之前一直没人消费，
    /// 现在是它的正式出口。
    [[nodiscard]] std::size_t stream_capacity() const noexcept;

private:
    Scheduler();
    struct Impl;
    std::unique_ptr<Impl> impl_;

    /// 停止接活并收尾。
    ///
    /// @param use_executor 是否通过 kairo 取消任务。**析构时必须传 false** ——
    ///        kairo 的 Executor 是函数级静态，比本单例更早析构，
    ///        那时碰它就是 use-after-free。
    void shutdown_impl(bool use_executor);
};

/// 流式合成的结束元数据（M6 真分块流式）。
///
/// 音频块本身只有波形，分段合成的阶段耗时、失败原因这些"整条流"
/// 的信息没有地方放 —— 都收在流尾。生产者在 close() 之前设置；
/// 消费者在读到"排空且已关闭"后取用。
struct StreamFooter {
    /// 全流分阶段耗时（分段合成逐段并入：同名阶段累加、count 为段内步数
    /// 之和）。取失败时也尽力保留已完成段的耗时。
    TimingReport timing;

    /// 非 nullopt 表示流因错误终止（模型推理失败、文本不合法等）。
    /// 协作式取消**不算**错误 —— 取消是调用方自己发起的，
    /// 通道关闭本身就是信号。
    std::optional<Error> error;
};

/// 有界音频分块通道 —— 流式 PCM 的分发通道。
///
/// 单生产者/多消费者语义中的"单消费者"：kairo 的 MpscChannel 明确只有一个逻辑消费者，
/// 这里保持一致：多个线程可以 try_push，但只有一个线程应当消费。
///
/// 背压策略：容量满时 try_push 返回 false，由调用方决定阻塞、丢弃还是降级。
class AudioChunkChannel {
public:
    explicit AudioChunkChannel(std::size_t capacity = 64);
    ~AudioChunkChannel();

    AudioChunkChannel(const AudioChunkChannel&) = delete;
    AudioChunkChannel& operator=(const AudioChunkChannel&) = delete;

    /// 非阻塞投递。通道满或已关闭返回 false。
    bool try_push(AudioBuffer chunk);

    /// 非阻塞取出一块。空或已关闭返回 false。
    bool try_pop(AudioBuffer& out);

    /// 阻塞投递，最多等待 timeout。
    bool push(AudioBuffer chunk, std::chrono::milliseconds timeout);

    /// 阻塞取出一块，最多等待 timeout。
    bool pop(AudioBuffer& out, std::chrono::milliseconds timeout);

    /// 关闭通道。消费者会收到"已排空"信号。
    void close() noexcept;

    /// 设置流尾元数据。生产者在 close() **之前**调用；重复设置以最后一次为准。
    void set_footer(StreamFooter footer);

    /// 流尾元数据。未设置时返回 nullptr。
    ///
    /// 指针指向通道内部存储，随通道存活；按"单消费者"约定，
    /// close() 之后不会再有写入，读取不需要额外同步。
    [[nodiscard]] const StreamFooter* footer() const noexcept;

    [[nodiscard]] bool closed() const noexcept;
    [[nodiscard]] bool drained() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] std::size_t capacity() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace genie

#endif  // GENIE_SCHEDULER_HPP
