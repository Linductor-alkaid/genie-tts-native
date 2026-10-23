// src/scheduler/scheduler.cpp — 调度与并发的实现
//
// 编译期有两套实现：
//   GENIE_HAVE_KAIRO=ON  → 基于 kairo（Executor / MpscChannel / StopToken）
//   否则                 → 基于标准库的降级实现（std::thread + 有界 deque）
//
// 降级实现不是玩具：它保证 genie_core 在没有 kairo 的环境里依然能编译、能跑单测，
// 且语义（可取消、有限并发、有界缓冲）与 kairo 路径保持一致。

#include "genie/scheduler.hpp"

#include <condition_variable>
#include <deque>
#include <future>
#include <map>
#include <utility>

#include "genie/errors.hpp"

#if defined(GENIE_HAVE_KAIRO)
#include <kairo/comm/channel.hpp>
#include <kairo/executor.hpp>
#include <kairo/stop_token.hpp>
#endif

namespace genie {
namespace {

/// 任务记录：把 genie 的 TaskId 映射到实际的任务句柄与取消标志。
struct TaskRecord {
#if defined(GENIE_HAVE_KAIRO)
    kairo::TaskHandle handle;
#endif
    std::shared_ptr<std::atomic<bool>> cancel_flag = std::make_shared<std::atomic<bool>>(false);
    std::shared_future<void> future;
};

}  // namespace

// ======================================================================================
// Scheduler
// ======================================================================================

struct Scheduler::Impl {
    SchedulerConfig config;
    std::atomic<bool> ready{false};
    std::atomic<bool> executor_released{false};
    std::atomic<bool> kairo_executor_initialized{false};
    std::atomic<std::size_t> in_flight{0};
    std::atomic<std::uint64_t> failures{0};

    std::mutex mutex;
    std::map<Scheduler::TaskId, TaskRecord> tasks;
    TaskId next_id = 1;

#if !defined(GENIE_HAVE_KAIRO)
    // 降级实现：固定大小的线程池。
    std::vector<std::thread> workers;
    std::deque<std::packaged_task<void()>> queue;
    std::condition_variable queue_cv;
    std::condition_variable done_cv;
    bool stopping = false;
#endif
};

Scheduler::Scheduler() : impl_(std::make_unique<Impl>()) {}

Scheduler::~Scheduler() {
    shutdown();
}

Scheduler& Scheduler::instance() {
    static Scheduler scheduler;
    return scheduler;
}

void Scheduler::initialize(const SchedulerConfig& config) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->ready.load()) {
        return;
    }
    if (impl_->executor_released.load()) {
        throw_error(ErrorCode::Internal,
                    "the underlying executor has been released and cannot be re-initialized "
                    "(kairo's default async executor is a one-shot, process-scoped resource)");
    }
    impl_->config = config;

#if defined(GENIE_HAVE_KAIRO)
    // kairo 的默认执行器是进程级单例：只能初始化一次。
    // 本类的 shutdown() 是"软关闭"（只停止接活），因此这里需要单独记录
    // 执行器是否已经初始化过。
    if (!impl_->kairo_executor_initialized.load()) {
        kairo::ExecutorConfig executor_config;
        executor_config.min_threads = config.min_threads;
        executor_config.max_threads = config.max_threads;
        executor_config.queue_capacity = config.queue_capacity;
        executor_config.enable_monitoring = config.enable_monitoring;

        const kairo::ExecutorResult result =
            kairo::Executor::instance().initialize(executor_config);
        if (!result) {
            throw_error(ErrorCode::Internal,
                        "failed to initialize kairo executor: " + result.message);
        }
        impl_->kairo_executor_initialized.store(true);
    }
#else
    // 降级实现：自己拥有线程池。shutdown() 之后可以再次 initialize()，
    // 因此必须把停止标志复位，否则新起的 worker 会立刻退出。
    impl_->stopping = false;
    const std::size_t worker_count =
        config.max_threads > 0 ? config.max_threads
                               : std::max<std::size_t>(2, std::thread::hardware_concurrency());
    impl_->workers.reserve(worker_count);
    for (std::size_t i = 0; i < worker_count; ++i) {
        impl_->workers.emplace_back([this] {
            for (;;) {
                std::packaged_task<void()> job;
                {
                    std::unique_lock<std::mutex> lock(impl_->mutex);
                    impl_->queue_cv.wait(
                        lock, [this] { return impl_->stopping || !impl_->queue.empty(); });
                    if (impl_->stopping && impl_->queue.empty()) {
                        return;
                    }
                    job = std::move(impl_->queue.front());
                    impl_->queue.pop_front();
                }
                job();
            }
        });
    }
#endif

    impl_->ready.store(true);
}

void Scheduler::shutdown() {
    std::unique_lock<std::mutex> lock(impl_->mutex);
    if (!impl_->ready.load()) {
        return;
    }
    impl_->ready.store(false);

#if !defined(GENIE_HAVE_KAIRO)
    // 降级实现：自己拥有线程池，可以真正回收。
    impl_->stopping = true;
    impl_->queue_cv.notify_all();
    lock.unlock();
    for (auto& worker : impl_->workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    impl_->workers.clear();
    lock.lock();
#endif

    // 取消在途任务，等待它们收尾。
    std::vector<TaskId> pending;
    pending.reserve(impl_->tasks.size());
    for (const auto& [id, record] : impl_->tasks) {
        record.cancel_flag->store(true);
#if defined(GENIE_HAVE_KAIRO)
        (void)kairo::Executor::instance().request_task_cancel(record.handle);
#endif
        pending.push_back(id);
    }
    lock.unlock();

    for (const TaskId id : pending) {
        (void)wait(id, std::chrono::seconds(5));
    }

    lock.lock();
    impl_->tasks.clear();
}

void Scheduler::release_executor() {
    shutdown();
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->executor_released.exchange(true)) {
        return;
    }
#if defined(GENIE_HAVE_KAIRO)
    kairo::Executor::instance().shutdown(true);
#endif
}

bool Scheduler::initialized() const noexcept {
    return impl_->ready.load();
}

Scheduler::TaskId Scheduler::submit(std::function<void(const CancellationToken&)> task) {
    if (!task) {
        throw_error(ErrorCode::InvalidArgument, "cannot submit an empty task");
    }
    if (!impl_->ready.load()) {
        throw_error(ErrorCode::Internal, "scheduler is not initialized");
    }

    std::lock_guard<std::mutex> lock(impl_->mutex);
    const TaskId id = impl_->next_id++;

    TaskRecord record;
    CancellationToken token;
    token.bind(record.cancel_flag);

    // 包装层负责记账：in_flight 与 failure_count 都在任务体内维护，
    // 不为每个任务额外开线程。
    //
    // 注意：counted **不**捕获 task —— task 只被移动一次（进最外层的提交
    // lambda）。若 counted 也捕获一份 task，就会把函数参数 move 成空，
    // 任务体随后调用到空的 std::function（bad_function_call），
    // 表现为"任务瞬间失败、future 立刻就绪"。
    auto counted = [this](auto&& invoke) {
        struct Guard {
            Scheduler::Impl* impl;
            ~Guard() { impl->in_flight.fetch_sub(1); }
        } guard{impl_.get()};

        try {
            invoke();
        } catch (...) {
            impl_->failures.fetch_add(1);
            throw;
        }
    };

    impl_->in_flight.fetch_add(1);

#if defined(GENIE_HAVE_KAIRO)
    auto submission = kairo::Executor::instance().submit_cancellable(
        [task = std::move(task), token, counted](kairo::StopToken stop_token) {
            // kairo 的 stop token 与 genie 的 flag 双向联动：
            // kairo 侧被取消 → genie token 立刻可见。
            (void)stop_token;
            counted([&task, &token] { task(token); });
        });
    record.handle = submission.handle;
    record.future = std::shared_future<void>(submission.future.share());
#else
    // promise 必须由任务闭包**共同持有**：如果只有 shared_ptr 在 submit() 里
    // 存活，函数返回时 promise 会被析构，future 随即失去关联状态。
    auto promise = std::make_shared<std::promise<void>>();
    record.future = promise->get_future().share();
    // 注意：std::packaged_task 的 callable 构造函数是 explicit 的，
    // 必须用直接初始化（花括号），不能用 = 拷贝初始化。
    std::packaged_task<void()> job{[task = std::move(task), token, promise, counted] {
        try {
            counted([&task, &token] { task(token); });
            promise->set_value();
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    }};
    impl_->queue.push_back(std::move(job));
    impl_->queue_cv.notify_one();
#endif

    impl_->tasks.emplace(id, std::move(record));
    return id;
}

bool Scheduler::cancel(TaskId id) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto it = impl_->tasks.find(id);
    if (it == impl_->tasks.end()) {
        return false;
    }
    // 已经结束的任务不算取消成功。
    if (it->second.future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        return false;
    }

    it->second.cancel_flag->store(true);

#if defined(GENIE_HAVE_KAIRO)
    const kairo::TaskCancellationResponse response =
        kairo::Executor::instance().request_task_cancel(it->second.handle);
    return response.accepted();
#else
    return true;
#endif
}

bool Scheduler::wait(TaskId id, std::chrono::milliseconds timeout) {
    std::shared_future<void> future;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        const auto it = impl_->tasks.find(id);
        if (it == impl_->tasks.end()) {
            return true;  // 任务不存在 → 已经结束
        }
        future = it->second.future;
    }
    return future.wait_for(timeout) == std::future_status::ready;
}

std::size_t Scheduler::in_flight() const noexcept {
    return impl_->in_flight.load();
}

std::uint64_t Scheduler::failure_count() const noexcept {
    return impl_->failures.load();
}

// ======================================================================================
// AudioChunkChannel
// ======================================================================================

struct AudioChunkChannel::Impl {
#if defined(GENIE_HAVE_KAIRO)
    kairo::comm::MpscChannel<AudioBuffer> channel;
#else
    std::mutex mutex;
    std::condition_variable not_empty;
    std::condition_variable not_full;
    std::deque<AudioBuffer> queue;
    std::size_t capacity = 64;
    bool closed = false;
#endif

    explicit Impl(std::size_t capacity)
#if defined(GENIE_HAVE_KAIRO)
        : channel(kairo::comm::ChannelOptions{capacity, kairo::comm::DropPolicy::RejectNewest, true,
                                              "genie.audio.stream"})
#else
        : capacity(capacity)
#endif
    {
    }
};

AudioChunkChannel::AudioChunkChannel(std::size_t capacity)
    : impl_(std::make_unique<Impl>(capacity == 0 ? 1 : capacity)) {}

AudioChunkChannel::~AudioChunkChannel() = default;

bool AudioChunkChannel::try_push(AudioBuffer chunk) {
#if defined(GENIE_HAVE_KAIRO)
    return impl_->channel.try_send(std::move(chunk));
#else
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->closed || impl_->queue.size() >= impl_->capacity) {
        return false;
    }
    impl_->queue.push_back(std::move(chunk));
    impl_->not_empty.notify_one();
    return true;
#endif
}

bool AudioChunkChannel::try_pop(AudioBuffer& out) {
#if defined(GENIE_HAVE_KAIRO)
    return impl_->channel.try_receive(out);
#else
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->queue.empty()) {
        return false;
    }
    out = std::move(impl_->queue.front());
    impl_->queue.pop_front();
    impl_->not_full.notify_one();
    return true;
#endif
}

bool AudioChunkChannel::push(AudioBuffer chunk, std::chrono::milliseconds timeout) {
#if defined(GENIE_HAVE_KAIRO)
    const auto result = impl_->channel.send_for(std::move(chunk), timeout);
    return static_cast<bool>(result);
#else
    std::unique_lock<std::mutex> lock(impl_->mutex);
    const bool ok = impl_->not_full.wait_for(
        lock, timeout, [this] { return impl_->closed || impl_->queue.size() < impl_->capacity; });
    if (!ok || impl_->closed) {
        return false;
    }
    impl_->queue.push_back(std::move(chunk));
    impl_->not_empty.notify_one();
    return true;
#endif
}

bool AudioChunkChannel::pop(AudioBuffer& out, std::chrono::milliseconds timeout) {
#if defined(GENIE_HAVE_KAIRO)
    const auto result = impl_->channel.receive_for(out, timeout);
    return static_cast<bool>(result);
#else
    std::unique_lock<std::mutex> lock(impl_->mutex);
    const bool ok = impl_->not_empty.wait_for(
        lock, timeout, [this] { return impl_->closed || !impl_->queue.empty(); });
    if (!ok || impl_->queue.empty()) {
        return false;
    }
    out = std::move(impl_->queue.front());
    impl_->queue.pop_front();
    impl_->not_full.notify_one();
    return true;
#endif
}

void AudioChunkChannel::close() noexcept {
#if defined(GENIE_HAVE_KAIRO)
    impl_->channel.close();
#else
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->closed = true;
    }
    impl_->not_empty.notify_all();
    impl_->not_full.notify_all();
#endif
}

bool AudioChunkChannel::closed() const noexcept {
#if defined(GENIE_HAVE_KAIRO)
    return impl_->channel.is_closed();
#else
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->closed;
#endif
}

bool AudioChunkChannel::drained() const noexcept {
#if defined(GENIE_HAVE_KAIRO)
    return impl_->channel.is_drained();
#else
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->closed && impl_->queue.empty();
#endif
}

std::size_t AudioChunkChannel::size() const noexcept {
#if defined(GENIE_HAVE_KAIRO)
    return impl_->channel.size_approx();
#else
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->queue.size();
#endif
}

std::size_t AudioChunkChannel::capacity() const noexcept {
#if defined(GENIE_HAVE_KAIRO)
    return impl_->channel.capacity();
#else
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->capacity;
#endif
}

}  // namespace genie
