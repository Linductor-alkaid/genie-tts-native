// src/scheduler/parallel.cpp

#include "scheduler/parallel.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>
#include <vector>

#include "genie/scheduler.hpp"

#if defined(GENIE_HAVE_KAIRO)
#include <kairo/executor.hpp>
#endif

namespace genie::detail {
namespace {

void run_serially(const std::size_t count, const std::function<void(std::size_t)>& body) {
    for (std::size_t i = 0; i < count; ++i) {
        body(i);
    }
}

}  // namespace

void parallel_for(const std::size_t count, const std::size_t max_workers,
                  const std::function<void(std::size_t)>& body) {
    const std::size_t workers = std::min(max_workers, count);
    if (workers <= 1) {
        run_serially(count, body);
        return;
    }

#if defined(GENIE_HAVE_KAIRO)
    // 辅助任务可能在调用方返回之后才被调度（领不到活、立刻退出）——
    // 它们碰的状态必须由共享指针撑住，不能放在调用方的栈上。body 只在
    // 领到活时才被调用，而调用方会等所有领到的活结束，所以 body 本身
    // 可以安全地引用调用方的栈。
    struct State {
        const std::function<void(std::size_t)>* body = nullptr;
        std::size_t count = 0;
        std::atomic<std::size_t> next{0};
        std::atomic<bool> failed{false};
        std::mutex mutex;
        std::condition_variable all_done;
        std::size_t finished = 0;
        std::vector<std::exception_ptr> errors;
    };
    auto state = std::make_shared<State>();
    state->body = &body;
    state->count = count;
    state->errors.resize(count);

    const auto drain = [](const std::shared_ptr<State>& shared) {
        for (;;) {
            const std::size_t index = shared->next.fetch_add(1);
            if (index >= shared->count) {
                return;
            }
            std::exception_ptr error;
            if (!shared->failed.load()) {
                try {
                    (*shared->body)(index);
                } catch (...) {
                    error = std::current_exception();
                    shared->failed.store(true);
                }
            }
            const std::lock_guard<std::mutex> lock(shared->mutex);
            shared->errors[index] = error;
            if (++shared->finished == shared->count) {
                shared->all_done.notify_all();
            }
        }
    };

    // 执行器不可用（已 release_executor，或初始化失败）时退化为调用线程独干：
    // 加载变慢，但不该因此失败。
    bool helpers_available = true;
    try {
        Scheduler::instance().initialize({});
    } catch (...) {
        helpers_available = false;
    }
    if (helpers_available) {
        for (std::size_t helper = 1; helper < workers; ++helper) {
            try {
                // future 丢弃即可：完成情况由 State 记账，不靠它等待
                //（kairo 的 future 来自 promise，析构不阻塞）。
                (void)kairo::Executor::instance().submit([state, drain] { drain(state); });
            } catch (...) {
                break;  // 排不上就少几路，调用线程会把剩下的活领走
            }
        }
    }

    drain(state);
    {
        std::unique_lock<std::mutex> lock(state->mutex);
        state->all_done.wait(lock, [&state] { return state->finished == state->count; });
    }
    for (const auto& error : state->errors) {
        if (error) {
            std::rethrow_exception(error);
        }
    }
#else
    run_serially(count, body);
#endif
}

}  // namespace genie::detail
