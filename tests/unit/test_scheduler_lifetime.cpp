// tests/unit/test_scheduler_lifetime.cpp
//
// 调度器**退出路径**的回归测试。
//
// 背景：kairo 的 Executor 是函数级静态 —— 它在 Scheduler::initialize() 里
// 才被首次引用，因此比 Scheduler 单例**更晚构造、更早析构**。
// 进程退出时 Executor 已经没了，Scheduler 的析构函数如果再去碰它
// （request_task_cancel），就是 use-after-free。
//
// 症状不是崩溃，而是**挂住**：kairo 的 shutdown 会等在途任务，最多 300 秒。
//
// 这个测试的作用是让**进程退出时调度器仍处于"就绪"状态** ——
// 也就是走那条有风险的析构路径。调度器的其余测试都在 TearDown 里
// shutdown()，反而绕开了它。
//
// 所以这个文件必须**排在单元测试源文件的最后**：gtest 按注册顺序执行，
// 这个测试要最后跑。

#include <gtest/gtest.h>

#include "genie/scheduler.hpp"

namespace genie {
namespace {

/// 重新初始化调度器，让进程退出时走"就绪状态下的析构"这条路径。
///
/// 如果析构函数又去碰已析构的 kairo Executor，进程会在退出阶段挂住
/// （kairo 最多等 300 秒）—— CI job 会超时失败。
TEST(SchedulerLifetime, LeavesSchedulerReadyAtExit) {
    auto& scheduler = Scheduler::instance();
    if (!scheduler.initialized()) {
        scheduler.initialize(SchedulerConfig{});
    }
    // 交一个马上结束的任务，确认执行器确实建起来了。
    const auto id = scheduler.submit([](const CancellationToken&) {});
    EXPECT_TRUE(scheduler.wait(id, std::chrono::seconds(5)));
    // 故意**不** shutdown —— 这正是要覆盖的退出路径。
}

}  // namespace
}  // namespace genie
