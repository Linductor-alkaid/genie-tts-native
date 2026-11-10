// src/scheduler/parallel.hpp
//
// 调度层内部的 fork-join 并行循环 —— 只在 src/ 内部用（公开 API 不暴露）。
//
// kairo 没有 parallel_for / 任务组；genie::Scheduler 的 submit 吞掉异常、
// 也没有 wait-all —— 都不适合"N 个互不依赖的活并行做完、出错要原样
// 抛给调用方"的加载期场景，因此在调度层补这一个原语（AGENTS.md §1.3：
// kairo 没有的才在本项目里补）。

#ifndef GENIE_SCHEDULER_PARALLEL_HPP
#define GENIE_SCHEDULER_PARALLEL_HPP

#include <cstddef>
#include <functional>

namespace genie::detail {

/// 并行执行 body(0) … body(count-1)，最多 max_workers 路并发，阻塞到全部结束。
///
/// - **调用线程自己也是一个 worker**：各路按原子计数器领活，辅助任务
///   排不上队、甚至根本没被调度时，调用线程独自也能把活干完 —— 等待的
///   只有"已经开工的项"，因此从执行器任务内部调用也不会死锁；
/// - 某一项抛异常后不再领新活；全部已开工的项结束后，重抛**下标最小**
///   的那个异常（结果与调度顺序无关）；
/// - max_workers ≤ 1、count ≤ 1、未编 kairo 的降级构建、或执行器已被
///   release_executor() 释放时，就地串行执行；
/// - 执行器尚未初始化时按默认配置初始化 genie::Scheduler（与流式合成的
///   惰性初始化同一语义）。
///
/// body 会被多个线程并发调用（不同下标），必须对不同下标线程安全。
void parallel_for(std::size_t count, std::size_t max_workers,
                  const std::function<void(std::size_t)>& body);

}  // namespace genie::detail

#endif  // GENIE_SCHEDULER_PARALLEL_HPP
