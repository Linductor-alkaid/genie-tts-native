// src/engine/stage_timer.hpp
//
// 阶段计时的 RAII 封装 —— 只在 src/ 内部用。
// 公开数据结构是 include/genie/timing.hpp 里的 TimingReport。

#ifndef GENIE_STAGE_TIMER_HPP
#define GENIE_STAGE_TIMER_HPP

#include <chrono>
#include <string>
#include <utility>

#include "genie/timing.hpp"

namespace genie::detail {

/// 构造时取起点，析构时把耗时记进 TimingReport。
///
/// 用法：把一段作用域包成一个阶段。循环迭代数在循环结束后用
/// set_count() 补上（析构时一并提交）；提前结算用 stop()。
class ScopedStageTimer {
public:
    ScopedStageTimer(TimingReport& report, std::string name)
        : report_(report), name_(std::move(name)), start_(std::chrono::steady_clock::now()) {}

    ~ScopedStageTimer() {
        if (!stopped_) {
            finish();
        }
    }

    ScopedStageTimer(const ScopedStageTimer&) = delete;
    ScopedStageTimer& operator=(const ScopedStageTimer&) = delete;

    /// 提前结算并返回耗时（毫秒），之后析构不再重复记录。
    double stop() {
        finish();
        return last_ms_;
    }

    /// 覆盖提交时的次数（如自回归解码的实际步数）。默认为 1。
    void set_count(std::uint32_t count) { count_ = count; }

private:
    void finish() {
        stopped_ = true;
        const auto end = std::chrono::steady_clock::now();
        last_ms_ = std::chrono::duration<double, std::milli>(end - start_).count();
        report_.record(std::move(name_), last_ms_, count_);
    }

    TimingReport& report_;
    std::string name_;
    std::chrono::steady_clock::time_point start_;
    std::uint32_t count_ = 1;
    double last_ms_ = 0.0;
    bool stopped_ = false;
};

}  // namespace genie::detail

#endif  // GENIE_STAGE_TIMER_HPP
