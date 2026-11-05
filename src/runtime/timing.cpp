// src/runtime/timing.cpp
//
// TimingReport 的实现。
//
// 格式化（to_string / to_json）与同名合并都放在 .cpp，是为了不让
// <cstdio> / snprintf 这类实现细节泄进公开头。

#include "genie/timing.hpp"

#include <cstdio>

namespace genie {

void TimingReport::record(std::string name, const double milliseconds, const std::uint32_t count) {
    for (auto& stage : stages_) {
        if (stage.name == name) {
            stage.milliseconds += milliseconds;
            stage.count += count;
            return;
        }
    }
    stages_.push_back(StageTiming{std::move(name), milliseconds, count});
}

void TimingReport::set_total_ms(const double milliseconds) noexcept {
    total_ms_ = milliseconds;
}

double TimingReport::total_milliseconds() const noexcept {
    if (total_ms_ > 0.0) {
        return total_ms_;
    }
    double sum = 0.0;
    for (const auto& stage : stages_) {
        sum += stage.milliseconds;
    }
    return sum;
}

const std::vector<StageTiming>& TimingReport::stages() const noexcept {
    return stages_;
}

std::string TimingReport::to_string() const {
    const double total = total_milliseconds();

    std::string out;
    char line[128];
    std::snprintf(line, sizeof(line), "%-34s %10s %8s %8s\n", "stage", "ms", "%", "count");
    out += line;

    for (const auto& stage : stages_) {
        const double percent = total > 0.0 ? 100.0 * stage.milliseconds / total : 0.0;
        // 子阶段（名字里有 '/'）缩进两格，视觉上表达"属于上面那个阶段"。
        const bool child = stage.name.find('/') != std::string::npos;
        const std::string label = (child ? "  " : "") + stage.name;
        std::snprintf(line, sizeof(line), "%-34s %10.2f %7.1f%% %8u\n", label.c_str(),
                      stage.milliseconds, percent, stage.count);
        out += line;
    }

    std::snprintf(line, sizeof(line), "%-34s %10.2f\n", "total", total);
    out += line;
    return out;
}

std::string TimingReport::to_json() const {
    static constexpr int kNumBufSize = 64;
    char number[kNumBufSize];

    std::string out = "{\"total_ms\":";
    std::snprintf(number, sizeof(number), "%.3f", total_milliseconds());
    out += number;

    out += ",\"stages\":[";
    bool first = true;
    for (const auto& stage : stages_) {
        if (!first) {
            out += ',';
        }
        first = false;
        out += "{\"name\":\"";
        out += stage.name;
        out += "\",\"ms\":";
        std::snprintf(number, sizeof(number), "%.3f", stage.milliseconds);
        out += number;
        out += ",\"count\":";
        std::snprintf(number, sizeof(number), "%u", stage.count);
        out += number;
        out += '}';
    }
    out += "]}";
    return out;
}

}  // namespace genie
