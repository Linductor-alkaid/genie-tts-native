// include/genie/timing.hpp
//
// 推理链路的分阶段耗时报告（roadmap M0.3 性能基线）。
//
// 设计取舍（见 docs/architecture.md 的性能计时一节）：
//   - 计时**常开**，没有开关。每个阶段只做两次 steady_clock 采样
//     （纳秒级），相对秒级的推理总开销可以忽略 —— 一个永远开着、
//     没人需要关的开关，比一个要沿整条调用链传递的布尔参数便宜。
//   - 报告随 SynthesisResult 返回、加载报告随 Engine 查询，而不是打到
//     日志：怎么呈现由调用方决定（CLI 打表格、基准工具算统计、SDK
//     使用者直接编程消费），库本身不引入任何日志依赖。
//   - 阶段名用 '/' 分层（"t2s/stage_decoder"），to_string() 据此缩进。

#ifndef GENIE_TIMING_HPP
#define GENIE_TIMING_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace genie {

/// 单个阶段的耗时。
struct StageTiming {
    /// 阶段名，'/' 分层（如 "t2s/stage_decoder"）。
    std::string name;
    /// 该阶段的累计耗时（毫秒）。
    double milliseconds = 0.0;
    /// 调用 / 迭代次数（如自回归解码的步数）；单次执行的阶段为 1。
    std::uint32_t count = 0;
};

/// 一次合成（或一次引擎加载）的分阶段耗时报告。
class TimingReport {
public:
    /// 记录一个阶段。同名阶段累加合并（保持首次出现的顺序）。
    void record(std::string name, double milliseconds, std::uint32_t count = 1);

    /// 报告覆盖的端到端耗时。
    ///
    /// 与各阶段之和的差值就是"未单独计时的编排开销"（张量拼装、
    /// 拷贝等）—— 这个差值本身也是 M5 优化的观测对象。
    void set_total_ms(double milliseconds) noexcept;
    /// 未显式设置时回退为各阶段之和。
    [[nodiscard]] double total_milliseconds() const noexcept;

    /// 各阶段（按记录顺序）。
    [[nodiscard]] const std::vector<StageTiming>& stages() const noexcept;

    /// 人类可读的对齐表格（含占比与次数）。每行以 '\n' 结束。
    [[nodiscard]] std::string to_string() const;

    /// 机器可读 JSON（基准工具落盘、脚本比对用）。
    /// 单行、无换行结尾；阶段名是内部标识符（受限字符集），不做转义。
    [[nodiscard]] std::string to_json() const;

private:
    std::vector<StageTiming> stages_;
    /// 显式设置的端到端时长；0 表示未设置（回退为各阶段之和）。
    double total_ms_ = 0.0;
};

}  // namespace genie

#endif  // GENIE_TIMING_HPP
