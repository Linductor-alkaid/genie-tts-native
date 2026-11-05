// tests/unit/test_timing.cpp
//
// TimingReport 的纯逻辑测试：同名合并、次数、total 语义、
// to_string / to_json 的稳定格式。不依赖模型、不依赖 ORT。

#include <gtest/gtest.h>

#include <string>

#include "genie/timing.hpp"

namespace genie {
namespace {

TEST(TimingReport, MergesEntriesWithSameName) {
    TimingReport report;
    report.record("frontend", 1.5);
    report.record("frontend", 0.5, 2);
    report.record("vocoder", 3.0);

    ASSERT_EQ(report.stages().size(), 2U);
    EXPECT_EQ(report.stages()[0].name, "frontend");
    EXPECT_DOUBLE_EQ(report.stages()[0].milliseconds, 2.0);
    EXPECT_EQ(report.stages()[0].count, 3U);
    EXPECT_EQ(report.stages()[1].name, "vocoder");
    EXPECT_DOUBLE_EQ(report.stages()[1].milliseconds, 3.0);
    EXPECT_EQ(report.stages()[1].count, 1U);
}

TEST(TimingReport, KeepsInsertionOrder) {
    TimingReport report;
    report.record("b", 1.0);
    report.record("a", 2.0);

    ASSERT_EQ(report.stages().size(), 2U);
    EXPECT_EQ(report.stages()[0].name, "b");
    EXPECT_EQ(report.stages()[1].name, "a");
}

TEST(TimingReport, TotalFallsBackToStageSum) {
    TimingReport report;
    report.record("a", 1.0);
    report.record("b", 2.5);

    EXPECT_DOUBLE_EQ(report.total_milliseconds(), 3.5);
}

TEST(TimingReport, ExplicitTotalWinsOverStageSum) {
    TimingReport report;
    report.record("a", 1.0);
    report.set_total_ms(10.0);

    EXPECT_DOUBLE_EQ(report.total_milliseconds(), 10.0);
}

TEST(TimingReport, EmptyReport) {
    const TimingReport report;

    EXPECT_TRUE(report.stages().empty());
    EXPECT_DOUBLE_EQ(report.total_milliseconds(), 0.0);
    EXPECT_EQ(report.to_json(), "{\"total_ms\":0.000,\"stages\":[]}");
}

TEST(TimingReport, JsonFormatIsStable) {
    // 机器可读格式的回归锚：基准工具落盘、脚本逐级比对都依赖它不变。
    TimingReport report;
    report.record("frontend", 12.5);
    report.record("t2s/stage_decoder", 1.0, 7);
    report.set_total_ms(13.5);

    EXPECT_EQ(report.to_json(),
              "{\"total_ms\":13.500,\"stages\":[{\"name\":\"frontend\",\"ms\":12.500,"
              "\"count\":1},{\"name\":\"t2s/stage_decoder\",\"ms\":1.000,\"count\":7}]}");
}

TEST(TimingReport, ToStringShowsStagesAndTotal) {
    TimingReport report;
    report.record("frontend", 1.0);
    report.record("t2s/stage_decoder", 3.0, 12);
    report.set_total_ms(4.0);

    const auto text = report.to_string();
    EXPECT_NE(text.find("frontend"), std::string::npos);
    EXPECT_NE(text.find("t2s/stage_decoder"), std::string::npos);
    EXPECT_NE(text.find("total"), std::string::npos);
    EXPECT_NE(text.find("12"), std::string::npos);  // count 列
}

}  // namespace
}  // namespace genie
