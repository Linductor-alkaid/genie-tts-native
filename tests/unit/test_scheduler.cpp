// tests/unit/test_scheduler.cpp

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include "genie/audio.hpp"
#include "genie/scheduler.hpp"

using namespace genie;
using namespace std::chrono_literals;

namespace {

class SchedulerTest : public ::testing::Test {
protected:
    void SetUp() override { Scheduler::instance().initialize(SchedulerConfig{}); }
    void TearDown() override { Scheduler::instance().shutdown(); }
};

}  // namespace

TEST_F(SchedulerTest, RunsSubmittedTask) {
    std::atomic<int> counter{0};
    const auto id = Scheduler::instance().submit(
        [&counter](const CancellationToken&) { counter.fetch_add(1); });
    EXPECT_TRUE(Scheduler::instance().wait(id, 5s));
    EXPECT_EQ(counter.load(), 1);
}

TEST_F(SchedulerTest, TaskReceivesCancellationToken) {
    std::atomic<bool> saw_token{false};
    const auto id = Scheduler::instance().submit(
        [&saw_token](const CancellationToken& token) { saw_token.store(!token.cancelled()); });
    EXPECT_TRUE(Scheduler::instance().wait(id, 5s));
    EXPECT_TRUE(saw_token.load());
}

TEST_F(SchedulerTest, CancelMarksTokenCancelled) {
    std::atomic<bool> observed{false};
    std::atomic<bool> started{false};

    const auto id = Scheduler::instance().submit([&](const CancellationToken& token) {
        started.store(true);
        // 忙等一小会儿，给 cancel() 留出时间窗。
        for (int i = 0; i < 1000 && !token.cancelled(); ++i) {
            std::this_thread::sleep_for(1ms);
        }
        observed.store(token.cancelled());
    });

    while (!started.load()) {
        std::this_thread::sleep_for(1ms);
    }
    EXPECT_TRUE(Scheduler::instance().cancel(id));
    EXPECT_TRUE(Scheduler::instance().wait(id, 5s));
    EXPECT_TRUE(observed.load());
}

TEST_F(SchedulerTest, CancelOfUnknownTaskReturnsFalse) {
    EXPECT_FALSE(Scheduler::instance().cancel(999999));
}

TEST_F(SchedulerTest, FailedTaskIsCounted) {
    const auto id = Scheduler::instance().submit(
        [](const CancellationToken&) { throw std::runtime_error("boom"); });
    EXPECT_TRUE(Scheduler::instance().wait(id, 5s));
    // 后台清理线程会记账，给它一点时间。
    for (int i = 0; i < 100 && Scheduler::instance().failure_count() == 0; ++i) {
        std::this_thread::sleep_for(10ms);
    }
    EXPECT_GE(Scheduler::instance().failure_count(), 1U);
}

TEST_F(SchedulerTest, InFlightReturnsToZero) {
    const auto id = Scheduler::instance().submit(
        [](const CancellationToken&) { std::this_thread::sleep_for(20ms); });
    EXPECT_TRUE(Scheduler::instance().wait(id, 5s));
    for (int i = 0; i < 100 && Scheduler::instance().in_flight() != 0; ++i) {
        std::this_thread::sleep_for(10ms);
    }
    EXPECT_EQ(Scheduler::instance().in_flight(), 0U);
}

TEST_F(SchedulerTest, ManyTasksAllComplete) {
    constexpr int kCount = 64;
    std::atomic<int> completed{0};
    std::vector<Scheduler::TaskId> ids;
    for (int i = 0; i < kCount; ++i) {
        ids.push_back(Scheduler::instance().submit(
            [&completed](const CancellationToken&) { completed.fetch_add(1); }));
    }
    for (const auto id : ids) {
        EXPECT_TRUE(Scheduler::instance().wait(id, 10s));
    }
    EXPECT_EQ(completed.load(), kCount);
}

TEST(AudioChunkChannelTest, PushAndPop) {
    AudioChunkChannel channel(4);
    AudioBuffer chunk;
    chunk.sample_rate = 32000;
    chunk.samples = {0.1F, 0.2F};

    EXPECT_TRUE(channel.try_push(chunk));
    AudioBuffer received;
    EXPECT_TRUE(channel.try_pop(received));
    EXPECT_EQ(received.samples.size(), 2U);
    EXPECT_FLOAT_EQ(received.samples[0], 0.1F);
}

TEST(AudioChunkChannelTest, PopOnEmptyReturnsFalse) {
    AudioChunkChannel channel(4);
    AudioBuffer received;
    EXPECT_FALSE(channel.try_pop(received));
}

TEST(AudioChunkChannelTest, FullChannelRejectsPush) {
    AudioChunkChannel channel(2);
    AudioBuffer chunk;
    chunk.samples = {0.0F};
    EXPECT_TRUE(channel.try_push(chunk));
    EXPECT_TRUE(channel.try_push(chunk));
    EXPECT_FALSE(channel.try_push(chunk));
}

TEST(AudioChunkChannelTest, CloseMarksChannelClosed) {
    AudioChunkChannel channel(4);
    EXPECT_FALSE(channel.closed());
    channel.close();
    EXPECT_TRUE(channel.closed());
    EXPECT_TRUE(channel.drained());
}

TEST(AudioChunkChannelTest, PopWithTimeoutReturnsFalseWhenEmpty) {
    AudioChunkChannel channel(4);
    AudioBuffer received;
    EXPECT_FALSE(channel.pop(received, 20ms));
}
