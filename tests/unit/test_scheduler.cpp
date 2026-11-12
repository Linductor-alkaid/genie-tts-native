// tests/unit/test_scheduler.cpp

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <vector>

#include "genie/audio.hpp"
#include "genie/errors.hpp"
#include "genie/scheduler.hpp"
#include "scheduler/parallel.hpp"

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

TEST_F(SchedulerTest, StreamCapacityReflectsSchedulerConfig) {
    // SchedulerConfig::stream_capacity 在 M6 之前是个没人消费的字段；
    // 现在它是流式音频通道容量的正式出口。单例按先到先得初始化，
    // 测试里其它用例都用的默认配置 → 64。
    EXPECT_EQ(Scheduler::instance().stream_capacity(), 64U);
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

TEST(AudioChunkChannelTest, FooterCarriesEndOfStreamMetadata) {
    // 流尾元数据（M6）：未设置时是 nullptr；设置后随通道存活，
    // 重复设置以最后一次为准。
    AudioChunkChannel channel(2);
    EXPECT_EQ(channel.footer(), nullptr);

    StreamFooter first;
    first.timing.record("frontend", 1.0);
    first.error = Error(ErrorCode::InferenceError, "boom");
    channel.set_footer(std::move(first));

    const auto* footer = channel.footer();
    ASSERT_NE(footer, nullptr);
    ASSERT_NE(footer->error, std::nullopt);
    EXPECT_EQ(footer->error->code(), ErrorCode::InferenceError);
    EXPECT_EQ(footer->timing.stages().size(), 1U);

    // close 之后的语义：footer 仍可读（单消费者约定：close 后不再有写入）。
    channel.close();
    EXPECT_EQ(channel.footer(), footer);

    // 最后一次设置生效。
    StreamFooter last;
    channel.set_footer(std::move(last));
    EXPECT_EQ(channel.footer()->error, std::nullopt);
}

TEST(AudioChunkChannelTest, PopWithTimeoutReturnsFalseWhenEmpty) {
    AudioChunkChannel channel(4);
    AudioBuffer received;
    EXPECT_FALSE(channel.pop(received, 20ms));
}

// --------------------------------------------------------------------------- parallel_for

TEST_F(SchedulerTest, ParallelForRunsEveryItem) {
    std::vector<std::atomic<int>> slots(64);
    for (auto& slot : slots) {
        slot.store(0);
    }
    detail::parallel_for(slots.size(), 8, [&slots](const std::size_t i) { slots[i].fetch_add(1); });
    for (std::size_t i = 0; i < slots.size(); ++i) {
        EXPECT_EQ(slots[i].load(), 1) << "item " << i;
    }
}

TEST_F(SchedulerTest, ParallelForPropagatesLowestIndexError) {
    // 两项都抛错：重抛的必须是下标最小的那个 —— 结果与调度顺序无关。
    try {
        detail::parallel_for(4, 4, [](const std::size_t i) {
            if (i == 3) {
                throw_error(ErrorCode::Internal, "later");
            }
            if (i == 1) {
                throw_error(ErrorCode::InvalidArgument, "earlier");
            }
        });
        FAIL() << "expected an error";
    } catch (const Error& e) {
        EXPECT_EQ(e.code(), ErrorCode::InvalidArgument);
        EXPECT_NE(std::string(e.what()).find("earlier"), std::string::npos);
    }
}

TEST_F(SchedulerTest, ParallelForSerialWhenWorkersIsOne) {
    // max_workers = 1：就地串行（不碰执行器）。顺序也必须是下标序。
    std::vector<std::size_t> order;
    detail::parallel_for(8, 1, [&order](const std::size_t i) { order.push_back(i); });
    ASSERT_EQ(order.size(), 8U);
    for (std::size_t i = 0; i < order.size(); ++i) {
        EXPECT_EQ(order[i], i);
    }
}

TEST_F(SchedulerTest, ParallelForEmptyIsNoop) {
    detail::parallel_for(0, 8, [](const std::size_t) { FAIL() << "no items, no calls"; });
}
