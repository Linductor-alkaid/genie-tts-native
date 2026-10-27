// tests/unit/test_hubert.cpp
//
// HuBERT 参考音频特征：输入输出契约与引擎接线。
//
// 上游：cn_hubert.run(None, {'input_values': audio_16k})[0] 原样喂给
// t2s_encoder 的 ssl_content —— 图内已是 [1, 768, T']。

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "engine/hubert.hpp"
#include "genie/audio.hpp"
#include "genie/errors.hpp"
#include "genie/reference.hpp"
#include "support/fake_session.hpp"

using namespace genie;
using namespace genie::test;

namespace {

constexpr std::int64_t kFrames = 50;  // 1 秒 16 kHz → 50 帧（步长 320）

}  // namespace

TEST(Hubert, Feeds16kWaveformAndPassesOutputThrough) {
    auto session = FakeSession::chinese_hubert();

    genie::AudioBuffer audio;
    audio.sample_rate = 16000;
    audio.samples.assign(static_cast<std::size_t>(kFrames) * 320, 0.25F);

    const auto ssl = hubert::compute_ssl(*session, audio);

    // 输入按名字绑定，形状 [1, N]。
    ASSERT_EQ(session->last_feed().size(), 1U);
    EXPECT_EQ(session->last_feed().front().first, "input_values");
    EXPECT_EQ(session->last_feed().front().second.rank(), 2);
    EXPECT_EQ(session->last_feed().front().second.dim(1), kFrames * 320);

    // 输出原样透传：[1, 768, T']，值来自假会话的填充（非零）。
    ASSERT_EQ(ssl.rank(), 3);
    EXPECT_EQ(ssl.dim(0), 1);
    EXPECT_EQ(ssl.dim(1), kSslChannels);
    EXPECT_EQ(ssl.dim(2), kFrames);
    EXPECT_FLOAT_EQ(ssl.floats()[0], 0.25F);
}

// 引擎接线测试在 test_engine.cpp —— TempVoicePackage 夹具在那里。
