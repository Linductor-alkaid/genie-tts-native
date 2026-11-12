// tests/unit/test_segmenter.cpp
//
// 合成分句器（src/text/segmenter.hpp）的纯逻辑测试。
// M6 流式合成的分段基础：终止符集合、引号粘连、软/硬切、UTF-8 边界。

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "text/segmenter.hpp"

namespace genie {
namespace {

/// 便捷包装：切分并返回可读的段列表。
std::vector<std::string> split(std::string_view text, std::size_t max_chars = 100) {
    return text::split_for_synthesis(text, max_chars);
}

TEST(SegmenterTest, EmptyTextProducesNoSegments) {
    EXPECT_TRUE(split("").empty());
    EXPECT_TRUE(split("   ").empty());           // 纯空白
    EXPECT_TRUE(split("\xE3\x80\x80").empty());  // 全角空格
}

TEST(SegmenterTest, SingleSentenceStaysWhole) {
    // 没有终止符、没超长 → 整段一段（整段合成路径的 parity 基础）。
    const auto segments =
        split("\xE4\xBB\x8A\xE5\xA4\xA9\xE5\xA4\xA9\xE6\xB0\x94\xE4\xB8\x8D\xE9\x94\x99");
    ASSERT_EQ(segments.size(), 1U);
    EXPECT_EQ(segments[0],
              "\xE4\xBB\x8A\xE5\xA4\xA9\xE5\xA4\xA9\xE6\xB0\x94\xE4\xB8\x8D\xE9\x94\x99");
}

TEST(SegmenterTest, SplitsOnHardTerminators) {
    const auto segments = split("你好。世界！再见？hi?");
    ASSERT_EQ(segments.size(), 4U);
    EXPECT_EQ(segments[0], "你好。");
    EXPECT_EQ(segments[1], "世界！");
    EXPECT_EQ(segments[2], "再见？");
    EXPECT_EQ(segments[3], "hi?");
}

TEST(SegmenterTest, ConsecutiveTerminatorsStayTogether) {
    // "你好！！！" 是一段 —— 连续终止符归并，不空切成三段。
    const auto segments = split("你好！！！今天天气不错。");
    ASSERT_EQ(segments.size(), 2U);
    EXPECT_EQ(segments[0], "你好！！！");
    EXPECT_EQ(segments[1], "今天天气不错。");
}

TEST(SegmenterTest, ClosingQuoteGluesToPreviousSentence) {
    // 收尾引号粘进前句，不落在下一段开头。
    const auto segments = split("他说完。”我们走吧。");
    ASSERT_EQ(segments.size(), 2U);
    EXPECT_EQ(segments[0], "他说完。”");
    EXPECT_EQ(segments[1], "我们走吧。");
}

TEST(SegmenterTest, EllipsisSplitsButPurePrefixIsDropped) {
    // 句中省略号是终止符；整段以省略号开头（没有实际内容）时前缀丢弃。
    const auto segments = split("你好……世界！");
    ASSERT_EQ(segments.size(), 2U);
    EXPECT_EQ(segments[0], "你好……");
    EXPECT_EQ(segments[1], "世界！");

    const auto prefixed = split("……你好。");
    ASSERT_EQ(prefixed.size(), 1U);
    EXPECT_EQ(prefixed[0], "你好。");
}

TEST(SegmenterTest, NewlineIsATerminator) {
    const auto segments = split("第一行\n第二行\r\n第三行");
    ASSERT_EQ(segments.size(), 3U);
    EXPECT_EQ(segments[0], "第一行");
    EXPECT_EQ(segments[1], "第二行");
    EXPECT_EQ(segments[2], "第三行");
}

TEST(SegmenterTest, SoftTerminatorsDoNotCutShortSegments) {
    // 逗号 / 分号 / 冒号是软终止符 —— 段没超长就保持完整（韵律更好）。
    // ASCII ':' 在时间 "15:30:45" 里也安全。
    const auto segments = split("时间是15:30:45，别迟到；注意：带上伞。");
    ASSERT_EQ(segments.size(), 1U);
    EXPECT_EQ(segments[0], "时间是15:30:45，别迟到；注意：带上伞。");
}

TEST(SegmenterTest, OverlongSegmentFallsBackToSoftTerminator) {
    // 超过 max_chars 时回退到最近的软终止符落刀。
    const auto segments = split("一二三四，五六七八九。", /*max_chars=*/5);
    ASSERT_EQ(segments.size(), 2U);
    EXPECT_EQ(segments[0], "一二三四，");
    EXPECT_EQ(segments[1], "五六七八九。");
}

TEST(SegmenterTest, OverlongSegmentWithoutSoftPointIsHardCut) {
    // 完全没有标点的长文本按字符硬切（计数按码点而非字节）。
    const auto segments = split("一二三四五六七八九十", /*max_chars=*/5);
    ASSERT_EQ(segments.size(), 2U);
    EXPECT_EQ(segments[0], "一二三四五");
    EXPECT_EQ(segments[1], "六七八九十");
}

TEST(SegmenterTest, ZeroMaxCharsMeansUnlimited) {
    // max_chars == 0 = 不限长：只按硬终止符切。
    const auto segments = split("一二三四五六七八九十", /*max_chars=*/0);
    ASSERT_EQ(segments.size(), 1U);
    EXPECT_EQ(segments[0], "一二三四五六七八九十");
}

TEST(SegmenterTest, AsciiDotIsNotATerminator) {
    // 小数 / 版本号里的点不能断句（与 normalizer 的决策一致）。
    // 代价是英文句尾的 '.' 也不触发分句 —— 上游本来就要求调用方分句。
    const auto segments = split("价格是3.14元，版本号是2.1.0。");
    ASSERT_EQ(segments.size(), 1U);
    EXPECT_EQ(segments[0], "价格是3.14元，版本号是2.1.0。");
}

TEST(SegmenterTest, EdgeWhitespaceIsTrimmed) {
    // 半角空格、全角空格、制表符、段边界的换行都修剪。
    const auto segments = split("  你好。　世界！\t\n");
    ASSERT_EQ(segments.size(), 2U);
    EXPECT_EQ(segments[0], "你好。");
    EXPECT_EQ(segments[1], "世界！");
}

TEST(SegmenterTest, PurePunctuationTextProducesNoSegments) {
    // 全是终止符的文本没有任何可合成的内容。
    EXPECT_TRUE(split("。。。 ！？ \n").empty());
}

// ------------------------------------------------------------------------------------------
// complete_prefix —— 增量喂入的"可成段前缀"。规则与 split_for_synthesis 同源，
// 喂入侧反复调用它必须产出与整段 split 相同的段序列。
// ------------------------------------------------------------------------------------------

TEST(SegmenterTest, CompletePrefixNoTerminatorMeansNothingYet) {
    const auto cut = text::complete_prefix("今天天气真好", 100);
    EXPECT_EQ(cut.end, 0U);
    EXPECT_EQ(cut.skip, 0U);
}

TEST(SegmenterTest, CompletePrefixCutsAtTerminator) {
    const auto cut = text::complete_prefix("你好。世界", 100);
    EXPECT_EQ(cut.skip, 0U);
    EXPECT_EQ(cut.end, 9U);  // "你好。"
}

TEST(SegmenterTest, CompletePrefixDiscardsPurePunctPrefix) {
    // "……" 是纯终止符前缀：skip=6（丢弃），段文本 [6,15) = "你好。"
    const auto cut = text::complete_prefix("……你好。世界", 100);
    EXPECT_EQ(cut.skip, 6U);
    EXPECT_EQ(cut.end, 15U);
}

TEST(SegmenterTest, CompletePrefixSoftCutMatchesSplit) {
    // 超长时回退软终止符 —— 与 split 的刀口完全一致。
    const auto cut = text::complete_prefix("一二三四，五六七八九十", 5);
    EXPECT_EQ(cut.skip, 0U);
    EXPECT_EQ(cut.end, 15U);  // "一二三四，"
}

TEST(SegmenterTest, CompletePrefixDefersToBoundary) {
    // 超限但下一个字符是句号 —— 这一刀让给它：句尾标点留在段里。
    const auto cut = text::complete_prefix("五六七八九。其余", 5);
    ASSERT_NE(cut.end, 0U);
    EXPECT_EQ(cut.skip, 0U);
    EXPECT_EQ(cut.end, 18U);  // "五六七八九。"
}

TEST(SegmenterTest, CompletePrefixMaxZeroOnlyHardTerminators) {
    const auto cut = text::complete_prefix("一二三四五六七八九十", 0);
    EXPECT_EQ(cut.end, 0U);  // 不限长：没有终止符就不成段
    const auto cut2 = text::complete_prefix("你好。世界。", 0);
    EXPECT_EQ(cut2.end, 9U);
}

TEST(SegmenterTest, IncrementalFeedingMatchesWholeSplit) {
    // 核心不变量：逐字节喂入反复调 complete_prefix，产出的段序列与
    // 整段 split_for_synthesis 完全一致（文本不含空白，段无修剪差异）。
    const std::string text = "今天天气真好。我们去公园散步，然后吃饭。好吗？好！";
    const std::vector<std::string> whole = split(text, 20);

    std::vector<std::string> incremental;
    std::string buffer;
    for (const char c : text) {
        buffer.push_back(c);
        for (;;) {
            const auto cut = text::complete_prefix(buffer, 20);
            if (cut.end == 0) {
                break;
            }
            incremental.push_back(buffer.substr(cut.skip, cut.end - cut.skip));
            buffer.erase(0, cut.end);
        }
    }
    if (!buffer.empty()) {
        incremental.push_back(buffer);  // finish 冲刷残余
    }

    ASSERT_EQ(incremental.size(), whole.size());
    for (std::size_t i = 0; i < whole.size(); ++i) {
        EXPECT_EQ(incremental[i], whole[i]) << "segment " << i;
    }
}

}  // namespace
}  // namespace genie
