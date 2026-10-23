// tests/unit/test_text.cpp
//
// 文本规范化与中文前端的测试。

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "genie/symbols.hpp"
#include "genie/text.hpp"

namespace genie::text {
namespace {

std::unique_ptr<TextFrontend> chinese_frontend() {
    return make_frontend(Language::Chinese);
}

TEST(Text, NormalizeConvertsFullWidthToHalfWidth) {
    // 全角数字 １２ → 半角 12
    EXPECT_EQ(normalize("\xEF\xBC\x91\xEF\xBC\x92", Language::English), "12");
    // 全角字母 Ａ → 半角 A
    EXPECT_EQ(normalize("\xEF\xBC\xA1", Language::English), "A");
}

TEST(Text, NormalizeMapsChinesePunctuation) {
    // 全角逗号 → 半角逗号（就地替换）
    EXPECT_EQ(normalize("\xE4\xBD\xA0\xEF\xBC\x8C\xE5\xA5\xBD", Language::Chinese),
              "\xE4\xBD\xA0,\xE5\xA5\xBD");
    // 句号 → "."
    EXPECT_EQ(normalize("\xE4\xBD\xA0\xE3\x80\x82", Language::Chinese), "\xE4\xBD\xA0.");
    // 顿号 → ","
    EXPECT_EQ(normalize("\xE4\xBD\xA0\xE3\x80\x81", Language::Chinese), "\xE4\xBD\xA0,");
}

TEST(Text, NormalizeCollapsesRepeatedPunctuation) {
    // 连续句号折叠成一个
    EXPECT_EQ(normalize("\xE4\xBD\xA0\xE3\x80\x82\xE3\x80\x82", Language::Chinese),
              "\xE4\xBD\xA0.");
    // 连续感叹号折叠成一个
    EXPECT_EQ(normalize("\xE4\xBD\xA0\xEF\xBC\x81\xEF\xBC\x81", Language::Chinese),
              "\xE4\xBD\xA0!");
    // 三连点 → 省略号
    EXPECT_EQ(normalize("...", Language::Chinese), "\xE2\x80\xA6");
}

TEST(Text, NormalizeReadsNumbersAsChinese) {
    // "123" → 一百二十三
    EXPECT_EQ(normalize("123", Language::Chinese),
              "\xE4\xB8\x80\xE7\x99\xBE\xE4\xBA\x8C\xE5\x8D\x81\xE4\xB8\x89");
    // "0" → 零
    EXPECT_EQ(normalize("0", Language::Chinese), "\xE9\x9B\xB6");
    // "10" → 十（一十缩写）
    EXPECT_EQ(normalize("10", Language::Chinese), "\xE5\x8D\x81");
    // "3.5" → 三点五
    EXPECT_EQ(normalize("3.5", Language::Chinese), "\xE4\xB8\x89\xE7\x82\xB9\xE4\xBA\x94");
}

TEST(Text, NormalizeDropsCharactersOutsideLanguage) {
    // 英文路径丢掉汉字
    EXPECT_EQ(normalize("ab\xE4\xBD\xA0", Language::English), "ab");
    // 中文路径丢掉英文字母
    EXPECT_EQ(normalize("a\xE4\xBD\xA0", Language::Chinese), "\xE4\xBD\xA0");
}

TEST(Text, NormalizeCollapsesWhitespace) {
    EXPECT_EQ(normalize("a   b", Language::English), "a b");
    EXPECT_EQ(normalize("  a  ", Language::English), "a");
}

TEST(Frontend, ConvertsChineseToPhonemeIds) {
    auto frontend = chinese_frontend();
    ASSERT_NE(frontend, nullptr);

    const auto sequence = frontend->convert("\xE4\xBD\xA0\xE5\xA5\xBD");
    ASSERT_EQ(sequence.ids.size(), 4U);
    // 你 → n + i3
    EXPECT_EQ(sequence.symbols[0], "n");
    EXPECT_EQ(sequence.symbols[1], "i3");
    // 好 → h + ao3
    EXPECT_EQ(sequence.symbols[2], "h");
    EXPECT_EQ(sequence.symbols[3], "ao3");

    for (const auto id : sequence.ids) {
        EXPECT_GE(id, 0);
        EXPECT_LT(static_cast<std::size_t>(id), symbols::size());
    }
}

TEST(Frontend, KeepsPunctuationAsPhonemes) {
    auto frontend = chinese_frontend();
    ASSERT_NE(frontend, nullptr);

    const auto sequence = frontend->convert("\xE4\xBD\xA0\xE3\x80\x82");
    ASSERT_EQ(sequence.ids.size(), 3U);
    EXPECT_EQ(sequence.symbols[0], "n");
    EXPECT_EQ(sequence.symbols[1], "i3");
    EXPECT_EQ(sequence.symbols[2], ".");
    EXPECT_EQ(sequence.ids[2], symbols::id_of("."));
}

TEST(Frontend, EmptyTextYieldsEmptySequence) {
    auto frontend = chinese_frontend();
    ASSERT_NE(frontend, nullptr);
    EXPECT_TRUE(frontend->convert("").empty());
    EXPECT_TRUE(frontend->convert("   ").empty());
}

TEST(Frontend, UnknownCharactersAreSkipped) {
    auto frontend = chinese_frontend();
    ASSERT_NE(frontend, nullptr);
    // 龘 在词典之外（12,010 常用字表里没有）→ 跳过。
    const auto sequence = frontend->convert("\xE9\xBE\x98");
    EXPECT_TRUE(sequence.empty());
}

TEST(Frontend, Word2PhCountsPhonesPerCharacter) {
    auto frontend = chinese_frontend();
    ASSERT_NE(frontend, nullptr);

    const auto sequence = frontend->convert("\xE4\xBD\xA0\xE5\xA5\xBD");
    // 两个汉字，每个两个音素。
    ASSERT_EQ(sequence.word2ph.size(), 2U);
    EXPECT_EQ(sequence.word2ph[0], 2);
    EXPECT_EQ(sequence.word2ph[1], 2);
}

TEST(Frontend, ReportsItsLanguage) {
    auto frontend = chinese_frontend();
    ASSERT_NE(frontend, nullptr);
    EXPECT_EQ(frontend->language(), Language::Chinese);
}

TEST(Frontend, UnsupportedLanguageReturnsNull) {
    // 英文 / 日文 / 韩文的上游实现依赖 Python 生态，本项目暂不内置。
    EXPECT_EQ(make_frontend(Language::English), nullptr);
    EXPECT_EQ(make_frontend(Language::Japanese), nullptr);
    EXPECT_EQ(make_frontend(Language::Korean), nullptr);
}

}  // namespace
}  // namespace genie::text
