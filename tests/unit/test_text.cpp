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
    // 三位以上的数字**逐位**读（"一"读成"幺"）—— 与上游一致。
    // 上游的 RE_DEFAULT_NUM = `\d{3}\d*` 在 RE_NUMBER 之前跑，
    // 所以 "123" 读作 "幺二三" 而不是 "一百二十三"。
    EXPECT_EQ(normalize("123", Language::Chinese), "\xE5\xB9\xBA\xE4\xBA\x8C\xE4\xB8\x89");
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

    // 故意用**长**文本：短于 16 字节的 std::string 走 SSO（数据在对象内部），
    // 悬垂引用读到的还是旧数据，ASAN 也抓不到。超过 16 字节才会落到堆上，
    // use-after-free 才能被 ASAN 逮住。
    const auto sequence = frontend->convert(
        "\xE4\xBD\xA0\xE5\xA5\xBD\xE4\xB8\x96\xE7\x95\x8C"
        "\xE4\xBD\xA0\xE5\xA5\xBD\xE4\xB8\x96\xE7\x95\x8C");  // 你好世界（两遍）
    ASSERT_EQ(sequence.ids.size(), 16U);
    // 你 → n + i2（三声连读： ToneSandhi 把"你好"的首字变二声）
    EXPECT_EQ(sequence.symbols[0], "n");
    EXPECT_EQ(sequence.symbols[1], "i2");
    // 好 → h + ao3
    EXPECT_EQ(sequence.symbols[2], "h");
    EXPECT_EQ(sequence.symbols[3], "ao3");
    // 世 → sh + ir4（翘舌韵母来自 opencpop 词表：shi → "sh ir"）
    EXPECT_EQ(sequence.symbols[4], "sh");
    EXPECT_EQ(sequence.symbols[5], "ir4");
    // 界 → j + ie4
    EXPECT_EQ(sequence.symbols[6], "j");
    EXPECT_EQ(sequence.symbols[7], "ie4");
    // 第二遍重复
    EXPECT_EQ(sequence.symbols[8], "n");
    EXPECT_EQ(sequence.symbols[15], "ie4");

    for (const auto id : sequence.ids) {
        EXPECT_GE(id, 0);
        EXPECT_LT(static_cast<std::size_t>(id), symbols::size());
    }
}

TEST(Frontend, KeepsPunctuationAsPhonemes) {
    auto frontend = chinese_frontend();
    ASSERT_NE(frontend, nullptr);

    const auto sequence = frontend->convert("\xE4\xBD\xA0\xE5\xA5\xBD\xE3\x80\x82");
    // 你(n i2，三声连读) 好(h ao3) 。(.)
    ASSERT_EQ(sequence.ids.size(), 5U);
    EXPECT_EQ(sequence.symbols[0], "n");
    EXPECT_EQ(sequence.symbols[1], "i2");
    EXPECT_EQ(sequence.symbols[2], "h");
    EXPECT_EQ(sequence.symbols[3], "ao3");
    EXPECT_EQ(sequence.symbols[4], ".");
    EXPECT_EQ(sequence.ids[4], symbols::id_of("."));
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

    const auto sequence = frontend->convert(
        "\xE4\xBD\xA0\xE5\xA5\xBD\xE4\xB8\x96\xE7\x95\x8C"
        "\xE4\xBD\xA0\xE5\xA5\xBD\xE4\xB8\x96\xE7\x95\x8C");  // 你好世界（两遍）
    // 八个汉字，每个两个音素。
    ASSERT_EQ(sequence.word2ph.size(), 8U);
    for (const auto count : sequence.word2ph) {
        EXPECT_EQ(count, 2);
    }
}

TEST(Frontend, ReportsItsLanguage) {
    auto frontend = chinese_frontend();
    ASSERT_NE(frontend, nullptr);
    EXPECT_EQ(frontend->language(), Language::Chinese);
}

TEST(Frontend, UnsupportedLanguageReturnsNull) {
    // 英文前端需要词典 —— 没有资源时返回 nullptr（不是静默产出空音素）。
    EXPECT_EQ(make_frontend(Language::English), nullptr);
    // 日文还没做（OpenJTalk + 103 MB 词典）。
    EXPECT_EQ(make_frontend(Language::Japanese), nullptr);
}

TEST(Frontend, KoreanFrontendNeedsNoResources) {
    auto frontend = make_frontend(Language::Korean);
    ASSERT_NE(frontend, nullptr);
    EXPECT_EQ(frontend->language(), Language::Korean);
}

}  // namespace
}  // namespace genie::text
