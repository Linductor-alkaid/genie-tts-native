// tests/unit/test_config.cpp

#include <gtest/gtest.h>

#include <string>

#include "genie/config.hpp"

using namespace genie;

TEST(Config, LanguageStringRoundTrip) {
    EXPECT_EQ(language_from_string("auto"), Language::Auto);
    EXPECT_EQ(language_from_string("chinese"), Language::Chinese);
    EXPECT_EQ(language_from_string("zh"), Language::Chinese);
    EXPECT_EQ(language_from_string("english"), Language::English);
    EXPECT_EQ(language_from_string("ja"), Language::Japanese);
    EXPECT_EQ(language_from_string("korean"), Language::Korean);
    EXPECT_EQ(language_from_string("hybrid-chinese-english"), Language::HybridChineseEnglish);

    EXPECT_EQ(language_to_string(Language::Chinese), "chinese");
    EXPECT_EQ(language_to_string(Language::HybridChineseEnglish), "hybrid-chinese-english");
}

TEST(Config, UnknownLanguageIsRejected) {
    EXPECT_FALSE(language_from_string("klingon").has_value());
    EXPECT_FALSE(language_from_string("").has_value());
}

TEST(Config, DetectsChinese) {
    EXPECT_EQ(detect_language("你好世界"), Language::Chinese);
}

TEST(Config, DetectsEnglish) {
    EXPECT_EQ(detect_language("hello world"), Language::English);
}

TEST(Config, DetectsJapanese) {
    EXPECT_EQ(detect_language("こんにちは"), Language::Japanese);
    EXPECT_EQ(detect_language("漢字とカタカナ"), Language::Japanese);
}

TEST(Config, DetectsKorean) {
    EXPECT_EQ(detect_language("안녕하세요"), Language::Korean);
}

// 上游规则：显式 chinese + 含 ASCII 字母 → 升级为混读模式。
// 见 upstream src/genie_tts/GetPhonesAndBert.py get_phones_and_bert()。
TEST(Config, ExplicitChineseWithLatinUpgradesToHybrid) {
    EXPECT_EQ(detect_language("你好 Mira", Language::Chinese), Language::HybridChineseEnglish);
    EXPECT_EQ(detect_language("你好世界", Language::Chinese), Language::Chinese);
}

TEST(Config, AutoDetectsChineseWithLatinAsHybrid) {
    EXPECT_EQ(detect_language("使用 ONNX 推理"), Language::HybridChineseEnglish);
}

TEST(Config, ExplicitLanguageIsRespected) {
    EXPECT_EQ(detect_language("hello", Language::English), Language::English);
    EXPECT_EQ(detect_language("你好", Language::English), Language::English);
}
