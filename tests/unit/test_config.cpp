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

TEST(Config, SessionDefaultsAreTheMeasuredOnes) {
    // 两个默认值都来自实测（docs/performance.md），改动需要新数据支撑：
    // 自旋关 —— 延迟持平、核占用降到 1/3–1/4；并行预打包关 —— 冷启动
    // -8% 换常驻内存 +0.5 GB，不值得默认开。
    const SessionConfig config;
    EXPECT_FALSE(config.allow_spinning);
    EXPECT_FALSE(config.parallel_prepack);
    EXPECT_EQ(config.graph_optimization_level, 3);
    EXPECT_TRUE(config.enable_cpu_mem_arena);
    EXPECT_TRUE(config.enable_mem_pattern);
    // profiler 是测量场景的采集端：每次 Run 都追加记录，常驻开销与文件
    // 体积都不小，绝不能默认开。
    EXPECT_TRUE(config.profile_output_dir.empty());
}
