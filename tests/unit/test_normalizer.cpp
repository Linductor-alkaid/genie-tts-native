// tests/unit/test_normalizer.cpp
//
// 中文文本规范化的黄金语料测试。
//
// 语料由**上游 Python 实现**生成（tools/verify_normalizer.py），
// 本实现逐条对齐 —— 见 docs/upstream-audit.md §15。
//
// 这层测试的意义：规范化规则有几十条，替换顺序还会互相影响。
// 靠人肉 review 迟早漏一条；靠上游当 oracle 全量比对就不会漏。

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "genie/json.hpp"
#include "genie/normalizer.hpp"

namespace genie {
namespace {

/// 黄金语料。
struct Case {
    std::string name;
    std::string input;
    std::string expected;
};

/// 读黄金语料文件。
std::vector<Case> load_cases() {
    std::vector<Case> cases;
    std::ifstream file(GENIE_NORMALIZER_GOLDEN_DIR "/normalizer.json");
    if (!file) {
        return cases;
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    const auto root = json::parse(buffer.str());

    const auto* list = root.find("cases");
    if (list == nullptr || !list->is_array()) {
        return cases;
    }
    for (const auto& item : list->as_array()) {
        Case entry;
        entry.name = item.get_string("name", "");
        entry.input = item.get_string("input", "");
        entry.expected = item.get_string("expected", "");
        cases.push_back(std::move(entry));
    }
    return cases;
}

TEST(Normalizer, GoldenCorpusIsPresent) {
    const auto cases = load_cases();
    ASSERT_FALSE(cases.empty()) << "黄金语料没读到：" GENIE_NORMALIZER_GOLDEN_DIR "/normalizer.json"
                                << "（用 tools/verify_normalizer.py 重新生成）";
    EXPECT_GE(cases.size(), 60U);
}

TEST(Normalizer, MatchesUpstreamGoldenCorpus) {
    const auto cases = load_cases();
    ASSERT_FALSE(cases.empty());

    for (const auto& entry : cases) {
        EXPECT_EQ(text::normalize_chinese(entry.input), entry.expected) << "用例：" << entry.name;
    }
}

// -----------------------------------------------------------------------------
// 几条关键规则的定点测试（黄金语料之外的"人肉可读"断言）
// -----------------------------------------------------------------------------

TEST(Normalizer, ReadsYearDigitByDigit) {
    // 年份是**逐位**读的：2024 → 二零二四（不是 二千零二十四）。
    EXPECT_EQ(text::normalize_chinese("2024年"), "二零二四年");
    EXPECT_EQ(text::normalize_chinese("2024年3月15日"), "二零二四年三月十五日");
}

TEST(Normalizer, ReadsPhoneDigitByDigit) {
    // 手机号逐位读，"一"读成"幺"。
    EXPECT_EQ(text::normalize_chinese("13812345678"), "幺三八幺二三四五六七八");
}

TEST(Normalizer, KeepsOnlyChineseAndPunctuation) {
    // 上游的 pattern_filter：只留中文与 !?…,.- 六种标点。
    EXPECT_EQ(text::normalize_chinese("hello world"), "");
    EXPECT_EQ(text::normalize_chinese("你好，世界！"), "你好,世界!");
    EXPECT_EQ(text::normalize_chinese("你好，世界！"), "你好,世界!");
}

TEST(Normalizer, CollapsesRepeatedPunctuation) {
    // 连续相同标点折叠成一个 —— **按字符**折叠，不是按字节。
    EXPECT_EQ(text::normalize_chinese("你好！！！"), "你好!");
    // "四"（E5 9B 9B）后两个字节相同，按字节折叠会把它吃掉半个。
    EXPECT_EQ(text::normalize_chinese("四"), "四");
}

TEST(Normalizer, ConvertsTraditionalToSimplified) {
    EXPECT_EQ(text::normalize_chinese("這個時間"), "这个时间");
}

TEST(Normalizer, HandlesFullwidthDigits) {
    EXPECT_EQ(text::normalize_chinese("２０２４年"), "二零二四年");
}

TEST(Normalizer, IsIdempotent) {
    // 规范化后再规范化，结果不变（上游的规则都是幂等的）。
    const std::string input = "2024年3月15日，我有12.5个苹果！";
    const auto once = text::normalize_chinese(input);
    EXPECT_EQ(text::normalize_chinese(once), once);
}

TEST(Normalizer, HandlesEmptyInput) {
    EXPECT_EQ(text::normalize_chinese(""), "");
}

}  // namespace
}  // namespace genie
