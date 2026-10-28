// tests/unit/test_korean.cpp —— 韩语 G2P 的单元测试。
//
// 覆盖：
//   1. 字母拆解与组装（jamo 的 h2j / j2hcj / compose）
//   2. 数字转韩文（汉字数词 / 固有数词）
//   3. 音变规则（连音、鼻音化、紧音化）
//   4. 前端：音素 → 符号 ID
//
// 黄金语料由 tools/g2p/verify_korean_g2p.py 用**上游的 Python 实现**
// 当 oracle 生成（53 句，逐音素比对）。
//
// 与上游的偏差（mecab-ko 形态标注那一步没有移植）见
// docs/upstream-audit.md §18。

#include <gtest/gtest.h>

#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

#include "genie/errors.hpp"
#include "genie/json.hpp"
#include "genie/symbols.hpp"
#include "genie/text.hpp"
#include "text/korean/jamo.hpp"
#include "text/korean/korean.hpp"

namespace {

using genie::text::korean::g2p;

}  // namespace

TEST(Korean, JamoDecomposition) {
    // "안" → ᄋ ᅡ ᆫ（组合字母 U+11xx ✓）
    const auto decomposed = genie::text::korean::h2j("안");
    const std::vector<std::string> expected = {"ᄋ", "ᅡ", "ᆫ"};
    std::vector<std::string> actual;
    std::size_t offset = 0;
    while (offset < decomposed.size()) {
        const std::string piece = decomposed.substr(offset, 3);
        actual.push_back(piece);
        offset += 3;
    }
    EXPECT_EQ(actual, expected);

    // 再组装回去 ✓
    EXPECT_EQ(genie::text::korean::compose(decomposed), "안");

    // 组合字母 → 兼容字母 ✓
    EXPECT_EQ(genie::text::korean::j2hcj(decomposed), "ㅇㅏㄴ");
}

TEST(Korean, PhonicsFollowUpstream) {
    // 上游 g2pk2 的几条代表性音变 ✓
    // （输出是**兼容字母** ✓ —— 上游最后一步把音节拆开了 ✓）
    EXPECT_EQ(g2p("읽다"), std::vector<std::string>({"ㅇ", "ㅣ", "ㄱ", "ㄸ", "ㅏ", "."}));
    EXPECT_EQ(g2p("있다"), std::vector<std::string>({"ㅇ", "ㅣ", "ㄷ", "ㄸ", "ㅏ", "."}));
    EXPECT_EQ(g2p("그것"), std::vector<std::string>({"ㄱ", "ㅡ", "ㄱ", "ㅓ", "ㄷ", "."}));
}

TEST(Korean, NumbersAreSpelledOut) {
    // 汉字数词 ✓（"2024" → 이천이십사 ✓，逐音素是 ㅇㅣㅊㅓㄴ... ✓）
    const auto year = g2p("2024년");
    EXPECT_NE(std::find(year.begin(), year.end(), "ㅊ"), year.end());
    EXPECT_NE(std::find(year.begin(), year.end(), "ㅂ"), year.end());

    // 固有数词 ✓（"3개" → 세개 ✓，ㅅㅔ ✓）
    const auto count = g2p("3개");
    EXPECT_NE(std::find(count.begin(), count.end(), "ㅅ"), count.end());
    EXPECT_NE(std::find(count.begin(), count.end(), "ㅔ"), count.end());
}

TEST(Korean, EmptyTextProducesNothing) {
    EXPECT_TRUE(g2p("").empty());
    EXPECT_TRUE(g2p("   ").empty());
}

TEST(Korean, GoldenCorpus) {
    std::ifstream file(GENIE_KOREAN_GOLDEN_FILE);
    ASSERT_TRUE(file.good()) << "找不到黄金语料：" << GENIE_KOREAN_GOLDEN_FILE;

    struct Case {
        std::string text;
        std::vector<std::string> phonemes;
    };
    std::vector<Case> cases;
    {
        const genie::json::Value root = genie::json::parse_file(GENIE_KOREAN_GOLDEN_FILE);
        ASSERT_TRUE(root.is_object());
        const genie::json::Value* list = root.find("cases");
        ASSERT_NE(list, nullptr);
        ASSERT_TRUE(list->is_array());
        for (const auto& item : list->as_array()) {
            ASSERT_TRUE(item.is_object());
            Case entry;
            entry.text = item.find("text")->as_string();
            for (const auto& phoneme : item.find("phonemes")->as_array()) {
                entry.phonemes.push_back(phoneme.as_string());
            }
            cases.push_back(std::move(entry));
        }
    }
    ASSERT_FALSE(cases.empty());

    for (const auto& entry : cases) {
        EXPECT_EQ(g2p(entry.text), entry.phonemes) << "音素不一致：" << entry.text;
    }
}

TEST(Korean, FrontendProducesSymbolIds) {
    const auto frontend = genie::text::make_frontend(genie::Language::Korean);
    ASSERT_NE(frontend, nullptr);
    const auto sequence = frontend->convert("안녕하세요.");
    ASSERT_FALSE(sequence.ids.empty());
    EXPECT_EQ(sequence.ids.size(), sequence.symbols.size());
    for (std::size_t i = 0; i < sequence.ids.size(); ++i) {
        EXPECT_EQ(genie::symbols::id_of(sequence.symbols[i]), sequence.ids[i]);
    }
    // 上游的韩文路径不产出 RoBERTa 特征 ✓
    EXPECT_TRUE(sequence.bert_chunks.empty());
}
