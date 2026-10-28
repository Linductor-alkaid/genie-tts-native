// tests/unit/test_tone_sandhi.cpp
//
// 变调（ToneSandhi）与儿化（Erhua）移植的测试。
//
// 分支测试直接构造 Unit（韵母手工给定，隔离拼音表差异）；
// 端到端测试走中文前端（真实拼音表 → 音素符号）。

#include <gtest/gtest.h>

#include <fstream>
#include <string>
#include <vector>

#include "genie/json.hpp"
#include "genie/pinyin.hpp"
#include "genie/symbols.hpp"
#include "genie/text.hpp"
#include "genie/utf8.hpp"
#include "text/tone_sandhi.hpp"

#ifndef GENIE_SANDHI_GOLDEN_FILE
#define GENIE_SANDHI_GOLDEN_FILE "tests/golden/sandhi.json"
#endif

namespace genie::text {
namespace {

/// 以字符列表构造单元（UTF-8 字面量直接切）。
sandhi::Unit make_unit(const std::vector<std::string>& chars,
                       const std::vector<std::string>& finals) {
    sandhi::Unit unit;
    unit.chars = chars;
    unit.finals = finals;
    return unit;
}

/// 把 UTF-8 串切成单字列表。
std::vector<std::string> chars_of(const std::string& text) {
    std::vector<std::string> chars;
    std::size_t offset = 0;
    utf8::Char ch;
    while (utf8::decode_next(text, offset, ch)) {
        chars.emplace_back(text.substr(ch.offset, ch.length));
    }
    return chars;
}

/// 端到端：文本 → 音素符号（空格分隔，便于断言）。
std::string symbols_of(const std::string& text) {
    const auto frontend = make_frontend(Language::Chinese);
    const auto sequence = frontend->convert(text);
    std::string joined;
    for (const auto& symbol : sequence.symbols) {
        if (!joined.empty()) {
            joined += ' ';
        }
        joined += symbol;
    }
    return joined;
}

// ---------------------------------------------------------------------------
// 分词
// ---------------------------------------------------------------------------

TEST(SandhiSegment, KeepsLexiconWordsWhole) {
    // "东西" 在 must_neural 词表里 → 保持整词。
    const auto units = sandhi::segment(chars_of("东西"));
    ASSERT_EQ(units.size(), 1U);
    EXPECT_EQ(units[0].word(), "东西");
}

TEST(SandhiSegment, FallsBackToSingleChars) {
    // 词表外的字按单字切。
    const auto units = sandhi::segment(chars_of("天空"));
    ASSERT_EQ(units.size(), 2U);
    EXPECT_EQ(units[0].word(), "天");
    EXPECT_EQ(units[1].word(), "空");
}

TEST(SandhiSegment, GreedyLongestMatch) {
    // "老头" 在词表里：优先取两字词，"儿"单独成单元等待 merge_er。
    const auto units = sandhi::segment(chars_of("老头儿"));
    ASSERT_EQ(units.size(), 2U);
    EXPECT_EQ(units[0].word(), "老头");
    EXPECT_EQ(units[1].word(), "儿");
}

TEST(SandhiSegment, MustErhuaWordStaysWhole) {
    // "媳妇儿" 本身就在 must_erhua 词表里 → 一整个单元。
    const auto units = sandhi::segment(chars_of("媳妇儿"));
    ASSERT_EQ(units.size(), 1U);
    EXPECT_EQ(units[0].word(), "媳妇儿");
}

// ---------------------------------------------------------------------------
// pre_merge（上游 pre_merge_for_modify）
// ---------------------------------------------------------------------------

TEST(SandhiPreMerge, MergesBareBu) {
    // 上游 _merge_bu：光杆"不"与后词合并（"看不懂"）。
    std::vector<sandhi::Unit> seg;
    seg.push_back(make_unit({"看"}, {"an4"}));
    seg.push_back(make_unit({"不"}, {"u4"}));
    seg.push_back(make_unit({"懂"}, {"ong3"}));
    sandhi::pre_merge_for_modify(seg);
    ASSERT_EQ(seg.size(), 2U);
    EXPECT_EQ(seg[1].word(), "不懂");
    // "不"的真实韵母要跟着合并走。
    ASSERT_EQ(seg[1].finals.size(), 2U);
    EXPECT_EQ(seg[1].finals[0], "u4");
}

TEST(SandhiPreMerge, MergesYiWithNextWord) {
    // 上游 _merge_yi function 2："一"吞掉后一个词（"一天"）。
    std::vector<sandhi::Unit> seg;
    seg.push_back(make_unit({"一"}, {"i1"}));
    seg.push_back(make_unit({"天"}, {"ian1"}));
    sandhi::pre_merge_for_modify(seg);
    ASSERT_EQ(seg.size(), 1U);
    EXPECT_EQ(seg[0].word(), "一天");
}

TEST(SandhiPreMerge, MergesContinuousThreeTones) {
    // 上游 _merge_continuous_three_tones：相邻全三声 ≤3 字并成一个变调域。
    std::vector<sandhi::Unit> seg;
    seg.push_back(make_unit({"你"}, {"i3"}));
    seg.push_back(make_unit({"好"}, {"ao3"}));
    sandhi::pre_merge_for_modify(seg);
    ASSERT_EQ(seg.size(), 1U);
    EXPECT_EQ(seg[0].word(), "你好");
}

TEST(SandhiPreMerge, DoesNotMergeAcrossNonThreeTone) {
    // 中间隔着非三声 → 不合并。
    std::vector<sandhi::Unit> seg;
    seg.push_back(make_unit({"你"}, {"i3"}));
    seg.push_back(make_unit({"是"}, {"@4"}));
    seg.push_back(make_unit({"好"}, {"ao3"}));
    sandhi::pre_merge_for_modify(seg);
    EXPECT_EQ(seg.size(), 3U);
}

TEST(SandhiPreMerge, MergesBareErIntoPrevious) {
    // 上游 _merge_er："儿"并入前一个词。
    std::vector<sandhi::Unit> seg;
    seg.push_back(make_unit({"花"}, {"ua1"}));
    seg.push_back(make_unit({"儿"}, {"er2"}));
    sandhi::pre_merge_for_modify(seg);
    ASSERT_EQ(seg.size(), 1U);
    EXPECT_EQ(seg[0].word(), "花儿");
    ASSERT_EQ(seg[0].finals.size(), 2U);
    EXPECT_EQ(seg[0].finals[1], "er2");
}

// ---------------------------------------------------------------------------
// 变调（上游 modified_tone）
// ---------------------------------------------------------------------------

TEST(SandhiTone, BuBeforeFourthToneBecomesSecond) {
    // 上游 _bu_sandhi："不怕" → bú pà。
    auto unit = make_unit({"不", "怕"}, {"u4", "a4"});
    sandhi::modified_tone(unit);
    EXPECT_EQ(unit.finals[0], "u2");
    EXPECT_EQ(unit.finals[1], "a4");
}

TEST(SandhiTone, BuStaysBeforeNonFourthTone) {
    auto unit = make_unit({"不", "听"}, {"u4", "ing1"});
    sandhi::modified_tone(unit);
    EXPECT_EQ(unit.finals[0], "u4");
}

TEST(SandhiTone, MidBuOfThreeCharWordIsNeutral) {
    // 上游 _bu_sandhi：三字中间的"不"读轻声（"看不懂"）。
    auto unit = make_unit({"看", "不", "懂"}, {"an4", "u4", "ong3"});
    sandhi::modified_tone(unit);
    EXPECT_EQ(unit.finals[1], "u5");
}

TEST(SandhiTone, YiBeforeFourthToneBecomesSecond) {
    // 上游 _yi_sandhi："一段" → yí duàn。
    auto unit = make_unit({"一", "段"}, {"i1", "uan4"});
    sandhi::modified_tone(unit);
    EXPECT_EQ(unit.finals[0], "i2");
}

TEST(SandhiTone, YiBeforeNonFourthToneBecomesFourth) {
    // "一天" → yì tiān。
    auto unit = make_unit({"一", "天"}, {"i1", "ian1"});
    sandhi::modified_tone(unit);
    EXPECT_EQ(unit.finals[0], "i4");
}

TEST(SandhiTone, YiInOrdinalStaysFirstTone) {
    // "第一" 的"一"读一声。
    auto unit = make_unit({"第", "一"}, {"i4", "i1"});
    sandhi::modified_tone(unit);
    EXPECT_EQ(unit.finals[1], "i1");
}

TEST(SandhiTone, YiBetweenReduplicationIsNeutral) {
    // "看一看" 的"一"读轻声。
    auto unit = make_unit({"看", "一", "看"}, {"an4", "i1", "an4"});
    sandhi::modified_tone(unit);
    EXPECT_EQ(unit.finals[1], "i5");
}

TEST(SandhiTone, YiInDigitSequenceUnchanged) {
    // 数字序列里的"一"不变调（"一零零"）。
    auto unit = make_unit({"一", "零", "零"}, {"i1", "ing2", "ing2"});
    sandhi::modified_tone(unit);
    EXPECT_EQ(unit.finals[0], "i1");
}

TEST(SandhiTone, MustNeuralWordList) {
    // 上游 must_neural_tone_words："东西" 的"西"读轻声。
    auto unit = make_unit({"东", "西"}, {"ong1", "i1"});
    sandhi::modified_tone(unit);
    EXPECT_EQ(unit.finals[1], "i5");
}

TEST(SandhiTone, ReduplicationSecondCharNeutral) {
    // 叠词第二字读轻声（上游 POS 限定 n/v/a，无 POS 直接应用）。
    auto unit = make_unit({"奶", "奶"}, {"ai3", "ai3"});
    sandhi::modified_tone(unit);
    EXPECT_EQ(unit.finals[1], "ai5");
}

TEST(SandhiTone, MustNotNeuralExempts) {
    // must_not_neural_tone_words："想想" 不轻声。
    auto unit = make_unit({"想", "想"}, {"iang3", "iang3"});
    sandhi::modified_tone(unit);
    EXPECT_EQ(unit.finals[1], "iang3");
}

TEST(SandhiTone, ClassifierGeIsNeutral) {
    // 量词"个"读轻声（"三个"）。
    auto unit = make_unit({"三", "个"}, {"an1", "e4"});
    sandhi::modified_tone(unit);
    EXPECT_EQ(unit.finals[1], "e5");
}

TEST(SandhiTone, LocativeSuffixIsNeutral) {
    // 方位词"上"读轻声（"桌上"）。
    auto unit = make_unit({"桌", "上"}, {"uo1", "ang4"});
    sandhi::modified_tone(unit);
    EXPECT_EQ(unit.finals[1], "ang5");
}

TEST(SandhiTone, TwoThirdTonesFirstBecomesSecond) {
    // 上游 _three_sandhi："你好" → ní hǎo。
    auto unit = make_unit({"你", "好"}, {"i3", "ao3"});
    sandhi::modified_tone(unit);
    EXPECT_EQ(unit.finals[0], "i2");
    EXPECT_EQ(unit.finals[1], "ao3");
}

TEST(SandhiTone, SinglePlusDoubleThirdTones) {
    // 上游 _three_sandhi："纸老虎"（单字 + 双字）→ 中字变二声。
    auto unit = make_unit({"纸", "老", "虎"}, {"@3", "ao3", "u3"});
    sandhi::modified_tone(unit);
    EXPECT_EQ(unit.finals[0], "@3");
    EXPECT_EQ(unit.finals[1], "ao2");
    EXPECT_EQ(unit.finals[2], "u3");
}

TEST(SandhiTone, FourCharIdiomSplitsInHalf) {
    // 上游 _three_sandhi：四字拆两半各自连读。
    // "你好" 与 "洗手" 两半都是全三声，各自首字变二声。
    auto unit = make_unit({"你", "好", "洗", "手"}, {"i3", "ao3", "i3", "ou3"});
    sandhi::modified_tone(unit);
    EXPECT_EQ(unit.finals[0], "i2");
    EXPECT_EQ(unit.finals[1], "ao3");
    EXPECT_EQ(unit.finals[2], "i2");
    EXPECT_EQ(unit.finals[3], "ou3");
}

// ---------------------------------------------------------------------------
// 儿化（上游 merge_erhua）
// ---------------------------------------------------------------------------

TEST(SandhiErhua, ErTakesPreviousTone) {
    // "味儿" 不在两张表里 → 儿改读前字声调（wèi → er4）。
    auto unit = make_unit({"味", "儿"}, {"ei4", "er2"});
    sandhi::merge_erhua(unit);
    EXPECT_EQ(unit.finals[1], "er4");
}

TEST(SandhiErhua, NotErhuaListSkipped) {
    // "花儿" 在 not_erhua 表里 → 不儿化，保持 er2。
    auto unit = make_unit({"花", "儿"}, {"ua1", "er2"});
    sandhi::merge_erhua(unit);
    EXPECT_EQ(unit.finals[1], "er2");
}

TEST(SandhiErhua, Er1CorrectedToEr2) {
    // 上游第 1 步：词尾"儿"读 er1 的先修正为 er2。
    auto unit = make_unit({"味", "儿"}, {"ei4", "er1"});
    sandhi::merge_erhua(unit);
    // 修正后继续走第 4 步 → 取前字声调。
    EXPECT_EQ(unit.finals[1], "er4");
}

TEST(SandhiErhua, WithoutErSuffixUnchanged) {
    auto unit = make_unit({"女", "儿"}, {"v3", "er2"});
    sandhi::merge_erhua(unit);
    // "女儿" 在 not_erhua 表里。
    EXPECT_EQ(unit.finals[1], "er2");
}

// ---------------------------------------------------------------------------
// 端到端（真实拼音表 → 音素符号）
// ---------------------------------------------------------------------------

TEST(SandhiFrontend, ThirdToneSandhiEndToEnd) {
    // 你好 → ní hǎo：n i2 h ao3
    EXPECT_EQ(symbols_of("你好"), "n i2 h ao3");
}

TEST(SandhiFrontend, BuSandhiEndToEnd) {
    // 不怕 → bú pà：b u2 p a4
    EXPECT_EQ(symbols_of("不怕"), "b u2 p a4");
}

TEST(SandhiFrontend, YiSandhiEndToEnd) {
    // 一天 → yì tiān：y i4 t ian1
    EXPECT_EQ(symbols_of("一天"), "y i4 t ian1");
    // 一段 → yí duàn：y i2 d uan4
    EXPECT_EQ(symbols_of("一段"), "y i2 d uan4");
}

TEST(SandhiFrontend, NeutralWordEndToEnd) {
    // 东西 → dōngxi：d ong1 x i5
    EXPECT_EQ(symbols_of("东西"), "d ong1 x i5");
}

TEST(SandhiFrontend, ClassifierEndToEnd) {
    // 三个 → sān gè：s an1 g e5
    EXPECT_EQ(symbols_of("三个"), "s an1 g e5");
}

TEST(SandhiFrontend, PunctuationBreaksSandhiScope) {
    // 标点打断变调域：两段各自处理，标点照旧输出。
    EXPECT_EQ(symbols_of("你好，老虎"), "n i2 h ao3 , l ao2 h u3");
}

TEST(SandhiFrontend, Word2phTracksPhonesPerChar) {
    const auto frontend = make_frontend(Language::Chinese);
    const auto sequence = frontend->convert("你好");
    // "你" → 声母+韵母 2 个音素；"好" 同。
    ASSERT_EQ(sequence.word2ph.size(), 2U);
    EXPECT_EQ(sequence.word2ph[0], 2);
    EXPECT_EQ(sequence.word2ph[1], 2);
}

// ---------------------------------------------------------------------------
// 黄金语料（tools/verify_sandhi.py 生成；与上游 oracle 的比对结论
// 见 docs/upstream-audit.md §9.6）
// ---------------------------------------------------------------------------

/// 与 chinese_frontend.cpp / tools/g2p/dump_sandhi.cpp 相同的流水线。
std::vector<sandhi::Unit> pipeline(const std::vector<std::string>& chars) {
    std::vector<sandhi::Unit> units = sandhi::segment(chars);
    std::size_t index = 0;
    for (auto& unit : units) {
        for (auto& final_with_tone : unit.finals) {
            if (const auto pinyin = pinyin::lookup(chars[index]); pinyin) {
                if (const auto syllable = pinyin::split(*pinyin); syllable.valid()) {
                    final_with_tone = syllable.final_with_tone;
                }
            }
            ++index;
        }
    }
    sandhi::pre_merge_for_modify(units);
    for (auto& unit : units) {
        sandhi::modified_tone(unit);
        sandhi::merge_erhua(unit);
    }
    return units;
}

TEST(SandhiGolden, CorpusMatchesLockedOutput) {
    std::ifstream file(GENIE_SANDHI_GOLDEN_FILE);
    ASSERT_TRUE(file.is_open()) << GENIE_SANDHI_GOLDEN_FILE;
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    const auto root = json::parse(content);

    std::size_t checked = 0;
    for (const auto& case_value : root.as_object().at("cases").as_array()) {
        const auto& obj = case_value.as_object();
        const auto& text = obj.at("text").as_string();
        const auto units = pipeline(chars_of(text));
        ASSERT_EQ(units.size(), obj.at("units").as_array().size()) << text;
        for (std::size_t i = 0; i < units.size(); ++i) {
            const auto& expected = obj.at("units").as_array()[i].as_object();
            EXPECT_EQ(units[i].word(), expected.at("word").as_string()) << text;
            ASSERT_EQ(units[i].finals.size(), expected.at("finals").as_array().size()) << text;
            for (std::size_t k = 0; k < units[i].finals.size(); ++k) {
                EXPECT_EQ(units[i].finals[k], expected.at("finals").as_array()[k].as_string())
                    << text << " unit " << i << " final " << k;
            }
        }
        ++checked;
    }
    EXPECT_GT(checked, 30U) << "黄金语料不应缩水";
}

}  // namespace
}  // namespace genie::text
