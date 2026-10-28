// tests/unit/test_english.cpp —— 英文 G2P 的单元测试。
//
// 覆盖三层：
//   1. 规范化：上游 English/Normalization.py 的 20 多条规则
//   2. 分词：nltk TweetTokenizer 的移植
//   3. 音素：CMU 词典查表 + 同形异音词 + GRU 兜底
//
// 黄金语料由 tools/g2p/verify_english_g2p.py 用**上游的 Python 实现**
// 当 oracle 生成（85 条，逐音素比对）。
//
// 数据容器（third_party/english/）随仓库发布：
//   cmudict.bin       —— CMU 发音词典（上游 engdict_cache.pickle + 热词表）
//   pos_tagger.bin    —— nltk averaged perceptron 词性标注
//   english_g2p.bin   —— g2p_en 的 seq2seq GRU 权重
// ngram 分词词典体积大，不随仓库发布 —— 缺失时 OOV 词直接交给 GRU，
// 与上游的行为有差异（见 docs/upstream-audit.md §17）。

#include <gtest/gtest.h>

#include <fstream>
#include <string>
#include <vector>

#include "genie/errors.hpp"
#include "genie/json.hpp"
#include "genie/symbols.hpp"
#include "genie/text.hpp"
#include "text/english/english.hpp"

namespace {

using genie::text::english::CmuDict;
using genie::text::english::PosTagger;
using genie::text::english::Seq2SeqG2p;
using genie::text::english::WordSegmenter;

CmuDict load_dict() {
    return CmuDict::load(GENIE_ENGLISH_CMU_DICT);
}

PosTagger load_tagger() {
    return PosTagger::load(GENIE_ENGLISH_POS_TAGGER);
}

Seq2SeqG2p load_seq2seq() {
    Seq2SeqG2p model;
    const bool ok = model.load(GENIE_ENGLISH_G2P_MODEL);
    EXPECT_TRUE(ok) << "english_g2p.bin 应该能加载";
    return model;
}

std::vector<std::string> phonemes_of(const std::string& text, const CmuDict& dict,
                                     const Seq2SeqG2p& seq2seq, const WordSegmenter& segmenter,
                                     const PosTagger* tagger) {
    return genie::text::english::phonemes(text, dict, seq2seq, segmenter, tagger);
}

}  // namespace

TEST(English, NormalizeFollowsUpstreamRules) {
    // 上游 Normalization.normalize 的几条代表性规则。
    EXPECT_EQ(genie::text::english::normalize("In 2024, the price is 50% up."),
              "in twenty twenty four, the price is fifty percent up.");
    EXPECT_EQ(genie::text::english::normalize("Call me at 555-123-4567."),
              "call me at five five five, one two three, four five six seven.");
    EXPECT_EQ(genie::text::english::normalize("Version 1.2.3 was released."),
              "version one point two dot three was released.");
    EXPECT_EQ(genie::text::english::normalize("The meeting is at 3:30 pm."),
              "the meeting is at three thirty pm.");
    EXPECT_EQ(genie::text::english::normalize("We need 5cm of wire."),
              "we need five centimeters of wire.");
    EXPECT_EQ(genie::text::english::normalize("Half of it is 1/2."), "half of it is one half.");
    EXPECT_EQ(genie::text::english::normalize("That's a complex presents problem."),
              "that's a complex presents problem.");

    // text_normalize 比 normalize 多两步：REP_MAP + 连续标点折叠。
    EXPECT_EQ(genie::text::english::text_normalize("Well... I don't know."), "well. i don't know.");
    EXPECT_EQ(genie::text::english::text_normalize("Go！ Now？"), "go! now?");
}

TEST(English, TokenizeFollowsTweetTokenizer) {
    const auto tokens = genie::text::english::tokenize("Don't stop, he's here!");
    std::vector<std::string> texts;
    for (const auto& token : tokens) {
        texts.push_back(token.text);
    }
    const std::vector<std::string> expected = {"Don't", "stop", ",", "he's", "here", "!"};
    EXPECT_EQ(texts, expected);
}

TEST(English, CmuDictLookup) {
    const auto dict = load_dict();
    const auto hello = dict.lookup("hello");
    const std::vector<std::string> expected = {"HH", "AH0", "L", "OW1"};
    EXPECT_EQ(hello, expected);
    EXPECT_TRUE(dict.lookup("zzzznotaword").empty());
}

TEST(English, PhonemesMatchUpstream) {
    const auto dict = load_dict();
    const auto tagger = load_tagger();
    const auto seq2seq = load_seq2seq();
    const WordSegmenter segmenter;

    const auto phonemes =
        phonemes_of("Hello world, this is a test.", dict, seq2seq, segmenter, &tagger);
    const std::vector<std::string> expected = {"HH",  "AH0", "L",   "OW1", "W", "ER1", "L",
                                               "D",   ",",   "DH",  "IH1", "S", "IH1", "Z",
                                               "AH0", "T",   "EH1", "S",   "T", "."};
    EXPECT_EQ(phonemes, expected);
}

TEST(English, HomographDependsOnPartOfSpeech) {
    const auto dict = load_dict();
    const auto tagger = load_tagger();
    const auto seq2seq = load_seq2seq();
    const WordSegmenter segmenter;

    // "read" 现在时（VB）→ R IY1 D；过去时（VBD）→ R EH1 D。
    const auto present = phonemes_of("I will read the book.", dict, seq2seq, segmenter, &tagger);
    EXPECT_NE(std::find(present.begin(), present.end(), "IY1"), present.end());
    const auto past =
        phonemes_of("She read the book yesterday.", dict, seq2seq, segmenter, &tagger);
    EXPECT_NE(std::find(past.begin(), past.end(), "EH1"), past.end());
}

TEST(English, GoldenCorpus) {
    // 黄金语料由 tools/g2p/verify_english_g2p.py 用上游 Python 生成。
    std::ifstream file(GENIE_ENGLISH_GOLDEN_FILE);
    ASSERT_TRUE(file.good()) << "找不到黄金语料：" << GENIE_ENGLISH_GOLDEN_FILE;

    // 用项目自带的 JSON 解析器读黄金语料（键序不敏感，不用手写扫描）。
    struct Case {
        std::string text;
        std::string normalized;
        std::string text_normalized;
        std::vector<std::string> phonemes;
        /// 依赖 ngram 分词词典的用例：词典不随仓库发布，单测只比规范化。
        bool needs_word_segment = false;
    };
    std::vector<Case> cases;
    {
        const genie::json::Value root = genie::json::parse_file(GENIE_ENGLISH_GOLDEN_FILE);
        ASSERT_TRUE(root.is_object());
        const genie::json::Value* list = root.find("cases");
        ASSERT_NE(list, nullptr);
        ASSERT_TRUE(list->is_array());
        for (const auto& item : list->as_array()) {
            ASSERT_TRUE(item.is_object());
            Case entry;
            entry.text = item.find("text")->as_string();
            entry.normalized = item.find("normalized")->as_string();
            entry.text_normalized = item.find("text_normalized")->as_string();
            for (const auto& phoneme : item.find("phonemes")->as_array()) {
                entry.phonemes.push_back(phoneme.as_string());
            }
            const genie::json::Value* flag = item.find("needs_word_segment");
            entry.needs_word_segment = flag != nullptr && flag->as_bool(false);
            cases.push_back(std::move(entry));
        }
    }
    ASSERT_FALSE(cases.empty());

    const auto dict = load_dict();
    const auto tagger = load_tagger();
    const auto seq2seq = load_seq2seq();
    const WordSegmenter segmenter;

    int compared = 0;
    for (const auto& entry : cases) {
        EXPECT_EQ(genie::text::english::normalize(entry.text), entry.normalized)
            << "规范化不一致：" << entry.text;
        EXPECT_EQ(genie::text::english::text_normalize(entry.text), entry.text_normalized)
            << "text_normalize 不一致：" << entry.text;
        if (entry.needs_word_segment) {
            continue;
        }
        EXPECT_EQ(phonemes_of(entry.text, dict, seq2seq, segmenter, &tagger), entry.phonemes)
            << "音素不一致：" << entry.text;
        ++compared;
    }
    // 至少要比掉大多数用例 —— 免得"全被跳过"也绿。
    EXPECT_GE(compared, 70);
}

TEST(English, FrontendProducesSymbolIds) {
    genie::text::EnglishG2pResources resources;
    resources.cmu_dict = GENIE_ENGLISH_CMU_DICT;
    resources.pos_tagger = GENIE_ENGLISH_POS_TAGGER;
    resources.g2p_model = GENIE_ENGLISH_G2P_MODEL;

    const auto frontend = genie::text::make_frontend(genie::Language::English, resources);
    ASSERT_NE(frontend, nullptr);
    const auto sequence = frontend->convert("Hello world.");
    ASSERT_FALSE(sequence.ids.empty());
    EXPECT_EQ(sequence.ids.size(), sequence.symbols.size());
    for (std::size_t i = 0; i < sequence.ids.size(); ++i) {
        EXPECT_EQ(genie::symbols::id_of(sequence.symbols[i]), sequence.ids[i]);
    }
    // 上游的英文路径不产出 RoBERTa 特征。
    EXPECT_TRUE(sequence.bert_chunks.empty());
}

TEST(English, FrontendNeedsDictionary) {
    genie::text::EnglishG2pResources resources;
    EXPECT_EQ(genie::text::make_frontend(genie::Language::English, resources), nullptr);
}
