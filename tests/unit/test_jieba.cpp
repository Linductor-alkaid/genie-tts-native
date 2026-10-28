// tests/unit/test_jieba.cpp
//
// jieba 分词移植的测试：容器加载、与上游 jieba Python 逐词一致
// （黄金语料，tools/g2p/verify_jieba.py 生成）、前端分词路径。
//
// 词典容器 third_party/jieba/jieba.bin 随仓库发布（8.3 MB，MIT），
// 黄金语料比对在 CI 也能跑。

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "genie/errors.hpp"
#include "genie/json.hpp"
#include "text/jieba.hpp"

#ifndef GENIE_JIEBA_MODEL_FILE
#define GENIE_JIEBA_MODEL_FILE "third_party/jieba/jieba.bin"
#endif

#ifndef GENIE_JIEBA_GOLDEN_FILE
#define GENIE_JIEBA_GOLDEN_FILE "tests/golden/jieba.json"
#endif

namespace genie::text {
namespace {

TEST(Jieba, LoadThrowsOnBadMagic) {
    const char* path = "/tmp/genie_jieba_bad.bin";
    std::ofstream file(path, std::ios::binary);
    file << "NOPE" << "whatever";
    file.close();
    EXPECT_THROW(jieba::Tokenizer::load(path), Error);
    std::remove(path);
}

TEST(Jieba, RouteAndHmmSplit) {
    const auto tokenizer = jieba::Tokenizer::load(GENIE_JIEBA_MODEL_FILE);
    // DP 路径 + OOV 的 HMM 细分（「敏而好学」「不耻下问」都不在词典，
    // HMM 把四字格各标成 BMMM/E，产出整词）。
    const auto words = tokenizer.cut("化学研究敏而好学");
    std::vector<std::string> texts;
    for (const auto& word : words) {
        texts.push_back(word.text);
    }
    EXPECT_EQ(texts, (std::vector<std::string>{"化学", "研究", "敏而好学"}));
}

TEST(JiebaGolden, CorpusMatchesLockedOutput) {
    std::ifstream file(GENIE_JIEBA_GOLDEN_FILE);
    ASSERT_TRUE(file.is_open()) << GENIE_JIEBA_GOLDEN_FILE;
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    const auto root = json::parse(content);

    const auto tokenizer = jieba::Tokenizer::load(GENIE_JIEBA_MODEL_FILE);
    std::size_t checked = 0;
    for (const auto& case_value : root.as_object().at("cases").as_array()) {
        const auto& obj = case_value.as_object();
        const auto& text = obj.at("text").as_string();
        const auto got = tokenizer.cut(text);
        const auto& expected = obj.at("words").as_array();
        ASSERT_EQ(got.size(), expected.size()) << text;
        for (std::size_t i = 0; i < got.size(); ++i) {
            EXPECT_EQ(got[i].text, expected[i].as_string()) << text << " word " << i;
        }
        ++checked;
    }
    EXPECT_GT(checked, 90U) << "黄金语料不应缩水";
}

}  // namespace
}  // namespace genie::text
