// tests/unit/test_g2pm.cpp
//
// g2pM 多音字消歧的测试：容器加载、模型输出与上游 Python 逐字一致
// （黄金语料，tools/g2p/verify_g2pm.py 生成）、中文前端的接线与降级。
//
// 模型容器 third_party/g2pm/g2pm.bin 随仓库发布（1.8 MB，Apache-2.0），
// 因此黄金语料比对在 CI 也能跑 —— 与六大模型"图壳"同先例。

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "engine/g2pm.hpp"
#include "genie/errors.hpp"
#include "genie/json.hpp"
#include "genie/text.hpp"
#include "text/frontend_factory_internal.hpp"
#include "text/jieba.hpp"

#ifndef GENIE_G2PM_MODEL_FILE
#define GENIE_G2PM_MODEL_FILE "third_party/g2pm/g2pm.bin"
#endif

#ifndef GENIE_G2PM_GOLDEN_FILE
#define GENIE_G2PM_GOLDEN_FILE "tests/golden/g2pm.json"
#endif

#ifndef GENIE_JIEBA_MODEL_FILE
#define GENIE_JIEBA_MODEL_FILE "third_party/jieba/jieba.bin"
#endif

namespace genie::text {
namespace {

std::string symbols_of(const TextFrontend& frontend, const std::string& text) {
    const auto sequence = frontend.convert(text);
    std::string joined;
    for (const auto& symbol : sequence.symbols) {
        if (!joined.empty()) {
            joined += ' ';
        }
        joined += symbol;
    }
    return joined;
}

/// 统计音素符号串里某个连续片段（"h ang2"）出现的次数。
std::size_t count_subsequence(const std::string& symbols, const std::string& piece) {
    std::size_t count = 0;
    for (auto pos = symbols.find(piece); pos != std::string::npos;
         pos = symbols.find(piece, pos + 1)) {
        ++count;
    }
    return count;
}

TEST(G2pm, LoadThrowsOnBadMagic) {
    const char* path = "/tmp/genie_g2pm_bad.bin";
    std::ofstream file(path, std::ios::binary);
    file << "NOPE" << "whatever";
    file.close();
    EXPECT_THROW(g2pm::Model::load(path), Error);
    std::remove(path);
}

TEST(G2pm, PassthroughCharsReturnedAsIs) {
    const auto model = g2pm::Model::load(GENIE_G2PM_MODEL_FILE);
    const auto pinyins = model.pinyins_of("重庆，你好");
    ASSERT_EQ(pinyins.size(), std::size_t{5});
    // 标点不在 cedict 里：原样透传（上游同款行为）。
    EXPECT_EQ(pinyins[2], "，");
    EXPECT_EQ(pinyins[0], "chong2");
    EXPECT_EQ(pinyins[1], "qing4");
}

TEST(G2pm, DisambiguatesByContext) {
    const auto model = g2pm::Model::load(GENIE_G2PM_MODEL_FILE);
    // 同一个「行」：银行分行 —— 前行读 háng、后行读 xíng。
    // 注意这是**模型层**的判定（未经词级读音修正）：模型把「分行」判成
    // xíng、「行长在银行上班」里把「银行」判成 xíng —— 上游同款行为，
    // 黄金语料逐字锁定，见 docs/upstream-audit.md §9.4。
    // 修正表会把这些改回来，所以**最终输出**两个「行」都读 háng
    //（见 G2pmFrontend.UsesContextReadings）。
    const auto pinyins = model.pinyins_of("银行分行下午三点关门");
    ASSERT_EQ(pinyins.size(), std::size_t{10});
    EXPECT_EQ(pinyins[1], "hang2");  // 银行
    EXPECT_EQ(pinyins[3], "xing2");  // 分行（模型层判定，修正表会改成 hang2）
}

TEST(G2pmGolden, CorpusMatchesLockedOutput) {
    std::ifstream file(GENIE_G2PM_GOLDEN_FILE);
    ASSERT_TRUE(file.is_open()) << GENIE_G2PM_GOLDEN_FILE;
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    const auto root = json::parse(content);

    const auto model = g2pm::Model::load(GENIE_G2PM_MODEL_FILE);
    std::size_t checked = 0;
    for (const auto& case_value : root.as_object().at("cases").as_array()) {
        const auto& obj = case_value.as_object();
        const auto& text = obj.at("text").as_string();
        const auto got = model.pinyins_of(text);
        const auto& expected = obj.at("pinyins").as_array();
        ASSERT_EQ(got.size(), expected.size()) << text;
        for (std::size_t i = 0; i < got.size(); ++i) {
            EXPECT_EQ(got[i], expected[i].as_string()) << text << " char " << i;
        }
        ++checked;
    }
    EXPECT_GT(checked, 40U) << "黄金语料不应缩水";
}

TEST(G2pmFrontend, UsesContextReadings) {
    const auto model = g2pm::Model::load(GENIE_G2PM_MODEL_FILE);
    const auto jieba = jieba::Tokenizer::load(GENIE_JIEBA_MODEL_FILE);
    const auto frontend = make_chinese_frontend(&model, &jieba);

    // 单字表把两个「行」都读成最常用的 xíng；g2pM + 词级修正后都读 háng
    //（模型把「分行」判成 xíng，词表兜底 —— 上游同款两层结构）。
    const std::string sentence = "银行分行下午三点关门。";
    EXPECT_NE(symbols_of(*frontend, sentence), symbols_of(*make_chinese_frontend(), sentence));
    EXPECT_EQ(count_subsequence(symbols_of(*frontend, sentence), "h ang2"), std::size_t{2});
}

TEST(G2pmFrontend, CorrectsModelMisjudgments) {
    const auto model = g2pm::Model::load(GENIE_G2PM_MODEL_FILE);
    const auto jieba = jieba::Tokenizer::load(GENIE_JIEBA_MODEL_FILE);
    const auto frontend = make_chinese_frontend(&model, &jieba);
    // g2pM 的已知误判由词级修正（polyphonic.pickle）兜底：
    //   长大 → zhǎngdà（模型给 cháng，单字表反而是对的）。
    // 原始模型输出保留在 Model::pinyins_of（见 G2pm.DisambiguatesByContext），
    // 修正在前端层 —— 与上游 ChineseG2P 的流水线位置一致。
    // （单字表对这句恰好也读 zhǎng —— 两路符号相同；
    //   差异要用原始模型输出对比才可见，见 G2pm.DisambiguatesByContext。）
    const std::string sentence = "他打算长大后当一名老师";
    EXPECT_NE(symbols_of(*frontend, sentence).find("zh ang3 d a4"), std::string::npos);
}

TEST(G2pmFrontend, DegradedWithoutModelKeepsLegacyBehavior) {
    const auto model = g2pm::Model::load(GENIE_G2PM_MODEL_FILE);
    const auto with_model = make_chinese_frontend(&model);
    const auto without_model = make_chinese_frontend();
    const auto via_public_factory = make_frontend(Language::Chinese);

    // 不带模型时与公开工厂（查单字表）完全一致 —— 降级路径。
    EXPECT_EQ(symbols_of(*without_model, "重庆很重要，行长在银行上班"),
              symbols_of(*via_public_factory, "重庆很重要，行长在银行上班"));
    // 带模型时多音字读音可能不同（但流水线其余部分不变）。
    EXPECT_NE(symbols_of(*with_model, "重庆很重要，行长在银行上班"),
              symbols_of(*via_public_factory, "重庆很重要，行长在银行上班"));
}

TEST(G2pmFrontend, ConvertsUColonToV) {
    const auto model = g2pm::Model::load(GENIE_G2PM_MODEL_FILE);
    const auto frontend = make_chinese_frontend(&model);
    // cedict 的 ü 记法是 "lu:"；上游 ChineseG2P 在消费侧替换为 "v"。
    // 「效率」的率 lu:4 → lv4 → 韵母 v4。
    EXPECT_NE(symbols_of(*frontend, "效率").find("v4"), std::string::npos);
}

}  // namespace
}  // namespace genie::text
