// tests/unit/test_roberta.cpp
//
// RoBERTa 韵律特征的两块基石：
//   1. 分词器 —— 与 HuggingFace tokenizers 逐 ID 对齐（黄金数据）；
//   2. word2ph 展开 —— 复刻上游 _expand_roberta_output 的三种布局。

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "engine/roberta.hpp"
#include "genie/errors.hpp"
#include "genie/json.hpp"
#include "genie/reference.hpp"
#include "support/fake_session.hpp"
#include "text/roberta_tokenizer.hpp"

using namespace genie;
using namespace genie::test;

namespace {

/// 黄金数据与词表都在 tests/golden/ 下；缺失时跳过（不许依赖网络）。
struct GoldenPaths {
    std::filesystem::path vocab;
    std::filesystem::path golden;
};

[[nodiscard]] std::optional<GoldenPaths> find_golden() {
    std::filesystem::path base;
#ifdef GENIE_TEST_SOURCE_DIR_DEFAULT
    base = std::filesystem::path(GENIE_TEST_SOURCE_DIR_DEFAULT) / "tests" / "golden";
#else
    base = std::filesystem::path("tests") / "golden";
#endif
    GoldenPaths paths{base / "vocab_roberta.txt", base / "roberta_tokenizer.json"};
    if (!std::filesystem::exists(paths.vocab) || !std::filesystem::exists(paths.golden)) {
        return std::nullopt;
    }
    return paths;
}

/// 展开测试里当占位行宽用的 bert 维度（真实值来自 reference.hpp）。
constexpr std::int64_t kTestBertDim = kBertDimension;

}  // namespace

// ---------------------------------------------------------------- 分词器

TEST(RobertaTokenizer, MatchesHuggingFaceGolden) {
    const auto paths = find_golden();
    if (!paths.has_value()) {
        GTEST_SKIP() << "golden files not present under tests/golden/";
    }
    const auto tokenizer = text::RobertaTokenizer::load(paths->vocab.string());
    const json::Value document = json::parse_file(paths->golden.string());
    ASSERT_TRUE(document.is_object());

    for (const auto& [text, expected] : document.as_object()) {
        const json::Value* ids = expected.find("ids");
        ASSERT_NE(ids, nullptr);
        ASSERT_TRUE(ids->is_array());
        const auto encoding = tokenizer->encode(text);
        ASSERT_EQ(encoding.ids.size(), ids->as_array().size()) << "text: " << text;
        for (std::size_t i = 0; i < ids->as_array().size(); ++i) {
            EXPECT_EQ(encoding.ids[i], ids->as_array()[i].as_int())
                << "text: " << text << " token #" << i;
        }
    }
}

TEST(RobertaTokenizer, EncodesChineseCharByCharWithWrapping) {
    const auto paths = find_golden();
    if (!paths.has_value()) {
        GTEST_SKIP() << "golden files not present under tests/golden/";
    }
    const auto tokenizer = text::RobertaTokenizer::load(paths->vocab.string());

    const auto encoding = tokenizer->encode("你好,世界.");
    ASSERT_GE(encoding.ids.size(), 8U);
    // [CLS] ... [SEP] 包裹；中文逐字、标点独立。
    EXPECT_EQ(encoding.ids.front(), tokenizer->cls_id());
    EXPECT_EQ(encoding.ids.back(), tokenizer->sep_id());
    EXPECT_EQ(encoding.tokens[1], "你");
    EXPECT_EQ(encoding.tokens[2], "好");
    EXPECT_EQ(encoding.tokens[3], ",");
    EXPECT_EQ(encoding.tokens[6], ".");
    EXPECT_EQ(encoding.tokens[7], "[SEP]");
    // 注意力掩码全 1（无填充）。
    for (const auto mask : encoding.attention_mask) {
        EXPECT_EQ(mask, 1);
    }
}

TEST(RobertaTokenizer, LowercasesAndStripsAccents) {
    const auto paths = find_golden();
    if (!paths.has_value()) {
        GTEST_SKIP() << "golden files not present under tests/golden/";
    }
    const auto tokenizer = text::RobertaTokenizer::load(paths->vocab.string());

    const auto encoding = tokenizer->encode("Caf\xC3\xA9");  // Café
    // 全部小写、变音符剥离（本次实现的覆盖范围内）。
    for (const auto& token : encoding.tokens) {
        EXPECT_NE(token, "C");
        EXPECT_NE(token, "\xC3\xA9");
    }
}

TEST(RobertaTokenizer, UnknownCharBecomesUnk) {
    const auto paths = find_golden();
    if (!paths.has_value()) {
        GTEST_SKIP() << "golden files not present under tests/golden/";
    }
    const auto tokenizer = text::RobertaTokenizer::load(paths->vocab.string());

    // 私有区字符一定不在词表里。
    const auto encoding = tokenizer->encode("\xEE\x80\x80");  // U+E000
    ASSERT_EQ(encoding.ids.size(), 3U);
    EXPECT_EQ(encoding.tokens[1], "[UNK]");
    EXPECT_EQ(encoding.ids[1], tokenizer->unk_id());
}

// ---------------------------------------------------------------- word2ph 展开

TEST(RobertaExpand, PhoneLevelOutputPassesThrough) {
    // 布局 1：T == 音素数，直接用。
    std::vector<float> token_bert;
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::int64_t col = 0; col < kTestBertDim; ++col) {
            token_bert.push_back(static_cast<float>(row) + 0.5F);
        }
    }
    const auto expanded = roberta::expand_by_word2ph(token_bert, {2, 1}, 3);
    ASSERT_EQ(expanded.rank(), 2);
    EXPECT_EQ(expanded.dim(0), 3);
    EXPECT_EQ(expanded.dim(1), kTestBertDim);
    EXPECT_FLOAT_EQ(expanded.floats()[0], 0.5F);
    EXPECT_FLOAT_EQ(expanded.floats()[1024], 1.5F);
}

TEST(RobertaExpand, TokenLevelWithSpecialTokensExpandsByWord2ph) {
    // 布局 2：T == len(word2ph) + 2（[CLS]/[SEP] 包裹）。
    // 行内容 = 行号，展开后能逐行核对。
    std::vector<float> token_bert;
    const std::size_t rows = 4;  // CLS + 2 tokens + SEP
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::int64_t col = 0; col < kTestBertDim; ++col) {
            token_bert.push_back(static_cast<float>(row));
        }
    }
    const auto expanded = roberta::expand_by_word2ph(token_bert, {1, 2}, 3);
    ASSERT_EQ(expanded.dim(0), 3);
    const auto& data = expanded.floats();
    // token1 重复 1 次，token2 重复 2 次；[CLS]/[SEP] 被丢弃。
    EXPECT_FLOAT_EQ(data[0], 1.0F);
    EXPECT_FLOAT_EQ(data[1024], 2.0F);
    EXPECT_FLOAT_EQ(data[2 * 1024], 2.0F);
}

TEST(RobertaExpand, TokenLevelWithoutSpecialTokensExpandsByWord2ph) {
    // 布局 3：T == len(word2ph)。
    std::vector<float> token_bert(2 * static_cast<std::size_t>(kTestBertDim), 0.0F);
    const auto expanded = roberta::expand_by_word2ph(token_bert, {1, 1}, 2);
    EXPECT_EQ(expanded.dim(0), 2);
}

TEST(RobertaExpand, RejectsUnknownLayout) {
    std::vector<float> token_bert(7 * static_cast<std::size_t>(kTestBertDim), 0.0F);
    EXPECT_THROW(roberta::expand_by_word2ph(token_bert, {1, 1}, 2), Error);
}

TEST(RobertaExpand, RejectsWord2phPhoneMismatch) {
    std::vector<float> token_bert(2 * static_cast<std::size_t>(kTestBertDim), 0.0F);
    // 布局 3（T == len(word2ph)）下，word2ph 合计 4 != 音素数 5。
    // 注意布局 1（T == 音素数）不查 word2ph —— 上游直接放行。
    EXPECT_THROW(roberta::expand_by_word2ph(token_bert, {2, 2}, 5), Error);
}

// ---------------------------------------------------------------- 会话装配

TEST(RobertaCompute, FeedsNamedInputsAndExpands) {
    const auto paths = find_golden();
    if (!paths.has_value()) {
        GTEST_SKIP() << "golden files not present under tests/golden/";
    }
    const auto tokenizer = text::RobertaTokenizer::load(paths->vocab.string());
    auto session = FakeSession::roberta();

    // "你好." → [CLS] 你 好 . [SEP]，5 行 token 级特征。
    // word2ph 给 {2,2,2}（合计 6 == 音素数 6）→ 命中布局 2（去 CLS/SEP
    // 后按 word2ph 重复），这样能验证"哪一行被重复了多少次"。
    const auto bert = roberta::compute_text_bert(*tokenizer, *session, "你好.", {2, 2, 2}, 6);

    // 假会话第 j 行 = token id；展开后：你×2、好×2、.×2。
    ASSERT_EQ(bert.rank(), 2);
    EXPECT_EQ(bert.dim(0), 6);
    EXPECT_EQ(bert.dim(1), kTestBertDim);
    const auto you_id = static_cast<float>(tokenizer->encode("你").ids[1]);
    const auto& data = bert.floats();
    EXPECT_FLOAT_EQ(data[0], you_id);
    EXPECT_FLOAT_EQ(data[1024], you_id);

    // 三个输入都按名字喂了，形状 [1, T]。
    ASSERT_EQ(session->last_feed().size(), 3U);
    for (const auto& [name, tensor] : session->last_feed()) {
        EXPECT_EQ(tensor.rank(), 2) << name;
        EXPECT_EQ(tensor.dim(0), 1) << name;
        EXPECT_EQ(tensor.dim(1), 5) << name;
    }
}

TEST(RobertaCompute, RejectsLayoutMismatchLoudly) {
    const auto paths = find_golden();
    if (!paths.has_value()) {
        GTEST_SKIP() << "golden files not present under tests/golden/";
    }
    const auto tokenizer = text::RobertaTokenizer::load(paths->vocab.string());
    auto session = FakeSession::roberta();

    // 编码出 5 行（你好. + CLS/SEP），却声称音素数是 6 —— 上游这一步
    // 是 ValueError，这里必须是 Error，不能安静输出垃圾。
    EXPECT_THROW((roberta::compute_text_bert(*tokenizer, *session, "你好.", {2, 2, 1}, 6)), Error);
}
