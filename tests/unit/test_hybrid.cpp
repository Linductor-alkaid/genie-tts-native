// tests/unit/test_hybrid.cpp
//
// 中英混读分块的测试：split_language 移植、前端分块路径、
// BERT 分块与 word2ph 的对齐，以及与上游 Python 逐条比对的黄金语料。
//
// 黄金语料由 tools/verify_language_split.py 生成（上游 split_language 的
// 正则逻辑为参照），锁定在 tests/golden/language_split.json。

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include "genie/json.hpp"
#include "genie/text.hpp"
#include "text/frontend_factory_internal.hpp"
#include "text/language_split.hpp"

#ifndef GENIE_HYBRID_GOLDEN_FILE
#define GENIE_HYBRID_GOLDEN_FILE "tests/golden/language_split.json"
#endif

namespace genie::text {
namespace {

/// 端到端：文本 → 音素符号（空格分隔，便于断言）。
std::string symbols_of(const std::string& text, Language language = Language::Chinese) {
    const auto frontend = make_frontend(language);
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

/// 混读前端的音素符号 + 分块摘要，用于断言。
struct HybridResult {
    std::string symbols;
    std::vector<BertChunk> chunks;
    std::size_t word2ph_size = 0;
};

HybridResult convert_hybrid(const std::string& text) {
    const auto frontend = make_frontend(Language::HybridChineseEnglish);
    const auto sequence = frontend->convert(text);
    HybridResult result;
    for (const auto& symbol : sequence.symbols) {
        if (!result.symbols.empty()) {
            result.symbols += ' ';
        }
        result.symbols += symbol;
    }
    result.chunks = sequence.bert_chunks;
    result.word2ph_size = sequence.word2ph.size();
    return result;
}

/// 分块的对齐不变量：各块的 [offset, offset+count) 必须连续覆盖 word2ph。
/// 纯中文走单块路径（chunks 为空、整段一次 RoBERTa），不在本不变量范围内。
void expect_chunks_partition(const HybridResult& result, const std::string& context) {
    if (result.chunks.empty()) {
        return;
    }
    std::size_t cursor = 0;
    for (const auto& chunk : result.chunks) {
        EXPECT_EQ(chunk.word_offset, cursor) << context;
        EXPECT_GT(chunk.word_count, std::size_t{0}) << context;
        EXPECT_FALSE(chunk.text.empty()) << context;
        cursor += chunk.word_count;
    }
    EXPECT_EQ(cursor, result.word2ph_size) << context;
}

// ---------------------------------------------------------------------------
// split_language 移植
// ---------------------------------------------------------------------------

TEST(LanguageSplit, PureChineseIsSingleChunk) {
    const auto chunks = lang_split::split_by_language("你好，世界。");
    ASSERT_EQ(chunks.size(), std::size_t{1});
    EXPECT_EQ(chunks[0].language, lang_split::Chunk::Language::Chinese);
    EXPECT_EQ(chunks[0].content, "你好，世界。");
}

TEST(LanguageSplit, EnglishRunSplitsChinese) {
    // 上游 re.split(r'[a-zA-Z]+') 的直接对应：字母串两侧各留一个中文块。
    const auto chunks = lang_split::split_by_language("你好world！");
    ASSERT_EQ(chunks.size(), std::size_t{3});
    EXPECT_EQ(chunks[0].language, lang_split::Chunk::Language::Chinese);
    EXPECT_EQ(chunks[0].content, "你好");
    EXPECT_EQ(chunks[1].language, lang_split::Chunk::Language::English);
    EXPECT_EQ(chunks[1].content, "world");
    EXPECT_EQ(chunks[2].language, lang_split::Chunk::Language::Chinese);
    EXPECT_EQ(chunks[2].content, "！");
}

TEST(LanguageSplit, LeadingEnglishDropsBlankHead) {
    // 开头的空白中文块被 strip 丢弃（上游 `if part.strip()`）。
    const auto chunks = lang_split::split_by_language(" hello");
    ASSERT_EQ(chunks.size(), std::size_t{1});
    EXPECT_EQ(chunks[0].language, lang_split::Chunk::Language::English);
    EXPECT_EQ(chunks[0].content, "hello");
}

TEST(LanguageSplit, SpacesBetweenEnglishWordsAreDropped) {
    // "hello world" 的空格属于中文块，strip 后为空 → 丢弃；
    // 两个单词成为两个独立的英文块（上游同款行为）。
    const auto chunks = lang_split::split_by_language("hello world");
    ASSERT_EQ(chunks.size(), std::size_t{2});
    EXPECT_EQ(chunks[0].content, "hello");
    EXPECT_EQ(chunks[1].content, "world");
    EXPECT_EQ(chunks[0].language, lang_split::Chunk::Language::English);
    EXPECT_EQ(chunks[1].language, lang_split::Chunk::Language::English);
}

TEST(LanguageSplit, SpacesStayWithChineseNeighbors) {
    // 非空白中文块保留自己的空白（交给各自的 TextNormalizer 过滤）。
    const auto chunks = lang_split::split_by_language("你好 hello 世界");
    ASSERT_EQ(chunks.size(), std::size_t{3});
    EXPECT_EQ(chunks[0].content, "你好 ");
    EXPECT_EQ(chunks[1].content, "hello");
    EXPECT_EQ(chunks[2].content, " 世界");
}

TEST(LanguageSplit, DigitsStayInChineseChunks) {
    // 数字不是字母：跟着所在的中文块走，由 TextNormalizer 处理。
    const auto chunks = lang_split::split_by_language("iPhone15很好");
    ASSERT_EQ(chunks.size(), std::size_t{2});
    EXPECT_EQ(chunks[0].content, "iPhone");
    EXPECT_EQ(chunks[1].content, "15很好");
    EXPECT_EQ(chunks[1].language, lang_split::Chunk::Language::Chinese);
}

TEST(LanguageSplit, EmptyTextGivesNoChunks) {
    EXPECT_TRUE(lang_split::split_by_language("").empty());
    EXPECT_TRUE(lang_split::split_by_language("   ").empty());
}

TEST(LanguageSplit, HasAsciiLetterMatchesUpstreamGate) {
    EXPECT_TRUE(lang_split::has_ascii_letter("你好Mira"));
    EXPECT_TRUE(lang_split::has_ascii_letter("WiFi"));
    EXPECT_FALSE(lang_split::has_ascii_letter("你好，世界。"));
    EXPECT_FALSE(lang_split::has_ascii_letter("2024年3月1号"));
    EXPECT_FALSE(lang_split::has_ascii_letter("！？…，。"));
}

// ---------------------------------------------------------------------------
// 前端混读路径
// ---------------------------------------------------------------------------

TEST(HybridFrontend, EnglishDroppedChineseKept) {
    // 裸前端（无英文资源）的降级路径：英文块整体丢弃 —— 与"音色包缺
    // 英文词典"的生产降级一致；中文块的音素必须原样保留。
    // （挂了英文前端的行为见 EnglishChunkEntersBertChunksAsZero。）
    //
    // 注意比对方式：块边界同时也是**变调的边界**（上游逐块调
    // chinese_to_phones，块间不做三声连读）—— 所以这里必须与
    // "逐块单独转换再拼接"比对，而不是与"删掉英文后的整段"比对：
    //   "你好Mira很高兴见到你" → 块 [你好][很高兴见到你]，你→2（块内闭环）
    //   整段 "你好很高兴见到你" → 一个汉字段，变调跨词段作用，好→2
    const auto hybrid = convert_hybrid("你好Mira很高兴见到你");
    const std::string expected = symbols_of("你好") + " " + symbols_of("很高兴见到你");
    EXPECT_EQ(hybrid.symbols, expected);
    EXPECT_EQ(hybrid.chunks.size(), std::size_t{2});
    expect_chunks_partition(hybrid, "你好Mira很高兴见到你");
}

TEST(HybridFrontend, SameAsPureWhenEnglishRemoved) {
    // 混读句 == 手工去掉英文后按纯中文合成的音素（变调不跨标点，
    // 块边界落在字母串处，不影响汉字段的变调结果）。
    const auto hybrid = convert_hybrid("你好，Mira。很高兴见到你");
    EXPECT_EQ(hybrid.symbols, symbols_of("你好，。很高兴见到你"));
}

TEST(HybridFrontend, PerChunkFullNormalization) {
    // 判别用例：电话号码在完整 TextNormalizer 里读「幺三八…」；
    // 旧的整段简化规则会读成数量（一百三十八亿…）。
    // 混读路径必须与"手工去掉英文后的纯中文路径"一致。
    const auto hybrid = convert_hybrid("Mira打13800138000了");
    EXPECT_EQ(hybrid.symbols, symbols_of("打幺三八零零幺三八零零零了"));
}

TEST(HybridFrontend, DateNormalizedPerChunk) {
    const auto hybrid = convert_hybrid("在2024年3月1号，Mira说很卡");
    EXPECT_EQ(hybrid.symbols, symbols_of("在二零二四年三月一号,说很卡"));
    EXPECT_EQ(hybrid.chunks.size(), std::size_t{2});
    expect_chunks_partition(hybrid, "日期混读");
}

TEST(HybridFrontend, PureChineseHasNoBertChunks) {
    // 纯中文走单块路径（上游 'chinese' 模式不分块），
    // 引擎据此用整段文本跑一次 RoBERTa。
    const auto hybrid = convert_hybrid("你好，世界。");
    EXPECT_FALSE(hybrid.symbols.empty());
    EXPECT_TRUE(hybrid.chunks.empty());
    EXPECT_EQ(hybrid.symbols, symbols_of("你好，世界。"));
}

TEST(HybridFrontend, ChunkWithoutPhonesIsSkipped) {
    // 英文块 + 空白：没有任何音素产出，也就没有 BERT 分块。
    const auto hybrid = convert_hybrid("hello   world");
    EXPECT_TRUE(hybrid.symbols.empty());
    EXPECT_TRUE(hybrid.chunks.empty());
    EXPECT_EQ(hybrid.word2ph_size, std::size_t{0});
}

/// 挂了**英文前端**的混读前端。
///
/// 缺陷回归的必要条件：上面的 convert_hybrid 走裸前端（没有英文
/// 资源），英文块被整体丢弃 —— 无音素、无 word2ph，分块不变量恒真，
/// 拦不住"英文块不进分块"这个缺陷。真实音色包带英文词典，英文块
/// 产出音素与 word2ph，覆盖缺口只有这条路径能暴露。
/// 返回 nullopt = 英文资源缺失（调用方据此 GTEST_SKIP，不许依赖网络）。
std::optional<HybridResult> try_convert_hybrid_with_english(const std::string& text) {
    text::EnglishG2pResources resources;
    resources.cmu_dict = GENIE_ENGLISH_CMU_DICT;
    resources.pos_tagger = GENIE_ENGLISH_POS_TAGGER;
    resources.g2p_model = GENIE_ENGLISH_G2P_MODEL;
    for (const std::string* path :
         {&resources.cmu_dict, &resources.pos_tagger, &resources.g2p_model}) {
        if (!std::filesystem::exists(*path)) {
            return std::nullopt;
        }
    }
    // 与引擎同款装配（engine.cpp 的 make_chinese_frontend 直连）：
    // 公开的 make_frontend(Hybrid, resources) 目前不接英文资源 ——
    // 见下方测试的对照断言与 PR 说明。
    const auto english = make_english_frontend(resources);
    if (english == nullptr) {
        return std::nullopt;
    }
    const auto frontend = make_chinese_frontend(nullptr, nullptr, english.get());
    const auto sequence = frontend->convert(text);

    HybridResult result;
    for (const auto& symbol : sequence.symbols) {
        if (!result.symbols.empty()) {
            result.symbols += ' ';
        }
        result.symbols += symbol;
    }
    result.chunks = sequence.bert_chunks;
    result.word2ph_size = sequence.word2ph.size();
    return result;
}

TEST(HybridFrontend, EnglishChunkEntersBertChunksAsZero) {
    // 缺陷回归（2026-11-05）：英文 G2P 接入后，英文块产出音素与
    // word2ph 却**没有进 bert_chunks** —— 分块不再连续覆盖 word2ph，
    // 混读合成在 RoBERTa 装配处报 "bert chunks cover N phones,
    // sequence has M"。
    //
    // 上游语义（audit §6.3）：英文块不跑 RoBERTa，拼接时是**全零行**，
    // 但必须占位 —— 块覆盖 == 序列长度。
    const auto maybe_hybrid = try_convert_hybrid_with_english("你好hello世界");
    if (!maybe_hybrid.has_value()) {
        GTEST_SKIP() << "english resources not present";
    }
    const auto& hybrid = *maybe_hybrid;
    ASSERT_FALSE(hybrid.symbols.empty());
    ASSERT_EQ(hybrid.chunks.size(), std::size_t{3});
    expect_chunks_partition(hybrid, "英文块占位");

    // 中间那块是英文：零特征标记，word2ph 覆盖英文音素。
    ASSERT_EQ(hybrid.chunks[1].text, "hello");
    EXPECT_TRUE(hybrid.chunks[1].zero_features);
    EXPECT_EQ(hybrid.chunks[1].word_offset, hybrid.chunks[0].word_count);
    EXPECT_GT(hybrid.chunks[1].word_count, std::size_t{0});

    // 与裸前端（英文被整体丢弃）对照：英文音素真的进了序列。
    EXPECT_NE(hybrid.symbols, convert_hybrid("你好hello世界").symbols);
}

// ---------------------------------------------------------------------------
// 黄金语料（tools/verify_language_split.py 以上游正则逻辑为参照生成）
// ---------------------------------------------------------------------------

TEST(HybridGolden, CorpusMatchesLockedOutput) {
    std::ifstream file(GENIE_HYBRID_GOLDEN_FILE);
    ASSERT_TRUE(file.is_open()) << GENIE_HYBRID_GOLDEN_FILE;
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    const auto root = json::parse(content);

    std::size_t checked = 0;
    for (const auto& case_value : root.as_object().at("cases").as_array()) {
        const auto& obj = case_value.as_object();
        const auto& text = obj.at("text").as_string();

        // 1) 切块本身与上游一致：黄金文件的 "split" 是上游
        //    split_language 正则逻辑的完整产出（含英文块）。
        const auto split = lang_split::split_by_language(text);
        const auto& expected_split = obj.at("split").as_array();
        ASSERT_EQ(split.size(), expected_split.size()) << text;
        for (std::size_t i = 0; i < split.size(); ++i) {
            const auto& expected = expected_split[i].as_object();
            const char* language =
                split[i].language == lang_split::Chunk::Language::Chinese ? "chinese" : "english";
            EXPECT_EQ(language, expected.at("language").as_string()) << text << " chunk " << i;
            EXPECT_EQ(split[i].content, expected.at("content").as_string())
                << text << " chunk " << i;
        }

        // 2) 前端的 BERT 分块 == 其中文块（黄金语料生成时英文块尚被
        //    丢弃；挂英文前端时英文块是零特征占位块，见
        //    EnglishChunkEntersBertChunksAsZero），且各块连续覆盖 word2ph。
        const auto result = convert_hybrid(text);
        expect_chunks_partition(result, text);
        const auto& expected_bert = obj.at("bert_chunks").as_array();
        ASSERT_EQ(result.chunks.size(), expected_bert.size()) << text;
        for (std::size_t i = 0; i < result.chunks.size(); ++i) {
            const auto& expected = expected_bert[i].as_object();
            EXPECT_EQ(result.chunks[i].text, expected.at("content").as_string())
                << text << " bert chunk " << i;
            EXPECT_EQ(result.chunks[i].word_offset,
                      static_cast<std::size_t>(expected.at("word_offset").as_int()))
                << text << " bert chunk " << i;
            EXPECT_EQ(result.chunks[i].word_count,
                      static_cast<std::size_t>(expected.at("word_count").as_int()))
                << text << " bert chunk " << i;
        }
        ++checked;
    }
    EXPECT_GT(checked, 15U) << "黄金语料不应缩水";
}

}  // namespace
}  // namespace genie::text
