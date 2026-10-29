// tests/unit/test_japanese.cpp —— 日语 G2P 的单元测试。
//
// 覆盖：
//   1. 文本规范化（% → パーセント、折叠连续标点、Unicode 小写）
//   2. 切段（哪些算"日语字符"—— Python 的 `\d` 是 Unicode 感知的）
//   3. 标签解析（OpenJTalk 全上下文标签 → 音素 + 韵律符号）
//   4. 黄金语料：124 句，逐音素比对
//
// 黄金语料由 tools/g2p/verify_japanese_g2p.py 用**上游的 Python 实现**
// （pyopenjtalk）当 oracle 生成。
//
// 注意：OpenJTalk 的词典（103 MB）**不在仓库里** ✓ —— 所以测试只跑
// "纯逻辑"那一层 ✓（文本 → 标签 → 音素 ✓），
// 标签本身由 oracle 生成后存在黄金语料里 ✓。
// 需要真词典的那条链路（OpenJTalk → 标签）由
// tools/g2p/verify_japanese_g2p.py 在本地跑 ✓。

#include <gtest/gtest.h>

#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

#include "genie/json.hpp"
#include "genie/symbols.hpp"
#include "text/frontend_factory_internal.hpp"
#include "text/japanese/japanese.hpp"
#include "text/japanese/openjtalk.hpp"

namespace {

using genie::text::japanese::labels_to_phonemes;
using genie::text::japanese::segment_japanese;
using genie::text::japanese::text_normalize;
using genie::text::japanese::text_to_phonemes;

}  // namespace

TEST(Japanese, TextNormalize) {
    // % / ％ → パーセント ✓
    EXPECT_EQ(text_normalize("50%"), "50パーセント");
    EXPECT_EQ(text_normalize("50％"), "50パーセント");
    // 连续标点折叠 ✓（上游 `([,./?!~…・])\1+` → `\1` ✓）
    EXPECT_EQ(text_normalize("あ,,い"), "あ,い");
    EXPECT_EQ(text_normalize("あ・・・い"), "あ・い");
    // 只折叠**同样字符**的连续串 ✓（上游 `([,./?!~…・])\1+` ✓）——
    // 不同标点挨在一起不折叠 ✓。
    EXPECT_EQ(text_normalize("あ?!い"), "あ?!い");
    EXPECT_EQ(text_normalize("あ,,..い"), "あ,.い");
    // 小写 ✓（Python 的 str.lower() 是 Unicode 感知的 ✓）
    EXPECT_EQ(text_normalize("ABC"), "abc");
    EXPECT_EQ(text_normalize("ＡＢＣ"), "ａｂｃ");
    // 汉字与假名不受影响 ✓
    EXPECT_EQ(text_normalize("日本語"), "日本語");
}

TEST(Japanese, Segmentation) {
    // 汉字 / 平假名 / 片假名 / ASCII 字母数字都是日语字符 ✓
    const auto seg = segment_japanese("漢字abc123");
    ASSERT_EQ(seg.segments.size(), 1U);
    EXPECT_EQ(seg.segments[0], "漢字abc123");
    EXPECT_TRUE(seg.marks.empty());

    // 标点把文本切开 ✓ —— marks[i] 在 segments[i] 之后 ✓
    const auto punct = segment_japanese("あ、い");
    ASSERT_EQ(punct.segments.size(), 2U);
    EXPECT_EQ(punct.segments[0], "あ");
    EXPECT_EQ(punct.segments[1], "い");
    ASSERT_EQ(punct.marks.size(), 1U);
    EXPECT_EQ(punct.marks[0], "、");

    // **全角 '０'（U+FF10）也是日语字符** ✓ ——
    // 上游的正则写的是 `\d` ✓，而 Python 3 的 `\d` 匹配所有 Nd 类数字 ✓，
    // 不只是显式列出的 \uff11-\uff19（那个区间从 １ 开始 ✓）。
    // 把它当标点的话切段位置就变了 → 音素序列完全不同 ✓。
    const auto full_width = segment_japanese("２０２３年");
    ASSERT_EQ(full_width.segments.size(), 1U);
    EXPECT_EQ(full_width.segments[0], "２０２３年");

    // ASCII 数字 ✓
    const auto ascii = segment_japanese("2023年");
    ASSERT_EQ(ascii.segments.size(), 1U);
    EXPECT_EQ(ascii.segments[0], "2023年");
}

TEST(Japanese, LabelsToPhonemes) {
    // OpenJTalk 的标准全上下文标签 ✓ —— 取第三个音素 ✓（上游 `-(.*?)\+` ✓）。
    // 首尾两个标签是 sil ✓（上游据此补 '^' 与 '$' ✓）。
    const std::vector<std::string> labels = {
        "xx^xx-sil+k=o/A:xx+xx+xx/B:xx-xx_xx/C:xx_xx+xx/D:xx+xx_xx/E:xx_xx!xx_xx-xx/"
        "F:xx_xx#xx_xx@xx_xx|xx_xx/"
        "G:5_5%0_xx_xx/H:xx_xx/I:xx-xx@xx+xx&xx-xx|xx+xx/J:1_5/K:1+1-5",
        "k^o-N+n=i/A:-3+2+4/B:xx-xx_xx/C:09_xx+xx/D:xx+xx_xx/E:xx_xx!xx_xx-xx/F:5_5#0_xx@1_1|1_5/"
        "G:xx_xx%xx_xx_xx/"
        "H:xx_xx/I:1-5@1+1&1-1|1+5/J:xx_xx/K:1+1-5",
        "w^a-sil+xx=xx/A:0+5+1/B:xx-xx_xx/C:xx_xx+xx/D:xx+xx_xx/E:5_5!0_xx-xx/"
        "F:xx_xx#xx_xx@xx_xx|xx_xx/G:xx_xx%"
        "xx_xx_xx/H:1_5/I:xx-xx@xx+xx&xx-xx|xx+xx/J:xx_xx/K:1+1-5",
    };
    const auto phones = labels_to_phonemes(labels);
    // 首个标签是 sil → '^' ✓（上游随后用 [1:-1] 砍掉 ✓）。
    ASSERT_FALSE(phones.empty());
    EXPECT_EQ(phones.front(), "^");
    // 末尾标签也是 sil → '$' ✓（'?' 只在 e3 == 1 时出现 ✓）。
    EXPECT_EQ(phones.back(), "$");
    // 中间的音素 ✓（"k^o-N+n=i" 里 `-(.*?)\+` 取到的是 'N' ✓）
    EXPECT_NE(std::find(phones.begin(), phones.end(), "N"), phones.end());
    EXPECT_EQ(phones.size(), 3U);
}

TEST(Japanese, GoldenCorpus) {
    std::ifstream file(GENIE_JAPANESE_GOLDEN_FILE);
    ASSERT_TRUE(file.good()) << "找不到黄金语料：" << GENIE_JAPANESE_GOLDEN_FILE;

    struct Case {
        std::string text;
        std::string normalized;
        std::vector<std::string> segments;
        std::vector<std::vector<std::string>> labels;
        std::vector<std::string> phonemes;
    };
    std::vector<Case> cases;
    {
        const genie::json::Value root = genie::json::parse_file(GENIE_JAPANESE_GOLDEN_FILE);
        ASSERT_TRUE(root.is_object());
        const genie::json::Value* list = root.find("cases");
        ASSERT_NE(list, nullptr);
        ASSERT_TRUE(list->is_array());
        for (const auto& item : list->as_array()) {
            ASSERT_TRUE(item.is_object());
            Case entry;
            entry.text = item.find("text")->as_string();
            entry.normalized = item.find("normalized")->as_string();
            for (const auto& segment : item.find("segments")->as_array()) {
                entry.segments.push_back(segment.as_string());
            }
            for (const auto& group : item.find("labels")->as_array()) {
                std::vector<std::string> labels;
                for (const auto& label : group.as_array()) {
                    labels.push_back(label.as_string());
                }
                entry.labels.push_back(std::move(labels));
            }
            for (const auto& phoneme : item.find("phonemes")->as_array()) {
                entry.phonemes.push_back(phoneme.as_string());
            }
            cases.push_back(std::move(entry));
        }
    }
    ASSERT_FALSE(cases.empty());

    for (const auto& entry : cases) {
        // 1. 规范化 ✓
        EXPECT_EQ(text_normalize(entry.text), entry.normalized) << "规范化不一致：" << entry.text;
        // 2. 切段 ✓
        const auto segmentation = segment_japanese(entry.text);
        EXPECT_EQ(segmentation.segments, entry.segments) << "切段不一致：" << entry.text;
        // 3. 标签 → 音素 → 完整序列 ✓
        EXPECT_EQ(text_to_phonemes(entry.text, entry.labels), entry.phonemes)
            << "音素不一致：" << entry.text;
    }
}

TEST(Japanese, FrontendNeedsDictionary) {
    // 没有词典就没有日文前端 ✓（词典 103 MB ✓ —— 由音色包给出 ✓）。
    genie::text::JapaneseG2pResources resources;
    EXPECT_EQ(genie::text::make_japanese_frontend(resources), nullptr);
    resources.dictionary_dir = "/definitely/not/a/dictionary";
    EXPECT_EQ(genie::text::make_japanese_frontend(resources), nullptr);

    // OpenJTalk 本身：没载入词典时 loaded() 为 false ✓
    genie::text::japanese::OpenJTalk engine;
    EXPECT_FALSE(engine.loaded());
    EXPECT_FALSE(engine.load(""));
}
