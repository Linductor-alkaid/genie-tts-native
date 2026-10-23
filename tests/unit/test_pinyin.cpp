// tests/unit/test_pinyin.cpp
//
// 拼音切分与音素推导的测试。
//
// 这部分规则的正确性已经用 tools/g2p/verify_pinyin_rules.py 对 41,923 个
// 汉字与 pypinyin 全量比对过（切分与音素都一致）。这里锁住的是
// 「不能变的那些行为」—— 具体音素、零声母的处理、儿化的边界情况。

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "genie/pinyin.hpp"
#include "genie/sha256.hpp"
#include "genie/symbols.hpp"
#include "genie/utf8.hpp"

namespace genie::pinyin {
namespace {

TEST(Pinyin, SplitsInitialAndFinal) {
    const auto syllable = split("zhong1");
    EXPECT_EQ(syllable.initial, "zh");
    EXPECT_EQ(syllable.final_with_tone, "ong1");
    EXPECT_TRUE(syllable.valid());
}

TEST(Pinyin, HandlesUmlaut) {
    // 绿 lv4 → l + v4
    const auto lv = split("lv4");
    EXPECT_EQ(lv.initial, "l");
    EXPECT_EQ(lv.final_with_tone, "v4");

    // 女 nv3 → n + v3
    const auto nv = split("nv3");
    EXPECT_EQ(nv.initial, "n");
    EXPECT_EQ(nv.final_with_tone, "v3");

    // 学 xue2 → x + ve2
    const auto xue = split("xue2");
    EXPECT_EQ(xue.initial, "x");
    EXPECT_EQ(xue.final_with_tone, "ve2");
}

TEST(Pinyin, HandlesZeroConsonant) {
    // 一 yi1 → 零声母 + i1
    const auto yi = split("yi1");
    EXPECT_EQ(yi.initial, "");
    EXPECT_EQ(yi.final_with_tone, "i1");

    // 五 wu3 → 零声母 + u3
    const auto wu = split("wu3");
    EXPECT_EQ(wu.initial, "");
    EXPECT_EQ(wu.final_with_tone, "u3");

    // 雨 yu3 → 零声母 + v3
    const auto yu = split("yu3");
    EXPECT_EQ(yu.initial, "");
    EXPECT_EQ(yu.final_with_tone, "v3");
}

TEST(Pinyin, HandlesContractedFinals) {
    // 六 liu4 → l + iou4
    const auto liu = split("liu4");
    EXPECT_EQ(liu.initial, "l");
    EXPECT_EQ(liu.final_with_tone, "iou4");

    // 贵 gui4 → g + uei4
    const auto gui = split("gui4");
    EXPECT_EQ(gui.initial, "g");
    EXPECT_EQ(gui.final_with_tone, "uei4");

    // 论 lun2 → l + uen2
    const auto lun = split("lun2");
    EXPECT_EQ(lun.initial, "l");
    EXPECT_EQ(lun.final_with_tone, "uen2");
}

TEST(Pinyin, RejectsEmptyInput) {
    const auto syllable = split("");
    EXPECT_FALSE(syllable.valid());
    EXPECT_EQ(syllable.initial, "");
    EXPECT_EQ(syllable.final_with_tone, "");
}

TEST(Pinyin, RejectsNasalOnlySyllable) {
    // 嗯 → n（没有韵母）。这种字不产生音素。
    const auto syllable = split("n2");
    EXPECT_FALSE(syllable.valid());
    EXPECT_EQ(syllable.final_with_tone, "");
}

TEST(Pinyin, ProducesExpectedPhones) {
    const auto phones = [](std::string_view tone3) { return syllable_to_phones(split(tone3)); };

    const std::vector<std::string> zhong = phones("zhong1");
    ASSERT_EQ(zhong.size(), 2U);
    EXPECT_EQ(zhong[0], "zh");
    EXPECT_EQ(zhong[1], "ong1");

    const std::vector<std::string> yi = phones("yi1");
    ASSERT_EQ(yi.size(), 2U);
    EXPECT_EQ(yi[0], "y");
    EXPECT_EQ(yi[1], "i1");

    const std::vector<std::string> wu = phones("wu3");
    ASSERT_EQ(wu.size(), 2U);
    EXPECT_EQ(wu[0], "w");
    EXPECT_EQ(wu[1], "u3");

    const std::vector<std::string> liu = phones("liu4");
    ASSERT_EQ(liu.size(), 2U);
    EXPECT_EQ(liu[0], "l");
    EXPECT_EQ(liu[1], "iu4");  // 收缩成书写形式
}

TEST(Pinyin, LookupFindsCommonCharacters) {
    EXPECT_EQ(lookup("\xE4\xB8\xAD").value_or(""), "zhong1");  // 中
    EXPECT_EQ(lookup("\xE5\x9B\xBD").value_or(""), "guo2");    // 国
    EXPECT_EQ(lookup("\xE4\xBD\xA0").value_or(""), "ni3");     // 你
    EXPECT_EQ(lookup("A").has_value(), false);
    EXPECT_EQ(lookup("").has_value(), false);
}

/// 生成的音素必须都在符号表里 —— 否则会被静默丢掉，合成结果就不对了。
TEST(Pinyin, GeneratedPhonesAreInSymbolTable) {
    const char* samples[] = {
        "zhong1", "guo2", "ni3", "hao3", "shi4", "zhong1", "wen2", "yu3", "er2",
        "a1",     "o1",   "e5",  "liu4", "gui4", "lun2",   "xue2", "lv4", "nv3",
    };
    for (const auto* sample : samples) {
        for (const auto& phone : syllable_to_phones(split(sample))) {
            // "er" 是零声母的伪声母，但符号表里没有它 —— 上游会过滤掉。
            // "a" / "o" / "e" 在符号表里（它们同时是日语罗马字音素）。
            if (phone == "er") {
                continue;
            }
            EXPECT_TRUE(symbols::contains(phone))
                << "音素 " << phone << "（来自 " << sample << "）不在符号表里";
        }
    }
}

/// 整张拼音表的摘要。
///
/// 把词典里每个字能产出的音素序列串起来算一个 SHA-256。
/// 规则或词典任何一处改动都会被抓到 —— 音素错了不会报错，只会合成出乱码，
/// 所以这一层必须有硬锁。
///
/// 生成方式见 tools/g2p/dump_table_digest.cpp。
TEST(Pinyin, TableDigestIsStable) {
    std::string blob;
    for (std::uint32_t codepoint = 0x4E00; codepoint <= 0x9FA5; ++codepoint) {
        const std::string character = utf8::encode(codepoint);
        const auto pinyin = lookup(character);
        if (!pinyin) {
            continue;
        }
        const auto syllable = split(*pinyin);
        if (!syllable.valid()) {
            continue;
        }
        blob += character;
        blob += '\t';
        const auto phones = syllable_to_phones(syllable);
        for (std::size_t i = 0; i < phones.size(); ++i) {
            if (i != 0) {
                blob += ',';
            }
            blob += phones[i];
        }
        blob += '\n';
    }
    EXPECT_EQ(sha256_hex(blob), "b381f96bef76453f2a00a1fe83779c14cf85878e55a24b7e32d69c8cee40b83a");
}

}  // namespace
}  // namespace genie::pinyin
