// tests/unit/test_symbols.cpp
//
// 符号表的回归测试。
//
// 符号表是整个项目最不能出错的地方 —— 顺序错一位，模型收到的就是
// 完全不同的音素序列，而且**不会报错**，只会合成出乱码。
//
// 所以这里不只测几个抽样，而是把整张表钉死：
//   - 数量必须是 732
//   - 全表摘要必须一致（任何顺序变动都会被抓到）
//   - 边界位置逐一检查

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <set>
#include <string>
#include <vector>

#include "genie/sha256.hpp"
#include "genie/symbols.hpp"

namespace genie::symbols {
namespace {

/// 整表摘要。生成方式：
///   sha256("\n".join(master_v2()))
/// 上游 Python 的同一张表算出同样的值 —— 见 docs/upstream-audit.md。
constexpr std::string_view kTableDigest =
    "0771f6cd6f6201e8c2b0de921dc69cc04caf3f346c599d4a41413967d5674da7";

TEST(Symbols, HasExpectedSize) {
    EXPECT_EQ(size(), 732U);
    EXPECT_EQ(master_v2().size(), 732U);
}

TEST(Symbols, TableDigestIsStable) {
    std::string joined;
    for (std::size_t i = 0; i < master_v2().size(); ++i) {
        if (i != 0) {
            joined += '\n';
        }
        joined += master_v2()[i];
    }
    EXPECT_EQ(sha256_hex(joined), kTableDigest);
}

TEST(Symbols, IsSortedUpToBrackets) {
    // 第 1 段是 sorted(set(...))，第 2 段是追加的 "[", "]"，
    // 后面两段各自有序。
    const auto& table = master_v2();
    const auto bracket = table[322];
    EXPECT_EQ(bracket, "[");
    EXPECT_EQ(table[323], "]");
    for (std::size_t i = 1; i < 322; ++i) {
        EXPECT_LT(table[i - 1], table[i])
            << "位置 " << i << ": " << table[i - 1] << " < " << table[i];
    }
}

TEST(Symbols, NoDuplicates) {
    const auto& table = master_v2();
    const std::set<std::string> unique(table.begin(), table.end());
    EXPECT_EQ(unique.size(), table.size());
}

TEST(Symbols, IdLookupRoundTrips) {
    for (std::size_t i = 0; i < master_v2().size(); ++i) {
        const auto& symbol = master_v2()[i];
        EXPECT_EQ(id_of(symbol), static_cast<std::int64_t>(i)) << "符号: " << symbol;
        EXPECT_EQ(name_of(static_cast<std::int64_t>(i)), symbol);
        EXPECT_TRUE(contains(symbol));
        EXPECT_TRUE(find(symbol).has_value());
    }
}

TEST(Symbols, UnknownSymbolIsRejected) {
    EXPECT_EQ(id_of("not-a-symbol"), -1);
    EXPECT_EQ(id_of(""), -1);
    EXPECT_FALSE(find("not-a-symbol").has_value());
    EXPECT_FALSE(contains("not-a-symbol"));
}

TEST(Symbols, OutOfRangeIdYieldsEmptyName) {
    EXPECT_TRUE(name_of(-1).empty());
    EXPECT_TRUE(name_of(732).empty());
    EXPECT_TRUE(name_of(100000).empty());
}

/// 抽样检查具体 ID —— 这些是模型里最常用的音素。
TEST(Symbols, KnownIdsAreCorrect) {
    EXPECT_EQ(id_of("!"), 0);
    EXPECT_EQ(id_of("z"), 319);
    EXPECT_EQ(id_of("zh"), 320);
    EXPECT_EQ(id_of("…"), 321);
    EXPECT_EQ(id_of("["), 322);
    EXPECT_EQ(id_of("]"), 323);
    EXPECT_EQ(id_of("ㄱ"), 324);
}

/// 上游符号表必须包含的东西。
TEST(Symbols, ContainsExpectedFamilies) {
    // 中文声母
    for (const auto* initial :
         {"b", "c", "ch", "d",  "f", "g", "h", "j", "k", "l",  "m",  "n",  "p",
          "q", "r", "s",  "sh", "t", "w", "x", "y", "z", "zh", "AA", "EE", "OO"}) {
        EXPECT_TRUE(contains(initial)) << "缺少声母: " << initial;
    }
    // 中文韵母（带调）
    // 注意 "i0" 本身**不在**表里：上游的生成规则是"基础韵母 + 声调"，
    // 所以 "i0" 只会派生出 "i01".."i05"。
    for (const auto* final : {"a1", "a5", "ai4", "an2", "ang4", "e5", "er2", "i1", "ong1", "u3",
                              "v4", "van2", "ve2", "vn3", "iu4", "ui4", "un2", "i01", "i05"}) {
        EXPECT_TRUE(contains(final)) << "缺少韵母: " << final;
    }
    // 标点
    for (const auto* mark : {"!", "?", "…", ",", ".", "-", "SP", "SP2", "SP3", "UNK"}) {
        EXPECT_TRUE(contains(mark)) << "缺少标点: " << mark;
    }
    // ARPAbet
    for (const auto* phone :
         {"AA", "AA0", "AA1", "ER", "IY1", "OW0", "SH", "TH", "ZH", "NG", "JH", "HH"}) {
        EXPECT_TRUE(contains(phone)) << "缺少 ARPAbet: " << phone;
    }
}

/// 标点的 ID 顺序在下游的文本前端里会被用到。
///
/// 标点落在主符号段里，按码位排序：ASCII 的标点挨在一起，
/// 而 "…"（U+2026）排在所有 ASCII 之后。
TEST(Symbols, PunctuationIdsAreContiguous) {
    EXPECT_EQ(id_of("!"), 0);
    EXPECT_EQ(id_of(","), 1);
    EXPECT_EQ(id_of("-"), 2);
    EXPECT_EQ(id_of("."), 3);
    EXPECT_EQ(id_of("?"), 4);
    EXPECT_EQ(id_of("…"), 321);
}

}  // namespace
}  // namespace genie::symbols
