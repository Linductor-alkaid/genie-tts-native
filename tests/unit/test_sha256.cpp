// tests/unit/test_sha256.cpp

#include <gtest/gtest.h>

#include <string>

#include "genie/sha256.hpp"

using namespace genie;

TEST(Sha256, MatchesKnownVectors) {
    // FIPS 180-4 / NIST 发布的标准测试向量
    EXPECT_EQ(sha256_hex(""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    EXPECT_EQ(sha256_hex("abc"),
              "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    EXPECT_EQ(sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
              "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
}

TEST(Sha256, HandlesLongInput) {
    const std::string million(1000000, 'a');
    EXPECT_EQ(sha256_hex(million),
              "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

TEST(Sha256, HandlesBlockBoundaryLengths) {
    // 长度恰好跨过 64 字节分组边界时最容易出错
    for (std::size_t length = 54; length <= 66; ++length) {
        const std::string input(length, 'x');
        const std::string digest = sha256_hex(input);
        EXPECT_EQ(digest.size(), 64U) << "length=" << length;
    }
}

TEST(Sha256, IsDeterministic) {
    EXPECT_EQ(sha256_hex("genie"), sha256_hex("genie"));
    EXPECT_NE(sha256_hex("genie"), sha256_hex("genie2"));
}

TEST(Sha256, ToHexIsLowercase) {
    const auto digest = sha256("abc");
    const std::string hex = to_hex(digest);
    EXPECT_EQ(hex.size(), 64U);
    for (const char ch : hex) {
        EXPECT_TRUE((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f'));
    }
}
