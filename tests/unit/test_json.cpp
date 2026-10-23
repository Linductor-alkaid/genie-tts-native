// tests/unit/test_json.cpp

#include <gtest/gtest.h>

#include <string>

#include "genie/errors.hpp"
#include "genie/json.hpp"

using namespace genie;
using namespace genie::json;

TEST(Json, ParsesScalars) {
    EXPECT_TRUE(parse("null").is_null());
    EXPECT_TRUE(parse("true").as_bool());
    EXPECT_FALSE(parse("false").as_bool());
    EXPECT_EQ(parse("42").as_int(), 42);
    EXPECT_EQ(parse("-7").as_int(), -7);
    EXPECT_DOUBLE_EQ(parse("3.5").as_double(), 3.5);
    EXPECT_DOUBLE_EQ(parse("1e3").as_double(), 1000.0);
    EXPECT_DOUBLE_EQ(parse("-2.5e-2").as_double(), -0.025);
    EXPECT_EQ(parse("\"hello\"").as_string(), "hello");
}

TEST(Json, ParsesNestedStructures) {
    const auto doc = parse(R"({
        "name": "voice",
        "sample_rate": 32000,
        "capabilities": {"v2proplus": true, "roberta": false},
        "models": [
            {"role": "t2s_encoder", "file": "models/a.onnx", "sha256": "ab"},
            {"role": "vits", "file": "models/b.onnx"}
        ]
    })");

    EXPECT_EQ(doc.get_string("name"), "voice");
    EXPECT_EQ(doc.get_int("sample_rate", 0), 32000);
    EXPECT_TRUE(doc.find("capabilities")->get_bool("v2proplus", false));
    EXPECT_FALSE(doc.find("capabilities")->get_bool("roberta", true));
    ASSERT_EQ(doc.find("models")->as_array().size(), 2U);
    EXPECT_EQ(doc.find("models")->as_array()[0].get_string("role"), "t2s_encoder");
    EXPECT_EQ(doc.find("models")->as_array()[1].get_string("file"), "models/b.onnx");
}

TEST(Json, MissingKeysReturnFallbacks) {
    const auto doc = parse(R"({"a": 1})");
    EXPECT_EQ(doc.get_int("missing", 99), 99);
    EXPECT_EQ(doc.get_string("missing", "fallback"), "fallback");
    EXPECT_TRUE(doc.get_bool("missing", true));
    EXPECT_EQ(doc.find("missing"), nullptr);
}

TEST(Json, HandlesEscapes) {
    const auto doc = parse(R"("a\"b\\c\/d\ne\tf")");
    EXPECT_EQ(doc.as_string(), "a\"b\\c/d\ne\tf");
}

TEST(Json, HandlesUnicodeEscapes) {
    // U+4F60 = 你，UTF-8 为 E4 BD A0
    const auto doc = parse(R"("\u4f60\u597d")");
    EXPECT_EQ(doc.as_string(), "\xE4\xBD\xA0\xE5\xA5\xBD");
}

TEST(Json, HandlesSurrogatePairs) {
    // U+1F600 😀 = D83D DE00
    const auto doc = parse(R"("\ud83d\ude00")");
    EXPECT_EQ(doc.as_string(), "\xF0\x9F\x98\x80");
}

TEST(Json, PreservesUtf8Literals) {
    const auto doc = parse("\"你好\"");
    EXPECT_EQ(doc.as_string(), "你好");
}

TEST(Json, RejectsMalformedInput) {
    EXPECT_THROW(parse("{"), Error);
    EXPECT_THROW(parse("}"), Error);
    EXPECT_THROW(parse("[1,]"), Error);
    EXPECT_THROW(parse("{\"a\"}"), Error);
    EXPECT_THROW(parse("tru"), Error);
    EXPECT_THROW(parse("\"unterminated"), Error);
    EXPECT_THROW(parse("01"), Error);
    EXPECT_THROW(parse(""), Error);
}

TEST(Json, RejectsUnescapedControlCharacters) {
    EXPECT_THROW(parse("\"a\nb\""), Error);
}

TEST(Json, RejectsDeeplyNestedInput) {
    std::string deep(200, '[');
    EXPECT_THROW(parse(deep), Error);
}

TEST(Json, RoundTripsThroughDump) {
    const auto doc = parse(R"({"a":[1,2,3],"b":"x","c":true,"d":null})");
    const auto reparsed = parse(doc.dump());
    EXPECT_EQ(reparsed.get_int("a", 0), 0);  // "a" 是数组
    EXPECT_EQ(reparsed.find("a")->as_array().size(), 3U);
    EXPECT_EQ(reparsed.get_string("b"), "x");
    EXPECT_TRUE(reparsed.get_bool("c", false));
    EXPECT_TRUE(reparsed.find("d")->is_null());
}

TEST(Json, DumpEscapesControlCharacters) {
    Value value(std::string("a\nb"));
    EXPECT_EQ(value.dump(), "\"a\\nb\"");
    EXPECT_EQ(parse(value.dump()).as_string(), "a\nb");
}

TEST(Json, EmptyContainers) {
    EXPECT_TRUE(parse("{}").is_object());
    EXPECT_TRUE(parse("{}").as_object().empty());
    EXPECT_TRUE(parse("[]").is_array());
    EXPECT_TRUE(parse("[]").as_array().empty());
}
