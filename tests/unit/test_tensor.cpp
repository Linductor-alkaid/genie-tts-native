// tests/unit/test_tensor.cpp

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "genie/errors.hpp"
#include "genie/tensor.hpp"

using namespace genie;

TEST(Tensor, Float32RoundTrip) {
    const auto tensor = Tensor::from_float32({1, 3}, {1.0F, 2.0F, 3.0F});
    EXPECT_EQ(tensor.type(), ElementType::Float32);
    EXPECT_EQ(tensor.rank(), 2U);
    EXPECT_EQ(tensor.element_count(), 3U);
    ASSERT_EQ(tensor.floats().size(), 3U);
    EXPECT_FLOAT_EQ(tensor.floats()[0], 1.0F);
    EXPECT_FLOAT_EQ(tensor.floats()[2], 3.0F);
    EXPECT_EQ(tensor.shape_string(), "1x3");
}

TEST(Tensor, Int64RoundTrip) {
    const auto tensor = Tensor::from_int64({1, 4}, {10, 20, 30, 40});
    EXPECT_EQ(tensor.type(), ElementType::Int64);
    ASSERT_EQ(tensor.ints().size(), 4U);
    EXPECT_EQ(tensor.ints()[3], 40);
    // 类型不匹配时返回空视图而不是 UB。
    EXPECT_TRUE(tensor.floats().empty());
}

TEST(Tensor, BoolRoundTrip) {
    const auto tensor = Tensor::from_bool({1, 3}, {true, false, true});
    EXPECT_EQ(tensor.type(), ElementType::Bool);
    ASSERT_EQ(tensor.bools().size(), 3U);
    EXPECT_TRUE(tensor.bools()[0]);
    EXPECT_FALSE(tensor.bools()[1]);
    EXPECT_TRUE(tensor.bools()[2]);
}

// 上游 stage_decoder 的 stop_condition_tensor 是 rank-0 bool 标量。
TEST(Tensor, ScalarBoolIsRankZero) {
    const auto tensor = Tensor::from_bool({}, {true});
    EXPECT_EQ(tensor.rank(), 0U);
    EXPECT_EQ(tensor.element_count(), 1U);
    const auto scalar = tensor.scalar();
    ASSERT_TRUE(scalar.has_value());
    EXPECT_TRUE(std::get<bool>(*scalar));
}

TEST(Tensor, ScalarRejectsMultiElement) {
    const auto tensor = Tensor::from_float32({2}, {1.0F, 2.0F});
    EXPECT_FALSE(tensor.scalar().has_value());
}

TEST(Tensor, ShapeMismatchThrows) {
    EXPECT_THROW(Tensor::from_float32({1, 3}, {1.0F}), Error);
    EXPECT_THROW(Tensor::from_int64({2}, {1, 2, 3}), Error);
}

TEST(Tensor, MatchesSupportsDynamicWildcard) {
    const auto tensor = Tensor::from_float32({1, 512}, std::vector<float>(512, 0.0F));
    EXPECT_TRUE(tensor.matches({1, 512}));
    EXPECT_TRUE(tensor.matches({1, -1}));
    EXPECT_TRUE(tensor.matches({-1, -1}));
    EXPECT_FALSE(tensor.matches({1, 511}));
    EXPECT_FALSE(tensor.matches({1}));
}

TEST(Tensor, ZerosAreZeroed) {
    const auto tensor = Tensor::zeros(ElementType::Float32, {2, 2});
    ASSERT_EQ(tensor.floats().size(), 4U);
    for (const float value : tensor.floats()) {
        EXPECT_FLOAT_EQ(value, 0.0F);
    }
}

TEST(Tensor, DimOutOfRangeIsZero) {
    const auto tensor = Tensor::from_float32({1, 2}, {1.0F, 2.0F});
    EXPECT_EQ(tensor.dim(0), 1);
    EXPECT_EQ(tensor.dim(1), 2);
    EXPECT_EQ(tensor.dim(2), 0);
}
