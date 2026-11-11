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

// from_raw 是后端把执行器输出缓冲转成 Tensor 的单次 memcpy 路径
// （ort_backend 的 from_ort_value 用它），必须与 from_float32 等价。
TEST(Tensor, RawRoundTripMatchesFactory) {
    const std::vector<float> data = {1.5F, -2.25F, 3.0F};
    const auto raw = Tensor::from_raw(ElementType::Float32, {1, 3}, data.data());
    const auto via_vector = Tensor::from_float32({1, 3}, data);
    EXPECT_EQ(raw.type(), ElementType::Float32);
    EXPECT_EQ(raw.shape_string(), via_vector.shape_string());
    ASSERT_EQ(raw.floats().size(), 3U);
    EXPECT_FLOAT_EQ(raw.floats()[1], -2.25F);

    const std::vector<std::int64_t> ints = {7, 8};
    const auto raw_int = Tensor::from_raw(ElementType::Int64, {2}, ints.data());
    ASSERT_EQ(raw_int.ints().size(), 2U);
    EXPECT_EQ(raw_int.ints()[1], 8);
}

TEST(Tensor, RawEmptyIsLegalNullptrIsNot) {
    // 零元素张量（动态维推理里真实存在）允许 nullptr。
    const auto empty = Tensor::from_raw(ElementType::Float32, {0}, nullptr);
    EXPECT_EQ(empty.element_count(), 0U);
    EXPECT_TRUE(empty.bytes().empty());
    EXPECT_THROW(Tensor::from_raw(ElementType::Float32, {2}, nullptr), Error);
}

// from_buffer 是后端把"执行器直接写入的自有输出缓冲"变成 Tensor 的零拷贝
// 接管路径（IOBinding 输出缓冲池用它）：vector 指针换手，无 memcpy。
TEST(Tensor, BufferAdoptionOwnsStorageWithoutCopy) {
    std::vector<std::byte> storage(3U * sizeof(float));
    const auto* raw_view = reinterpret_cast<const float*>(storage.data());
    const float first = raw_view[0];  // 接管前读一眼；接管后原缓冲失效
    const auto tensor = Tensor::from_buffer(std::move(storage), ElementType::Float32, {1, 3});
    EXPECT_EQ(tensor.type(), ElementType::Float32);
    EXPECT_EQ(tensor.shape_string(), "1x3");
    ASSERT_EQ(tensor.floats().size(), 3U);
    EXPECT_FLOAT_EQ(tensor.floats()[0], first);
    // 接管是移动：原 vector 的容量归零（标准未定内容，但长度必为 0 ——
    // 被移动后的 vector 处于有效未指定态，这里只断言 tensor 侧所有权完整）。
    EXPECT_EQ(tensor.byte_size(), 3U * sizeof(float));
}

TEST(Tensor, BufferAdoptionRejectsSizeMismatch) {
    std::vector<std::byte> storage(2U * sizeof(float));
    EXPECT_THROW(Tensor::from_buffer(std::move(storage), ElementType::Float32, {3}), Error);
    // 标量（rank-0）：恰好 1 个元素。
    std::vector<std::byte> one(sizeof(float));
    EXPECT_NO_THROW(Tensor::from_buffer(std::move(one), ElementType::Float32, {}));
    std::vector<std::byte> one_int64(sizeof(std::int64_t));
    EXPECT_THROW(Tensor::from_buffer(std::move(one_int64), ElementType::Float32, {}), Error);
}
