// tests/unit/test_errors.cpp

#include <gtest/gtest.h>

#include <string>

#include "genie/errors.hpp"

using namespace genie;

TEST(Errors, ToStringCoversEveryCode) {
    EXPECT_STREQ(to_string(ErrorCode::Ok), "ok");
    EXPECT_STREQ(to_string(ErrorCode::InvalidArgument), "invalid_argument");
    EXPECT_STREQ(to_string(ErrorCode::NotFound), "not_found");
    EXPECT_STREQ(to_string(ErrorCode::IoError), "io_error");
    EXPECT_STREQ(to_string(ErrorCode::ModelLoadError), "model_load_error");
    EXPECT_STREQ(to_string(ErrorCode::InferenceError), "inference_error");
    EXPECT_STREQ(to_string(ErrorCode::Unsupported), "unsupported");
    EXPECT_STREQ(to_string(ErrorCode::Cancelled), "cancelled");
    EXPECT_STREQ(to_string(ErrorCode::Timeout), "timeout");
    EXPECT_STREQ(to_string(ErrorCode::BackendUnavailable), "backend_unavailable");
    EXPECT_STREQ(to_string(ErrorCode::Internal), "internal");
}

TEST(Errors, RoundTripsThroughString) {
    for (const auto code :
         {ErrorCode::Ok, ErrorCode::InvalidArgument, ErrorCode::NotFound, ErrorCode::IoError,
          ErrorCode::ModelLoadError, ErrorCode::InferenceError, ErrorCode::Unsupported,
          ErrorCode::Cancelled, ErrorCode::Timeout, ErrorCode::BackendUnavailable,
          ErrorCode::Internal}) {
        EXPECT_EQ(error_code_from_string(to_string(code)), code);
    }
}

TEST(Errors, UnknownStringMapsToInternal) {
    EXPECT_EQ(error_code_from_string("not-a-real-code"), ErrorCode::Internal);
}

TEST(Errors, ThrownErrorCarriesCode) {
    try {
        throw_error(ErrorCode::NotFound, "missing thing");
        FAIL() << "throw_error should have thrown";
    } catch (const Error& error) {
        EXPECT_EQ(error.code(), ErrorCode::NotFound);
        EXPECT_STREQ(error.what(), "missing thing");
    }
}

TEST(Errors, ErrorIsARuntimeError) {
    try {
        throw_error(ErrorCode::Internal, "boom");
    } catch (const std::runtime_error& error) {
        EXPECT_STREQ(error.what(), "boom");
        return;
    }
    FAIL() << "expected std::runtime_error";
}
