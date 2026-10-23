// src/c/c_api.cpp
//
// C ABI 实现。
//
// 边界规则（见 include/genie/c/genie_tts.h）：
//   - **异常绝不穿过 ABI 边界**。所有入口都 try/catch，
//     C++ 异常转成 genie_status 返回。
//   - 返回的指针由本库分配，用 genie_free 释放。
//   - 字符串一律 UTF-8。
//
// 错误码与 C++ 侧 genie::ErrorCode 共用同一套枚举值 —— 两套语义漂移是最
// 难查的一类 bug，所以这里直接做静态断言把它们钉在一起。

#include "genie/c/genie_tts.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include "genie/audio.hpp"
#include "genie/engine.hpp"
#include "genie/errors.hpp"
#include "genie/text.hpp"
#include "genie/version.hpp"

/// C 侧的引擎句柄。
///
/// 必须定义在**全局命名空间** —— 头文件里前置声明的就是全局的
/// `struct genie_engine`，放到匿名命名空间里会变成另一个类型。
struct genie_engine {
    std::unique_ptr<genie::Engine> impl;
};

namespace {

using genie::ErrorCode;

/// 把 C++ 异常转成状态码。永不抛出。
genie_status to_status(const std::exception& error) noexcept {
    if (const auto* known = dynamic_cast<const genie::Error*>(&error); known != nullptr) {
        return static_cast<genie_status>(static_cast<int>(known->code()));
    }
    return GENIE_ERROR_INTERNAL;
}

genie::Language from_c_language(genie_language language) noexcept {
    const auto value = static_cast<int>(language);
    if (value < 0 || value > static_cast<int>(genie::Language::HybridChineseEnglish)) {
        return genie::Language::Auto;
    }
    return static_cast<genie::Language>(value);
}

/// 释放由 malloc 分配的缓冲区。
void free_buffer(void* ptr) noexcept {
    std::free(ptr);
}

}  // namespace

// 错误码与 C++ 侧必须一一对应。两边漂移了就在编译期炸掉。
static_assert(static_cast<int>(ErrorCode::Ok) == GENIE_OK);
static_assert(static_cast<int>(ErrorCode::InvalidArgument) == GENIE_ERROR_INVALID_ARGUMENT);
static_assert(static_cast<int>(ErrorCode::NotFound) == GENIE_ERROR_NOT_FOUND);
static_assert(static_cast<int>(ErrorCode::IoError) == GENIE_ERROR_IO);
static_assert(static_cast<int>(ErrorCode::ModelLoadError) == GENIE_ERROR_MODEL_LOAD);
static_assert(static_cast<int>(ErrorCode::InferenceError) == GENIE_ERROR_INFERENCE);
static_assert(static_cast<int>(ErrorCode::Unsupported) == GENIE_ERROR_UNSUPPORTED);
static_assert(static_cast<int>(ErrorCode::Cancelled) == GENIE_ERROR_CANCELLED);
static_assert(static_cast<int>(ErrorCode::Timeout) == GENIE_ERROR_TIMEOUT);
static_assert(static_cast<int>(ErrorCode::BackendUnavailable) == GENIE_ERROR_BACKEND_UNAVAILABLE);
static_assert(static_cast<int>(ErrorCode::Internal) == GENIE_ERROR_INTERNAL);

extern "C" {

const char* genie_version(void) {
    return genie::version_string();
}

genie_status genie_engine_create(const char* model_dir, genie_engine** out) {
    if (out == nullptr) {
        return GENIE_ERROR_INVALID_ARGUMENT;
    }
    *out = nullptr;
    if (model_dir == nullptr || *model_dir == '\0') {
        return GENIE_ERROR_INVALID_ARGUMENT;
    }

    try {
        auto handle = std::make_unique<genie_engine>();
        handle->impl = std::make_unique<genie::Engine>(std::string(model_dir));
        *out = handle.release();
        return GENIE_OK;
    } catch (const std::exception& error) {
        return to_status(error);
    } catch (...) {
        return GENIE_ERROR_INTERNAL;
    }
}

void genie_engine_destroy(genie_engine* engine) {
    delete engine;
}

genie_status genie_engine_synthesize(genie_engine* engine, const char* text,
                                     genie_language language, uint64_t seed, float** out_samples,
                                     size_t* out_count, int* out_rate) {
    if (out_samples == nullptr || out_count == nullptr || out_rate == nullptr) {
        return GENIE_ERROR_INVALID_ARGUMENT;
    }
    *out_samples = nullptr;
    *out_count = 0;
    *out_rate = 0;

    if (engine == nullptr || engine->impl == nullptr) {
        return GENIE_ERROR_INVALID_ARGUMENT;
    }
    if (text == nullptr) {
        return GENIE_ERROR_INVALID_ARGUMENT;
    }

    try {
        genie::SynthesisOptions options;
        options.language = from_c_language(language);
        if (seed != 0) {
            options.seed = seed;
        }

        const auto result = engine->impl->synthesize(std::string(text), options);

        auto* buffer =
            static_cast<float*>(std::malloc(result.audio.samples.size() * sizeof(float)));
        if (buffer == nullptr && !result.audio.samples.empty()) {
            return GENIE_ERROR_INTERNAL;
        }
        if (!result.audio.samples.empty()) {
            std::memcpy(buffer, result.audio.samples.data(),
                        result.audio.samples.size() * sizeof(float));
        }
        *out_samples = buffer;
        *out_count = result.audio.samples.size();
        *out_rate = result.audio.sample_rate;
        return GENIE_OK;
    } catch (const std::exception& error) {
        return to_status(error);
    } catch (...) {
        return GENIE_ERROR_INTERNAL;
    }
}

genie_status genie_engine_synthesize_wav(genie_engine* engine, const char* text,
                                         genie_language language, uint64_t seed,
                                         const char* output_path) {
    if (engine == nullptr || engine->impl == nullptr || text == nullptr || output_path == nullptr) {
        return GENIE_ERROR_INVALID_ARGUMENT;
    }

    try {
        genie::SynthesisOptions options;
        options.language = from_c_language(language);
        if (seed != 0) {
            options.seed = seed;
        }
        const auto result = engine->impl->synthesize(std::string(text), options);
        if (options.format == genie::AudioFormat::Float32) {
            genie::WavCodec::write_float32(std::string(output_path), result.audio);
        } else {
            genie::WavCodec::write_pcm16(std::string(output_path), result.audio);
        }
        return GENIE_OK;
    } catch (const std::exception& error) {
        return to_status(error);
    } catch (...) {
        return GENIE_ERROR_INTERNAL;
    }
}

genie_status genie_phonemize(const char* text, genie_language language, int64_t** out_ids,
                             size_t* out_count) {
    if (out_ids == nullptr || out_count == nullptr) {
        return GENIE_ERROR_INVALID_ARGUMENT;
    }
    *out_ids = nullptr;
    *out_count = 0;
    if (text == nullptr) {
        return GENIE_ERROR_INVALID_ARGUMENT;
    }

    try {
        auto frontend = genie::text::make_frontend(from_c_language(language));
        if (!frontend) {
            return GENIE_ERROR_UNSUPPORTED;
        }
        const auto sequence = frontend->convert(std::string(text));
        if (sequence.ids.empty()) {
            return GENIE_OK;
        }
        auto* buffer = static_cast<int64_t*>(std::malloc(sequence.ids.size() * sizeof(int64_t)));
        if (buffer == nullptr) {
            return GENIE_ERROR_INTERNAL;
        }
        std::memcpy(buffer, sequence.ids.data(), sequence.ids.size() * sizeof(int64_t));
        *out_ids = buffer;
        *out_count = sequence.ids.size();
        return GENIE_OK;
    } catch (const std::exception& error) {
        return to_status(error);
    } catch (...) {
        return GENIE_ERROR_INTERNAL;
    }
}

void genie_free(void* ptr) {
    free_buffer(ptr);
}

const char* genie_status_string(genie_status status) {
    return genie::to_string(static_cast<ErrorCode>(static_cast<int>(status)));
}

}  // extern "C"
