/*
 * genie/c/genie_tts.h —— C ABI
 *
 * 面向不想直接链接 C++ SDK 的调用方（Go / Rust / Python ctypes / C）。
 *
 * 约定：
 *   - 所有函数返回 genie_status（0 = 成功），不跨边界抛异常。
 *   - 所有返回的指针由本库分配，必须用对应的 *_free 释放。
 *   - 字符串一律 UTF-8。
 *   - 线程安全：genie_engine 句柄本身可被多个线程使用，
 *     但同一个句柄上的单个操作需要调用方自行串行化。
 */

#ifndef GENIE_C_GENIE_TTS_H
#define GENIE_C_GENIE_TTS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(GENIE_SHARED)
#define GENIE_API __declspec(dllexport)
#elif defined(__GNUC__) && __GNUC__ >= 4
#define GENIE_API __attribute__((visibility("default")))
#else
#define GENIE_API
#endif

/** 状态码。数值与 C++ 侧 genie::ErrorCode 一一对应。 */
typedef enum genie_status {
    GENIE_OK = 0,
    GENIE_ERROR_INVALID_ARGUMENT = 1,
    GENIE_ERROR_NOT_FOUND = 2,
    GENIE_ERROR_IO = 3,
    GENIE_ERROR_MODEL_LOAD = 4,
    GENIE_ERROR_INFERENCE = 5,
    GENIE_ERROR_UNSUPPORTED = 6,
    GENIE_ERROR_CANCELLED = 7,
    GENIE_ERROR_TIMEOUT = 8,
    GENIE_ERROR_BACKEND_UNAVAILABLE = 9,
    GENIE_ERROR_INTERNAL = 10
} genie_status;

/** 语言。数值与 C++ 侧 genie::Language 一一对应。 */
typedef enum genie_language {
    GENIE_LANGUAGE_AUTO = 0,
    GENIE_LANGUAGE_CHINESE = 1,
    GENIE_LANGUAGE_ENGLISH = 2,
    GENIE_LANGUAGE_JAPANESE = 3,
    GENIE_LANGUAGE_KOREAN = 4,
    GENIE_LANGUAGE_HYBRID_CHINESE_ENGLISH = 5
} genie_language;

/** 不透明的引擎句柄。 */
typedef struct genie_engine genie_engine;

/** 不透明的合成请求句柄（流式）。 */
typedef struct genie_request genie_request;

/** 库版本字符串（静态存储，调用方不得释放）。 */
GENIE_API const char* genie_version(void);

/**
 * 创建一个引擎实例。
 *
 * @param model_dir  模型包目录（含 manifest.json）
 * @param out        输出引擎句柄；失败时为 NULL
 * @return 状态码
 */
GENIE_API genie_status genie_engine_create(const char* model_dir, genie_engine** out);

/**
 * 销毁引擎实例。传入 NULL 是无操作。
 */
GENIE_API void genie_engine_destroy(genie_engine* engine);

/**
 * 同步合成。
 *
 * @param text        UTF-8 文本
 * @param language    语言
 * @param seed        随机种子；传 0 表示不固定
 * @param out_samples 输出的 float32 单声道采样（调用方需用 genie_free 释放）
 * @param out_count   输出采样数
 * @param out_rate    输出采样率
 * @return 状态码
 */
GENIE_API genie_status genie_engine_synthesize(genie_engine* engine, const char* text,
                                               genie_language language, uint64_t seed,
                                               float** out_samples, size_t* out_count,
                                               int* out_rate);

/**
 * 合成并直接写成 WAV 文件。
 *
 * 比 genie_engine_synthesize 更省事：调用方不用管浮点缓冲区与采样率。
 *
 * @param output_path 输出 WAV 路径
 * @return 状态码
 */
GENIE_API genie_status genie_engine_synthesize_wav(genie_engine* engine, const char* text,
                                                   genie_language language, uint64_t seed,
                                                   const char* output_path);

/**
 * 把文本转成音素符号 ID —— **不需要模型**。
 *
 * 排查文本前端的问题时很有用：直接看到模型会收到的 text_seq。
 *
 * @param out_ids 输出的符号 ID 数组（用 genie_free 释放）
 * @param out_count 输出元素个数
 * @return 状态码；语言不受支持时返回 GENIE_ERROR_UNSUPPORTED
 */
GENIE_API genie_status genie_phonemize(const char* text, genie_language language, int64_t** out_ids,
                                       size_t* out_count);

/**
 * 释放由本库分配的缓冲区。
 */
GENIE_API void genie_free(void* ptr);

/** 状态码的稳定文本描述。 */
GENIE_API const char* genie_status_string(genie_status status);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* GENIE_C_GENIE_TTS_H */
