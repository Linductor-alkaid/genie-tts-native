/*
 * tests/c/test_c_api.c
 *
 * C ABI 的测试。
 *
 * 这个文件**用 C 编译器编译**（不是 C++）—— 头文件如果不小心引入了
 * C++ 语法，这里就会编译失败。ABI 的使用者（Go / Rust / Python ctypes）
 * 拿到的就是纯 C 头文件，所以这个约束必须由测试来守。
 *
 * 测试范围刻意避开需要 ONNX Runtime 的路径：没有后端时
 * genie_engine_create() 应当返回 GENIE_ERROR_BACKEND_UNAVAILABLE，
 * 这本身也是一个要测的行为。
 */

#include "genie/c/genie_tts.h"

#include <stdio.h>
#include <string.h>

/* 挂死时 stdout 缓冲区里的内容会丢 —— 每步都 flush，才知道卡在哪。 */
static void step(const char* name) {
    printf("[%s]\n", name);
    fflush(stdout);
}

static int g_failures = 0;

#define CHECK(cond, message)                                             \
    do {                                                                 \
        if (!(cond)) {                                                   \
            printf("FAIL: %s (%s:%d)\n", (message), __FILE__, __LINE__); \
            ++g_failures;                                                \
        }                                                                \
    } while (0)

static void test_version(void) {
    const char* version = genie_version();
    CHECK(version != NULL, "version is not null");
    CHECK(version != NULL && strlen(version) > 0, "version is not empty");
    printf("version: %s\n", version != NULL ? version : "(null)");
}

static void test_status_string(void) {
    /* 状态码文本与 C++ 侧 genie::to_string(ErrorCode) 是同一套（snake_case）。 */
    CHECK(strcmp(genie_status_string(GENIE_OK), "ok") == 0, "GENIE_OK is ok");
    CHECK(strcmp(genie_status_string(GENIE_ERROR_INVALID_ARGUMENT), "invalid_argument") == 0,
          "invalid_argument");
    CHECK(strcmp(genie_status_string(GENIE_ERROR_BACKEND_UNAVAILABLE), "backend_unavailable") == 0,
          "backend_unavailable");
    /* 未知码也不能崩。 */
    CHECK(genie_status_string((genie_status)999) != NULL, "unknown status has a string");
}

static void test_null_arguments(void) {
    genie_engine* engine = NULL;
    CHECK(genie_engine_create(NULL, &engine) == GENIE_ERROR_INVALID_ARGUMENT,
          "create(NULL) is invalid argument");
    CHECK(engine == NULL, "out handle stays NULL on failure");
    CHECK(genie_engine_create("", &engine) == GENIE_ERROR_INVALID_ARGUMENT,
          "create(\"\") is invalid argument");
    CHECK(genie_engine_create("x", NULL) == GENIE_ERROR_INVALID_ARGUMENT,
          "create(x, NULL) is invalid argument");

    /* 传 NULL 引擎不能崩。 */
    float* samples = NULL;
    size_t count = 123;
    int rate = 0;
    CHECK(genie_engine_synthesize(NULL, "hi", GENIE_LANGUAGE_CHINESE, 0, &samples, &count, &rate) ==
              GENIE_ERROR_INVALID_ARGUMENT,
          "synthesize(NULL engine) is invalid argument");
    CHECK(samples == NULL, "samples stays NULL");
    CHECK(count == 0, "count is zeroed");

    /* 销毁 NULL 是无操作。 */
    genie_engine_destroy(NULL);
}

static void test_phonemize(void) {
    int64_t* ids = NULL;
    size_t count = 0;
    const char* text = "\xE4\xBD\xA0\xE5\xA5\xBD"; /* 你好 */

    const genie_status status = genie_phonemize(text, GENIE_LANGUAGE_CHINESE, &ids, &count);
    CHECK(status == GENIE_OK, "phonemize succeeds");
    CHECK(count > 0, "phonemize produced phonemes");
    if (ids != NULL && count > 0) {
        printf("phonemes: %zu\n", count);
        genie_free(ids);
    }

    /* 不受支持的语言 → UNSUPPORTED，不是崩溃。 */
    int64_t* bad = NULL;
    size_t bad_count = 0;
    CHECK(genie_phonemize("hello", GENIE_LANGUAGE_ENGLISH, &bad, &bad_count) ==
              GENIE_ERROR_UNSUPPORTED,
          "english is not supported yet");
    CHECK(bad == NULL, "no buffer on failure");

    /* NULL 参数。 */
    CHECK(genie_phonemize(NULL, GENIE_LANGUAGE_CHINESE, &bad, &bad_count) ==
              GENIE_ERROR_INVALID_ARGUMENT,
          "phonemize(NULL) is invalid argument");
}

static void test_engine_create_without_backend(void) {
    /* 没有 ONNX Runtime 时创建引擎应当报 BackendUnavailable，而不是崩。 */
    genie_engine* engine = NULL;
    const genie_status status = genie_engine_create("/nonexistent/voice-package", &engine);
    CHECK(status != GENIE_OK, "create on a bogus path fails");
    CHECK(engine == NULL, "no engine on failure");
    printf("create status: %d (%s)\n", (int)status, genie_status_string(status));
}

int main(void) {
    step("start");
    test_version();
    step("version");
    test_status_string();
    step("status_string");
    test_null_arguments();
    step("null_arguments");
    test_phonemize();
    step("phonemize");
    test_engine_create_without_backend();
    step("engine_create");

    if (g_failures != 0) {
        printf("\n%d check(s) FAILED\n", g_failures);
        return 1;
    }
    printf("\nall checks passed\n");
    return 0;
}
