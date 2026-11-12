// tools/bench/main.cpp —— genie-bench 性能基准工具
//
// 用途：复现性地测量一条合成链路的分阶段耗时（roadmap M0.3 / M7）。
//
// 与 CLI `synth --timing` 的区别：synth --timing 是"跑一次、看一眼"，
// 这里是"warmup 消抖 + N 次重复 + 统计汇总（mean/min/max/median） +
// RTF + 峰值 RSS + CPU 核占用"，数字可以直接写进 docs/performance.md。
//
// 计时数据全部来自埋点（SynthesisResult.timing / Engine::load_timing），
// 本工具只做采样与统计，不自己插桩。
//
// JSON 输出自包含采集环境（ORT 版本 / OS / CPU 型号）、每 run 原始样本
// 与解码步数：采样在 ONNX 图内完成、不可复现，每次运行的语义 token 数
// 都不同，跨版本对比必须按 ms/token 归一（docs/performance.md）。

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
// windows.h **必须**先于 psapi.h（后者依赖前者定义的 BOOL / DWORD）。
// clang-format 会按字母序重排同一个 include 块（它曾把 psapi.h 排到
// windows.h 前面，Windows CI 因此红过一次）—— 所以这里拆成两块。
#include <windows.h>

#include <psapi.h>
#else
#include <sys/resource.h>
#include <sys/utsname.h>
#endif

#include "genie/audio.hpp"
#include "genie/config.hpp"
#include "genie/engine.hpp"
#include "genie/errors.hpp"
#include "genie/timing.hpp"
#include "genie/version.hpp"

namespace {

constexpr int kExitOk = 0;
constexpr int kExitError = 1;
constexpr int kExitUsage = 2;

// ------------------------------------------------------------------------------------------
// 内置样例文本：短 / 中 / 长三档，覆盖"调用开销为主"与"解码长度为主"
// 两种工况。--text 可覆盖（可重复传入，逐条独立测量）。
//
// 默认样例保持**纯中文**是为了与既有基线（docs/performance.md）可比。
// 混读链路的缺陷已修复（#34），需要时用
// `--language hybrid-chinese-english --text ...` 单独测混读工况。
// ------------------------------------------------------------------------------------------
const std::vector<std::string> kDefaultTexts = {
    "今天天气真不错。",
    "今天天气真不错，我们出去散步吧，顺便去公园里看看花。",
    "二零二五年秋天，原生推理引擎完成了从脚本语言到本地机器码的迁移，"
    "它不依赖任何在线服务，在一台普通的台式机上就能实时合成出自然的语音。",
};

void print_usage() {
    std::cout << R"(genie-bench — per-stage TTS benchmark (M0.3 baseline)

Usage:
  genie-bench --model <dir> [options]

Options:
  --model <dir>     Voice package directory (contains manifest.json)
  --text <text>     Text to synthesize (repeatable; defaults to 3 built-in samples)
  --language <lang> auto | chinese | english | japanese | korean | hybrid-chinese-english
  --threads <n>     ORT intra-op threads per session (default 8)
  --spinning <0|1>  ORT intra-op spin-wait (session.intra_op.allow_spinning, default 0)
  --graph-opt <n>   Graph optimization level 0-3 (default 3 = ORT_ENABLE_ALL)
  --arena <0|1>     ORT CPU memory arena (default 1)
  --mem-pattern <0|1> ORT memory pattern (default 1)
  --parallel-prepack <0|1> Pre-pack weights across the intra-op pool while
                    creating sessions (session.prepack.enable_parallel, default 0)
  --no-verify       Skip voice package SHA-256 verification (faster cold start)
  --warmup <n>      Warmup runs per text, excluded from stats (default 1)
  --runs <n>        Measured runs per text (default 5)
  --json <file>     Also write the aggregate as JSON to this path
  --io-binding <0|1> Bind decoder outputs to pooled buffers (M5 round 2;
                    session learns output-shape growth; identical results).
  --profile <dir>   Enable the ORT op profiler, writing per-session Chrome
                    traces into <dir> (summarize with
                    tools/bench/summarize_profile.py). Timings under
                    profiling are NOT a baseline — analysis only.
  --stream          Streaming benchmark (M6): measures TTFA (time to first
                    audio chunk), chunk pacing and total drain time instead
                    of per-stage sync timings.
  --segment-chars <n>  StreamOptions.max_segment_chars (default 100; 0 =
                    no segmentation — one chunk, legacy behavior)
  --silence-ms <n>  Silence appended to non-final chunks (default 150;
                    0 = none)
  --save <file>     (with --stream) write the first measured run's audio
                    as 16-bit PCM WAV (listening check for chunk joins)
  -h, --help        Show this help

Notes:
  The JSON includes the collection environment (ORT version / OS / CPU),
  per-run samples and decode iterations. Per-text failures do not change
  the exit code — check the "errors" count and "error" fields.
)";
}

struct Options {
    std::string model_dir;
    std::string language = "auto";
    std::string json_path;
    std::string profile_dir;
    std::vector<std::string> texts;
    int threads = 8;
    int spinning = 0;
    int graph_opt = 3;
    int arena = 1;
    int mem_pattern = 1;
    int parallel_prepack = 0;
    int io_binding = 0;
    int warmup = 1;
    int runs = 5;
    bool no_verify = false;
    bool help = false;
    // 流式模式（M6）
    bool stream = false;
    int segment_chars = 100;
    int silence_ms = 150;
    std::string save_wav;
};

bool parse_options(int argc, char** argv, Options& options) {
    if (argc < 2) {
        return false;
    }
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](std::string& target) {
            if (i + 1 >= argc) {
                return false;
            }
            target = argv[++i];
            return true;
        };
        if (arg == "-h" || arg == "--help") {
            options.help = true;
        } else if (arg == "--model") {
            if (!next(options.model_dir)) {
                return false;
            }
        } else if (arg == "--text") {
            std::string text;
            if (!next(text)) {
                return false;
            }
            options.texts.push_back(text);
        } else if (arg == "--language") {
            if (!next(options.language)) {
                return false;
            }
        } else if (arg == "--threads" || arg == "--spinning" || arg == "--graph-opt" ||
                   arg == "--arena" || arg == "--mem-pattern" || arg == "--parallel-prepack" ||
                   arg == "--io-binding" || arg == "--warmup" || arg == "--runs") {
            std::string value;
            if (!next(value)) {
                return false;
            }
            int parsed = std::atoi(value.c_str());
            if (arg == "--threads") {
                options.threads = parsed;
            } else if (arg == "--spinning") {
                options.spinning = parsed;
            } else if (arg == "--graph-opt") {
                options.graph_opt = parsed;
            } else if (arg == "--arena") {
                options.arena = parsed;
            } else if (arg == "--mem-pattern") {
                options.mem_pattern = parsed;
            } else if (arg == "--parallel-prepack") {
                options.parallel_prepack = parsed;
            } else if (arg == "--io-binding") {
                options.io_binding = parsed;
            } else if (arg == "--warmup") {
                options.warmup = parsed;
            } else {
                options.runs = parsed;
            }
        } else if (arg == "--no-verify") {
            options.no_verify = true;
        } else if (arg == "--stream") {
            options.stream = true;
        } else if (arg == "--segment-chars" || arg == "--silence-ms") {
            std::string value;
            if (!next(value)) {
                return false;
            }
            const int parsed = std::atoi(value.c_str());
            if (parsed < 0) {
                std::cerr << "--segment-chars/--silence-ms must be >= 0\n";
                return false;
            }
            if (arg == "--segment-chars") {
                options.segment_chars = parsed;
            } else {
                options.silence_ms = parsed;
            }
        } else if (arg == "--save") {
            if (!next(options.save_wav)) {
                return false;
            }
        } else if (arg == "--json") {
            if (!next(options.json_path)) {
                return false;
            }
        } else if (arg == "--profile") {
            if (!next(options.profile_dir)) {
                return false;
            }
        } else {
            std::cerr << "unknown option: " << arg << "\n";
            return false;
        }
    }
    return true;
}

// ------------------------------------------------------------------------------------------
// 系统指标。读不到时返回 -1，输出里显示 n/a —— 工具必须在所有 CI
// 平台上可编译、可运行（psapi.lib 在 tools/CMakeLists.txt 里按平台链接）。
// ------------------------------------------------------------------------------------------

/// 进程峰值 RSS（KB）。
double peak_rss_kb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS counters{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
        return static_cast<double>(counters.PeakWorkingSetSize) / 1024.0;
    }
    return -1.0;
#else
    std::ifstream status("/proc/self/status");
    if (!status) {
        return -1.0;
    }
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmHWM:", 0) == 0) {
            // 形如 "VmHWM:    1234 kB"
            long kb = 0;
            if (std::sscanf(line.c_str(), "VmHWM: %ld", &kb) == 1) {
                return static_cast<double>(kb);
            }
        }
    }
    return -1.0;
#endif
}

/// 进程当前 RSS（KB）—— 峰值之外的另一半：加载完成后的常驻量才是
/// 长期占用，峰值只说明瞬时需求。
double current_rss_kb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS counters{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
        return static_cast<double>(counters.WorkingSetSize) / 1024.0;
    }
    return -1.0;
#else
    std::ifstream status("/proc/self/status");
    if (!status) {
        return -1.0;
    }
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            long kb = 0;
            if (std::sscanf(line.c_str(), "VmRSS: %ld", &kb) == 1) {
                return static_cast<double>(kb);
            }
        }
    }
    return -1.0;
#endif
}

/// 进程累计 CPU 时间（user + sys，秒）。
double process_cpu_seconds() {
#ifdef _WIN32
    FILETIME create_time{};
    FILETIME exit_time{};
    FILETIME kernel_time{};
    FILETIME user_time{};
    if (GetProcessTimes(GetCurrentProcess(), &create_time, &exit_time, &kernel_time, &user_time)) {
        auto to_seconds = [](const FILETIME& ft) {
            ULARGE_INTEGER value;
            value.LowPart = ft.dwLowDateTime;
            value.HighPart = ft.dwHighDateTime;
            return static_cast<double>(value.QuadPart) * 1e-7;  // 100 ns 粒度
        };
        return to_seconds(kernel_time) + to_seconds(user_time);
    }
    return -1.0;
#else
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    const double user = usage.ru_utime.tv_sec + usage.ru_utime.tv_usec * 1e-6;
    const double sys = usage.ru_stime.tv_sec + usage.ru_stime.tv_usec * 1e-6;
    return user + sys;
#endif
}

// ------------------------------------------------------------------------------------------
// 环境元数据。基准数字只在同代硬件 + 同 ORT 版本上可比（docs/performance.md
// 开头的告诫），所以 JSON 记录必须自包含采集环境。读不到的字段写
// "unknown" —— 工具必须在所有 CI 平台上可编译、可运行。
// ------------------------------------------------------------------------------------------

/// OS 与内核标识（如 "Linux 6.8.0-124-generic x86_64"）。
std::string os_identifier() {
#if defined(_WIN32)
    // Windows 没有轻量可移植的版本查询（GetVersionEx 已废弃），先只写平台名。
    return "Windows";
#else
    utsname info{};
    if (uname(&info) != 0) {
        return "unknown";
    }
    return std::string(info.sysname) + " " + info.release + " " + info.machine;
#endif
}

/// CPU 型号（如 "13th Gen Intel(R) Core(TM) i9-13900KF"）。
///
/// 只实现了 Linux 的 /proc/cpuinfo 解析。bench 不在 Windows CI 里跑，
/// 真要采 Windows 基线时再补 cpuid brand string，不为它提前写平台分支。
std::string cpu_model() {
#if !defined(_WIN32)
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        if (line.rfind("model name", 0) != 0) {
            continue;
        }
        const auto sep = line.find(':');
        if (sep == std::string::npos) {
            continue;
        }
        const std::string value = line.substr(sep + 1);
        const auto first = value.find_first_not_of(" \t");
        const auto last = value.find_last_not_of(" \t\r\n");
        if (first != std::string::npos) {
            return value.substr(first, last - first + 1);
        }
    }
#endif
    return "unknown";
}

// ------------------------------------------------------------------------------------------
// 统计与格式化。
// ------------------------------------------------------------------------------------------

struct Stats {
    double mean = 0.0;
    double min = 0.0;
    double max = 0.0;
    double median = 0.0;
};

Stats stats_of(std::vector<double> values) {
    Stats stats{};
    if (values.empty()) {
        return stats;
    }
    std::sort(values.begin(), values.end());
    stats.min = values.front();
    stats.max = values.back();
    double sum = 0.0;
    for (const double value : values) {
        sum += value;
    }
    stats.mean = sum / static_cast<double>(values.size());
    const std::size_t mid = values.size() / 2;
    stats.median = values.size() % 2 == 1 ? values[mid] : (values[mid - 1] + values[mid]) / 2.0;
    return stats;
}

std::string json_escape(const std::string& text) {
    std::string out;
    for (const char ch : text) {
        switch (ch) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (static_cast<unsigned char>(ch) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04X", ch);
                    out += buf;
                } else {
                    out += ch;
                }
        }
    }
    return out;
}

/// 单次运行的原始样本。
///
/// 汇总统计会抹掉分布信息，而对比优化前后恰恰要看分布（方差、离群
/// run）。iterations 尤其关键：采样在 ONNX 图内、不可复现，每次运行
/// 的语义 token 数都不同，总耗时里混着生成长度的随机波动 —— 不记录
/// 步数就无法按 ms/token 归一，跨版本对比无从谈起。
struct RunSample {
    double wall_ms = 0.0;
    double rtf = 0.0;
    double cpu_cores = -1.0;  // (user+sys) / wall；不可用时为负
    /// t2s/stage_decoder 的解码步数（== 最终语义 token 数）。
    std::uint32_t iterations = 0;
    /// 本次运行各阶段毫秒数（按阶段首见顺序）。
    std::vector<std::pair<std::string, double>> stage_ms;
};

/// 一条文本的聚合结果：每个阶段一组采样（按首次出现顺序），加上
/// 一组运行级原始样本。
struct TextAggregate {
    std::string text;
    /// 阶段名 → 各次运行的毫秒数。用 vector 保持首见顺序（map 会按字典序打乱，
    /// 与 CLI --timing 的阶段顺序不一致）。
    std::vector<std::pair<std::string, std::vector<double>>> stages;
    std::vector<double> audio_seconds;
    std::vector<RunSample> samples;
    std::string error;  // 非空表示这条文本合成失败
};

/// 抽取样本的某个数值字段，喂给 stats_of。
template <typename Member>
std::vector<double> sample_values(const std::vector<RunSample>& samples, Member member) {
    std::vector<double> values;
    values.reserve(samples.size());
    for (const auto& sample : samples) {
        values.push_back(static_cast<double>(sample.*member));
    }
    return values;
}

void collect_run(TextAggregate& aggregate, const genie::SynthesisResult& result,
                 const double wall_seconds, const double cpu_seconds) {
    RunSample sample;
    sample.wall_ms = wall_seconds * 1000.0;
    const double audio = result.audio.duration_seconds();
    sample.rtf = audio > 0.0 ? wall_seconds / audio : 0.0;
    sample.cpu_cores = wall_seconds > 0.0 && cpu_seconds >= 0.0 ? cpu_seconds / wall_seconds : -1.0;

    for (const auto& stage : result.timing.stages()) {
        bool found = false;
        for (auto& entry : aggregate.stages) {
            if (entry.first == stage.name) {
                entry.second.push_back(stage.milliseconds);
                found = true;
                break;
            }
        }
        if (!found) {
            aggregate.stages.emplace_back(stage.name, std::vector<double>{stage.milliseconds});
        }
        sample.stage_ms.emplace_back(stage.name, stage.milliseconds);
        if (stage.name == "t2s/stage_decoder") {
            sample.iterations = stage.count;
        }
    }
    aggregate.audio_seconds.push_back(audio);
    aggregate.samples.push_back(std::move(sample));
}

void print_aggregate(const TextAggregate& aggregate, const Options& options) {
    std::cout << "  " << options.runs << " runs, warmup " << options.warmup << "\n";

    // 阶段表：与 CLI --timing 同款布局，加统计列。
    std::cout << "  stage                            mean      min      max   median\n";
    for (const auto& [name, samples] : aggregate.stages) {
        const Stats stats = stats_of(samples);
        char line[160];
        const bool child = name.find('/') != std::string::npos;
        const std::string label = (child ? "    " : "  ") + name;
        std::snprintf(line, sizeof(line), "%-34s %8.2f %8.2f %8.2f %8.2f\n", label.c_str(),
                      stats.mean, stats.min, stats.max, stats.median);
        std::cout << line;
    }

    const Stats wall_stats = stats_of(sample_values(aggregate.samples, &RunSample::wall_ms));
    char line[160];
    std::snprintf(line, sizeof(line), "  %-34s %8.2f %8.2f %8.2f %8.2f\n", "  total(wall)",
                  wall_stats.mean, wall_stats.min, wall_stats.max, wall_stats.median);
    std::cout << line;

    const Stats rtf_stats = stats_of(sample_values(aggregate.samples, &RunSample::rtf));
    std::snprintf(line, sizeof(line),
                  "  RTF (wall / audio, lower is better): mean %.3f  min %.3f  median %.3f  max "
                  "%.3f\n",
                  rtf_stats.mean, rtf_stats.min, rtf_stats.median, rtf_stats.max);
    std::cout << line;

    // 解码步数与 ms/token：跨版本对比的主指标。stage_decoder 的总耗时
    // 混着"这次生成了多少 token"的随机性，只有除以步数才可比。
    const Stats iter_stats = stats_of(sample_values(aggregate.samples, &RunSample::iterations));
    if (iter_stats.mean > 0.0) {
        std::snprintf(line, sizeof(line), "  decode iterations: mean %.1f  min %.0f  max %.0f\n",
                      iter_stats.mean, iter_stats.min, iter_stats.max);
        std::cout << line;
        const auto decoder =
            std::find_if(aggregate.stages.begin(), aggregate.stages.end(),
                         [](const auto& entry) { return entry.first == "t2s/stage_decoder"; });
        if (decoder != aggregate.stages.end()) {
            const Stats decoder_stats = stats_of(decoder->second);
            std::snprintf(line, sizeof(line), "  stage_decoder ms/token: mean %.2f\n",
                          decoder_stats.mean / iter_stats.mean);
            std::cout << line;
        }
    }

    const Stats audio_stats = stats_of(aggregate.audio_seconds);
    const Stats cpu_stats = stats_of(sample_values(aggregate.samples, &RunSample::cpu_cores));
    if (!aggregate.samples.empty() && aggregate.samples.front().cpu_cores >= 0.0) {
        std::snprintf(line, sizeof(line), "  audio: mean %.2f s   cpu cores used: mean %.1f\n",
                      audio_stats.mean, cpu_stats.mean);
    } else {
        std::snprintf(line, sizeof(line), "  audio: mean %.2f s   cpu cores used: n/a\n",
                      audio_stats.mean);
    }
    std::cout << line;
}

/// 数值字段的 JSON 片段。变量长度的行（samples / 阶段对象）不用单个
/// 大 snprintf 拼 —— 缓冲截断会产出"看起来正常"的坏 JSON。
std::string json_number(double value, const char* format) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), format, value);
    return buf;
}

/// 内存快照（KB；读不到为负，JSON 里写 null）。
struct MemorySnapshot {
    double peak_kb = -1.0;
    double after_load_kb = -1.0;
    double end_kb = -1.0;
};

std::string to_json(const std::vector<TextAggregate>& aggregates, const Options& options,
                    const MemorySnapshot& memory, const genie::TimingReport& load) {
    // 固定长度的统计行（名字 + 四个统计值 + 缩进）能超过 120 字节 ——
    // 缓冲必须留足。
    static constexpr int kJsonBufSize = 512;
    char number[kJsonBufSize];

    std::size_t error_count = 0;
    for (const auto& aggregate : aggregates) {
        if (!aggregate.error.empty()) {
            ++error_count;
        }
    }

    std::string out;
    out += "{\n";
    out += "  \"model\": \"" + json_escape(options.model_dir) + "\",\n";
    out += "  \"version\": \"" + std::string(genie::version_string()) + "\",\n";
    // 采集环境：基准数字只在同硬件 + 同 ORT 版本上可比（ORT 版本对齐
    // 是上游钉死 1.22.1 的原因，见 docs/upstream-audit.md §7）。
    out += "  \"environment\": {\"ort\": \"" + std::string(genie::onnxruntime_version_string()) +
           "\", \"os\": \"" + json_escape(os_identifier()) + "\", \"cpu\": \"" +
           json_escape(cpu_model()) + "\"},\n";
    std::snprintf(number, sizeof(number),
                  "  \"threads\": %d, \"spinning\": %d, \"graph_opt\": %d, \"arena\": %d, "
                  "\"mem_pattern\": %d, \"parallel_prepack\": %d, \"io_binding\": %d, "
                  "\"runs\": %d, \"warmup\": %d,\n",
                  options.threads, options.spinning, options.graph_opt, options.arena,
                  options.mem_pattern, options.parallel_prepack, options.io_binding, options.runs,
                  options.warmup);
    out += number;
    const auto kb_field = [&out, &number](const char* name, const double kb) {
        if (kb < 0.0) {
            out += std::string("  \"") + name + "\": null,\n";
        } else {
            std::snprintf(number, sizeof(number), "  \"%s\": %.0f,\n", name, kb);
            out += number;
        }
    };
    kb_field("peak_rss_kb", memory.peak_kb);
    kb_field("rss_after_load_kb", memory.after_load_kb);
    kb_field("rss_end_kb", memory.end_kb);
    out += "  \"errors\": " + std::to_string(error_count) + ",\n";
    // 冷启动：是否跳过包校验必须进记录（它是冷启动里最大的一块可开关
    // 开销）；加载的分阶段耗时随之落盘 —— 只打在终端上的表没法事后比对。
    out += std::string("  \"verify_voice_package\": ") + (options.no_verify ? "false" : "true") +
           ",\n";
    // profiler 开关同样进记录：开着 profiler 的计时数字不可当基线。
    out += "  \"profile_output_dir\": \"" + json_escape(options.profile_dir) + "\",\n";
    out += "  \"load\": " + load.to_json() + ",\n";
    out += "  \"texts\": [\n";
    for (std::size_t i = 0; i < aggregates.size(); ++i) {
        const auto& aggregate = aggregates[i];
        out += "    {\"text\": \"" + json_escape(aggregate.text) + "\", ";
        if (!aggregate.error.empty()) {
            out += "\"error\": \"" + json_escape(aggregate.error) + "\"";
        } else {
            const Stats rtf_stats = stats_of(sample_values(aggregate.samples, &RunSample::rtf));
            std::snprintf(number, sizeof(number),
                          "\"rtf\": {\"mean\": %.3f, \"min\": %.3f, \"median\": %.3f, \"max\": "
                          "%.3f}, ",
                          rtf_stats.mean, rtf_stats.min, rtf_stats.median, rtf_stats.max);
            out += number;
            const Stats audio_stats = stats_of(aggregate.audio_seconds);
            std::snprintf(number, sizeof(number), "\"audio_seconds\": {\"mean\": %.3f}, ",
                          audio_stats.mean);
            out += number;
            const Stats wall_stats =
                stats_of(sample_values(aggregate.samples, &RunSample::wall_ms));
            std::snprintf(number, sizeof(number),
                          "\"total_wall_ms\": {\"mean\": %.1f, \"min\": %.1f, \"max\": %.1f}, ",
                          wall_stats.mean, wall_stats.min, wall_stats.max);
            out += number;
            const Stats cpu_stats =
                stats_of(sample_values(aggregate.samples, &RunSample::cpu_cores));
            if (!aggregate.samples.empty() && aggregate.samples.front().cpu_cores >= 0.0) {
                std::snprintf(number, sizeof(number), "\"cpu_cores\": {\"mean\": %.2f}, ",
                              cpu_stats.mean);
                out += number;
            }
            const Stats iter_stats =
                stats_of(sample_values(aggregate.samples, &RunSample::iterations));
            if (iter_stats.mean > 0.0) {
                std::snprintf(number, sizeof(number),
                              "\"iterations\": {\"mean\": %.1f, \"min\": %.0f, \"max\": %.0f}, ",
                              iter_stats.mean, iter_stats.min, iter_stats.max);
                out += number;
            }
            out += "\"stages\": [\n";
            for (std::size_t s = 0; s < aggregate.stages.size(); ++s) {
                const auto& [name, samples] = aggregate.stages[s];
                const Stats stats = stats_of(samples);
                std::snprintf(number, sizeof(number),
                              "      {\"name\": \"%s\", \"ms\": {\"mean\": %.3f, \"min\": %.3f, "
                              "\"max\": %.3f, \"median\": %.3f}}%s\n",
                              name.c_str(), stats.mean, stats.min, stats.max, stats.median,
                              s + 1 < aggregate.stages.size() ? "," : "");
                out += number;
            }
            out += "    ],\n    \"samples\": [\n";
            for (std::size_t s = 0; s < aggregate.samples.size(); ++s) {
                const auto& sample = aggregate.samples[s];
                std::string stages_obj = "{";
                for (std::size_t k = 0; k < sample.stage_ms.size(); ++k) {
                    stages_obj += "\"" + sample.stage_ms[k].first +
                                  "\": " + json_number(sample.stage_ms[k].second, "%.3f") +
                                  (k + 1 < sample.stage_ms.size() ? ", " : "");
                }
                stages_obj += "}";
                // 阶段数可变，逐段拼接而不是塞进同一个 snprintf 缓冲。
                out += "      {\"wall_ms\": " + json_number(sample.wall_ms, "%.1f") +
                       ", \"rtf\": " + json_number(sample.rtf, "%.4f") +
                       ", \"cpu_cores\": " + json_number(sample.cpu_cores, "%.2f") +
                       ", \"iterations\": " + std::to_string(sample.iterations) +
                       ", \"stages\": " + stages_obj + "}";
                out += s + 1 < aggregate.samples.size() ? ",\n" : "\n";
            }
            out += "    ]";
        }
        out += i + 1 < aggregates.size() ? "},\n" : "}\n";
    }
    out += "  ]\n}\n";
    return out;
}

// ------------------------------------------------------------------------------------------
// 流式基准（M6：--stream）。
//
// 与同步模式分开统计：流式的核心指标不是"总耗时"，而是首块延迟（TTFA）
// 与块到达节奏（最大间隔 = 欠载风险的观察窗口）。计时从
// synthesize_streaming() 调用前一刻起算 —— 提交进调度器的排队时间也算在
// 内，那是消费者视角的真实首块延迟。
// ------------------------------------------------------------------------------------------

struct StreamRunSample {
    double ttfa_ms = 0.0;   // 首块到达（含排队）
    double total_ms = 0.0;  // 提交 → 排空关闭
    double rtf = 0.0;       // (total_ms / 1000) / audio_seconds
    double audio_seconds = 0.0;
    int chunks = 0;
    double max_gap_ms = 0.0;       // 相邻块到达间隔的最大值；单块流为 0
    std::uint32_t iterations = 0;  // 全流 t2s/stage_decoder 步数（footer）
};

struct StreamAggregate {
    std::string text;
    std::vector<StreamRunSample> samples;
    std::string error;
};

/// 排空一条流并按到达时刻采样。返回 (ttfa_ms, arrivals, audio)。
struct StreamDrain {
    double ttfa_ms = -1.0;
    double total_ms = 0.0;
    double audio_seconds = 0.0;
    std::vector<double> arrivals_ms;
    std::vector<float> recorded;  // --save 时收集波形；nullptr 时不收集
};

StreamDrain drain_stream_channel(const std::shared_ptr<genie::AudioChunkChannel>& channel,
                                 const std::chrono::steady_clock::time_point submit_start,
                                 std::vector<float>* recorded) {
    StreamDrain drain;
    genie::AudioBuffer chunk;
    for (;;) {
        bool got = channel->try_pop(chunk);
        if (!got && channel->closed()) {
            // 关闭之后队列里可能还有残余块 —— 排空为止。
            got = channel->try_pop(chunk);
            if (!got) {
                break;
            }
        }
        if (!got) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - submit_start)
                              .count();
        if (drain.ttfa_ms < 0.0) {
            drain.ttfa_ms = ms;
        }
        drain.arrivals_ms.push_back(ms);
        drain.audio_seconds += chunk.duration_seconds();
        if (recorded != nullptr) {
            recorded->insert(recorded->end(), chunk.samples.begin(), chunk.samples.end());
        }
    }
    drain.total_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - submit_start)
            .count();
    return drain;
}

/// 跑一次流式合成并采样。footer 里的错误以异常上抛（与同步路径的
/// catch 汇合）。
StreamRunSample run_stream_once(genie::Engine& engine, const std::string& text,
                                const genie::SynthesisOptions& synthesis,
                                const genie::StreamOptions& stream_options,
                                std::vector<float>* recorded) {
    StreamRunSample sample;
    const auto submit_start = std::chrono::steady_clock::now();
    const auto channel = engine.synthesize_streaming(text, synthesis, stream_options);
    const auto drain = drain_stream_channel(channel, submit_start, recorded);

    if (const auto* footer = channel->footer()) {
        if (footer->error.has_value()) {
            throw *footer->error;
        }
        for (const auto& stage : footer->timing.stages()) {
            if (stage.name == "t2s/stage_decoder") {
                sample.iterations = stage.count;
            }
        }
    }

    sample.ttfa_ms = drain.ttfa_ms;
    sample.total_ms = drain.total_ms;
    sample.audio_seconds = drain.audio_seconds;
    sample.chunks = static_cast<int>(drain.arrivals_ms.size());
    for (std::size_t i = 1; i < drain.arrivals_ms.size(); ++i) {
        sample.max_gap_ms =
            std::max(sample.max_gap_ms, drain.arrivals_ms[i] - drain.arrivals_ms[i - 1]);
    }
    sample.rtf = drain.audio_seconds > 0.0 ? (drain.total_ms / 1000.0) / drain.audio_seconds : 0.0;
    return sample;
}

template <typename Member>
std::vector<double> stream_sample_values(const std::vector<StreamRunSample>& samples,
                                         Member member) {
    std::vector<double> values;
    values.reserve(samples.size());
    for (const auto& sample : samples) {
        values.push_back(sample.*member);
    }
    return values;
}

void print_stream_aggregate(const StreamAggregate& aggregate, const Options& options) {
    std::cout << "  " << options.runs << " runs, warmup " << options.warmup
              << " (stream, segment-chars " << options.segment_chars << ", silence-ms "
              << options.silence_ms << ")\n";
    char line[160];

    const Stats ttfa = stats_of(stream_sample_values(aggregate.samples, &StreamRunSample::ttfa_ms));
    std::snprintf(line, sizeof(line),
                  "  TTFA (time to first chunk, ms): mean %8.2f  min %8.2f  median %8.2f  max "
                  "%8.2f\n",
                  ttfa.mean, ttfa.min, ttfa.median, ttfa.max);
    std::cout << line;

    const Stats total =
        stats_of(stream_sample_values(aggregate.samples, &StreamRunSample::total_ms));
    std::snprintf(line, sizeof(line),
                  "  total (submit -> drained, ms):  mean %8.2f  min %8.2f  max %8.2f\n",
                  total.mean, total.min, total.max);
    std::cout << line;

    const Stats rtf = stats_of(stream_sample_values(aggregate.samples, &StreamRunSample::rtf));
    std::snprintf(line, sizeof(line),
                  "  RTF (total / audio): mean %.3f  min %.3f  median %.3f  max %.3f\n", rtf.mean,
                  rtf.min, rtf.median, rtf.max);
    std::cout << line;

    const Stats chunks =
        stats_of(stream_sample_values(aggregate.samples, &StreamRunSample::chunks));
    std::snprintf(line, sizeof(line), "  chunks per run: mean %.1f\n", chunks.mean);
    std::cout << line;

    const Stats gaps =
        stats_of(stream_sample_values(aggregate.samples, &StreamRunSample::max_gap_ms));
    std::snprintf(line, sizeof(line),
                  "  max inter-chunk gap (ms): mean %.2f  max %.2f  (underrun-risk window)\n",
                  gaps.mean, gaps.max);
    std::cout << line;

    const Stats audio =
        stats_of(stream_sample_values(aggregate.samples, &StreamRunSample::audio_seconds));
    std::snprintf(line, sizeof(line), "  audio: mean %.2f s\n", audio.mean);
    std::cout << line;

    const Stats iter =
        stats_of(stream_sample_values(aggregate.samples, &StreamRunSample::iterations));
    if (iter.mean > 0.0) {
        std::snprintf(line, sizeof(line), "  decode iterations (whole stream): mean %.1f\n",
                      iter.mean);
        std::cout << line;
    }
}

std::string to_stream_json(const std::vector<StreamAggregate>& aggregates, const Options& options,
                           const MemorySnapshot& memory, const genie::TimingReport& load) {
    static constexpr int kJsonBufSize = 512;
    char number[kJsonBufSize];

    std::size_t error_count = 0;
    for (const auto& aggregate : aggregates) {
        if (!aggregate.error.empty()) {
            ++error_count;
        }
    }

    std::string out;
    out += "{\n";
    out += "  \"mode\": \"stream\",\n";
    out += "  \"model\": \"" + json_escape(options.model_dir) + "\",\n";
    out += "  \"version\": \"" + std::string(genie::version_string()) + "\",\n";
    out += "  \"environment\": {\"ort\": \"" + std::string(genie::onnxruntime_version_string()) +
           "\", \"os\": \"" + json_escape(os_identifier()) + "\", \"cpu\": \"" +
           json_escape(cpu_model()) + "\"},\n";
    std::snprintf(number, sizeof(number),
                  "  \"threads\": %d, \"spinning\": %d, \"graph_opt\": %d, \"arena\": %d, "
                  "\"mem_pattern\": %d, \"parallel_prepack\": %d, \"io_binding\": %d, "
                  "\"runs\": %d, \"warmup\": %d, \"max_segment_chars\": %d, "
                  "\"sentence_silence_ms\": %d,\n",
                  options.threads, options.spinning, options.graph_opt, options.arena,
                  options.mem_pattern, options.parallel_prepack, options.io_binding, options.runs,
                  options.warmup, options.segment_chars, options.silence_ms);
    out += number;
    const auto kb_field = [&out, &number](const char* name, const double kb) {
        if (kb < 0.0) {
            out += std::string("  \"") + name + "\": null,\n";
        } else {
            std::snprintf(number, sizeof(number), "  \"%s\": %.0f,\n", name, kb);
            out += number;
        }
    };
    kb_field("peak_rss_kb", memory.peak_kb);
    kb_field("rss_after_load_kb", memory.after_load_kb);
    kb_field("rss_end_kb", memory.end_kb);
    out += "  \"errors\": " + std::to_string(error_count) + ",\n";
    out += std::string("  \"verify_voice_package\": ") + (options.no_verify ? "false" : "true") +
           ",\n";
    out += "  \"load\": " + load.to_json() + ",\n";
    out += "  \"texts\": [\n";
    for (std::size_t i = 0; i < aggregates.size(); ++i) {
        const auto& aggregate = aggregates[i];
        out += "    {\"text\": \"" + json_escape(aggregate.text) + "\", ";
        if (!aggregate.error.empty()) {
            out += "\"error\": \"" + json_escape(aggregate.error) + "\"";
        } else {
            const auto field = [&](const char* name, auto member, const char* format) {
                const Stats stats = stats_of(stream_sample_values(aggregate.samples, member));
                std::snprintf(number, sizeof(number),
                              "\"%s\": {\"mean\": %s, \"min\": %s, "
                              "\"median\": %s, \"max\": %s}",
                              name, json_number(stats.mean, format).c_str(),
                              json_number(stats.min, format).c_str(),
                              json_number(stats.median, format).c_str(),
                              json_number(stats.max, format).c_str());
                out += number;
            };
            field("ttfa_ms", &StreamRunSample::ttfa_ms, "%.2f");
            out += ", ";
            field("total_ms", &StreamRunSample::total_ms, "%.1f");
            out += ", ";
            field("rtf", &StreamRunSample::rtf, "%.4f");
            out += ", ";
            field("chunks", &StreamRunSample::chunks, "%.1f");
            out += ", ";
            field("max_gap_ms", &StreamRunSample::max_gap_ms, "%.2f");
            out += ", ";
            field("audio_seconds", &StreamRunSample::audio_seconds, "%.3f");
            out += ", ";
            field("iterations", &StreamRunSample::iterations, "%.1f");
            out += ", \"samples\": [\n";
            for (std::size_t s = 0; s < aggregate.samples.size(); ++s) {
                const auto& sample = aggregate.samples[s];
                std::snprintf(number, sizeof(number),
                              "      {\"ttfa_ms\": %.2f, \"total_ms\": %.1f, \"rtf\": %.4f, "
                              "\"chunks\": %d, \"max_gap_ms\": %.2f, \"audio_seconds\": %.3f, "
                              "\"iterations\": %u}",
                              sample.ttfa_ms, sample.total_ms, sample.rtf, sample.chunks,
                              sample.max_gap_ms, sample.audio_seconds, sample.iterations);
                out += number;
                out += s + 1 < aggregate.samples.size() ? ",\n" : "\n";
            }
            out += "    ]";
        }
        out += i + 1 < aggregates.size() ? "},\n" : "}\n";
    }
    out += "  ]\n}\n";
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse_options(argc, argv, options)) {
        print_usage();
        return kExitUsage;
    }
    if (options.help) {
        print_usage();
        return kExitOk;
    }
    if (options.model_dir.empty()) {
        std::cerr << "genie-bench requires --model <dir>\n";
        return kExitUsage;
    }
    if (options.runs < 1 || options.warmup < 0 || options.threads < 1 ||
        (options.spinning != 0 && options.spinning != 1) ||
        (options.arena != 0 && options.arena != 1) ||
        (options.mem_pattern != 0 && options.mem_pattern != 1) ||
        (options.parallel_prepack != 0 && options.parallel_prepack != 1) ||
        (options.io_binding != 0 && options.io_binding != 1) || options.graph_opt < 0 ||
        options.graph_opt > 3) {
        std::cerr << "--runs/--threads must be >= 1, --warmup must be >= 0, "
                     "--spinning/--arena/--mem-pattern/--parallel-prepack must be 0 or 1, "
                     "--graph-opt must be 0-3\n";
        return kExitUsage;
    }
    if (!options.save_wav.empty() && !options.stream) {
        std::cerr << "--save requires --stream\n";
        return kExitUsage;
    }
    if (!genie::has_onnxruntime_backend()) {
        std::cerr << "this build has no ONNX Runtime backend — reconfigure with "
                     "-DGENIE_ENABLE_ORT=ON\n";
        return kExitError;
    }
    const auto language = genie::language_from_string(options.language);
    if (!language.has_value()) {
        std::cerr << "unknown language: " << options.language << "\n";
        return kExitUsage;
    }
    if (options.texts.empty()) {
        options.texts = kDefaultTexts;
    }

    const auto features = genie::build_features();
    std::cout << "genie-bench — genie-tts-native " << genie::version_string()
              << " (onnxruntime: " << (features.onnxruntime ? "yes" : "no")
              << ", kairo: " << (features.kairo ? "yes" : "no") << ")\n";

    genie::SynthesisOptions synthesis;
    synthesis.language = *language;

    // 线程预算与 CLI synth 同款：每会话一个保守的 intra-op 预算，
    // 否则 4-6 个会话各开满全部核心，多核机器上互相踩踏。
    genie::EngineConfig config;
    config.session.intra_op_threads = options.threads;
    config.session.allow_spinning = options.spinning != 0;
    // M5.3 参数扫描的另外三个维度（图优化级别 / arena / 内存模式）：
    // SessionConfig 字段早已存在，bench 此前只暴露了线程与自旋。
    config.session.graph_optimization_level = options.graph_opt;
    config.session.enable_cpu_mem_arena = options.arena != 0;
    config.session.enable_mem_pattern = options.mem_pattern != 0;
    config.session.parallel_prepack = options.parallel_prepack != 0;
    config.session.enable_io_binding = options.io_binding != 0;
    if (!options.profile_dir.empty()) {
        // 算子级 profiler（M5 性能分析采集端）：目录自建，trace 按会话
        // （模型文件 stem）分文件落盘，tools/bench/summarize_profile.py 聚合。
        std::filesystem::create_directories(options.profile_dir);
        config.session.profile_output_dir = options.profile_dir;
        std::cerr << "warning: ORT profiler enabled (--profile " << options.profile_dir
                  << ") —— 计时数字含 profiler 开销，仅供算子分布分析，不作基线\n";
    }
    if (options.no_verify) {
        // 基准关心的是"引擎冷启动"，校验是其中最大的一块可开关开销；
        // 跳过与否都必须写进测量记录，不能靠猜。
        config.verify_voice_package = false;
        std::cerr << "warning: skipping voice package SHA-256 verification (--no-verify)\n";
    }

    std::cout << "constructing engine (cold start) from: " << options.model_dir << "\n";
    genie::Engine engine(options.model_dir, config);

    const auto& load_report = engine.load_timing();
    std::cout << "\n[load]\n" << load_report.to_string();
    MemorySnapshot memory;
    memory.after_load_kb = current_rss_kb();
    if (memory.after_load_kb >= 0.0) {
        std::printf("RSS after load: %.0f MB\n", memory.after_load_kb / 1024.0);
    }

    std::vector<TextAggregate> aggregates;
    std::vector<StreamAggregate> stream_aggregates;
    genie::StreamOptions stream_options;
    stream_options.max_segment_chars = static_cast<std::size_t>(options.segment_chars);
    stream_options.sentence_silence_ms = options.silence_ms;
    for (std::size_t i = 0; i < options.texts.size(); ++i) {
        const auto& text = options.texts[i];
        TextAggregate aggregate;
        aggregate.text = text;

        std::cout << "\n--- text " << (i + 1) << "/" << options.texts.size() << " (" << text.size()
                  << " bytes) ---\n";
        std::cout << "  " << text << "\n";

        try {
            if (options.stream) {
                // 预热（与同步同理由：磁盘缓存 / 首次执行路径抖动不计统计）。
                for (int w = 0; w < options.warmup; ++w) {
                    const auto warm_start = std::chrono::steady_clock::now();
                    (void)drain_stream_channel(
                        engine.synthesize_streaming(text, synthesis, stream_options), warm_start,
                        nullptr);
                }

                StreamAggregate aggregate;
                aggregate.text = text;
                for (int r = 0; r < options.runs; ++r) {
                    // --save：第一轮实测的波形落盘（听测分块拼接的边界）。
                    std::vector<float> recorded;
                    std::vector<float>* collect =
                        (r == 0 && !options.save_wav.empty()) ? &recorded : nullptr;
                    aggregate.samples.push_back(
                        run_stream_once(engine, text, synthesis, stream_options, collect));
                    if (collect != nullptr && !collect->empty()) {
                        genie::AudioBuffer wav;
                        wav.samples = std::move(recorded);
                        wav.sample_rate = engine.sample_rate();
                        genie::WavCodec::write_pcm16(options.save_wav, wav);
                        std::cout << "  first-run audio written to " << options.save_wav << "\n";
                    }
                }
                print_stream_aggregate(aggregate, options);
                stream_aggregates.push_back(std::move(aggregate));
                aggregates.push_back(TextAggregate{.text = text});
                continue;
            }

            // 预热：磁盘缓存 / ORT 首次执行路径抖动很大，不计进统计。
            // 预热失败没有继续测的意义，异常直接传出。
            for (int w = 0; w < options.warmup; ++w) {
                (void)engine.synthesize(text, synthesis);
            }

            for (int r = 0; r < options.runs; ++r) {
                const double cpu_before = process_cpu_seconds();
                const auto wall_start = std::chrono::steady_clock::now();

                const auto result = engine.synthesize(text, synthesis);

                const double wall_seconds =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start)
                        .count();
                const double cpu_after = process_cpu_seconds();
                const double cpu_seconds =
                    cpu_before < 0.0 || cpu_after < 0.0 ? -1.0 : cpu_after - cpu_before;
                collect_run(aggregate, result, wall_seconds, cpu_seconds);
            }

            print_aggregate(aggregate, options);
        } catch (const genie::Error& error) {
            aggregate.error = error.what();
            std::cout << "  FAILED: [" << genie::to_string(error.code()) << "] " << error.what()
                      << "\n";
        }
        aggregates.push_back(std::move(aggregate));
    }

    std::size_t error_count = 0;
    for (const auto& aggregate : aggregates) {
        if (!aggregate.error.empty()) {
            ++error_count;
        }
    }
    for (const auto& aggregate : stream_aggregates) {
        if (!aggregate.error.empty()) {
            ++error_count;
        }
    }
    if (error_count > 0) {
        // 退出码保持 0（单条文本失败不影响其他条的测量）—— 但失败必须
        // 在终端上可见，不能只藏在 JSON 的 error 字段里。
        std::cerr << "warning: " << error_count << "/" << aggregates.size()
                  << " text(s) failed to synthesize\n";
    }

    memory.peak_kb = peak_rss_kb();
    memory.end_kb = current_rss_kb();
    if (memory.peak_kb < 0.0) {
        std::cout << "\npeak RSS: n/a\n";
    } else {
        std::printf("\npeak RSS: %.0f MB (%.0f kB), at exit: %.0f MB\n", memory.peak_kb / 1024.0,
                    memory.peak_kb, memory.end_kb / 1024.0);
    }

    if (!options.json_path.empty()) {
        std::ofstream json(options.json_path);
        if (!json) {
            std::cerr << "cannot write " << options.json_path << "\n";
            return kExitError;
        }
        json << (options.stream
                     ? to_stream_json(stream_aggregates, options, memory, engine.load_timing())
                     : to_json(aggregates, options, memory, engine.load_timing()));
        std::cout << "aggregate written to " << options.json_path << "\n";
    }

    return kExitOk;
}
