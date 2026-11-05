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

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <psapi.h>
#include <windows.h>
#else
#include <sys/resource.h>
#endif

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
// 默认样例刻意保持**纯中文**：混读（拉丁字母相邻中文）目前会触发
// bert_chunks 与音素序列对不上的缺陷（见 docs/performance.md 已知问题），
// 基准工具的默认路径必须是绿的。
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
  --warmup <n>      Warmup runs per text, excluded from stats (default 1)
  --runs <n>        Measured runs per text (default 5)
  --json <file>     Also write the aggregate as JSON to this path
  -h, --help        Show this help
)";
}

struct Options {
    std::string model_dir;
    std::string language = "auto";
    std::string json_path;
    std::vector<std::string> texts;
    int threads = 8;
    int warmup = 1;
    int runs = 5;
    bool help = false;
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
        } else if (arg == "--threads" || arg == "--warmup" || arg == "--runs") {
            std::string value;
            if (!next(value)) {
                return false;
            }
            int parsed = std::atoi(value.c_str());
            if (arg == "--threads") {
                options.threads = parsed;
            } else if (arg == "--warmup") {
                options.warmup = parsed;
            } else {
                options.runs = parsed;
            }
        } else if (arg == "--json") {
            if (!next(options.json_path)) {
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

/// 一条文本的聚合结果：每个阶段一组采样（按首次出现顺序），加上
/// RTF / 墙钟 / 音频时长 / CPU 核占用四条运行级指标。
struct TextAggregate {
    std::string text;
    /// 阶段名 → 各次运行的毫秒数。用 vector 保持首见顺序（map 会按字典序打乱，
    /// 与 CLI --timing 的阶段顺序不一致）。
    std::vector<std::pair<std::string, std::vector<double>>> stages;
    std::vector<double> rtf;           // 合成墙钟 / 音频时长（越低越好）
    std::vector<double> wall_seconds;  // 调用级墙钟（比 timing.total 多一层调用开销）
    std::vector<double> audio_seconds;
    std::vector<double> cpu_cores;  // (user+sys) / wall；不可用时为负
    std::string error;              // 非空表示这条文本合成失败
};

void collect_run(TextAggregate& aggregate, const genie::SynthesisResult& result,
                 const double wall_seconds, const double cpu_seconds) {
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
    }
    const double audio = result.audio.duration_seconds();
    aggregate.wall_seconds.push_back(wall_seconds);
    aggregate.audio_seconds.push_back(audio);
    aggregate.rtf.push_back(audio > 0.0 ? wall_seconds / audio : 0.0);
    aggregate.cpu_cores.push_back(
        wall_seconds > 0.0 && cpu_seconds >= 0.0 ? cpu_seconds / wall_seconds : -1.0);
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

    const Stats wall_stats = stats_of(aggregate.wall_seconds);
    char line[160];
    std::snprintf(line, sizeof(line), "  %-34s %8.2f %8.2f %8.2f %8.2f\n", "  total(wall)",
                  wall_stats.mean * 1000.0, wall_stats.min * 1000.0, wall_stats.max * 1000.0,
                  wall_stats.median * 1000.0);
    std::cout << line;

    const Stats rtf_stats = stats_of(aggregate.rtf);
    std::snprintf(line, sizeof(line),
                  "  RTF (wall / audio, lower is better): mean %.3f  min %.3f  max %.3f\n",
                  rtf_stats.mean, rtf_stats.min, rtf_stats.max);
    std::cout << line;

    const Stats audio_stats = stats_of(aggregate.audio_seconds);
    const Stats cpu_stats = stats_of(aggregate.cpu_cores);
    if (!aggregate.cpu_cores.empty() && aggregate.cpu_cores.front() >= 0.0) {
        std::snprintf(line, sizeof(line), "  audio: mean %.2f s   cpu cores used: mean %.1f\n",
                      audio_stats.mean, cpu_stats.mean);
    } else {
        std::snprintf(line, sizeof(line), "  audio: mean %.2f s   cpu cores used: n/a\n",
                      audio_stats.mean);
    }
    std::cout << line;
}

std::string to_json(const std::vector<TextAggregate>& aggregates, const Options& options,
                    const double rss_kb) {
    // 单条 stage 记录（名字 + 四个统计值 + 缩进）能超过 120 字节 ——
    // 缓冲必须留足；snprintf 截断会产出"看起来正常"的坏 JSON。
    static constexpr int kJsonBufSize = 512;
    char number[kJsonBufSize];

    std::string out;
    out += "{\n";
    out += "  \"model\": \"" + json_escape(options.model_dir) + "\",\n";
    out += "  \"version\": \"" + std::string(genie::version_string()) + "\",\n";
    std::snprintf(number, sizeof(number), "  \"threads\": %d, \"runs\": %d, \"warmup\": %d,\n",
                  options.threads, options.runs, options.warmup);
    out += number;
    if (rss_kb < 0.0) {
        out += "  \"peak_rss_kb\": null,\n";
    } else {
        std::snprintf(number, sizeof(number), "  \"peak_rss_kb\": %.0f,\n", rss_kb);
        out += number;
    }
    out += "  \"texts\": [\n";
    for (std::size_t i = 0; i < aggregates.size(); ++i) {
        const auto& aggregate = aggregates[i];
        out += "    {\"text\": \"" + json_escape(aggregate.text) + "\", ";
        if (!aggregate.error.empty()) {
            out += "\"error\": \"" + json_escape(aggregate.error) + "\"";
        } else {
            const Stats rtf_stats = stats_of(aggregate.rtf);
            std::snprintf(number, sizeof(number),
                          "\"rtf\": {\"mean\": %.3f, \"min\": %.3f, \"max\": %.3f}, ",
                          rtf_stats.mean, rtf_stats.min, rtf_stats.max);
            out += number;
            const Stats audio_stats = stats_of(aggregate.audio_seconds);
            std::snprintf(number, sizeof(number), "\"audio_seconds\": {\"mean\": %.3f}, ",
                          audio_stats.mean);
            out += number;
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
    if (options.runs < 1 || options.warmup < 0 || options.threads < 1) {
        std::cerr << "--runs/--threads must be >= 1, --warmup must be >= 0\n";
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
    genie::SessionConfig config;
    config.intra_op_threads = options.threads;

    std::cout << "constructing engine (cold start) from: " << options.model_dir << "\n";
    genie::Engine engine(options.model_dir, config);

    const auto& load_report = engine.load_timing();
    std::cout << "\n[load]\n" << load_report.to_string();

    std::vector<TextAggregate> aggregates;
    for (std::size_t i = 0; i < options.texts.size(); ++i) {
        const auto& text = options.texts[i];
        TextAggregate aggregate;
        aggregate.text = text;

        std::cout << "\n--- text " << (i + 1) << "/" << options.texts.size() << " (" << text.size()
                  << " bytes) ---\n";
        std::cout << "  " << text << "\n";

        try {
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

    const double rss = peak_rss_kb();
    if (rss < 0.0) {
        std::cout << "\npeak RSS: n/a\n";
    } else {
        std::printf("\npeak RSS: %.0f MB (%.0f kB)\n", rss / 1024.0, rss);
    }

    if (!options.json_path.empty()) {
        std::ofstream json(options.json_path);
        if (!json) {
            std::cerr << "cannot write " << options.json_path << "\n";
            return kExitError;
        }
        json << to_json(aggregates, options, rss);
        std::cout << "aggregate written to " << options.json_path << "\n";
    }

    return kExitOk;
}
