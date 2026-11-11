// tools/gate/main.cpp —— genie-gate 精度门控工具（roadmap M0.2 / M5 量化）
//
// 为什么需要它：量化（动态 INT8 等）会改变每个模型的数值输出，而"听觉效果
// 没有损失"不能靠耳朵试出来的每一版都人肉裁决。这里把裁决变成可复现的两步：
//
//   1. capture —— 用**包装会话工厂**跑一次真实合成，把每个模型真实收到的
//      输入与产生的输出逐张量落盘。引擎零改动：Engine 本来就接受自定义
//      SessionFactory（include/genie/engine.hpp），真会话由公开的
//      create_session() 创建，装饰器只负责旁路记录。
//   2. e2e     —— 基线包与变体包各完整合成，比较语义 token 一致率并导出
//      双份 WAV 供听测。
//
// 张量级数值比对（abs/rel/余弦 + 逐角色容差门控）在 tools/gate/precision_gate.py，
// 它消费 capture 的产物，把变体模型放在捕获输入上重放后对比捕获的 fp32 输出。
// 两步分开是因为变体 .onnx 的重放用 Python 侧 onnxruntime 最直接（量化工具链
// 也在 Python），而捕获必须发生在真实 C++ 流水线里才有意义。
//
// 确定性事实（2026-11-11 实测，docs/performance.md 量化一节）：
//   - 采样在 ONNX 图内、种子在会话内固定：同进程里两个同配置引擎逐 run
//     序号完全一致（fp32 自比 e2e 的 WAV 逐字节相同）；跨进程则会漂移。
//   - fp32 重放也会翻近平局的采样 token —— 所以 token 类门控是地板相对的
//     （precision_gate.py --floor-report），不是绝对 1.0。
//
// 捕获格式：每个张量一个裸二进制文件（行主序、小端、无头部），
// 元数据（名字 / dtype / 形状 / 文件路径）在 capture.json —— Python 侧用
// np.fromfile + reshape 即可还原。

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "gate/capture.hpp"
#include "genie/audio.hpp"
#include "genie/config.hpp"
#include "genie/engine.hpp"
#include "genie/errors.hpp"
#include "genie/json.hpp"
#include "genie/session.hpp"
#include "genie/version.hpp"

namespace {

constexpr int kExitOk = 0;
constexpr int kExitError = 1;
constexpr int kExitUsage = 2;

void print_usage() {
    std::cout << R"(genie-gate — precision gate harness (M0.2 golden tolerances / M5 quantization)

Usage:
  genie-gate capture --model <dir> --text <text> [--text ...] [options]
  genie-gate e2e --baseline <dir> --variant <dir> --text <text> [--text ...] [options]

capture options:
  --model <dir>        Voice package directory (fp32 / int8 weights; fp16 is
                       rejected — its load path bypasses custom session factories)
  --text <text>        Text to synthesize (repeatable; one synthesis per text)
  --language <lang>    auto | chinese | english | japanese | korean | hybrid-chinese-english
  --out <dir>          Capture output directory (default: ./genie-gate-capture)
  --stage-steps <n>    Fully capture the first N stage_decoder calls (default 1;
                       later calls only keep iy*/y/stop tensors to skip KV cache)
  --threads <n>        ORT intra-op threads per session (default 8)
  --no-verify          Skip voice package SHA-256 verification
  --json <file>        Also write capture metadata to this path (default <out>/capture.json)

e2e options:
  --baseline <dir>     Reference voice package
  --variant <dir>      Package under test (may equal --baseline to measure the
                       run-to-run noise floor)
  --runs <n>           Syntheses per package per text (default 1; use 2+ to see
                       in-graph sampling variance)
  --out-prefix <p>     Output prefix for WAV pairs and e2e.json (default ./genie-gate-e2e)
  --fail-below <r>     Exit 1 if any pair's token agreement < r (default: never fail)

Common options:
  -h, --help           Show this help
)";
}

struct Options {
    std::string command;
    std::string model_dir;
    std::string baseline_dir;
    std::string variant_dir;
    std::string out_dir = "genie-gate-capture";
    std::string json_path;
    std::string out_prefix = "genie-gate-e2e";
    std::string language = "auto";
    std::vector<std::string> texts;
    int threads = 8;
    int stage_steps = 1;
    int runs = 1;
    bool no_verify = false;
    double fail_below = -1.0;
    bool help = false;
};

bool parse_options(int argc, char** argv, Options& options) {
    if (argc < 2) {
        return false;
    }
    options.command = argv[1];
    if (options.command != "capture" && options.command != "e2e") {
        return false;
    }
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](std::string& target) {
            if (i + 1 >= argc) {
                return false;
            }
            target = argv[++i];
            return true;
        };
        auto next_num = [&](int& target) {
            std::string value;
            if (!next(value)) {
                return false;
            }
            target = std::atoi(value.c_str());
            return true;
        };
        if (arg == "-h" || arg == "--help") {
            options.help = true;
        } else if (arg == "--model") {
            if (!next(options.model_dir)) {
                return false;
            }
        } else if (arg == "--baseline") {
            if (!next(options.baseline_dir)) {
                return false;
            }
        } else if (arg == "--variant") {
            if (!next(options.variant_dir)) {
                return false;
            }
        } else if (arg == "--out") {
            if (!next(options.out_dir)) {
                return false;
            }
        } else if (arg == "--json") {
            if (!next(options.json_path)) {
                return false;
            }
        } else if (arg == "--out-prefix") {
            if (!next(options.out_prefix)) {
                return false;
            }
        } else if (arg == "--language") {
            if (!next(options.language)) {
                return false;
            }
        } else if (arg == "--text") {
            std::string text;
            if (!next(text)) {
                return false;
            }
            options.texts.push_back(std::move(text));
        } else if (arg == "--stage-steps") {
            if (!next_num(options.stage_steps)) {
                return false;
            }
        } else if (arg == "--runs") {
            if (!next_num(options.runs)) {
                return false;
            }
        } else if (arg == "--threads") {
            if (!next_num(options.threads)) {
                return false;
            }
        } else if (arg == "--no-verify") {
            options.no_verify = true;
        } else if (arg == "--fail-below") {
            std::string value;
            if (!next(value)) {
                return false;
            }
            options.fail_below = std::atof(value.c_str());
        } else {
            std::cerr << "unknown option: " << arg << "\n";
            return false;
        }
    }
    return !options.help;
}

genie::json::Value load_manifest_json(const std::string& package_dir) {
    return genie::json::parse_file((std::filesystem::path(package_dir) / "manifest.json").string());
}

/// fp16 包的打包权重在 load_model_sessions 里**绕过**自定义工厂直接建会话
/// （src/engine/model_loader.cpp：注入内存权重的路径走 create_session 重载），
/// 捕获会静默漏掉这些模型 —— 与其产出不完整的捕获，不如拒绝。
void reject_fp16_package(const genie::json::Value& manifest, const std::string& dir) {
    const auto* format = manifest.find("weight_format");
    if (format != nullptr && format->is_string() && format->as_string() == "fp16") {
        throw genie::Error(genie::ErrorCode::InvalidArgument,
                           "genie-gate does not support fp16 packages ('" + dir +
                               "'): their packed weights bypass custom session factories, so "
                               "capture would silently miss models. Use the fp32 package.");
    }
}

genie::EngineConfig make_engine_config(const Options& options) {
    genie::EngineConfig config;
    if (options.threads > 0) {
        config.session.intra_op_threads = options.threads;
    }
    config.verify_voice_package = !options.no_verify;
    return config;
}

/// 包装工厂：路径 → 角色名 → 捕获会话（内部用公开 create_session 建真会话）。
genie::SessionFactory make_capturing_factory(const std::filesystem::path& out_dir,
                                             std::vector<genie::gate::CallRecord>& log,
                                             std::atomic<int>& phase, int stage_full_steps) {
    return [&out_dir, &log, &phase, stage_full_steps](const std::string& path,
                                                      const genie::SessionConfig& config) {
        auto inner = genie::create_session(path, config);
        if (inner == nullptr) {
            return inner;
        }
        return std::unique_ptr<genie::ModelSession>(std::make_unique<genie::gate::CapturingSession>(
            std::move(inner), genie::gate::role_from_path(path), out_dir, log, phase,
            stage_full_steps));
    };
}

int command_capture(const Options& options) {
    if (options.model_dir.empty() || options.texts.empty()) {
        std::cerr << "capture requires --model <dir> and at least one --text\n";
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
    reject_fp16_package(load_manifest_json(options.model_dir), options.model_dir);

    const std::filesystem::path out_dir(options.out_dir);
    std::error_code ec;
    std::filesystem::create_directories(out_dir, ec);

    std::vector<genie::gate::CallRecord> log;
    std::atomic<int> phase{0};  // 0 = Engine 构造期（加载），1 = 合成期。
    auto factory = make_capturing_factory(out_dir, log, phase, options.stage_steps);

    std::cout << "loading voice package: " << options.model_dir << "\n";
    genie::Engine engine(options.model_dir, std::move(factory), make_engine_config(options));

    phase.store(1);
    genie::SynthesisOptions synthesis;
    synthesis.language = *language;
    for (const auto& text : options.texts) {
        std::cout << "synthesizing: " << text << "\n";
        const auto result = engine.synthesize(text, synthesis);
        std::cout << "  semantic tokens: " << result.semantic_tokens.size() << ", audio "
                  << result.audio.duration_seconds() << " s\n";
    }

    const auto root = genie::gate::capture_log_to_json(log, options.model_dir, options.texts,
                                                       options.language, options.stage_steps);
    const auto json_path = options.json_path.empty() ? out_dir / "capture.json"
                                                     : std::filesystem::path(options.json_path);
    std::ofstream out(json_path);
    if (!out.good()) {
        throw genie::Error(genie::ErrorCode::IoError,
                           "cannot write capture.json: " + json_path.string());
    }
    out << root.dump() << "\n";
    std::cout << "capture metadata: " << json_path.string() << "\n";
    std::cout << "captured " << log.size() << " model calls into " << out_dir.string() << "\n";
    return kExitOk;
}

// --------------------------------------------------------------------------------------------
// e2e：基线包与变体包各合成若干次，比较语义 token 序列。
//
// 一致率定义：两序列的**最长公共前缀长度 / 较短序列长度**。采样在图内，
// 同进程内逐 run 序号可复现（见文件头注释），所以有意义的配对是同 run 序号
// （i==j）；跨 run 序号的配对反映"执行计数错位"下的漂移，仅作参考。
// --------------------------------------------------------------------------------------------

struct RunResult {
    std::vector<std::int64_t> tokens;
    double seconds = 0.0;
    /// 只留第一个 run 的音频落盘听测用；其余 run 的波形直接丢弃，省内存。
    genie::AudioBuffer audio;
};

RunResult synthesize_once(genie::Engine& engine, const std::string& text,
                          const genie::SynthesisOptions& options, bool keep_audio) {
    const auto result = engine.synthesize(text, options);
    RunResult run;
    run.tokens = result.semantic_tokens;
    run.seconds = result.audio.duration_seconds();
    if (keep_audio) {
        run.audio = result.audio;
    }
    return run;
}

double prefix_agreement(const std::vector<std::int64_t>& a, const std::vector<std::int64_t>& b,
                        std::int64_t* matched_out = nullptr) {
    if (a.empty() || b.empty()) {
        return 0.0;
    }
    const std::size_t limit = std::min(a.size(), b.size());
    std::size_t matched = 0;
    while (matched < limit && a[matched] == b[matched]) {
        ++matched;
    }
    if (matched_out != nullptr) {
        *matched_out = static_cast<std::int64_t>(matched);
    }
    return static_cast<double>(matched) / static_cast<double>(std::min(a.size(), b.size()));
}

int command_e2e(const Options& options) {
    if (options.baseline_dir.empty() || options.variant_dir.empty() || options.texts.empty()) {
        std::cerr << "e2e requires --baseline <dir>, --variant <dir> and at least one --text\n";
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

    const auto config = make_engine_config(options);
    genie::SynthesisOptions synthesis;
    synthesis.language = *language;

    std::cout << "loading baseline: " << options.baseline_dir << "\n";
    genie::Engine baseline(options.baseline_dir, config);
    std::cout << "loading variant:  " << options.variant_dir << "\n";
    genie::Engine variant(options.variant_dir, config);

    genie::json::ArrayType pair_log;
    bool any_below = false;

    for (std::size_t text_index = 0; text_index < options.texts.size(); ++text_index) {
        const auto& text = options.texts[text_index];
        std::vector<RunResult> baseline_runs;
        std::vector<RunResult> variant_runs;
        for (int i = 0; i < options.runs; ++i) {
            baseline_runs.push_back(synthesize_once(baseline, text, synthesis, i == 0));
            variant_runs.push_back(synthesize_once(variant, text, synthesis, i == 0));
        }

        // 每个基线 run 对每个变体 run —— 同序号对是主裁决，跨序号对是参考。
        for (std::size_t i = 0; i < static_cast<std::size_t>(options.runs); ++i) {
            for (std::size_t j = 0; j < static_cast<std::size_t>(options.runs); ++j) {
                std::int64_t matched = 0;
                const double agreement =
                    prefix_agreement(baseline_runs[i].tokens, variant_runs[j].tokens, &matched);
                genie::json::ObjectType pair;
                pair["text"] = genie::json::Value(text);
                pair["baseline_run"] = genie::json::Value(static_cast<std::int64_t>(i));
                pair["variant_run"] = genie::json::Value(static_cast<std::int64_t>(j));
                pair["baseline_tokens"] =
                    genie::json::Value(static_cast<std::int64_t>(baseline_runs[i].tokens.size()));
                pair["variant_tokens"] =
                    genie::json::Value(static_cast<std::int64_t>(variant_runs[j].tokens.size()));
                pair["matched_prefix"] = genie::json::Value(matched);
                pair["agreement"] = genie::json::Value(agreement);
                pair["baseline_seconds"] = genie::json::Value(baseline_runs[i].seconds);
                pair["variant_seconds"] = genie::json::Value(variant_runs[j].seconds);
                pair_log.push_back(genie::json::Value(std::move(pair)));

                std::cout << "[" << text << "] b" << i << "×v" << j << ": tokens "
                          << baseline_runs[i].tokens.size() << " vs "
                          << variant_runs[j].tokens.size() << ", prefix agreement " << agreement
                          << "\n";
                if (options.fail_below >= 0.0 && agreement < options.fail_below) {
                    any_below = true;
                }
            }
        }

        // 首个 run 的 WAV 落盘供听测；命名带文本序号避免互相覆盖。
        // PCM16 是 CLI synth 的默认交付格式，听测不需要 float32 的精度余量。
        const auto text_tag = std::to_string(text_index);
        const auto baseline_wav = options.out_prefix + "_t" + text_tag + "_baseline.wav";
        const auto variant_wav = options.out_prefix + "_t" + text_tag + "_variant.wav";
        genie::WavCodec::write_pcm16(baseline_wav, baseline_runs.front().audio);
        genie::WavCodec::write_pcm16(variant_wav, variant_runs.front().audio);
    }

    genie::json::ObjectType root;
    root["schema"] = genie::json::Value(static_cast<std::int64_t>(1));
    root["tool"] = genie::json::Value(std::string("genie-gate"));
    root["genie_version"] = genie::json::Value(std::string(genie::version_string()));
    root["baseline"] = genie::json::Value(options.baseline_dir);
    root["variant"] = genie::json::Value(options.variant_dir);
    root["runs"] = genie::json::Value(static_cast<std::int64_t>(options.runs));
    genie::json::ArrayType texts;
    for (const auto& text : options.texts) {
        texts.push_back(genie::json::Value(text));
    }
    root["texts"] = genie::json::Value(std::move(texts));
    root["pairs"] = genie::json::Value(std::move(pair_log));

    const auto json_path = options.out_prefix + ".json";
    std::ofstream out(json_path);
    if (!out.good()) {
        throw genie::Error(genie::ErrorCode::IoError, "cannot write e2e json: " + json_path);
    }
    out << genie::json::Value(root).dump() << "\n";
    std::cout << "e2e report: " << json_path << "\n";

    return any_below ? kExitError : kExitOk;
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse_options(argc, argv, options)) {
        print_usage();
        return options.help ? kExitOk : kExitUsage;
    }
    try {
        return options.command == "capture" ? command_capture(options) : command_e2e(options);
    } catch (const genie::Error& error) {
        std::cerr << "genie-gate: " << error.what() << "\n";
        return kExitError;
    } catch (const std::exception& error) {
        std::cerr << "genie-gate: unexpected error: " << error.what() << "\n";
        return kExitError;
    }
}
