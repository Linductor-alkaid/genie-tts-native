// tools/cli/main.cpp —— genie-tts 命令行工具
//
// 子命令：
//   version            打印版本与构建特性
//   info               打印构建信息与可用后端
//   synth              合成语音（需要模型包）
//   inspect            检查模型包的 manifest 并列出模型
//   g2p                把文本转成音素序列并打印（不需要模型）

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "genie/audio.hpp"
#include "genie/config.hpp"
#include "genie/engine.hpp"
#include "genie/errors.hpp"
#include "genie/onnx_graph.hpp"
#include "genie/scheduler.hpp"
#include "genie/session.hpp"
#include "genie/symbols.hpp"
#include "genie/text.hpp"
#include "genie/version.hpp"
#include "genie/voice_package.hpp"

namespace {

constexpr int kExitOk = 0;
constexpr int kExitError = 1;
constexpr int kExitUsage = 2;

void print_usage() {
    std::cout << R"(genie-tts — native GPT-SoVITS inference engine

Usage:
  genie-tts <command> [options]

Commands:
  version              Print version and build features
  info                 Print build information
  synth                Synthesize speech from text
  inspect <model-dir>  Inspect a voice package manifest
  g2p --text <text>    Convert text to a phoneme sequence (no model needed)

Options:
  --model <dir>        Voice package directory (contains manifest.json)
  --text <text>        Text to synthesize
  --output <file>      Output WAV path
  --language <lang>    auto | chinese | english | japanese | korean | hybrid-chinese-english
  --seed <n>           Reserved (see note below)
  --threads <n>        ORT intra-op threads per session (default 8)
  -h, --help           Show this help

Note on --seed: sampling (TopK / temperature) lives inside the ONNX graphs
and ONNX Runtime exposes no public seed for its in-graph samplers, so the
seed is accepted for API compatibility but cannot make runs reproducible yet.
)";
}

struct Options {
    std::string command;
    std::string model_dir;
    std::string text;
    std::string output;
    std::string language = "auto";
    std::uint64_t seed = 0;
    int threads = 8;
    bool help = false;
};

bool parse_options(int argc, char** argv, Options& options) {
    if (argc < 2) {
        return false;
    }
    options.command = argv[1];
    for (int i = 2; i < argc; ++i) {
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
            if (!next(options.text)) {
                return false;
            }
        } else if (arg == "--output") {
            if (!next(options.output)) {
                return false;
            }
        } else if (arg == "--language") {
            if (!next(options.language)) {
                return false;
            }
        } else if (arg == "--seed") {
            std::string value;
            if (!next(value)) {
                return false;
            }
            options.seed = std::strtoull(value.c_str(), nullptr, 10);
        } else if (arg == "--threads") {
            std::string value;
            if (!next(value)) {
                return false;
            }
            options.threads = std::atoi(value.c_str());
        } else if (!arg.empty() && arg[0] == '-') {
            std::cerr << "unknown option: " << arg << "\n";
            return false;
        } else if (options.model_dir.empty()) {
            // 位置参数：inspect 的模型文件或目录
            options.model_dir = arg;
        } else {
            std::cerr << "unexpected argument: " << arg << "\n";
            return false;
        }
    }
    return true;
}

/// 把文本转成音素序列并打印。
///
/// 排查文本前端的问题时用得上：不需要模型就能看到模型会收到的
/// text_seq 到底长什么样。
int command_g2p(const Options& options) {
    if (options.text.empty()) {
        std::cerr << "g2p requires --text <text>\n";
        return kExitUsage;
    }

    const auto requested = genie::language_from_string(options.language);
    if (!requested) {
        std::cerr << "unknown language: " << options.language << "\n";
        return kExitUsage;
    }

    const auto language = genie::detect_language(options.text, *requested);
    auto frontend = genie::text::make_frontend(language);
    if (!frontend) {
        std::cerr << "no text frontend for language: " << genie::language_to_string(language)
                  << "\n";
        return kExitUsage;
    }

    const auto sequence = frontend->convert(options.text);
    std::cout << "language: " << genie::language_to_string(language) << "\n";
    std::cout << "normalized: " << genie::text::normalize(options.text, language) << "\n";
    std::cout << "phonemes (" << sequence.ids.size() << "):\n";
    for (std::size_t i = 0; i < sequence.symbols.size(); ++i) {
        std::cout << "  " << i << "\t" << sequence.symbols[i] << "\t" << sequence.ids[i] << "\n";
    }
    return kExitOk;
}

int command_version() {
    std::cout << "genie-tts-native " << genie::version_string() << "\n";
    const auto features = genie::build_features();
    std::cout << "  onnxruntime: " << (features.onnxruntime ? "yes" : "no") << "\n";
    std::cout << "  kairo:       " << (features.kairo ? "yes" : "no") << "\n";
    std::cout << "  sanitizers:  " << (features.sanitizers ? "yes" : "no") << "\n";
    return kExitOk;
}

int command_info() {
    command_version();
    std::cout << "  backend:     "
              << (genie::has_onnxruntime_backend() ? "onnxruntime-cpu" : "unavailable") << "\n";
    return kExitOk;
}

/// 打印一个 ONNX 模型的 I/O 契约。
///
/// 用的是 protobuf 解析器而不是 ONNX Runtime —— 上游发布的模型是"图壳"
/// （权重在外部 .bin 里），ORT 加载不了，但图结构是完整的。
int inspect_onnx(const std::string& path) {
    const auto spec = genie::read_onnx_graph(path);
    std::cout << path << "\n";
    std::cout << "  ir_version: " << spec.ir_version << "\n";
    std::cout << "  inputs (" << spec.inputs.size() << "):\n";
    for (const auto& input : spec.inputs) {
        std::cout << "    " << input.name << ": " << genie::to_string(input.type) << " "
                  << input.shape_string() << "\n";
    }
    std::cout << "  outputs (" << spec.outputs.size() << "):\n";
    for (const auto& output : spec.outputs) {
        std::cout << "    " << output.name << ": " << genie::to_string(output.type) << " "
                  << output.shape_string() << "\n";
    }
    if (spec.uses_external_data) {
        std::cout << "  external weights:";
        for (const auto& file : spec.external_data_files) {
            std::cout << " " << file;
        }
        std::cout << "\n";
    }
    return kExitOk;
}

/// 打印一个音色包的清单。
int inspect_voice_package(const std::string& root) {
    const auto package = genie::VoicePackage::load(root, /*verify_hash=*/true);
    const auto& manifest = package.manifest();

    std::cout << "voice package: " << root << "\n";
    std::cout << "  name:          " << manifest.name << "\n";
    std::cout << "  model_version: " << manifest.model_version << "\n";
    std::cout << "  language:      " << genie::language_to_string(manifest.language) << "\n";
    std::cout << "  sample_rate:   " << manifest.sample_rate << "\n";
    std::cout << "  v2proplus:     " << (manifest.capabilities.v2proplus ? "yes" : "no") << "\n";
    std::cout << "  roberta:       " << (manifest.capabilities.roberta ? "yes" : "no") << "\n";
    std::cout << "  models:\n";
    for (const auto& [role, entry] : manifest.models) {
        const std::string verified =
            entry.sha256.empty() ? std::string("(no checksum)") : std::string("(sha256 ok)");
        std::cout << "    " << genie::to_string(role) << ": " << entry.file << "  " << verified
                  << "\n";
    }
    return kExitOk;
}

int command_inspect(const Options& options) {
    if (options.model_dir.empty()) {
        std::cerr << "inspect requires a model file or voice package directory\n";
        return kExitUsage;
    }

    // 传入的是模型文件 → 直接打印它的契约。
    if (std::filesystem::is_regular_file(options.model_dir)) {
        return inspect_onnx(options.model_dir);
    }

    // 传入的是目录 → 优先当作音色包；没有 manifest 就退回列出 .onnx 文件。
    if (std::filesystem::exists(std::filesystem::path(options.model_dir) / "manifest.json")) {
        return inspect_voice_package(options.model_dir);
    }

    std::cout << "voice package: " << options.model_dir << "\n";
    std::vector<std::filesystem::path> models;
    for (const auto& entry : std::filesystem::directory_iterator(options.model_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".onnx") {
            models.push_back(entry.path());
        }
    }
    if (models.empty()) {
        std::cout << "  (no .onnx files found)\n";
        return kExitOk;
    }
    std::sort(models.begin(), models.end());
    for (const auto& model : models) {
        if (const int code = inspect_onnx(model.string()); code != kExitOk) {
            return code;
        }
    }
    return kExitOk;
}

int command_synth(const Options& options) {
    if (options.model_dir.empty() || options.text.empty()) {
        std::cerr << "synth requires --model <dir> and --text <text>\n";
        return kExitUsage;
    }
    const auto language = genie::language_from_string(options.language);
    if (!language.has_value()) {
        std::cerr << "unknown language: " << options.language << "\n";
        return kExitUsage;
    }
    if (!genie::has_onnxruntime_backend()) {
        std::cerr << "this build has no ONNX Runtime backend — reconfigure with "
                     "-DGENIE_ENABLE_ORT=ON\n";
        return kExitError;
    }

    genie::SynthesisOptions synthesis;
    synthesis.language = *language;
    if (options.seed != 0) {
        // 诚实语义：ORT 没有公开的图内采样种子接口，seed 目前只是透传
        // 给 SynthesisOptions（C ABI 同一套字段），不改变随机性。
        // 审计记录见 docs/upstream-audit.md §10。
        synthesis.seed = options.seed;
    }

    // 每个 ORT 会话默认开满全部核心的线程池；Engine 要建 4-6 个会话，
    // 在多核机器上是上百个自旋线程 —— 调度器一忙就互相踩踏
    // （实测 32 核机器上高负载时段错误/浮点异常都出在 ORT 内部）。
    // CLI 单请求场景给一个保守的每会话预算，可用 --threads 覆盖。
    genie::SessionConfig config;
    if (options.threads > 0) {
        config.intra_op_threads = options.threads;
    }

    genie::Engine engine(options.model_dir, config);
    const auto result = engine.synthesize(options.text, synthesis);

    if (options.output.empty()) {
        std::cerr << "synth requires --output <file.wav>\n";
        return kExitUsage;
    }
    if (synthesis.format == genie::AudioFormat::Float32) {
        genie::WavCodec::write_float32(options.output, result.audio);
    } else {
        genie::WavCodec::write_pcm16(options.output, result.audio);
    }

    std::cout << "wrote " << options.output << " (" << result.audio.frame_count() << " frames, "
              << result.audio.sample_rate << " Hz, " << result.audio.duration_seconds() << " s)\n";
    std::cout << "  phonemes: " << result.phoneme_ids.size() << "\n";
    std::cout << "  semantic tokens: " << result.semantic_tokens.size() << "\n";
    return kExitOk;
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

    try {
        if (options.command == "version") {
            return command_version();
        }
        if (options.command == "info") {
            return command_info();
        }
        if (options.command == "inspect") {
            return command_inspect(options);
        }
        if (options.command == "synth") {
            return command_synth(options);
        }
        if (options.command == "g2p") {
            return command_g2p(options);
        }
    } catch (const genie::Error& error) {
        std::cerr << "error [" << genie::to_string(error.code()) << "]: " << error.what() << "\n";
        return kExitError;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return kExitError;
    }

    std::cerr << "unknown command: " << options.command << "\n";
    print_usage();
    return kExitUsage;
}
