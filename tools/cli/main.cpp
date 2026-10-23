// tools/cli/main.cpp —— genie-tts 命令行工具
//
// 子命令：
//   version            打印版本与构建特性
//   info               打印构建信息与可用后端
//   synth              合成语音（需要模型包）
//   inspect            检查模型包的 manifest 并列出模型

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "genie/audio.hpp"
#include "genie/config.hpp"
#include "genie/errors.hpp"
#include "genie/scheduler.hpp"
#include "genie/session.hpp"
#include "genie/version.hpp"

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

Options:
  --model <dir>        Voice package directory (contains manifest.json)
  --text <text>        Text to synthesize
  --output <file>      Output WAV path
  --language <lang>    auto | chinese | english | japanese | korean | hybrid-chinese-english
  --seed <n>           Random seed (0 = not fixed)
  -h, --help           Show this help
)";
}

struct Options {
    std::string command;
    std::string model_dir;
    std::string text;
    std::string output;
    std::string language = "auto";
    std::uint64_t seed = 0;
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
        } else {
            std::cerr << "unknown option: " << arg << "\n";
            return false;
        }
    }
    return true;
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

int command_inspect(const Options& options) {
    if (options.model_dir.empty()) {
        std::cerr << "inspect requires --model <dir>\n";
        return kExitUsage;
    }
    std::cout << "voice package: " << options.model_dir << "\n";
    std::cout << "  (manifest parsing lands with the voice-package milestone)\n";
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
    std::cerr << "synthesis pipeline lands with the inference milestone\n";
    return kExitError;
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
