// tests/regression/test_regression.cpp
//
// 回归层：把**喂给每个模型的东西**锁成黄金数据。
//
// 为什么锁这个？
//
// 上游的模型契约（输入名、dtype、形状）一旦错位，**模型不会报错** ——
// 它会安静地输出垃圾。这类 bug 单测抓不到，因为单测只验证控制流。
//
// 所以这里记录每次推理时**每个模型的全部输入**（名字 / dtype / 形状），
// 存成黄金文件，任何偏离都会让测试失败。
//
// 换模型版本、换前端实现、改引擎结构时，这个测试会第一个叫出来。
//
// 重新生成黄金数据：
//     GENIE_UPDATE_GOLDEN=1 ctest -R regression
// 改完记得 review diff —— 形状变化往往是有意的，但需要有人签字。

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "genie/audio.hpp"
#include "genie/engine.hpp"
#include "genie/json.hpp"
#include "genie/symbols.hpp"
#include "genie/text.hpp"
#include "support/fake_session.hpp"

namespace genie {
namespace {

using test::FakeSession;

/// 一条模型输入的记录。
struct InputRecord {
    std::string name;
    std::string dtype;
    std::vector<std::int64_t> shape;
};

/// 一个模型的记录。
struct ModelRecord {
    std::string model;
    std::vector<InputRecord> inputs;
};

/// 一次推理的完整记录。
struct Trace {
    std::string name;
    std::string text;
    std::vector<ModelRecord> models;
};

/// 把张量的 dtype / 形状变成可比较的字符串。
std::string dtype_name(ElementType type) {
    switch (type) {
        case ElementType::Float32:
            return "float32";
        case ElementType::Int64:
            return "int64";
        case ElementType::Bool:
            return "bool";
    }
    return "unknown";
}

std::vector<InputRecord> records_from(const TensorFeed& feed) {
    std::vector<InputRecord> records;
    for (const auto& [name, tensor] : feed) {
        InputRecord record;
        record.name = name;
        record.dtype = dtype_name(tensor.type());
        record.shape = tensor.shape();
        records.push_back(std::move(record));
    }
    // feed 是有序的（vector），但仍然按名字排一遍 —— 顺序变了也不该算回归。
    std::sort(records.begin(), records.end(),
              [](const InputRecord& a, const InputRecord& b) { return a.name < b.name; });
    return records;
}

/// 临时音色包（含参考音频）。
class TempPackage {
public:
    TempPackage() {
        static int counter = 0;
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        root_ = std::filesystem::temp_directory_path() /
                ("genie_regression_" + std::to_string(stamp) + "_" + std::to_string(counter++));
        std::filesystem::create_directories(root_ / "models");
        std::filesystem::create_directories(root_ / "reference");
        root_str_ = root_.string();

        for (const auto* name : {"t2s_encoder.onnx", "t2s_first_stage_decoder.onnx",
                                 "t2s_stage_decoder.onnx", "vits.onnx"}) {
            std::ofstream file(root_ / "models" / name);
            file << "placeholder";
        }

        AudioBuffer audio;
        audio.sample_rate = 32000;
        audio.samples.assign(32000, 0.1F);
        WavCodec::write_pcm16((root_ / "reference" / "audio.wav").string(), audio);

        std::ofstream text(root_ / "reference" / "text.txt");
        text << "\xE5\x8F\x82\xE8\x80\x83";  // 参考

        std::ofstream manifest(root_ / "manifest.json");
        manifest << R"({
  "name": "regression",
  "format_version": 1,
  "model_version": "v2",
  "language": "chinese",
  "sample_rate": 32000,
  "models": [
    {"role": "t2s_encoder", "file": "models/t2s_encoder.onnx"},
    {"role": "t2s_first_stage_decoder", "file": "models/t2s_first_stage_decoder.onnx"},
    {"role": "t2s_stage_decoder", "file": "models/t2s_stage_decoder.onnx"},
    {"role": "vits", "file": "models/vits.onnx"}
  ]
})";
    }

    ~TempPackage() {
        std::error_code ec;
        std::filesystem::remove_all(root_, ec);
    }

    TempPackage(const TempPackage&) = delete;
    TempPackage& operator=(const TempPackage&) = delete;

    [[nodiscard]] const std::string& root() const noexcept { return root_str_; }

private:
    std::filesystem::path root_;
    std::string root_str_;
};

/// 记录每个模型的输入。
class Recorder {
public:
    /// 记一次：模型名 + 它的输入。
    void capture(const std::string& model, const TensorFeed& feed) {
        ModelRecord record;
        record.model = model;
        record.inputs = records_from(feed);
        models_.push_back(std::move(record));
    }

    [[nodiscard]] const std::vector<ModelRecord>& models() const noexcept { return models_; }

private:
    std::vector<ModelRecord> models_;
};

/// 跑一次推理，把 trace 记下来。
Trace run_trace(const std::string& case_name, const std::string& text) {
    TempPackage package;

    // 每个模型一个假会话；引擎还没析构时这些指针是有效的。
    std::map<std::string, FakeSession*> sessions;
    const auto factory = [&sessions](const std::string& path, const SessionConfig&) {
        std::unique_ptr<ModelSession> session;
        std::string key;
        if (path.find("t2s_encoder") != std::string::npos) {
            key = "t2s_encoder";
            session = FakeSession::encoder();
        } else if (path.find("first_stage") != std::string::npos) {
            key = "t2s_first_stage_decoder";
            session = FakeSession::first_stage();
        } else if (path.find("stage_decoder") != std::string::npos) {
            key = "t2s_stage_decoder";
            session = FakeSession::stage(3);
        } else if (path.find("vits") != std::string::npos) {
            key = "vits";
            session = FakeSession::vocoder(64);
        } else {
            return session;
        }
        sessions[key] = static_cast<FakeSession*>(session.get());
        return session;
    };

    Engine engine(package.root(), factory);
    (void)engine.synthesize(text);

    Trace trace;
    trace.name = case_name;
    trace.text = text;

    // 按固定顺序记录，跟调用顺序无关 —— 顺序变了不算回归。
    Recorder recorder;
    for (const auto* model :
         {"t2s_encoder", "t2s_first_stage_decoder", "t2s_stage_decoder", "vits"}) {
        const auto it = sessions.find(model);
        if (it == sessions.end() || it->second == nullptr) {
            continue;
        }
        recorder.capture(model, it->second->last_feed());
    }
    trace.models = recorder.models();
    return trace;
}

/// 黄金文件路径。
std::filesystem::path golden_path() {
    // 编译期把源码目录传进来（CMAKE_CURRENT_SOURCE_DIR）。
    return std::filesystem::path(GENIE_REGRESSION_GOLDEN_DIR) / "model_io_trace.json";
}

/// 把 trace 序列化成 JSON（给黄金文件用）。
std::string trace_to_json(const std::vector<Trace>& traces) {
    std::string out = "{\n  \"schema\": 1,\n  \"cases\": [\n";
    for (std::size_t i = 0; i < traces.size(); ++i) {
        const auto& trace = traces[i];
        out += "    {\n      \"name\": \"" + trace.name + "\",\n";
        out += "      \"text\": \"" + trace.text + "\",\n";
        out += "      \"models\": [\n";
        for (std::size_t m = 0; m < trace.models.size(); ++m) {
            const auto& model = trace.models[m];
            out += "        {\n          \"model\": \"" + model.model + "\",\n";
            out += "          \"inputs\": [\n";
            for (std::size_t k = 0; k < model.inputs.size(); ++k) {
                const auto& input = model.inputs[k];
                out += "            {\"name\": \"" + input.name + "\", \"dtype\": \"" +
                       input.dtype + "\", \"shape\": [";
                for (std::size_t d = 0; d < input.shape.size(); ++d) {
                    out += std::to_string(input.shape[d]);
                    if (d + 1 < input.shape.size()) {
                        out += ", ";
                    }
                }
                out += "]}";
                out += (k + 1 < model.inputs.size()) ? ",\n" : "\n";
            }
            out += "          ]\n        }";
            out += (m + 1 < trace.models.size()) ? ",\n" : "\n";
        }
        out += "      ]\n    }";
        out += (i + 1 < traces.size()) ? ",\n" : "\n";
    }
    out += "  ]\n}\n";
    return out;
}

/// 回归用例集合。
std::vector<Trace> run_all_cases() {
    return {
        run_trace("chinese_basic", "你好，世界。"),
        run_trace("chinese_single", "啊"),
        run_trace("chinese_digits", "一二三"),
        run_trace("mixed_punctuation", "你好！世界？"),
    };
}

TEST(Regression, ModelIoTraceMatchesGolden) {
    const auto traces = run_all_cases();

    const auto path = golden_path();
    const bool update = std::getenv("GENIE_UPDATE_GOLDEN") != nullptr;

    if (update) {
        std::ofstream file(path);
        ASSERT_TRUE(file.good());
        file << trace_to_json(traces);
        SUCCEED() << "golden updated: " << path.string();
        return;
    }

    ASSERT_TRUE(std::filesystem::exists(path))
        << "golden file missing: " << path.string() << "（用 GENIE_UPDATE_GOLDEN=1 重新生成）";

    std::ifstream file(path);
    ASSERT_TRUE(file.good());
    const auto actual = trace_to_json(traces);

    std::ostringstream expected;
    expected << file.rdbuf();

    EXPECT_EQ(expected.str(), actual) << "模型 I/O 契约变了。确认是有意的改动后，用 "
                                         "GENIE_UPDATE_GOLDEN=1 重新生成黄金数据并 review diff。";
}

}  // namespace
}  // namespace genie
