// tools/gate/capture.hpp —— 精度门控的捕获基础设施
//
// 从 main.cpp 拆出来只为可测：装饰器、角色识别、张量落盘、捕获日志都要被
// 单测覆盖（用 tests/support/fake_session.hpp 的假会话，无需真权重）。
// e2e 与 CLI 参数解析留在 main.cpp。
//
// 捕获格式见 main.cpp 顶部注释与 docs/regression.md 精度门控一节。

#ifndef GENIE_GATE_CAPTURE_HPP
#define GENIE_GATE_CAPTURE_HPP

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "genie/errors.hpp"
#include "genie/json.hpp"
#include "genie/session.hpp"
#include "genie/tensor.hpp"

namespace genie::gate {

/// 每个模型最多完整捕获多少次调用。
///
/// RoBERTa 会被分块编码多次调用（#47），自回归解码器会调用上百次 ——
/// 全捕会把输出目录撑到 GB 级。对门控而言，每个模型几次调用足够统计；
/// stage_decoder 另有专门策略（见 CapturingSession）。
constexpr int kMaxCallsPerRole = 8;

/// 模型文件路径 → 角色名（与 manifest 的 role 命名一致）。
///
/// 按子串匹配即可：包内文件名都含角色名（t2s_first_stage_decoder_fp32.onnx 等）。
/// 顺序只影响别名冲突 —— "first_stage" 与 "stage_decoder" 都含 "stage"，
/// 所以更长的模式必须在前。未知模型用文件名主干兜底，不该静默丢失。
[[nodiscard]] inline std::string role_from_path(const std::string& path) {
    static const std::vector<std::pair<const char*, const char*>> kPatterns = {
        {"first_stage", "t2s_first_stage_decoder"},
        {"stage_decoder", "t2s_stage_decoder"},
        {"t2s_encoder", "t2s_encoder"},
        {"vits", "vits"},
        {"prompt_encoder", "prompt_encoder"},
        {"speaker", "speaker_verification"},
        {"hubert", "chinese_hubert"},
        {"roberta", "roberta"},
    };
    for (const auto& [pattern, role] : kPatterns) {
        if (path.find(pattern) != std::string::npos) {
            return role;
        }
    }
    const auto stem = std::filesystem::path(path).stem().string();
    return stem.empty() ? "unknown" : stem;
}

[[nodiscard]] inline const char* dtype_ext(ElementType type) {
    switch (type) {
        case ElementType::Float32:
            return "f32";
        case ElementType::Int64:
            return "i64";
        case ElementType::Bool:
            return "b8";
    }
    return "bin";
}

[[nodiscard]] inline const char* dtype_name(ElementType type) {
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

/// 一次推理调用的完整记录。
struct CallRecord {
    std::string role;
    int call = 0;
    /// "load"（Engine 构造期：参考音频条件 / 参考文本特征）或 "synth"。
    std::string phase = "synth";
    /// 是否只记录了部分张量（stage_decoder 的后续步）。
    bool partial = false;
    bool truncated = false;  ///< 超出 kMaxCallsPerRole，只记账不落盘。
    std::vector<json::Value> inputs;
    std::vector<json::Value> outputs;
};

/// 把捕获日志序列化成 capture.json 的 JSON 值。
[[nodiscard]] inline json::Value capture_log_to_json(const std::vector<CallRecord>& log,
                                                     const std::string& package,
                                                     const std::vector<std::string>& texts,
                                                     const std::string& language,
                                                     int stage_full_steps) {
    json::ArrayType calls;
    for (const auto& record : log) {
        json::ObjectType entry;
        entry["role"] = json::Value(record.role);
        entry["call"] = json::Value(static_cast<std::int64_t>(record.call));
        entry["phase"] = json::Value(record.phase);
        entry["partial"] = json::Value(record.partial);
        entry["truncated"] = json::Value(record.truncated);
        entry["inputs"] = json::Value(json::ArrayType(record.inputs.begin(), record.inputs.end()));
        entry["outputs"] =
            json::Value(json::ArrayType(record.outputs.begin(), record.outputs.end()));
        calls.push_back(json::Value(std::move(entry)));
    }
    json::ObjectType root;
    root["schema"] = json::Value(static_cast<std::int64_t>(1));
    // 注意：Value 没有 const char* 重载，裸指针会走"指针→bool"的标准转换
    // 落进 Value(bool) —— 字符串必须显式包 std::string。
    root["tool"] = json::Value(std::string("genie-gate"));
    root["package"] = json::Value(package);
    json::ArrayType text_values;
    for (const auto& text : texts) {
        text_values.push_back(json::Value(text));
    }
    root["texts"] = json::Value(std::move(text_values));
    root["language"] = json::Value(language);
    root["stage_full_steps"] = json::Value(static_cast<std::int64_t>(stage_full_steps));
    root["calls"] = json::Value(std::move(calls));
    return json::Value(std::move(root));
}

/// 捕获会话：包装真会话，把每次 run 的输入输出按张量落盘。
///
/// 落盘是裸字节拷贝（Tensor::bytes 视图 → ofstream），不做任何格式转换 ——
/// 门控的对比基准就是这些字节，中间任何"顺手"的规范化都会污染结论。
///
/// phase：0 = Engine 构造期（load），非 0 = 合成期。由调用方在构造引擎前置 0、
/// 合成前置 1 —— Engine 不是线程安全的，捕获工具是单线程顺序流程。
class CapturingSession final : public ModelSession {
public:
    CapturingSession(std::unique_ptr<ModelSession> inner, std::string role,
                     const std::filesystem::path& out_dir, std::vector<CallRecord>& log,
                     const std::atomic<int>& phase, int stage_full_steps)
        : inner_(std::move(inner)),
          role_(std::move(role)),
          out_dir_(out_dir),
          log_(log),
          phase_(phase),
          stage_full_steps_(stage_full_steps) {}

    [[nodiscard]] const std::vector<TensorSpec>& inputs() const override {
        return inner_->inputs();
    }
    [[nodiscard]] const std::vector<TensorSpec>& outputs() const override {
        return inner_->outputs();
    }
    [[nodiscard]] const std::string& model_path() const override { return inner_->model_path(); }

    [[nodiscard]] std::vector<Tensor> run(const TensorFeed& feed) override {
        const int call = calls_++;
        const bool is_stage_decoder = role_ == "t2s_stage_decoder";

        CallRecord record;
        record.role = role_;
        record.call = call;
        record.phase = phase_.load(std::memory_order_relaxed) == 0 ? "load" : "synth";

        // stage_decoder 的后续调用里 KV cache 随步数线性膨胀，完整落盘会把
        // 捕获目录撑到 GB 级；门控只用首步做教师强制比对，后续步保留
        // token 相关的小张量（iy / y / stop）用于序列一致性统计。
        const bool full = !is_stage_decoder || call < stage_full_steps_;
        record.partial = !full;
        record.truncated = call >= kMaxCallsPerRole;

        if (record.truncated) {
            log_.push_back(std::move(record));
            return inner_->run(feed);
        }
        const auto call_dir = out_dir_ / role_ / ("call_" + std::to_string(call));
        std::error_code ec;
        std::filesystem::create_directories(call_dir / "inputs", ec);
        record.inputs = dump_feed(call_dir / "inputs", feed, !full);
        std::vector<Tensor> result = inner_->run(feed);
        std::filesystem::create_directories(call_dir / "outputs", ec);
        record.outputs = dump_tensors(call_dir / "outputs", result, full);
        log_.push_back(std::move(record));
        return result;
    }

private:
    /// 候选输入名（partial 模式下只捕获这些 —— stage_decoder 的 token 流）。
    [[nodiscard]] static bool partial_kept(const std::string& name) {
        return name.rfind("iy", 0) == 0 || name == "y" || name == "stop_condition_tensor";
    }

    std::vector<json::Value> dump_feed(const std::filesystem::path& dir, const TensorFeed& feed,
                                       bool partial) {
        std::vector<json::Value> entries;
        for (const auto& [name, tensor] : feed) {
            if (partial && !partial_kept(name)) {
                continue;
            }
            entries.push_back(dump_one(dir, name, tensor));
        }
        return entries;
    }

    /// full 模式下输出全留（含 KV cache —— 教师强制比对需要首步的 present_*）；
    /// partial 模式只留 token 流小张量，KV cache 不落盘。
    std::vector<json::Value> dump_tensors(const std::filesystem::path& dir,
                                          const std::vector<Tensor>& tensors, bool full) {
        std::vector<json::Value> entries;
        const auto& specs = inner_->outputs();
        for (std::size_t i = 0; i < tensors.size(); ++i) {
            const auto& name = i < specs.size() ? specs[i].name : "out_" + std::to_string(i);
            if (!full && !partial_kept(name)) {
                continue;
            }
            entries.push_back(dump_one(dir, name, tensors[i]));
        }
        return entries;
    }

    json::Value dump_one(const std::filesystem::path& dir, const std::string& name,
                         const Tensor& tensor) {
        const auto file = dir / (name + "." + dtype_ext(tensor.type()));
        std::ofstream out(file, std::ios::binary);
        if (!out.good()) {
            throw Error(ErrorCode::IoError, "cannot write capture file: " + file.string());
        }
        out.write(reinterpret_cast<const char*>(tensor.bytes().data()),
                  static_cast<std::streamsize>(tensor.byte_size()));
        json::ObjectType entry;
        entry["name"] = json::Value(name);
        entry["dtype"] = json::Value(std::string(dtype_name(tensor.type())));
        json::ArrayType shape;
        for (const auto dim : tensor.shape()) {
            shape.push_back(json::Value(static_cast<std::int64_t>(dim)));
        }
        entry["shape"] = json::Value(std::move(shape));
        entry["bytes"] = json::Value(static_cast<std::int64_t>(tensor.byte_size()));
        entry["file"] = json::Value(std::filesystem::relative(file, out_dir_).string());
        return json::Value(std::move(entry));
    }

    std::unique_ptr<ModelSession> inner_;
    std::string role_;
    std::filesystem::path out_dir_;
    std::vector<CallRecord>& log_;
    const std::atomic<int>& phase_;
    int stage_full_steps_;
    int calls_ = 0;
};

}  // namespace genie::gate

#endif  // GENIE_GATE_CAPTURE_HPP
