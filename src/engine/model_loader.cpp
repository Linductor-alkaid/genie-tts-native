// src/engine/model_loader.cpp

#include "engine/model_loader.hpp"

#include <cstdint>
#include <filesystem>
#include <map>
#include <system_error>
#include <utility>

#include "engine/stage_timer.hpp"
#include "genie/errors.hpp"
#include "genie/onnx_graph.hpp"

namespace genie::detail {
namespace {

/// 一个模型要注入的 fp16 打包文件：图里的 location + 磁盘上的绝对路径。
struct PackedWeightRef {
    std::string location;
    std::string path;
};

}  // namespace

void load_model_sessions(const std::vector<ModelLoadRequest>& requests,
                         const SessionFactory& factory, const SessionConfig& config,
                         const bool upcast_fp16_weights, TimingReport& timing) {
    // 每个请求要注入的 fp16 打包文件；上转结果按磁盘路径去重。
    std::vector<std::vector<PackedWeightRef>> packed(requests.size());
    std::map<std::string, Fp32WeightBuffer> upcast;

    if (upcast_fp16_weights) {
        {
            ScopedStageTimer timer(timing, "load/fp16_scan");
            timer.set_count(static_cast<std::uint32_t>(requests.size()));
            for (std::size_t i = 0; i < requests.size(); ++i) {
                const auto directory = std::filesystem::path(requests[i].path).parent_path();
                for (const auto& file : scan_external_weights(requests[i].path)) {
                    const auto path = (directory / file.location).string();
                    std::error_code error;
                    const auto size = std::filesystem::file_size(path, error);
                    if (error) {
                        throw_error(ErrorCode::ModelLoadError,
                                    "external weight file '" + path + "' (referenced by " +
                                        requests[i].path + ") is not readable: " + error.message());
                    }
                    if (classify_external_weights(file, size) == ExternalWeightLayout::Fp16Packed) {
                        packed[i].push_back({file.location, path});
                        upcast.try_emplace(path);
                    }
                }
            }
        }
        for (auto& [path, buffer] : upcast) {
            ScopedStageTimer timer(timing, "load/fp16_upcast");
            buffer = read_fp16_weights_as_fp32(path);
        }
    }

    for (std::size_t i = 0; i < requests.size(); ++i) {
        const auto& request = requests[i];
        ScopedStageTimer timer(timing, request.stage);
        std::unique_ptr<ModelSession> session;
        if (packed[i].empty()) {
            session = factory(request.path, config);
        } else {
            std::vector<ExternalWeightBuffer> weights;
            weights.reserve(packed[i].size());
            for (const auto& ref : packed[i]) {
                const auto& buffer = upcast.at(ref.path);
                weights.push_back({ref.location, buffer.data.get(), buffer.size_bytes()});
            }
            // 直接走真实后端：SessionFactory 的 (path, config) 签名带不了
            // 内存权重，注入的假会话也消费不了它们。
            session = create_session(request.path, config, weights);
        }
        if (!session) {
            throw_error(ErrorCode::ModelLoadError,
                        "session factory returned no session for: " + request.path);
        }
        *request.target = std::move(session);
    }
}

}  // namespace genie::detail
