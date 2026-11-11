// src/engine/model_loader.cpp

#include "engine/model_loader.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <map>
#include <system_error>
#include <utility>

#include "engine/stage_timer.hpp"
#include "genie/errors.hpp"
#include "genie/onnx_graph.hpp"
#include "scheduler/parallel.hpp"

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
                         const std::size_t concurrency, TimingReport& timing) {
    // 每个请求要注入的 fp16 打包文件；上转结果按磁盘路径去重。
    std::vector<std::vector<PackedWeightRef>> packed(requests.size());
    std::map<std::string, Fp32WeightBuffer> upcast;

    // 上转请求在扫描阶段才逐个判定（classify 按文件实际字节长识别 fp16
    // 打包），所以"有没有 fp16 请求"要先看一遍 —— 没有（纯 fp32/int8 包）
    // 就连扫描计时条目都不留。
    const bool has_fp16_requests =
        std::any_of(requests.begin(), requests.end(),
                    [](const ModelLoadRequest& request) { return request.upcast_fp16; });

    if (has_fp16_requests) {
        {
            ScopedStageTimer timer(timing, "load/fp16_scan");
            timer.set_count(static_cast<std::uint32_t>(requests.size()));
            for (std::size_t i = 0; i < requests.size(); ++i) {
                if (!requests[i].upcast_fp16) {
                    continue;  // 混合精度包：int8/fp32 请求不参与扫描。
                }
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
        // 上转按文件并行（条目已就位，各写各的 —— map 结构不再变化）。
        std::vector<std::pair<const std::string, Fp32WeightBuffer>*> files;
        files.reserve(upcast.size());
        for (auto& entry : upcast) {
            files.push_back(&entry);
        }
        ScopedStageTimer timer(timing, "load/fp16_upcast");
        timer.set_count(static_cast<std::uint32_t>(files.size()));
        parallel_for(files.size(), concurrency, [&files](const std::size_t i) {
            files[i]->second = read_fp16_weights_as_fp32(files[i]->first);
        });
    }

    // TimingReport 不是线程安全的：各模型的耗时先记进自己的槽位，
    // 阶段结束后按请求顺序提交。
    std::vector<double> milliseconds(requests.size(), 0.0);
    {
        ScopedStageTimer timer(timing, "load/sessions");
        parallel_for(requests.size(), concurrency, [&](const std::size_t i) {
            const auto start = std::chrono::steady_clock::now();
            const auto& request = requests[i];
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
            milliseconds[i] =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                    .count();
        });
    }
    for (std::size_t i = 0; i < requests.size(); ++i) {
        timing.record(requests[i].stage, milliseconds[i]);
    }
}

}  // namespace genie::detail
