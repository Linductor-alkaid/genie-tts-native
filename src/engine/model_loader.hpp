// src/engine/model_loader.hpp
//
// 引擎构造期的批量模型会话加载 —— 只在 src/ 内部用。
//
// 为什么单独成模块：会话初始化是冷启动的大头（fp32 包里占 ~85%），fp16
// 包还多一步外部权重上转。把"加载哪些模型"（Engine 的装配职责）与
// "怎么加载"（扫描 / 上转 / 建会话的编排）分开，后者才能单独优化
// 而不动 Engine 的装配逻辑。

#ifndef GENIE_ENGINE_MODEL_LOADER_HPP
#define GENIE_ENGINE_MODEL_LOADER_HPP

#include <memory>
#include <string>
#include <vector>

#include "genie/config.hpp"
#include "genie/engine.hpp"
#include "genie/session.hpp"
#include "genie/timing.hpp"

namespace genie::detail {

/// 一个待加载的模型会话。
struct ModelLoadRequest {
    /// 模型文件路径。
    std::string path;
    /// 计时阶段名（"load/<角色>"）。
    std::string stage;
    /// 加载结果写到这里（由调用方持有，须在加载期间有效）。
    std::unique_ptr<ModelSession>* target = nullptr;
};

/// 批量加载模型会话，阶段耗时记进 timing。
///
/// - fp32 包：逐个走 factory（测试可注入假会话）。
/// - fp16 包（upcast_fp16_weights）：先流式扫描各模型的外部权重（毫秒级），
///   按文件判定存储格式；fp16 打包的文件**各上转一次** —— 多个模型共用的
///   文件（t2s 两个解码器的 t2s_shared）只转一次 —— 再建会话并以内存缓冲
///   替换这些文件。不需要上转的模型（自包含、fp32 外部文件）照常走
///   factory。上转缓冲在全部会话建完后释放（后端建会话时已拷贝）。
///
/// @throws genie::Error 任一模型加载失败（调用方持有的、已建成的会话
///         随调用方的异常路径释放）
void load_model_sessions(const std::vector<ModelLoadRequest>& requests,
                         const SessionFactory& factory, const SessionConfig& config,
                         bool upcast_fp16_weights, TimingReport& timing);

}  // namespace genie::detail

#endif  // GENIE_ENGINE_MODEL_LOADER_HPP
