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

#include <cstddef>
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
    /// 该模型的权重是否需要 fp16 上转。
    ///
    /// 这是**逐模型**的标志而不是包级开关：混合精度包（M5 量化，逐模型
    /// weight_format 覆盖）里 fp16 与 int8/fp32 可以共存；int8 权重必须
    /// 原样交给 ORT，上转只该发生在真正 fp16 打包的请求上。
    bool upcast_fp16 = false;
};

/// 批量加载模型会话，阶段耗时记进 timing。
///
/// - fp32 / int8 权重：逐个走 factory（测试可注入假会话）—— int8 外部
///   数据由 ORT 原样直读，与 fp32 同一条路径。
/// - 标记 upcast_fp16 的请求：先流式扫描外部权重（毫秒级），按文件判定
///   存储格式；fp16 打包的文件**各上转一次** —— 多个模型共用的文件
///   （t2s 两个解码器的 t2s_shared）只转一次 —— 再建会话并以内存缓冲
///   替换这些文件。不需要上转的模型（自包含、fp32 外部文件）照常走
///   factory。上转缓冲在全部会话建完后释放（后端建会话时已拷贝）。
///
/// 上转（按文件）与建会话（按模型）各自最多 concurrency 路并行
/// （detail::parallel_for；1 = 串行）。各项按下标领活，调用方应把耗时
/// 大的请求排在前面 —— 并行阶段的墙钟时间由最慢的那个决定。并行时
/// factory 会被多个线程并发调用（kairo 默认池不绑核，ORT 会话的
/// intra-op 线程不受影响 —— architecture.md §2.14）。
///
/// 计时："load/sessions" 是建会话阶段的墙钟时间；各模型自己的条目
/// （requests[i].stage）在并行时相互重叠，之和会超过它。
///
/// @throws genie::Error 任一模型加载失败（下标最小的那个错误；调用方
///         持有的、已建成的会话随调用方的异常路径释放）
void load_model_sessions(const std::vector<ModelLoadRequest>& requests,
                         const SessionFactory& factory, const SessionConfig& config,
                         std::size_t concurrency, TimingReport& timing);

}  // namespace genie::detail

#endif  // GENIE_ENGINE_MODEL_LOADER_HPP
