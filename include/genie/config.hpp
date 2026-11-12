// genie/config.hpp — 合成配置与语言模型
//
// 重要事实（见 docs/upstream-audit.md）：
//   上游 Genie-TTS 的 T2S 采样（TopK / 温度 / 重复惩罚）**在 ONNX 图内完成**，
//   first_stage_decoder 与 stage_decoder 各含一个 TopK 算子。因此这里没有
//   主机侧 top_k / top_p / temperature 参数 —— 那些属于图内常量，改动它们
//   需要重新导出模型，而不是改配置。
//
//   唯一可由主机控制的随机性来源是 ONNX Runtime 的全局随机种子。

#ifndef GENIE_CONFIG_HPP
#define GENIE_CONFIG_HPP

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace genie {

/// 支持的语言。映射到上游 G2P 的四条前端路径。
enum class Language {
    Auto,                  ///< 自动检测（中文文本含 ASCII 字母时进入混读模式）
    Chinese,               ///< 中文
    English,               ///< 英文
    Japanese,              ///< 日文
    Korean,                ///< 韩文
    HybridChineseEnglish,  ///< 中英混读（上游 'hybrid-chinese-english'）
};

/// 语言名 ↔ 枚举互转。未知语言返回 std::nullopt。
[[nodiscard]] std::optional<Language> language_from_string(std::string_view text) noexcept;

/// 规范化的小写语言名（"zh" / "en" / "ja" / "ko" / "hybrid-chinese-english"）。
[[nodiscard]] std::string language_to_string(Language language);

/// 根据文本内容猜测语言。规则与上游 GetPhonesAndBert.get_phones_and_bert 一致：
///   - 显式指定 Chinese 且文本含 ASCII 字母 → 升级为 HybridChineseEnglish
///   - Auto：含假名 → Japanese；含谚文 → Korean；含汉字 → Chinese；否则 English
[[nodiscard]] Language detect_language(std::string_view text, Language requested = Language::Auto);

/// 输出音频格式。
enum class AudioFormat {
    WavPcm16,  ///< WAV / PCM 16-bit
    Float32,   ///< 裸 float32（不做 WAV 封装）
};

/// 一次合成请求的参数。
struct SynthesisOptions {
    /// 语言。默认 Auto。
    Language language = Language::Auto;

    /// 随机种子。设置后通过 ONNX Runtime 全局种子影响图内采样，可复现。
    std::optional<std::uint64_t> seed;

    /// 语义 token 上限。上游 MAX_T2S_LEN = 1000，解码循环上限 500。
    int max_semantic_tokens = 1000;

    /// 是否启用中文 RoBERTa 韵律特征。缺失模型时自动降级为全零特征。
    bool enable_bert = true;

    /// 输出采样率（Hz）。VITS 输出固定 32000。
    int output_sample_rate = 32000;

    /// 输出格式。
    AudioFormat format = AudioFormat::WavPcm16;

    /// 归一化输出音量到 [-1, 1]。仅影响幅度，不改变时长。
    bool normalize_output = false;
};

/// 流式合成的分段参数（M6 低延迟流式）。
///
/// 分段只在**流式**路径生效：同步 `synthesize()` 不分句（上游一致性优先，
/// 整段文本仍是一次解码）；流式路径把文本切成句段、逐段合成后立刻
/// 推送音频 —— 等价于上游"长文本调用方自己分句多次合成再拼音频"
/// 的用法（docs/upstream-audit.md §24.1/§24.3）。
struct StreamOptions {
    /// 单段字符数上限（按码点计，非字节）。0 = 不分句（整段一个块，
    /// 即 M6 之前的旧行为）。默认 100：一段 100 字的语义 token 约
    /// 100~200 个，离解码上限（500 次）还有安全余量。
    std::size_t max_segment_chars = 100;

    /// 句间静音时长（毫秒）。插在**前一段**音频的尾部（除最后一段），
    /// 模拟自然句间停顿；0 = 不插。静音按引擎的实际采样率折算样本数
    /// （不是 `output_sample_rate` 的标称值 —— 后者目前只是改标）。
    int sentence_silence_ms = 150;
};

/// ONNX Runtime 会话级配置。用于 CPU 线程与图优化调优（M5.3）。
struct SessionConfig {
    /// intra-op 线程数。0 = 由 ORT 自行决定。
    int intra_op_threads = 0;
    /// inter-op 线程数。0 = 由 ORT 自行决定。
    int inter_op_threads = 0;
    /// 图优化级别。0=禁用 1=基础 2=扩展 3=全部。
    int graph_optimization_level = 3;
    /// 执行模式：false = Sequential，true = Parallel。
    bool parallel_execution = false;
    /// 是否启用内存模式（复用分配器）。
    bool enable_mem_pattern = true;
    /// 是否启用 CPU 内存 arena。
    bool enable_cpu_mem_arena = true;
    /// intra-op 线程空闲时是否自旋等待（ORT 的
    /// session.intra_op.allow_spinning）。自旋用 CPU 常驻占用换同步延迟：
    /// 引擎一次合成只有**一个**会话在干活，其余会话的自旋线程纯属内耗。
    /// M5.3 实测（i9-13900KF，docs/performance.md §参数扫描）：关掉自旋
    /// 延迟持平或更优（短句 12.5 vs 14.4 ms/token），核占用降至 1/3–1/4；
    /// 高线程数 × 自旋还是本机已知的不稳定区（ORT 线程超订阅历史上在
    /// 这台机器上出过段错误/浮点异常，见 tools/cli 的注释）。默认**关**。
    bool allow_spinning = false;
    /// 建会话时把各节点的权重预打包（PrePack）摊到本会话的 intra-op 线程池上
    /// 并行做（ORT 的 session.prepack.enable_parallel —— **1.31 起**才有这个
    /// 键，更早的版本（含上游钉的 1.22）会忽略它、等价于关闭；且只在
    /// intra_op_threads > 1 时生效）。
    ///
    /// 实测（i9-13900KF，ORT 1.31，docs/performance.md §加载期并行）：冷启动
    /// -8%（fp32 包三轮均值 3.77 → 3.46 s），但**常驻**内存 +0.5 GB（RSS
    /// 时间线显示加载后不回落）—— 用内存换启动时间，默认**关**；内存宽裕、
    /// 冷启动敏感的部署可以打开。
    bool parallel_prepack = false;
    /// ORT 内建 profiler 的 trace 输出目录（M5 子模型级性能分析的采集端）。
    /// 空 = 关闭（默认）。非空时每个会话在该目录下落一份 Chrome trace JSON
    /// （文件名以模型文件 stem 开头，ORT 追加时间戳与 pid 后缀 —— 一个进程
    /// 的多个会话互不覆盖），用 tools/bench/summarize_profile.py 聚合成
    /// 算子级耗时表。profiler 每次 Run 都追加记录，只该在测量场景开启；
    /// 目录由调用方保证存在（genie-bench --profile 会自建）。
    std::string profile_output_dir;
    /// IOBinding 输出缓冲池（M5 解码热路径第二轮）。开启后，重复推理的
    /// 会话（stage_decoder 逐步循环是唯一显著场景）在学到相邻步的输出
    /// 形状增长规律后，把输出绑定到自管缓冲，ORT 直接写入、Tensor 接管
    /// —— 宿主侧每步 51 份输出拷贝（KV 随长度线性增长）消失。
    ///
    /// 形状靠会话内**自学习**：前两次 run 走普通路径观测，从相邻差里学
    /// "哪个维度 + 增多少"；预测失准（会话跨合成复用时形状跳变）自动
    /// 回退普通路径重学；连续失准且从未成功过的会话（单次调用的
    /// first_stage 等）永久禁用绑定 —— 两条路径逐字节等价（集成测试
    /// 锁定），等价性不依赖开关。实测（i9-13900KF，ORT 1.31，
    /// docs/performance.md §IOBinding）：长文本 **ms/token -12%**（20.24 →
    /// 17.81）、短文本 -2%，RSS 持平 —— 默认**开**；单次调用会话的绑定
    /// 尝试在内核执行前就被 ORT 拒绝，失败成本可忽略。
    bool enable_io_binding = true;
};

}  // namespace genie

#endif  // GENIE_CONFIG_HPP
