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
};

}  // namespace genie

#endif  // GENIE_CONFIG_HPP
