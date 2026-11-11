// genie/voice_package.hpp — 音色包（voice package）
//
// 音色包是本项目的**部署单元**：一个目录，里面放着一次合成需要的全部资源。
//
// ```text
// voice-package/
// ├── manifest.json          ← 清单（必需）
// ├── models/
// │   ├── t2s_encoder.onnx
// │   ├── t2s_first_stage_decoder.onnx
// │   ├── t2s_stage_decoder.onnx
// │   ├── vits.onnx
// │   ├── prompt_encoder.onnx        # V2ProPlus
// │   ├── chinese-hubert-base.onnx
// │   └── RoBERTa.onnx               # 可选
// ├── tokenizer/
// └── reference/
// ```
//
// manifest 记录：模型版本、语言、采样率、模型文件清单、**校验值**、可选能力。
// 校验值是必须的：模型文件被替换是最常见的部署事故，
// 而 ONNX 加载失败时的报错信息通常看不出根因。

#ifndef GENIE_VOICE_PACKAGE_HPP
#define GENIE_VOICE_PACKAGE_HPP

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "genie/config.hpp"

namespace genie {

/// 模型在推理链路中的角色。
enum class ModelRole {
    T2sEncoder,
    T2sFirstStageDecoder,
    T2sStageDecoder,
    Vits,           ///< 声码器（V2）
    PromptEncoder,  ///< V2ProPlus
    SpeakerVerification,
    ChineseHubert,
    Roberta,
    G2pm,   ///< g2pM 多音字消歧（可选；缺失时回退单字最常用读音）
    Jieba,  ///< jieba 分词（可选；缺失时变调边界退回词表匹配）
    /// 英文 CMU 发音词典容器（可选；缺失时英文 G2P 不可用）。
    EnglishCmuDict,
    /// 英文词性标注权重容器（可选；缺失时同形异音词取默认读音）。
    EnglishPosTagger,
    /// 英文 seq2seq GRU 权重容器（可选；缺失时 OOV 词没有发音）。
    EnglishG2pModel,
    /// 英文 ngram 分词词典目录（可选；缺失时 OOV 词直接交给 GRU）。
    EnglishWordSegment,
    /// OpenJTalk 词典目录（open_jtalk_dic_utf_8-1.11，可选；缺失时日文没有音素）。
    JapaneseDictionary,
};

/// 角色 ↔ 稳定字符串互转。未知角色返回空字符串。
[[nodiscard]] std::string to_string(ModelRole role);
[[nodiscard]] std::optional<ModelRole> model_role_from_string(std::string_view text);

/// 外部权重文件的存储精度。
enum class WeightFormat {
    Fp32,  ///< fp32 权重（.onnx 图直接引用，运行时原样加载）
    Fp16,  ///< fp16 权重（HF 上游的 `*_fp16.bin`）—— 加载期在内存里
           ///< 上转为 FP32 再建会话，包体积减半、运行时仍是 FP32。
           ///< 借鉴上游 Genie-TTS 的分发格式（docs/architecture.md）。
    Int8,  ///< 动态 INT8 权重（onnxruntime.quantization quantize_dynamic 的
           ///< 产物，M5 量化阶梯）。权重常驻从 4 字节/元素降到 1 字节，
           ///< 运行时由 MLAS 的 u8s8 内核计算（AVX_VNNI 直配）。加载路径
           ///< 与 fp32 相同 —— ORT 直读外部数据，没有上转这一步。
           ///< 精度是否无损由精度门控裁决（tools/gate/）。
};

/// 一个模型文件的描述。
struct ModelEntry {
    ModelRole role = ModelRole::T2sEncoder;
    /// 相对音色包根目录的路径。
    std::string file;
    /// 小写十六进制 SHA-256。空表示不校验（不推荐）。
    std::string sha256;

    /// 逐模型精度覆盖（manifest 模型条目的 "weight_format"，缺省跟包级）。
    ///
    /// 混合精度包的核心：量化门控没过的模型保持 fp32，过了的写 int8 ——
    /// 包级 weight_format 只描述默认值，真正的加载行为按逐模型裁决。
    std::optional<WeightFormat> weight_format;

    /// 解析后的绝对路径（由 VoicePackage 填充）。
    std::string absolute_path;
};

/// 音色包清单。
struct VoiceManifest {
    /// 音色包名称。
    std::string name;
    /// 清单格式版本。当前为 1。
    int format_version = 1;
    /// 模型版本（GPT-SoVITS V2 / V2ProPlus 等）。
    std::string model_version;
    /// 默认语言。
    Language language = Language::Auto;
    /// 输出采样率。VITS 固定 32000。
    int sample_rate = 32000;
    /// 能力开关。
    struct Capabilities {
        bool v2proplus = false;  ///< 是否包含 prompt_encoder
        bool roberta = false;    ///< 是否包含中文 RoBERTa
    } capabilities;
    /// 外部权重精度（manifest 的 "weight_format"，缺省 fp32）。
    ///
    /// 自包含模型（如 RoBERTa 的 1.2 GB 单文件）不受此开关影响 ——
    /// fp16 模式只改写带 external data 的模型。
    WeightFormat weight_format = WeightFormat::Fp32;
    /// 模型清单，按角色索引。
    std::map<ModelRole, ModelEntry> models;
};

/// 音色包。
class VoicePackage {
public:
    /// 加载音色包。
    ///
    /// @param root        音色包根目录（含 manifest.json）
    /// @param verify_hash 是否校验模型文件的 SHA-256
    ///
    /// @throws genie::Error NotFound      —— manifest.json 不存在
    /// @throws genie::Error InvalidArgument —— manifest 格式错误或缺少必需模型
    /// @throws genie::Error IoError        —— 清单引用的文件不存在
    /// @throws genie::Error ModelLoadError —— 校验值不匹配
    [[nodiscard]] static VoicePackage load(const std::string& root, bool verify_hash = true);

    /// 从内存中的 manifest 文本加载（便于测试与远程包）。
    ///
    /// @param root          根目录（用于解析相对路径）
    /// @param manifest_json manifest 文本
    [[nodiscard]] static VoicePackage load_from_manifest(const std::string& root,
                                                         const std::string& manifest_json,
                                                         bool verify_hash = true);

    [[nodiscard]] const std::string& root() const noexcept { return root_; }
    [[nodiscard]] const VoiceManifest& manifest() const noexcept { return manifest_; }

    /// 取某个角色的模型条目。
    [[nodiscard]] const ModelEntry* find(ModelRole role) const;

    /// 取某个角色的模型绝对路径。缺失时抛 NotFound。
    ///
    /// @throws genie::Error NotFound
    [[nodiscard]] std::string model_path(ModelRole role) const;

    /// 是否具备某个角色。
    [[nodiscard]] bool has(ModelRole role) const;

private:
    std::string root_;
    VoiceManifest manifest_;
};

}  // namespace genie

#endif  // GENIE_VOICE_PACKAGE_HPP
