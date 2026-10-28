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
};

/// 角色 ↔ 稳定字符串互转。未知角色返回空字符串。
[[nodiscard]] std::string to_string(ModelRole role);
[[nodiscard]] std::optional<ModelRole> model_role_from_string(std::string_view text);

/// 一个模型文件的描述。
struct ModelEntry {
    ModelRole role = ModelRole::T2sEncoder;
    /// 相对音色包根目录的路径。
    std::string file;
    /// 小写十六进制 SHA-256。空表示不校验（不推荐）。
    std::string sha256;

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
