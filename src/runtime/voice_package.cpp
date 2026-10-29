// src/runtime/voice_package.cpp — 音色包加载

#include "genie/voice_package.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <utility>

#include "genie/errors.hpp"
#include "genie/json.hpp"
#include "genie/sha256.hpp"

namespace genie {
namespace {

/// 必需的模型角色。缺任何一个都无法完成端到端合成。
///
/// 注意：chinese-hubert-base 虽不在这个列表里，但缺少它时
/// `ssl_content` 只能退化为零张量 —— 属于"可降级"，不是"不可用"。
constexpr ModelRole kRequiredRoles[] = {
    ModelRole::T2sEncoder,
    ModelRole::T2sFirstStageDecoder,
    ModelRole::T2sStageDecoder,
    ModelRole::Vits,
};

/// 解析 manifest 中的 models 数组。
std::map<ModelRole, ModelEntry> parse_models(const json::Value& root) {
    std::map<ModelRole, ModelEntry> models;
    const json::Value* array = root.find("models");
    if (array == nullptr || !array->is_array()) {
        throw_error(ErrorCode::InvalidArgument, "manifest: 'models' must be an array");
    }

    for (const auto& item : array->as_array()) {
        if (!item.is_object()) {
            throw_error(ErrorCode::InvalidArgument, "manifest: each model entry must be an object");
        }
        const std::string role_name = item.get_string("role");
        const auto role = model_role_from_string(role_name);
        if (!role.has_value()) {
            throw_error(ErrorCode::InvalidArgument,
                        "manifest: unknown model role '" + role_name + "'");
        }
        const std::string file = item.get_string("file");
        if (file.empty()) {
            throw_error(ErrorCode::InvalidArgument,
                        "manifest: model '" + role_name + "' has no 'file'");
        }
        ModelEntry entry;
        entry.role = *role;
        entry.file = file;
        entry.sha256 = item.get_string("sha256");
        models[*role] = std::move(entry);
    }
    return models;
}

VoiceManifest parse_manifest(const json::Value& root) {
    VoiceManifest manifest;
    manifest.name = root.get_string("name", "unnamed");
    manifest.format_version = static_cast<int>(root.get_int("format_version", 1));
    if (manifest.format_version != 1) {
        throw_error(ErrorCode::InvalidArgument, "manifest: unsupported format_version " +
                                                    std::to_string(manifest.format_version) +
                                                    " (this build supports 1)");
    }
    manifest.model_version = root.get_string("model_version", "unknown");

    const std::string language_name = root.get_string("language", "auto");
    const auto language = language_from_string(language_name);
    if (!language.has_value()) {
        throw_error(ErrorCode::InvalidArgument,
                    "manifest: unknown language '" + language_name + "'");
    }
    manifest.language = *language;

    manifest.sample_rate = static_cast<int>(root.get_int("sample_rate", 32000));
    if (manifest.sample_rate <= 0) {
        throw_error(ErrorCode::InvalidArgument, "manifest: sample_rate must be positive");
    }

    if (const json::Value* capabilities = root.find("capabilities"); capabilities != nullptr) {
        if (!capabilities->is_object()) {
            throw_error(ErrorCode::InvalidArgument, "manifest: 'capabilities' must be an object");
        }
        manifest.capabilities.v2proplus = capabilities->get_bool("v2proplus", false);
        manifest.capabilities.roberta = capabilities->get_bool("roberta", false);
    }

    manifest.models = parse_models(root);

    for (const ModelRole role : kRequiredRoles) {
        if (manifest.models.find(role) == manifest.models.end()) {
            throw_error(ErrorCode::InvalidArgument,
                        "manifest: missing required model role '" + to_string(role) + "'");
        }
    }
    return manifest;
}

}  // namespace

std::string to_string(ModelRole role) {
    switch (role) {
        case ModelRole::T2sEncoder:
            return "t2s_encoder";
        case ModelRole::T2sFirstStageDecoder:
            return "t2s_first_stage_decoder";
        case ModelRole::T2sStageDecoder:
            return "t2s_stage_decoder";
        case ModelRole::Vits:
            return "vits";
        case ModelRole::PromptEncoder:
            return "prompt_encoder";
        case ModelRole::SpeakerVerification:
            return "speaker_verification";
        case ModelRole::ChineseHubert:
            return "chinese_hubert";
        case ModelRole::Roberta:
            return "roberta";
        case ModelRole::G2pm:
            return "g2pm";
        case ModelRole::Jieba:
            return "jieba";
        case ModelRole::EnglishCmuDict:
            return "english_cmu_dict";
        case ModelRole::EnglishPosTagger:
            return "english_pos_tagger";
        case ModelRole::EnglishG2pModel:
            return "english_g2p_model";
        case ModelRole::EnglishWordSegment:
            return "english_word_segment";
        case ModelRole::JapaneseDictionary:
            return "japanese_dictionary";
    }
    return "";
}

std::optional<ModelRole> model_role_from_string(std::string_view text) {
    constexpr ModelRole kAll[] = {
        ModelRole::T2sEncoder,
        ModelRole::T2sFirstStageDecoder,
        ModelRole::T2sStageDecoder,
        ModelRole::Vits,
        ModelRole::PromptEncoder,
        ModelRole::SpeakerVerification,
        ModelRole::ChineseHubert,
        ModelRole::Roberta,
        ModelRole::G2pm,
        ModelRole::Jieba,
        ModelRole::EnglishCmuDict,
        ModelRole::EnglishPosTagger,
        ModelRole::EnglishG2pModel,
        ModelRole::EnglishWordSegment,
        ModelRole::JapaneseDictionary,
    };
    for (const ModelRole role : kAll) {
        if (to_string(role) == text) {
            return role;
        }
    }
    return std::nullopt;
}

VoicePackage VoicePackage::load(const std::string& root, bool verify_hash) {
    const std::filesystem::path manifest_path = std::filesystem::path(root) / "manifest.json";
    std::ifstream file(manifest_path, std::ios::binary);
    if (!file) {
        throw_error(ErrorCode::NotFound,
                    "voice package manifest not found: " + manifest_path.string());
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    return load_from_manifest(root, buffer.str(), verify_hash);
}

VoicePackage VoicePackage::load_from_manifest(const std::string& root,
                                              const std::string& manifest_json, bool verify_hash) {
    const json::Value document = json::parse(manifest_json);
    if (!document.is_object()) {
        throw_error(ErrorCode::InvalidArgument, "manifest: root must be a JSON object");
    }

    VoicePackage package;
    package.root_ = root;
    package.manifest_ = parse_manifest(document);

    const std::filesystem::path base(root);
    for (auto& [role, entry] : package.manifest_.models) {
        const std::filesystem::path path = base / entry.file;
        entry.absolute_path = path.string();

        std::error_code ec;
        if (!std::filesystem::exists(path, ec)) {
            throw_error(ErrorCode::IoError,
                        "voice package: model file not found: " + path.string());
        }

        if (verify_hash && !entry.sha256.empty()) {
            const std::string actual = to_hex(sha256_file(path.string()));
            if (actual != entry.sha256) {
                throw_error(ErrorCode::ModelLoadError, "voice package: checksum mismatch for " +
                                                           path.string() + " (expected " +
                                                           entry.sha256 + ", got " + actual + ")");
            }
        }
    }
    return package;
}

const ModelEntry* VoicePackage::find(ModelRole role) const {
    const auto it = manifest_.models.find(role);
    return it == manifest_.models.end() ? nullptr : &it->second;
}

std::string VoicePackage::model_path(ModelRole role) const {
    const ModelEntry* entry = find(role);
    if (entry == nullptr) {
        throw_error(ErrorCode::NotFound,
                    "voice package has no model for role '" + to_string(role) + "'");
    }
    return entry->absolute_path;
}

bool VoicePackage::has(ModelRole role) const {
    return find(role) != nullptr;
}

}  // namespace genie
