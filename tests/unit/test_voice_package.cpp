// tests/unit/test_voice_package.cpp

#include <gtest/gtest.h>

#include <unistd.h>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "genie/errors.hpp"
#include "genie/sha256.hpp"
#include "genie/voice_package.hpp"

using namespace genie;

namespace {

/// 临时音色包夹具。
class TempPackage {
public:
    TempPackage() {
        static int counter = 0;
        root_ = std::filesystem::temp_directory_path() /
                ("genie_voicepkg_" + std::to_string(::getpid()) + "_" + std::to_string(counter++));
        std::filesystem::create_directories(root_ / "models");
        root_str_ = root_.string();
    }

    ~TempPackage() {
        std::error_code ec;
        std::filesystem::remove_all(root_, ec);
    }

    void write_model(const std::string& name, const std::string& content) const {
        std::ofstream file(root_ / "models" / name, std::ios::binary);
        file << content;
    }

    void write_manifest(const std::string& content) const {
        std::ofstream file(root_ / "manifest.json", std::ios::binary);
        file << content;
    }

    [[nodiscard]] const std::string& root() const { return root_str_; }

private:
    std::filesystem::path root_;
    std::string root_str_;
};

/// 生成一份 manifest。参数控制模型清单与校验值。
struct ManifestOptions {
    bool include_vits = true;
    bool include_prompt_encoder = false;
    std::string encoder_sha256;         ///< 非空则写入 t2s_encoder 的校验值
    bool wrong_encoder_sha256 = false;  ///< 写入一个错误的校验值
    bool omit_vits_role = false;        ///< 缺少必需角色
};

std::string build_manifest(const ManifestOptions& options) {
    // 手工拼 JSON 字符串：转义普通字符串比用 raw string 更容易看出边界。
    const std::string quote = "\"";
    std::vector<std::string> entries;

    std::string encoder = "{\"role\":\"t2s_encoder\",\"file\":\"models/t2s_encoder.onnx\"";
    if (!options.encoder_sha256.empty()) {
        encoder += ",\"sha256\":" + quote +
                   (options.wrong_encoder_sha256 ? std::string(64, '0') : options.encoder_sha256) +
                   quote;
    }
    encoder += "}";
    entries.push_back(encoder);

    entries.push_back(
        "{\"role\":\"t2s_first_stage_decoder\",\"file\":\"models/t2s_first_stage_decoder.onnx\"}");
    entries.push_back(
        "{\"role\":\"t2s_stage_decoder\",\"file\":\"models/t2s_stage_decoder.onnx\"}");
    if (options.include_vits && !options.omit_vits_role) {
        entries.push_back("{\"role\":\"vits\",\"file\":\"models/vits.onnx\"}");
    }
    if (options.include_prompt_encoder) {
        entries.push_back("{\"role\":\"prompt_encoder\",\"file\":\"models/prompt_encoder.onnx\"}");
    }

    std::string models = "[";
    for (std::size_t i = 0; i < entries.size(); ++i) {
        if (i != 0) {
            models += ",";
        }
        models += entries[i];
    }
    models += "]";

    std::string manifest = "{";
    manifest += "\"name\":\"test-voice\",";
    manifest += "\"format_version\":1,";
    manifest += "\"model_version\":\"v2\",";
    manifest += "\"language\":\"chinese\",";
    manifest += "\"sample_rate\":32000,";
    manifest += "\"capabilities\":{\"v2proplus\":" +
                std::string(options.include_prompt_encoder ? "true" : "false") + "},";
    manifest += "\"models\":" + models;
    manifest += "}";
    return manifest;
}

/// 填好所有模型文件。
void populate_models(const TempPackage& package) {
    package.write_model("t2s_encoder.onnx", "encoder-bytes");
    package.write_model("t2s_first_stage_decoder.onnx", "first-stage-bytes");
    package.write_model("t2s_stage_decoder.onnx", "stage-bytes");
    package.write_model("vits.onnx", "vits-bytes");
}

}  // namespace

TEST(VoicePackage, LoadsManifestAndResolvesPaths) {
    TempPackage package;
    populate_models(package);
    package.write_manifest(build_manifest({}));

    const auto voice = VoicePackage::load(package.root());

    EXPECT_EQ(voice.manifest().name, "test-voice");
    EXPECT_EQ(voice.manifest().model_version, "v2");
    EXPECT_EQ(voice.manifest().language, Language::Chinese);
    EXPECT_EQ(voice.manifest().sample_rate, 32000);
    EXPECT_FALSE(voice.manifest().capabilities.v2proplus);

    EXPECT_TRUE(voice.has(ModelRole::T2sEncoder));
    EXPECT_TRUE(voice.has(ModelRole::Vits));
    EXPECT_FALSE(voice.has(ModelRole::PromptEncoder));

    const std::string path = voice.model_path(ModelRole::T2sEncoder);
    EXPECT_NE(path.find("t2s_encoder.onnx"), std::string::npos);
    EXPECT_TRUE(std::filesystem::exists(path));
}

TEST(VoicePackage, MissingManifestThrowsNotFound) {
    TempPackage package;
    EXPECT_THROW(VoicePackage::load(package.root()), Error);
}

TEST(VoicePackage, MissingModelFileThrowsIoError) {
    TempPackage package;
    package.write_model("t2s_encoder.onnx", "x");
    // 故意不写 t2s_stage_decoder.onnx
    package.write_manifest(build_manifest({}));
    EXPECT_THROW(VoicePackage::load(package.root()), Error);
}

TEST(VoicePackage, MissingRequiredRoleThrowsInvalidArgument) {
    TempPackage package;
    populate_models(package);
    ManifestOptions options;
    options.omit_vits_role = true;
    package.write_manifest(build_manifest(options));
    EXPECT_THROW(VoicePackage::load(package.root()), Error);
}

TEST(VoicePackage, VerifiesChecksum) {
    TempPackage package;
    populate_models(package);
    ManifestOptions options;
    options.encoder_sha256 = sha256_hex("encoder-bytes");
    package.write_manifest(build_manifest(options));

    EXPECT_NO_THROW(VoicePackage::load(package.root(), /*verify_hash=*/true));
}

TEST(VoicePackage, ChecksumMismatchThrowsModelLoadError) {
    TempPackage package;
    populate_models(package);
    ManifestOptions options;
    options.encoder_sha256 = sha256_hex("encoder-bytes");
    options.wrong_encoder_sha256 = true;
    package.write_manifest(build_manifest(options));

    EXPECT_THROW(VoicePackage::load(package.root(), /*verify_hash=*/true), Error);
}

TEST(VoicePackage, SkipsChecksumVerificationWhenDisabled) {
    TempPackage package;
    populate_models(package);
    ManifestOptions options;
    options.encoder_sha256 = sha256_hex("encoder-bytes");
    options.wrong_encoder_sha256 = true;
    package.write_manifest(build_manifest(options));

    EXPECT_NO_THROW(VoicePackage::load(package.root(), /*verify_hash=*/false));
}

TEST(VoicePackage, RejectsUnknownModelRole) {
    TempPackage package;
    populate_models(package);
    package.write_manifest(R"({
        "name": "bad",
        "models": [
            {"role":"not_a_role","file":"models/x.onnx"},
            {"role":"t2s_encoder","file":"models/t2s_encoder.onnx"},
            {"role":"t2s_first_stage_decoder","file":"models/t2s_first_stage_decoder.onnx"},
            {"role":"t2s_stage_decoder","file":"models/t2s_stage_decoder.onnx"},
            {"role":"vits","file":"models/vits.onnx"}
        ]
    })");
    EXPECT_THROW(VoicePackage::load(package.root()), Error);
}

TEST(VoicePackage, RejectsUnsupportedFormatVersion) {
    TempPackage package;
    populate_models(package);
    package.write_manifest(R"({
        "name": "future",
        "format_version": 99,
        "models": [
            {"role":"t2s_encoder","file":"models/t2s_encoder.onnx"},
            {"role":"t2s_first_stage_decoder","file":"models/t2s_first_stage_decoder.onnx"},
            {"role":"t2s_stage_decoder","file":"models/t2s_stage_decoder.onnx"},
            {"role":"vits","file":"models/vits.onnx"}
        ]
    })");
    EXPECT_THROW(VoicePackage::load(package.root()), Error);
}

TEST(VoicePackage, ModelPathForMissingRoleThrowsNotFound) {
    TempPackage package;
    populate_models(package);
    package.write_manifest(build_manifest({}));
    const auto voice = VoicePackage::load(package.root());
    EXPECT_THROW(voice.model_path(ModelRole::Roberta), Error);
}

TEST(VoicePackage, ModelRoleStringRoundTrip) {
    for (const ModelRole role :
         {ModelRole::T2sEncoder, ModelRole::T2sFirstStageDecoder, ModelRole::T2sStageDecoder,
          ModelRole::Vits, ModelRole::PromptEncoder, ModelRole::SpeakerVerification,
          ModelRole::ChineseHubert, ModelRole::Roberta}) {
        EXPECT_EQ(model_role_from_string(to_string(role)), role);
    }
    EXPECT_FALSE(model_role_from_string("bogus").has_value());
}
