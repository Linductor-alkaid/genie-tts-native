// src/runtime/reference.cpp

#include "genie/reference.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "genie/errors.hpp"

namespace genie {
namespace {

/// 读整个文本文件。
[[nodiscard]] std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw_error(ErrorCode::IoError, "reference file not found: " + path.string());
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

}  // namespace

ReferenceCondition load_reference(const std::string& root, const text::TextFrontend& frontend,
                                  int sample_rate) {
    // 注意写法：写成 `path base(root) / "reference"` 会被解析成函数声明
    // （最令人烦恼的解析），这里显式给初值。
    const std::filesystem::path base = std::filesystem::path(root) / "reference";
    const std::filesystem::path audio_path = base / "audio.wav";
    const std::filesystem::path text_path = base / "text.txt";

    ReferenceCondition condition;

    // 1) 参考波形。
    if (!std::filesystem::exists(audio_path)) {
        throw_error(ErrorCode::IoError,
                    "reference audio not found: " + audio_path.string() +
                        "（音色包需要 reference/audio.wav 作为克隆音色的前提）");
    }
    AudioBuffer audio = WavCodec::read(audio_path.string());
    if (audio.sample_rate != sample_rate) {
        audio.samples = resample(audio.samples, audio.sample_rate, sample_rate);
        audio.sample_rate = sample_rate;
    }
    condition.audio = std::move(audio);

    // 16 kHz 版本：HuBERT 与说话人验证模型要这个采样率。
    condition.audio_16k.samples =
        resample(condition.audio.samples, condition.audio.sample_rate, 16000);
    condition.audio_16k.sample_rate = 16000;

    // 2) 参考文本 → 音素。
    if (std::filesystem::exists(text_path)) {
        condition.text = read_text_file(text_path);
        const auto sequence = frontend.convert(condition.text);
        condition.phoneme_ids = sequence.ids;
    }

    // 3) 可选特征：没有对应模型时退化为**零张量**。
    //
    // 上游在没有 chinese-roberta / chinese-hubert 时也是走全零特征这条路
    // （见上游 GetPhonesAndBert.py 的降级分支）。
    //
    // 形状必须显式给对 —— 形状错了模型不会报错，只会输出垃圾：
    //   ref_bert     [T, 1024]     T = 参考音素数（没有则 1）—— **秩 2**
    //   ssl_content  [1, 768, T']   T' = 参考音频帧数
    //
    // 注意 ref_bert 是秩 2：上游 ONNX 图声明的是 `ref_bert: [ref_length, 1024]`，
    // 前面**没有** batch 维。
    const auto phoneme_length = condition.phoneme_ids.empty()
                                    ? std::int64_t{1}
                                    : static_cast<std::int64_t>(condition.phoneme_ids.size());
    condition.bert = Tensor::zeros(ElementType::Float32, {phoneme_length, kBertDimension});

    // 帧数语义对齐上游 HuBERT：输入是 16 kHz 波形、步长 320 采样 →
    // 50 帧/秒（T' = 16k 采样数 / 320）。上游的 ssl_content 由
    // chinese-hubert-base.run({'input_values': audio_16k}) 产出
    //（见上游 ReferenceAudio.__init__），帧数跟着 **16 kHz** 波形走 ——
    // 不能拿 32 kHz 波形除 320，那会多出一倍的零帧，稀释韵律注意力。
    const auto ssl_frames = std::max<std::int64_t>(
        1, static_cast<std::int64_t>(condition.audio_16k.samples.size()) / 320);
    condition.ssl = Tensor::zeros(ElementType::Float32, {1, kSslChannels, ssl_frames});

    return condition;
}

}  // namespace genie
