// src/engine/engine.cpp

#include "genie/engine.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "genie/errors.hpp"
#include "genie/onnx_graph.hpp"
#include "genie/symbols.hpp"

namespace genie {
namespace {

/// 上游在 `tts()` 里的第一行：`text = '。' + text`（防止漏第一句）。
///
/// 这是上游的实际行为，会影响首音素分布 —— 照搬。
constexpr std::string_view kUpstreamTextPrefix = "\xE3\x80\x82";  // 。

/// 语义 token 的 EOS 阈值。上游：`np.where(semantic_tokens >= 1024)`。
constexpr std::int64_t kEosThreshold = 1024;

/// 上游 T2S 解码循环的上限。
constexpr std::int32_t kMaxT2SIterations = 500;

/// 语义 token 序列的形状：[1, 1, L]。
///
/// 上游用 `np.expand_dims(y[:, -idx:], axis=0)` 得到这个形状，
/// 而 `vits` 的 `pred_semantic` 输入就是它。
[[nodiscard]] Tensor make_pred_semantic(const std::vector<std::int64_t>& tokens) {
    return Tensor::from_int64({1, 1, static_cast<std::int64_t>(tokens.size())}, tokens);
}

/// 截掉第一个 EOS 之后的 token。
///
/// 上游：
/// ```python
/// eos_indices = np.where(semantic_tokens >= 1024)
/// if len(eos_indices[0]) > 0:
///     first_eos_index = eos_indices[-1][0]
///     semantic_tokens = semantic_tokens[..., :first_eos_index]
/// ```
///
/// 注意上游取的是 `eos_indices[-1][0]` —— 形状是 [1, 1, L]，
/// 所以 `eos_indices[0]` 是第 0 维（长度 1），`[-1]` 取最后一个元素。
/// 也就是"第一个 EOS 的位置"。照搬这个语义。
[[nodiscard]] std::vector<std::int64_t> truncate_at_eos(std::vector<std::int64_t> tokens) {
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i] >= kEosThreshold) {
            tokens.resize(i);
            break;
        }
    }
    return tokens;
}

}  // namespace

struct Engine::Impl {
    std::string root;
    VoicePackage package;
    VoiceManifest manifest;
    int sample_rate = 32000;

    std::unique_ptr<ModelSession> encoder;
    std::unique_ptr<ModelSession> first_stage;
    std::unique_ptr<ModelSession> stage;
    std::unique_ptr<ModelSession> vocoder;
    std::unique_ptr<ModelSession> prompt_encoder;    // V2ProPlus
    std::unique_ptr<ModelSession> speaker_verifier;  // V2ProPlus

    T2SDecoder decoder;
    std::unique_ptr<text::TextFrontend> frontend;

    /// 最近一次流式合成的任务 id。
    std::atomic<Scheduler::TaskId> streaming_task{0};

    /// 参考音频条件（来自音色包）。
    /// 没有参考音频时退化为零张量 —— 与上游 `ssl_content` 的缺省行为一致。
    Tensor ref_seq;
    Tensor ref_bert;
    Tensor ssl_content;
    Tensor ref_audio;

    /// V2ProPlus 的全局 embedding。
    Tensor global_emb;
    Tensor global_emb_advanced;

    [[nodiscard]] bool v2proplus() const noexcept { return manifest.capabilities.v2proplus; }
};

Engine::Engine(std::string root) : Engine(std::move(root), SessionConfig{}) {}

Engine::Engine(std::string root, const SessionConfig& config) {
    Engine(
        std::move(root),
        [](const std::string& path, const SessionConfig& cfg) { return create_session(path, cfg); },
        config);
}

Engine::Engine(std::string root, SessionFactory factory, const SessionConfig& config) {
    auto impl = std::make_unique<Impl>();
    impl->root = std::move(root);

    // 1) 音色包。
    impl->package = VoicePackage::load(impl->root, /*verify_hash=*/true);
    impl->manifest = impl->package.manifest();
    impl->sample_rate = impl->manifest.sample_rate > 0 ? impl->manifest.sample_rate : 32000;

    // 2) 模型会话。
    const auto load = [&factory, &config](const std::string& path) {
        auto session = factory(path, config);
        if (!session) {
            throw_error(ErrorCode::ModelLoadError,
                        "session factory returned no session for: " + path);
        }
        return session;
    };

    impl->encoder = load(impl->package.model_path(ModelRole::T2sEncoder));
    impl->first_stage = load(impl->package.model_path(ModelRole::T2sFirstStageDecoder));
    impl->stage = load(impl->package.model_path(ModelRole::T2sStageDecoder));
    impl->vocoder = load(impl->package.model_path(ModelRole::Vits));

    if (impl->v2proplus()) {
        impl->prompt_encoder = load(impl->package.model_path(ModelRole::PromptEncoder));
        impl->speaker_verifier = load(impl->package.model_path(ModelRole::SpeakerVerification));
    }

    // 3) 文本前端。
    impl->frontend = text::make_frontend(impl->manifest.language);
    impl->decoder = T2SDecoder(T2SOptions{kMaxT2SIterations, kEosThreshold});

    impl_ = std::move(impl);
}

Engine::~Engine() {
    // 流式任务捕获了 this（引擎不是线程安全的，一个引擎同时只跑一个流式合成）。
    // 引擎析构时必须**等任务收尾**，否则任务会访问已析构的对象。
    //
    // 注意：即使 cancel_streaming() 已经取消过，这里也要再等一次 ——
    // 取消只是置标志位，任务可能还在跑。
    const auto task_id = impl_->streaming_task.load();
    if (task_id == 0) {
        return;
    }
    auto& scheduler = Scheduler::instance();
    scheduler.cancel(task_id);
    scheduler.wait(task_id, std::chrono::seconds(5));
    impl_->streaming_task.store(0);
}

const VoiceManifest& Engine::manifest() const noexcept {
    return impl_->manifest;
}

int Engine::sample_rate() const noexcept {
    return impl_->sample_rate;
}

SynthesisResult Engine::synthesize(std::string_view text, const SynthesisOptions& options) {
    return synthesize_impl(text, options, nullptr);
}

SynthesisResult Engine::synthesize_impl(std::string_view text, const SynthesisOptions& options,
                                        const CancellationToken* cancelled) {
    if (cancelled != nullptr && cancelled->cancelled()) {
        throw_error(ErrorCode::Cancelled, "synthesis cancelled before it started");
    }
    if (text.empty()) {
        throw_error(ErrorCode::InvalidArgument, "synthesis text is empty");
    }
    if (impl_->frontend == nullptr) {
        throw_error(
            ErrorCode::BackendUnavailable,
            "no text frontend for language: " + language_to_string(impl_->manifest.language));
    }

    // 1) 语言与文本前端。
    const auto language = detect_language(text, options.language);
    const std::string prepared = std::string(kUpstreamTextPrefix) + std::string(text);
    const auto phonemes = impl_->frontend->convert(prepared);

    SynthesisResult result;
    result.language = language;
    result.phoneme_ids = phonemes.ids;

    if (phonemes.ids.empty()) {
        throw_error(ErrorCode::InvalidArgument, "text produced no phonemes");
    }

    // 2) T2S：音素 + 参考条件 → 语义 token。
    TensorFeed encoder_feed;
    encoder_feed.emplace_back("text_seq", make_pred_semantic(phonemes.ids));
    encoder_feed.emplace_back("ref_seq", impl_->ref_seq);
    encoder_feed.emplace_back("text_bert", impl_->ref_bert);
    encoder_feed.emplace_back("ref_bert", impl_->ref_bert);
    encoder_feed.emplace_back("ssl_content", impl_->ssl_content);

    auto semantic = impl_->decoder.decode(*impl_->encoder, *impl_->first_stage, *impl_->stage,
                                          encoder_feed, cancelled);
    semantic = truncate_at_eos(std::move(semantic));
    result.semantic_tokens = semantic;

    if (semantic.empty()) {
        throw_error(ErrorCode::InvalidArgument, "t2s produced no semantic tokens");
    }

    // 3) 声码器：语义 token → 波形。
    TensorFeed vocoder_feed;
    vocoder_feed.emplace_back("text_seq", make_pred_semantic(phonemes.ids));
    vocoder_feed.emplace_back("pred_semantic", make_pred_semantic(semantic));
    if (impl_->v2proplus()) {
        vocoder_feed.emplace_back("ge", impl_->global_emb);
        vocoder_feed.emplace_back("ge_advanced", impl_->global_emb_advanced);
    } else {
        vocoder_feed.emplace_back("ref_audio", impl_->ref_audio);
    }

    const auto audio = impl_->vocoder->run(vocoder_feed);
    const auto* wave = find_output(audio, impl_->vocoder->outputs(), "audio");
    if (wave == nullptr) {
        throw_error(ErrorCode::InvalidArgument, "vocoder has no 'audio' output");
    }

    const auto samples = wave->floats();
    result.audio.samples.assign(samples.begin(), samples.end());
    result.audio.sample_rate =
        options.output_sample_rate > 0 ? options.output_sample_rate : impl_->sample_rate;
    return result;
}

std::shared_ptr<AudioChunkChannel> Engine::synthesize_streaming(std::string_view text,
                                                                const SynthesisOptions& options) {
    auto channel = std::make_shared<AudioChunkChannel>();

    // 调度器是进程级单例，initialize() 幂等 —— 这里保证它已经就绪，
    // 让调用方不必自己初始化。
    auto& scheduler = Scheduler::instance();
    if (!scheduler.initialized()) {
        scheduler.initialize({});
    }

    // 合成任务在调度器（kairo）上跑；调用线程只负责消费音频。
    // 引擎本身不是线程安全的，所以这里捕获的是 this —— 同一个 Engine
    // 不应该同时跑两个流式合成。
    const std::string owned_text(text);
    auto* target = this;
    // 任务持有通道的 shared_ptr：调用方先放手也不会让通道在任务还在写的时候析构。
    const auto task_id = Scheduler::instance().submit(
        [target, owned_text, options, channel](const CancellationToken& token) {
            try {
                const auto result = target->synthesize_impl(owned_text, options, &token);
                channel->try_push(result.audio);
            } catch (...) {
                // 错误通过"通道提前关闭"传达给消费者。
            }
            channel->close();
        });

    impl_->streaming_task.store(task_id);
    return channel;
}

bool Engine::cancel_streaming() {
    // 注意：**不**清空 task id —— 析构时还要靠它等待任务收尾。
    const auto task_id = impl_->streaming_task.load();
    if (task_id == 0) {
        return false;
    }
    return Scheduler::instance().cancel(task_id);
}

}  // namespace genie
