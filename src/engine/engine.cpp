// src/engine/engine.cpp

#include "genie/engine.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <exception>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "engine/g2pm.hpp"
#include "engine/hubert.hpp"
#include "engine/roberta.hpp"
#include "genie/errors.hpp"
#include "genie/onnx_graph.hpp"
#include "genie/reference.hpp"
#include "genie/symbols.hpp"
#include "text/frontend_factory_internal.hpp"

namespace genie {
namespace {

/// 这个语言是否走**中文** RoBERTa ✓。
///
/// 上游 `_get_phones_and_bert_pure_lang` 只在 chinese / hybrid 两个分支
/// 真的跑 RoBERTa ✓；english / korean / japanese 给的是全零 ✓
/// —— RoBERTa 是中文模型 ✓，喂别的语言的音素只会得到对不上的布局 ✓。
[[nodiscard]] bool uses_chinese_bert(Language language) noexcept {
    return language == Language::Chinese || language == Language::HybridChineseEnglish;
}

/// 缺前端时给一条**能指路**的报错 ✓。
///
/// 以前这里是静默回退到中文前端 ✓ —— 日文没有词典时，日语文本被中文
/// 前端的 pattern_filter 过滤成空 ✓，最后表现为"合成出来是静音"✓，
/// 而真正的原因（缺 103 MB 的 OpenJTalk 词典 ✓）没人看得见 ✓。
[[nodiscard]] std::string missing_frontend_hint(Language language) {
    switch (language) {
        case Language::English:
            return "音色包缺少英文发音词典（角色 english_cmu_dict）——"
                   "用 make_voice_package.py --english-cmu-dict 接入";
        case Language::Japanese:
            return "音色包缺少 OpenJTalk 词典（角色 japanese_dictionary，约 103 MB）——"
                   "用 make_voice_package.py --japanese-dictionary 接入";
        case Language::Korean:
            return "韩文前端不需要外部资源，出现这个错误说明构建有问题";
        default:
            return "音色包里没有可用的前端资源";
    }
}

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

/// 音素序列张量：**秩是 2**，形状 `[1, N]`。
///
/// 上游的 `phones_seq = np.array([phones], dtype=np.int64)` → `[1, N]`。
/// 这个张量喂给 t2s_encoder 的 `text_seq` 和 vits 的 `text_seq` —— 两处都是秩 2。
///
/// 别和 `pred_semantic` 搞混：那个是秩 3（`[1, 1, L]`），
/// 因为上游对它做了 `np.expand_dims(y[:, -idx:], axis=0)`。
[[nodiscard]] Tensor make_phones_tensor(const std::vector<std::int64_t>& phonemes) {
    return Tensor::from_int64({1, static_cast<std::int64_t>(phonemes.size())}, phonemes);
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
    std::unique_ptr<ModelSession> roberta;           // 中文韵律（可选）
    std::unique_ptr<ModelSession> hubert;            // 参考音频 ssl_content（可选）

    /// RoBERTa 的分词器，跟随 roberta 会话一起加载。
    std::unique_ptr<text::RobertaTokenizer> roberta_tokenizer;
    std::unique_ptr<g2pm::Model> g2pm_model;         // 多音字消歧（可选角色）
    std::unique_ptr<text::jieba::Tokenizer> jieba_;  // 分词（可选角色）

    T2SDecoder decoder;
    std::unique_ptr<text::TextFrontend> frontend;
    /// 按语言挑前端 ✓。
    ///
    /// **缺资源时返回 nullptr** ✓（而不是静默回退到中文前端 ✓ ——
    /// 静默回退会让"日文没词典"表现为"合成出来是静音" ✓）。
    [[nodiscard]] const text::TextFrontend* frontend_for(Language language) const {
        switch (language) {
            case Language::English:
                return english_frontend.get();
            case Language::Korean:
                return korean_frontend.get();
#ifdef GENIE_HAVE_OPENJTALK
            case Language::Japanese:
                return japanese_frontend.get();
#endif
            default:
                return frontend.get();
        }
    }

    /// 英文前端（音色包里没有英文词典时为空）。
    std::unique_ptr<text::TextFrontend> english_frontend;
    /// 韩文前端（不需要外部资源 ✓ —— 总是创建 ✓）。
    std::unique_ptr<text::TextFrontend> korean_frontend;
#ifdef GENIE_HAVE_OPENJTALK
    /// 日文前端（词典缺失时为空 ✓ —— 前端自己持有 OpenJTalk ✓）。
    std::unique_ptr<text::TextFrontend> japanese_frontend;
#endif

    /// 最近一次流式合成的任务 id。
    std::atomic<Scheduler::TaskId> streaming_task{0};

    /// 参考音频条件（来自音色包）。
    ///
    /// **形状必须显式给对** —— 形状错了模型不会报错，只会输出垃圾。
    /// 韵律 / HuBERT 特征在对应的可选模型缺失时是零张量，
    /// 与上游的降级行为一致。
    ReferenceCondition reference;

    /// V2ProPlus 的全局 embedding。
    Tensor global_emb;
    Tensor global_emb_advanced;

    [[nodiscard]] bool v2proplus() const noexcept { return manifest.capabilities.v2proplus; }

    /// 计算 V2ProPlus 的全局说话人嵌入（ge / ge_advanced）。
    ///
    /// 上游的顺序（ReferenceAudio.update_global_emb）：
    ///   sv_emb = sv_model.run({'waveform': audio_16k})[0]
    ///   ge, ge_advanced = prompt_encoder.run({'ref_audio': audio_32k, 'sv_emb': sv_emb})
    ///
    /// 上游是**第一次推理时**算（算完缓存），这里提前到加载期 ——
    /// 形状不对的话早失败比合成出垃圾好。
    void compute_global_embeddings() {
        // 1) 说话人向量：16 kHz 波形 → sv_emb。
        TensorFeed sv_feed;
        sv_feed.emplace_back(
            "waveform",
            Tensor::from_float32({1, static_cast<std::int64_t>(reference.audio_16k.samples.size())},
                                 reference.audio_16k.samples));
        const auto sv_outputs = speaker_verifier->run(sv_feed);
        if (sv_outputs.empty()) {
            throw_error(ErrorCode::ModelLoadError, "说话人验证模型没有输出");
        }

        // 2) prompt encoder：32 kHz 波形 + sv_emb → ge / ge_advanced。
        TensorFeed encoder_feed;
        encoder_feed.emplace_back(
            "ref_audio",
            Tensor::from_float32({1, static_cast<std::int64_t>(reference.audio.samples.size())},
                                 reference.audio.samples));
        encoder_feed.emplace_back("sv_emb", sv_outputs.front());
        const auto encoder_outputs = prompt_encoder->run(encoder_feed);
        if (encoder_outputs.size() < 2) {
            throw_error(ErrorCode::ModelLoadError,
                        "prompt_encoder 至少要两个输出（ge / ge_advanced），实际 " +
                            std::to_string(encoder_outputs.size()));
        }
        global_emb = encoder_outputs[0];
        global_emb_advanced = encoder_outputs[1];
    }
};

Engine::Engine(std::string root) : Engine(std::move(root), SessionConfig{}) {}

Engine::Engine(std::string root, const SessionConfig& config)
    : Engine(
          std::move(root),
          [](const std::string& path, const SessionConfig& cfg) {
              return create_session(path, cfg);
          },
          config) {}

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
        // 说话人验证模型是 V2ProPlus 的**硬依赖**：
        // 上游的 update_global_emb() 只有在 SV 模型加载成功时才会算
        // global_emb；算不出来就没法喂 vits 的 ge / ge_advanced。
        // 这里在**加载期**就报清楚，而不是等到推理时炸。
        if (!impl->package.has(ModelRole::SpeakerVerification)) {
            throw_error(ErrorCode::NotFound,
                        "V2ProPlus 音色包缺少说话人验证模型（speaker_verification）—— "
                        "没有它算不出 ge / ge_advanced，V2ProPlus 分支不可用");
        }
        impl->speaker_verifier = load(impl->package.model_path(ModelRole::SpeakerVerification));
    }

    // 3) 文本前端。g2pM（多音字消歧）先于前端装配 —— 前端持有它做
    //    上下文读音判定；模型缺失时前端退化为单字最常用读音
    //    （与 RoBERTa / HuBERT 的可选降级同款模式）。
    if (impl->package.has(ModelRole::Jieba)) {
        impl->jieba_ = std::make_unique<text::jieba::Tokenizer>(
            text::jieba::Tokenizer::load(impl->package.model_path(ModelRole::Jieba)));
    }
    // 3.1) 英文 G2P 的资源（都是数据文件，不是模型）。
    //      词典是硬依赖 —— 缺了它就没有英文前端；其余资源缺失时降级
    //      （见 docs/upstream-audit.md §17）。
    text::EnglishG2pResources english_resources;
    if (impl->package.has(ModelRole::EnglishCmuDict)) {
        english_resources.cmu_dict = impl->package.model_path(ModelRole::EnglishCmuDict);
        if (impl->package.has(ModelRole::EnglishPosTagger)) {
            english_resources.pos_tagger = impl->package.model_path(ModelRole::EnglishPosTagger);
        }
        if (impl->package.has(ModelRole::EnglishG2pModel)) {
            english_resources.g2p_model = impl->package.model_path(ModelRole::EnglishG2pModel);
        }
        if (impl->package.has(ModelRole::EnglishWordSegment)) {
            english_resources.word_segment_dir =
                impl->package.model_path(ModelRole::EnglishWordSegment);
        }
        impl->english_frontend = text::make_english_frontend(english_resources);
    }

    // 3.2) 韩文前端（不需要外部资源 ✓ —— 规则与数据都内置 ✓）。
    impl->korean_frontend = text::make_korean_frontend();

    if (impl->package.has(ModelRole::G2pm)) {
        impl->g2pm_model = std::make_unique<g2pm::Model>(
            g2pm::Model::load(impl->package.model_path(ModelRole::G2pm)));
        impl->frontend = text::make_chinese_frontend(impl->g2pm_model.get(), impl->jieba_.get(),
                                                     impl->english_frontend.get());
    } else {
        impl->frontend =
            text::make_chinese_frontend(nullptr, impl->jieba_.get(), impl->english_frontend.get());
    }
    impl->decoder = T2SDecoder(T2SOptions{kMaxT2SIterations, kEosThreshold});

    // 3.5) RoBERTa（中文韵律）：模型与分词器一起加载，二者缺一不可 ——
    // 分词器词表与模型必须配套，加载期就报清楚。
    if (impl->package.has(ModelRole::Roberta)) {
        const auto model_path = impl->package.model_path(ModelRole::Roberta);
        impl->roberta = load(model_path);
        const auto vocab_path = std::filesystem::path(model_path).parent_path() / "vocab.txt";
        impl->roberta_tokenizer = text::RobertaTokenizer::load(vocab_path.string());
    }

    // 3.6) HuBERT（参考音频内容特征）：缺失时 ssl_content 保持全零降级。
    if (impl->package.has(ModelRole::ChineseHubert)) {
        impl->hubert = load(impl->package.model_path(ModelRole::ChineseHubert));
    }

    // 4) 参考音频条件。
    //
    // 参考文本按**音色包声明的语言**转音素 ✓ —— 上游的
    // `ReferenceAudio.set_text(prompt_text, language)` 用的就是它 ✓。
    // 以前固定用中文前端 ✓ —— 日文音色包的参考文本会被转成拼音 ✓。
    const text::TextFrontend* reference_frontend = impl->frontend_for(impl->manifest.language);
    if (reference_frontend == nullptr) {
        throw_error(
            ErrorCode::BackendUnavailable,
            "no text frontend for language: " + language_to_string(impl->manifest.language) +
                " —— " + missing_frontend_hint(impl->manifest.language));
    }
    {
        impl->reference = load_reference(impl->root, *reference_frontend, impl->sample_rate);

        // 参考音频的内容特征：有 HuBERT 就算真的，没有保持全零降级
        // （与上游 ReferenceAudio.__init__ 一致）。
        if (impl->hubert) {
            impl->reference.ssl = hubert::compute_ssl(*impl->hubert, impl->reference.audio_16k);
        }

        // 参考文本的韵律特征：有 RoBERTa 就算真的，没有保持全零降级
        // （与上游 GetPhonesAndBert.py 的降级分支一致）。
        //
        // **只有中文路径才跑 RoBERTa** ✓ —— 上游的
        // `_get_phones_and_bert_pure_lang` 里 english / korean / japanese
        // 三个分支都给全零 ✓（RoBERTa 是中文模型 ✓，喂别的语言的音素
        // 只会得到垃圾 ✓）。无条件跑的话，非中文路径会因为
        // "RoBERTa 的 token 数与音素数对不上"而直接报错 ✓。
        if (impl->roberta_tokenizer && !impl->reference.text.empty() &&
            uses_chinese_bert(impl->manifest.language)) {
            const auto sequence = impl->frontend->convert(impl->reference.text);
            if (!sequence.ids.empty()) {
                impl->reference.bert = roberta::compute_text_bert(
                    *impl->roberta_tokenizer, *impl->roberta, sequence, impl->reference.text);
            }
        }
    }

    // 5) V2ProPlus 的全局说话人嵌入。
    //
    // 上游是在第一次推理时算（ReferenceAudio.update_global_emb 会缓存结果），
    // 这里提前到加载期 —— 形状不对的话早失败比合成出垃圾好。
    if (impl->prompt_encoder != nullptr) {
        impl->compute_global_embeddings();
    }

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
    //    纯英文文本走英文前端；中文 / 混读走中文前端（混读路径内部
    //    再按语言块分派）。
    const auto language = detect_language(text, options.language);
    const text::TextFrontend* frontend = impl_->frontend_for(language);
    if (frontend == nullptr) {
        // 以前这里是静默回退到中文前端 ✓ —— 后果是"别的语言没有对应资源"
        // 变成"合成出来是静音"✓，看不出根因 ✓。现在直接说清楚 ✓。
        throw_error(ErrorCode::BackendUnavailable,
                    "no text frontend for language: " + language_to_string(language) + " —— " +
                        missing_frontend_hint(language));
    }
    const std::string prepared = std::string(kUpstreamTextPrefix) + std::string(text);
    const auto phonemes = frontend->convert(prepared);

    SynthesisResult result;
    result.language = language;
    result.phoneme_ids = phonemes.ids;

    if (phonemes.ids.empty()) {
        throw_error(ErrorCode::InvalidArgument, "text produced no phonemes");
    }

    // 2) T2S：音素 + 参考条件 → 语义 token。
    //
    // 注意 text_bert 与 ref_bert 是**两个不同的张量** ——
    // 上游喂的是两段不同的韵律特征。有 RoBERTa 时各自算真的，
    // 缺失时零张量（与上游降级一致），形状按各自的音素长度给。
    const auto& reference = impl_->reference;
    // text_bert 是 [音素数, 1024]（**秩 2**，没有 batch 维）——
    // 上游 ONNX 图声明的是 `text_bert: [text_length, 1024]`。
    auto text_bert = Tensor::zeros(
        ElementType::Float32, {static_cast<std::int64_t>(phonemes.ids.size()), kBertDimension});
    // **只有中文路径才跑 RoBERTa** ✓ —— 与上游一致 ✓：
    // english / korean / japanese 三个分支在上游都给全零 ✓
    // （RoBERTa 是中文模型 ✓）。无条件跑的话，别的语言会因为
    // "RoBERTa 的 token 行数与音素数对不上"而抛异常 ✓ ——
    // 上游 expand_by_word2ph 遇到对不上的布局也是直接 raise ✓。
    if (impl_->roberta_tokenizer && uses_chinese_bert(language)) {
        // 混读序列按语言块逐块编码（前端在 bert_chunks 里给出分块与对齐），
        // 单块（纯中文）路径用整段规范化文本 —— 两者与上游逐块 / 单块一致。
        const auto normalized = text::normalize(prepared, language);
        text_bert = roberta::compute_text_bert(*impl_->roberta_tokenizer, *impl_->roberta, phonemes,
                                               normalized);
    }
    const auto ref_seq = Tensor::from_int64(
        {1, static_cast<std::int64_t>(reference.phoneme_ids.size())}, reference.phoneme_ids);

    TensorFeed encoder_feed;
    encoder_feed.emplace_back("text_seq", make_phones_tensor(phonemes.ids));
    encoder_feed.emplace_back("ref_seq", ref_seq);
    encoder_feed.emplace_back("text_bert", text_bert);
    encoder_feed.emplace_back("ref_bert", reference.bert);
    encoder_feed.emplace_back("ssl_content", reference.ssl);

    auto semantic = impl_->decoder.decode(*impl_->encoder, *impl_->first_stage, *impl_->stage,
                                          encoder_feed, cancelled);
    semantic = truncate_at_eos(std::move(semantic));
    result.semantic_tokens = semantic;

    if (semantic.empty()) {
        throw_error(ErrorCode::InvalidArgument, "t2s produced no semantic tokens");
    }

    // 3) 声码器：语义 token → 波形。
    TensorFeed vocoder_feed;
    vocoder_feed.emplace_back("text_seq", make_phones_tensor(phonemes.ids));
    vocoder_feed.emplace_back("pred_semantic", make_pred_semantic(semantic));
    if (impl_->v2proplus()) {
        vocoder_feed.emplace_back("ge", impl_->global_emb);
        vocoder_feed.emplace_back("ge_advanced", impl_->global_emb_advanced);
    } else {
        // 参考波形：32 kHz 单声道。
        vocoder_feed.emplace_back(
            "ref_audio",
            Tensor::from_float32({1, static_cast<std::int64_t>(reference.audio.samples.size())},
                                 reference.audio.samples));
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
