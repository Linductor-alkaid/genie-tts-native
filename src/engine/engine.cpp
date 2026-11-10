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

#if defined(__GLIBC__)
#include <malloc.h>
#endif

#include "engine/g2pm.hpp"
#include "engine/hubert.hpp"
#include "engine/model_loader.hpp"
#include "engine/roberta.hpp"
#include "engine/stage_timer.hpp"
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
#ifdef GENIE_HAVE_OPENJTALK
            return "音色包缺少 OpenJTalk 词典（角色 japanese_dictionary，约 103 MB）——"
                   "用 make_voice_package.py --japanese-dictionary 接入";
#else
            // 没编 OpenJTalk 的构建 ✓（MSVC 上默认如此 ✓）——
            // 这时候给词典也没用 ✓，缺的是运行时本身 ✓。
            return "这个构建没有编 OpenJTalk —— 日语音素转换需要它，"
                   "重新构建时打开 GENIE_ENABLE_OPENJTALK";
#endif
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

/// 加载期并行会话数的默认值（intra_op_threads 未配置时）。与 ORT 会话的
/// 默认预算对齐：8 路并行建会话的瞬时负载与一次合成同量级（kairo 默认
/// 池不绑核，ORT 线程由 OS 自由调度 —— architecture.md §2.14）。
constexpr std::size_t kDefaultLoadConcurrency = 8;

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

/// 把堆上的空闲内存还给系统。
///
/// glibc 的 malloc 会把释放的内存留在进程的 arena 里备用：ORT 会话里几 MB
/// 一块的权重张量释放后，常驻内存（RSS）并不会回落。malloc_trim 把这些
/// 空闲页交还内核。只在加载期调一次，其它平台不需要（Windows 的大块分配
/// 释放即归还）。
void return_free_memory_to_os() noexcept {
#if defined(__GLIBC__)
    (void)malloc_trim(0);
#endif
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
    // 下面三个只在加载期用（算完参考特征 / 全局嵌入即释放，构造结束后为空）。
    std::unique_ptr<ModelSession> prompt_encoder;    // V2ProPlus
    std::unique_ptr<ModelSession> speaker_verifier;  // V2ProPlus
    std::unique_ptr<ModelSession> hubert;            // 参考音频 ssl_content（可选）
    std::unique_ptr<ModelSession> roberta;           // 中文韵律（可选，合成期使用）

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
            case Language::Japanese:
#ifdef GENIE_HAVE_OPENJTALK
                return japanese_frontend.get();
#else
                // 没编 OpenJTalk 的构建 ✓（MSVC 上默认如此 ✓）——
                // 日文前端**不存在** ✓。这里必须返回 nullptr ✓：
                // 落进 default 就会静默回退到中文前端 ✓ ——
                // 那正是这次要修的"日文合成出来是静音"✓。
                return nullptr;
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

    /// 构造期的分阶段耗时（模型加载 / 参考特征 / 全局嵌入），经
    /// Engine::load_timing() 公开 —— 冷启动基线只采这一次。
    TimingReport load_timing;

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

Engine::Engine(std::string root) : Engine(std::move(root), EngineConfig{}) {}

Engine::Engine(std::string root, const SessionConfig& config)
    : Engine(
          std::move(root),
          [](const std::string& path, const SessionConfig& cfg) {
              return create_session(path, cfg);
          },
          EngineConfig{.session = config}) {}

Engine::Engine(std::string root, const EngineConfig& config)
    : Engine(
          std::move(root),
          [](const std::string& path, const SessionConfig& cfg) {
              return create_session(path, cfg);
          },
          config) {}

Engine::Engine(std::string root, SessionFactory factory, const SessionConfig& config)
    : Engine(std::move(root), std::move(factory), EngineConfig{.session = config}) {}

Engine::Engine(std::string root, SessionFactory factory, const EngineConfig& config) {
    auto impl = std::make_unique<Impl>();
    impl->root = std::move(root);

    // 冷启动总时长与各阶段耗时都记进 impl->load_timing（M0.3 基线）。
    const auto load_start = std::chrono::steady_clock::now();

    // 1) 音色包。校验是逐文件全量读盘（成本与包体积线性），受
    //    EngineConfig::verify_voice_package 控制；存在性检查不受影响。
    {
        detail::ScopedStageTimer timer(impl->load_timing, "load/voice_package");
        impl->package = VoicePackage::load(impl->root, config.verify_voice_package);
    }
    impl->manifest = impl->package.manifest();
    impl->sample_rate = impl->manifest.sample_rate > 0 ? impl->manifest.sample_rate : 32000;

    // 2) 模型会话。会话初始化是冷启动的大头（fp32 包里占 ~85%），按角色
    //    单独计时；怎么加载（fp16 包的扫描 / 上转、建会话）由
    //    detail::load_model_sessions 编排。fp16 权重经内存上转后运行时仍是
    //    FP32 —— 精度影响只来自存储格式，与 HF 上游发布的 fp16 bin 一致。
    //
    // 说话人验证模型是 V2ProPlus 的**硬依赖**：上游的 update_global_emb()
    // 只有在 SV 模型加载成功时才会算 global_emb；算不出来就没法喂 vits 的
    // ge / ge_advanced。这里在**加载前**就报清楚，而不是等到推理时炸。
    if (impl->v2proplus() && !impl->package.has(ModelRole::SpeakerVerification)) {
        throw_error(ErrorCode::NotFound,
                    "V2ProPlus 音色包缺少说话人验证模型（speaker_verification）—— "
                    "没有它算不出 ge / ge_advanced，V2ProPlus 分支不可用");
    }
    {
        std::vector<detail::ModelLoadRequest> requests;
        const auto request = [&impl, &requests](const ModelRole role, std::string stage,
                                                std::unique_ptr<ModelSession>& target) {
            requests.push_back({impl->package.model_path(role), std::move(stage), &target});
        };
        request(ModelRole::T2sEncoder, "load/t2s_encoder", impl->encoder);
        request(ModelRole::T2sFirstStageDecoder, "load/t2s_first_stage_decoder", impl->first_stage);
        request(ModelRole::T2sStageDecoder, "load/t2s_stage_decoder", impl->stage);
        request(ModelRole::Vits, "load/vits", impl->vocoder);
        if (impl->v2proplus()) {
            request(ModelRole::PromptEncoder, "load/prompt_encoder", impl->prompt_encoder);
            request(ModelRole::SpeakerVerification, "load/speaker_verification",
                    impl->speaker_verifier);
        }
        // RoBERTa（中文韵律）与 HuBERT（参考音频内容特征）是可选角色：
        // 缺失时对应特征保持全零降级（与上游一致，见 4)）。
        if (impl->package.has(ModelRole::Roberta)) {
            request(ModelRole::Roberta, "load/roberta", impl->roberta);
        }
        if (impl->package.has(ModelRole::ChineseHubert)) {
            request(ModelRole::ChineseHubert, "load/chinese_hubert", impl->hubert);
        }
        // 并发路数与会话线程数同源（SessionConfig::intra_op_threads，默认 8）：
        // 峰值 ≈ 路数 × 线程，不引入新的满载形态。0 视为未配置，取默认。
        const std::size_t load_concurrency =
            config.session.intra_op_threads > 0
                ? static_cast<std::size_t>(config.session.intra_op_threads)
                : kDefaultLoadConcurrency;
        detail::load_model_sessions(requests, factory, config.session,
                                    impl->manifest.weight_format == WeightFormat::Fp16,
                                    load_concurrency, impl->load_timing);
    }

    // 3) 文本前端。g2pM（多音字消歧）先于前端装配 —— 前端持有它做
    //    上下文读音判定；模型缺失时前端退化为单字最常用读音
    //    （与 RoBERTa / HuBERT 的可选降级同款模式）。
    if (impl->package.has(ModelRole::Jieba)) {
        detail::ScopedStageTimer timer(impl->load_timing, "load/jieba");
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
        detail::ScopedStageTimer timer(impl->load_timing, "load/english_frontend");
        impl->english_frontend = text::make_english_frontend(english_resources);
    }

    // 3.2) 韩文前端（不需要外部资源 ✓ —— 规则与数据都内置 ✓）。
    //      纯内存构造，没有可测的加载成本，不计时。
    impl->korean_frontend = text::make_korean_frontend();

    // 3.3) 日文前端（OpenJTalk）：词典（约 103 MB ✓）由音色包给出 ✓ ——
    //      前端必须在这里**显式装配** ✓：只声明 japanese_frontend 成员
    //      而不赋值的话 ✓，frontend_for(Japanese) 永远返回 nullptr ✓，
    //      日文即使包里带了词典 ✓，也只会报"音色包缺少 OpenJTalk 词典"✓。
#ifdef GENIE_HAVE_OPENJTALK
    if (impl->package.has(ModelRole::JapaneseDictionary)) {
        detail::ScopedStageTimer timer(impl->load_timing, "load/japanese_frontend");
        text::JapaneseG2pResources japanese_resources;
        japanese_resources.dictionary_dir = impl->package.model_path(ModelRole::JapaneseDictionary);
        impl->japanese_frontend = text::make_japanese_frontend(japanese_resources);
    }
#endif

    if (impl->package.has(ModelRole::G2pm)) {
        detail::ScopedStageTimer timer(impl->load_timing, "load/g2pm");
        impl->g2pm_model = std::make_unique<g2pm::Model>(
            g2pm::Model::load(impl->package.model_path(ModelRole::G2pm)));
        impl->frontend = text::make_chinese_frontend(impl->g2pm_model.get(), impl->jieba_.get(),
                                                     impl->english_frontend.get());
    } else {
        impl->frontend =
            text::make_chinese_frontend(nullptr, impl->jieba_.get(), impl->english_frontend.get());
    }
    impl->decoder = T2SDecoder(T2SOptions{kMaxT2SIterations, kEosThreshold});

    // 3.5) RoBERTa 的分词器：词表必须与模型配套，二者缺一不可（会话已在
    //      2) 里加载）—— 加载期就报清楚。
    if (impl->roberta) {
        detail::ScopedStageTimer timer(impl->load_timing, "load/roberta_tokenizer");
        const auto vocab_path =
            std::filesystem::path(impl->package.model_path(ModelRole::Roberta)).parent_path() /
            "vocab.txt";
        impl->roberta_tokenizer = text::RobertaTokenizer::load(vocab_path.string());
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
        {
            detail::ScopedStageTimer timer(impl->load_timing, "load/reference");
            impl->reference = load_reference(impl->root, *reference_frontend, impl->sample_rate);
        }

        // 参考音频的内容特征：有 HuBERT 就算真的，没有保持全零降级
        // （与上游 ReferenceAudio.__init__ 一致）。
        if (impl->hubert) {
            detail::ScopedStageTimer timer(impl->load_timing, "load/reference_ssl");
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
            detail::ScopedStageTimer timer(impl->load_timing, "load/reference_bert");
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
        detail::ScopedStageTimer timer(impl->load_timing, "load/global_embeddings");
        impl->compute_global_embeddings();
    }

    // 6) 只在加载期用到的会话：用完即释放。
    //
    // HuBERT 只算参考音频的 ssl 特征（4)），说话人验证与 prompt_encoder 只算
    // V2ProPlus 的全局嵌入（5)）—— 结果已缓存进 reference / global_emb，合成
    // 路径不再碰这三个会话。它们常驻约 0.6 GB（docs/performance.md §内存）。
    // 上游 Genie-TTS 一直持有它们（支持运行时换参考音频）；本引擎的参考音频
    // 在构造期绑定、没有换参考的 API —— 将来要加，需要重新加载这三个模型
    // （docs/upstream-audit.md §23）。
    {
        detail::ScopedStageTimer timer(impl->load_timing, "load/release_load_only_models");
        impl->hubert.reset();
        impl->speaker_verifier.reset();
        impl->prompt_encoder.reset();
        return_free_memory_to_os();
    }

    // 端到端总时长。它与各阶段之和的差 = manifest 校验、前端装配这些
    // 未单独计时的开销 —— 这个差值本身也是 M0.3 要观测的对象。
    impl->load_timing.set_total_ms(
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - load_start)
            .count());

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

const TimingReport& Engine::load_timing() const noexcept {
    return impl_->load_timing;
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

    // 各阶段耗时报告（M0.3 性能基线）。埋点常开 —— 每个阶段只做两次
    // steady_clock 采样（纳秒级），相对秒级的合成开销可以忽略。
    TimingReport report;
    const auto synth_start = std::chrono::steady_clock::now();

    // 1) 语言与文本前端。
    //    纯英文文本走英文前端；中文 / 混读走中文前端（混读路径内部
    //    再按语言块分派）。
    Language language;
    {
        detail::ScopedStageTimer timer(report, "language_detect");
        language = detect_language(text, options.language);
    }
    const text::TextFrontend* frontend = impl_->frontend_for(language);
    if (frontend == nullptr) {
        // 以前这里是静默回退到中文前端 ✓ —— 后果是"别的语言没有对应资源"
        // 变成"合成出来是静音"✓，看不出根因 ✓。现在直接说清楚 ✓。
        throw_error(ErrorCode::BackendUnavailable,
                    "no text frontend for language: " + language_to_string(language) + " —— " +
                        missing_frontend_hint(language));
    }
    const std::string prepared = std::string(kUpstreamTextPrefix) + std::string(text);
    text::PhonemeSequence phonemes;
    {
        detail::ScopedStageTimer timer(report, "frontend");
        phonemes = frontend->convert(prepared);
    }

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
        detail::ScopedStageTimer timer(report, "text_bert");
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

    // T2S 是链路的成本大头；decode() 内部还会把三个子阶段
    // （encoder / first_stage / 自回归循环）记进同一份报告。
    std::vector<std::int64_t> semantic;
    {
        detail::ScopedStageTimer timer(report, "t2s");
        semantic = impl_->decoder.decode(*impl_->encoder, *impl_->first_stage, *impl_->stage,
                                         encoder_feed, cancelled, &report);
        semantic = truncate_at_eos(std::move(semantic));
        result.semantic_tokens = semantic;

        if (semantic.empty()) {
            throw_error(ErrorCode::InvalidArgument, "t2s produced no semantic tokens");
        }
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

    std::vector<Tensor> audio;
    {
        detail::ScopedStageTimer timer(report, "vocoder");
        audio = impl_->vocoder->run(vocoder_feed);
    }
    const auto* wave = find_output(audio, impl_->vocoder->outputs(), "audio");
    if (wave == nullptr) {
        throw_error(ErrorCode::InvalidArgument, "vocoder has no 'audio' output");
    }

    const auto samples = wave->floats();
    result.audio.samples.assign(samples.begin(), samples.end());
    result.audio.sample_rate =
        options.output_sample_rate > 0 ? options.output_sample_rate : impl_->sample_rate;

    report.set_total_ms(
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - synth_start)
            .count());
    result.timing = std::move(report);
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
                // 各阶段耗时报告随 result 析构丢弃 —— 通道目前只传音频块。
                // M6 真分块流式时，报告应挂在最后一个分块上交给消费者。
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
