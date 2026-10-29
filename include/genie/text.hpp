// include/genie/text.hpp
//
// 文本前端：文本 → 音素序列。
//
// 模型输入的 `text_seq` 是**符号 ID 序列**，不是文字。所以从用户输入到
// 模型输入，中间这一段是整个链路里最容易出错、也最影响听感的地方：
//
//   原始文本
//     → 规范化（全角/数字/标点）
//     → G2P（文字 → 音素）
//     → 音素 → 符号 ID
//
// 这一层被拆成两个可替换的部分：
//
//   - `normalize()`：与语言无关的文本清洗
//   - `TextFrontend`：语言相关的 G2P
//
// 之所以要留 `TextFrontend` 这个接口：上游的 G2P 依赖 g2pM（BERT）、
// jieba（分词）、nltk（英文）、pyopenjtalk（日文）—— 全都是 Python 生态。
// 本实现自带一套确定性的中文前端，但如果使用者手上有更准的 G2P
// （比如把 g2pM 导出来），可以自己实现这个接口接上去。

#ifndef GENIE_TEXT_HPP
#define GENIE_TEXT_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "genie/config.hpp"

namespace genie::text {

/// 混读路径的一个 BERT 分块。
///
/// 上游对 'hybrid-chinese-english' 文本是**按语言块**分别跑 RoBERTa 的
/// （GetPhonesAndBert.py 对每个语言块单独调 `_get_phones_and_bert_pure_lang`
/// 再 `np.concatenate`），块与块之间没有跨块的注意力 —— 所以韵律特征
/// 不能拿"把所有中文块拼成一段"的文本算一次，必须逐块编码、逐块展开。
struct BertChunk {
    /// 该块的文本（中文块是逐块规范化后的结果，喂给 RoBERTa tokenizer）。
    std::string text;
    /// 该块在 `PhonemeSequence::word2ph` 里的起始下标。
    std::size_t word_offset = 0;
    /// 该块占的 word2ph 条数；这些条目的和 = 该块的音素数。
    std::size_t word_count = 0;
};

/// 音素序列。
struct PhonemeSequence {
    /// 音素符号名（"zh"、"ong1"…）。
    std::vector<std::string> symbols;
    /// 音素对应的符号 ID —— 这才是模型要吃的东西。
    std::vector<std::int64_t> ids;
    /// 每个"词"占几个音素。上游按分词结果统计，用于韵律特征。
    /// 本实现按**单字**统计（没有分词），见 docs/upstream-audit.md。
    std::vector<std::int32_t> word2ph;
    /// 混读路径的 BERT 分块。各块的 [word_offset, word_offset+word_count)
    /// 连续覆盖整个 word2ph。**空** = 单块路径：整段文本跑一次 RoBERTa
    /// （纯中文走这条，与上游 'chinese' 模式一致）。
    std::vector<BertChunk> bert_chunks;

    [[nodiscard]] bool empty() const noexcept { return ids.empty(); }
    [[nodiscard]] std::size_t size() const noexcept { return ids.size(); }
};

/// 文本规范化。
///
/// 会做这些事（按语言略有差异）：
///   - 全角 ASCII → 半角（上游 TextNormalizer 的 F2H_* 规则）
///   - 阿拉伯数字 → 中文读法（中文路径；上游 num.py 的规则）
///   - 标点映射（上游 ChineseG2P.PUNCTUATION_REPLACEMENTS）
///   - 目标语言之外的字符过滤
///   - 连续标点折叠成一个
///
/// @param text  原始文本。
/// @param language 目标语言。
/// @return 规范化后的文本。
[[nodiscard]] std::string normalize(std::string_view text, Language language);

/// 文本前端（文字 → 音素）。
class TextFrontend {
public:
    virtual ~TextFrontend() = default;

    /// 把文本转成音素序列。
    [[nodiscard]] virtual PhonemeSequence convert(std::string_view text) const = 0;

    /// 这个前端处理的语言。
    [[nodiscard]] virtual Language language() const noexcept = 0;
};

/// 英文 G2P 需要的资源。
///
/// 这几个都是**数据文件**（不是模型），由 tools/deploy/ 下的转换脚本生成，
/// 落在音色包目录里（见 docs/upstream-audit.md §17）。
struct EnglishG2pResources {
    /// CMU 发音词典容器（third_party/english/cmudict.bin）。**必需**。
    std::string cmu_dict;

    /// nltk 词性标注权重容器（third_party/english/pos_tagger.bin）。
    /// 可选 —— 缺失时同形异音词（read / complex / lead / presents）
    /// 固定取第二个读音。
    std::string pos_tagger;

    /// seq2seq GRU 权重容器（third_party/english/english_g2p.bin）。
    /// 可选 —— 缺失时词典里没有的词没有发音（会被跳过）。
    std::string g2p_model;

    /// ngram 分词词典目录（上游 English_G2P/wordsegment/）。
    /// 可选 —— 缺失时词典里没有的词直接交给 GRU。
    std::string word_segment_dir;
};

/// 日语 G2P 需要的资源。
///
/// 只有一个：OpenJTalk 的词典目录 ✓（open_jtalk_dic_utf_8-1.11 ✓）。
/// 词典约 103 MB ✓ —— **不入仓库** ✓（与英文的 wordsegment 词典同理 ✓），
/// 由音色包在运行时给出路径 ✓。
struct JapaneseG2pResources {
    /// OpenJTalk 词典目录。**必需** —— 没有它就没有日文前端。
    std::string dictionary_dir;
};

/// 创建指定语言的前端。
///
/// `Language::Auto` 会按文本内容自动判断（与 `detect_language()` 一致）。
/// `Language::HybridChineseEnglish` 由中文前端承担：按语言分块
/// （`split_language`），每个中文块走完整规范化 + G2P；
/// 英文块走英文前端（需要提供 `resources`）。
///
/// @return 前端实例；语言不受支持或资源缺失时返回 nullptr。
[[nodiscard]] std::unique_ptr<TextFrontend> make_frontend(Language language);

/// 创建英文前端。
///
/// @param resources 英文 G2P 的资源路径。
/// @return 前端实例；CMU 词典缺失或读不出来时返回 nullptr。
[[nodiscard]] std::unique_ptr<TextFrontend> make_frontend(Language language,
                                                          const EnglishG2pResources& resources);

/// 创建日文前端。
///
/// @param resources 日语 G2P 的资源路径（OpenJTalk 词典目录 ✓）。
/// @return 前端实例；词典缺失或读不出来时返回 nullptr。
[[nodiscard]] std::unique_ptr<TextFrontend> make_frontend(Language language,
                                                          const JapaneseG2pResources& resources);

}  // namespace genie::text

#endif  // GENIE_TEXT_HPP
