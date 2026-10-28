// src/text/english/english.hpp
//
// 英文 G2P 的内部接口（上游 G2P/English/）。
//
// 上游的英文链路是：
//
//   原始文本
//     → REP_MAP（中文标点 → 英文标点）
//     → Normalization.normalize（数字/缩写/货币/日期/度量… 20 条规则）
//     → 折叠连续标点（注意：连**空白**也一起折叠）
//     → TweetTokenizer 分词（原始词表 + 规范化后词表两遍）
//     → nltk POS 标注（只为 4 个同形异音词服务）
//     → CMU 词典查表 → 失败则 's / 连字符 / 分词 / GRU 兜底
//     → 过滤掉空格与不在符号表里的音素
//
// 本实现逐段移植，见 docs/upstream-audit.md §17。

#ifndef GENIE_SRC_TEXT_ENGLISH_ENGLISH_HPP
#define GENIE_SRC_TEXT_ENGLISH_ENGLISH_HPP

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace genie::text::english {

/// 一个分词结果。
struct Token {
    std::string text;
    /// 起始字节偏移（规范化后的文本里）。
    std::size_t offset = 0;
};

/// TweetTokenizer 移植。
///
/// 只保留对 TTS 有意义的那几条分支：URL、邮箱、@handle、#hashtag、
/// HTML 标签、带撇号/连字符的词、带分隔符的数字、普通词，
/// 其余字符各自成一个 token。上游的正则还包含表情符号、国旗、
/// emoji 肤色修饰符等分支 —— 它们产出的 token 同样查不到音素，
/// 对最终音素序列没有影响，故不移植（差异见 audit §17）。
[[nodiscard]] std::vector<Token> tokenize(std::string_view text);

/// 英文文本规范化（上游 Normalization.normalize + REP_MAP + 折叠连续标点）。
[[nodiscard]] std::string normalize(std::string_view text);

/// 上游 `EnglishG2P.text_normalize`：REP_MAP → normalize() → 连续标点折叠。
///
/// 与 normalize() 的区别：REP_MAP 把中文/全角标点换成半角，
/// 连续标点折叠把 "..." 压成 "." —— 这两步在 G2P 之前做。
[[nodiscard]] std::string text_normalize(std::string_view text);

/// CMU 发音词典。
///
/// 容器格式由 tools/deploy/convert_cmu_dict.py 生成，
/// 落盘为 third_party/english/cmudict.bin。
class CmuDict {
public:
    /// 加载容器。文件不存在 / magic 不符 / 版本不符 → 抛 genie::Error。
    [[nodiscard]] static CmuDict load(const std::string& path);

    /// 查一个词的首选发音（上游 `_read_cmu_dict` 只保留每个词的第一个读音）。
    /// @return 音素序列；查不到返回空。
    [[nodiscard]] std::vector<std::string> lookup(std::string_view word) const;

    [[nodiscard]] std::size_t size() const noexcept { return count_; }

private:
    struct Entry {
        std::uint32_t word_offset = 0;  // 词在词表 blob 里的字节偏移
        std::uint32_t word_length = 0;
        std::uint32_t pron_offset = 0;  // 发音在音素 blob 里的条目下标
        std::uint32_t pron_length = 0;
    };

    std::string words_;
    std::vector<Entry> entries_;  // 按词字典序排好，二分查找
    std::vector<std::uint16_t> phoneme_ids_;
    std::vector<std::string> phoneme_names_;
    std::size_t count_ = 0;
};

/// 发音词典里存不下的词，用 seq2seq GRU 猜（上游 `CleanG2p.predict`）。
///
/// 权重来自上游的 checkpoint20.npz（g2p_en），由
/// tools/deploy/convert_english_g2p.py 转成裸二进制
/// third_party/english/english_g2p.bin。没加载到权重时不可用。
class Seq2SeqG2p {
public:
    /// 加载权重容器。文件不存在 → 返回 false（调用方降级，不算错误）。
    [[nodiscard]] bool load(const std::string& path);

    /// 预测一个词的发音。
    [[nodiscard]] std::vector<std::string> predict(std::string_view word) const;

    [[nodiscard]] bool available() const noexcept { return !encoder_.enc_emb.empty(); }

private:
    struct GruWeights {
        std::vector<float> enc_emb;   // [29, 32]
        std::vector<float> enc_w_ih;  // [64, 32]
        std::vector<float> enc_w_hh;  // [64, 32]
        std::vector<float> enc_b_ih;  // [64]
        std::vector<float> enc_b_hh;  // [64]
        std::vector<float> dec_emb;   // [74, 32]
        std::vector<float> dec_w_ih;  // [64, 32]
        std::vector<float> dec_w_hh;  // [64, 32]
        std::vector<float> dec_b_ih;  // [64]
        std::vector<float> dec_b_hh;  // [64]
        std::vector<float> fc_w;      // [71, 32]
        std::vector<float> fc_b;      // [71]
    };

    GruWeights encoder_;
    std::vector<std::string> phonemes_;  // p2idx 的逆映射
};

/// 英文词性标注（上游 `nltk.pos_tag`，nltk 的 averaged perceptron）。
///
/// 上游只用它给 4 个同形异音词挑读音，但标注结果是整句算出来的
/// （前一个词的词性会影响当前词），所以这里照搬 nltk 的算法：
/// 特征提取 + 权重点积 + 平局取字典序最大的标签。
///
/// 权重由 tools/deploy/convert_pos_tagger.py 打包成
/// third_party/english/pos_tagger.bin。
class PosTagger {
public:
    [[nodiscard]] static PosTagger load(const std::string& path);

    /// 标注一句话。返回与输入等长的标签序列。
    [[nodiscard]] std::vector<std::string> tag(const std::vector<std::string>& words) const;

    [[nodiscard]] bool available() const noexcept { return !classes_.empty(); }

private:
    struct FeatureWeights {
        std::vector<std::pair<std::uint8_t, float>> weights;
    };

    [[nodiscard]] std::string normalize_word(std::string_view word) const;
    [[nodiscard]] const FeatureWeights* find_feature(std::string_view feature) const;
    [[nodiscard]] const std::pair<std::string, std::uint8_t>* find_tagdict(
        std::string_view word) const;

    std::vector<std::string> classes_;
    std::vector<std::pair<std::string, std::uint8_t>> tagdict_;
    std::vector<std::pair<std::string, FeatureWeights>> features_;
};

/// 用 ngram 词典做 Viterbi 分词（上游 WordSegment.py 的 WordSegmenter）。
///
/// 数据（unigrams.txt / bigrams.txt / words.txt）来自上游的
/// English_G2P 目录，体积很大，不随仓库发布；没数据时不可用。
class WordSegmenter {
public:
    /// 加载词典目录。目录不存在 → 返回 false。
    [[nodiscard]] bool load(const std::string& directory);

    /// 切一个词。切不成（只有一个片段）时返回空。
    [[nodiscard]] std::vector<std::string> segment(std::string_view word) const;

    [[nodiscard]] bool available() const noexcept { return !unigrams_.empty(); }

private:
    [[nodiscard]] double score(std::string_view word, std::string_view previous) const;

    std::vector<std::pair<std::string, double>> unigrams_;
    std::vector<std::pair<std::string, double>> bigrams_;
    double total_ = 0.0;
};

/// 英文文本 → 音素序列（符号名，与上游 `g2p()` 的输出一致）。
///
/// @param text      原始文本
/// @param dict      CMU 发音词典
/// @param seq2seq   OOV 兜底（GRU）；没加载权重时不可用
/// @param segmenter OOV 兜底（ngram 分词）；没有词典数据时不可用
/// @param tagger    词性标注（给同形异音词挑读音）；可为 nullptr
[[nodiscard]] std::vector<std::string> phonemes(std::string_view text, const CmuDict& dict,
                                                const Seq2SeqG2p& seq2seq,
                                                const WordSegmenter& segmenter,
                                                const PosTagger* tagger);

}  // namespace genie::text::english

#endif  // GENIE_SRC_TEXT_ENGLISH_ENGLISH_HPP
