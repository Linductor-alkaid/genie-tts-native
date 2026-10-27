// src/text/roberta_tokenizer.hpp — RoBERTa 的 BERT WordPiece 分词器（内部实现）
//
// 复刻上游中文韵律路径用的 HuggingFace tokenizers 流水线
// （tokenizer.json：BertNormalizer → BertPreTokenizer → WordPiece，
// 后处理 [CLS] ... [SEP]）。上游见 GetPhonesAndBert.py 的
// `model_manager.roberta_tokenizer.encode(text_clean)`。
//
// 这是 src/ 内部头：公开 API 只暴露"引擎会用 RoBERTa 特征"这一事实，
// 分词器本身不是契约的一部分。

#ifndef GENIE_SRC_TEXT_ROBERTA_TOKENIZER_HPP
#define GENIE_SRC_TEXT_ROBERTA_TOKENIZER_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace genie::text {

/// BERT WordPiece 分词器（chinese-roberta-wwm-ext 系）。
///
/// 词表来自 BERT 经典的 vocab.txt（一行一个 token，行号即 ID）——
/// 与 tokenizer.json 里 model.vocab 的顺序一致（本仓库的黄金数据测试锁定）。
class RobertaTokenizer {
public:
    /// 编码结果。
    struct Encoding {
        /// token ID 序列，首尾是 [CLS] / [SEP]。
        std::vector<std::int64_t> ids;
        /// 与 ids 等长的注意力掩码（本分词器全 1）。
        std::vector<std::int64_t> attention_mask;
        /// token 字符串（诊断用）。
        std::vector<std::string> tokens;
    };

    /// 从 vocab.txt 加载。
    ///
    /// @param vocab_path vocab.txt 路径
    /// @throws genie::Error 文件不可读或词表为空
    [[nodiscard]] static std::unique_ptr<RobertaTokenizer> load(const std::string& vocab_path);

    /// 编码一段**规范化后**的文本（与上游 text_clean 同语义）。
    [[nodiscard]] Encoding encode(std::string_view text) const;

    /// 特殊 token 的 ID（加载后有效）。
    [[nodiscard]] std::int64_t cls_id() const noexcept { return cls_id_; }
    [[nodiscard]] std::int64_t sep_id() const noexcept { return sep_id_; }
    [[nodiscard]] std::int64_t unk_id() const noexcept { return unk_id_; }

private:
    RobertaTokenizer() = default;

    std::vector<std::string> vocab_;
    std::int64_t cls_id_ = 0;
    std::int64_t sep_id_ = 0;
    std::int64_t unk_id_ = 0;
};

}  // namespace genie::text

#endif  // GENIE_SRC_TEXT_ROBERTA_TOKENIZER_HPP
