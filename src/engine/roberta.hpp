// src/engine/roberta.hpp — RoBERTa 韵律特征（内部实现）
//
// 上游中文路径里 text_bert / ref_bert 由 Chinese RoBERTa 产出
// （GetPhonesAndBert.py 的 `_get_phones_and_bert_pure_lang`）：
// tokenizer 编码规范化文本 → RoBERTa 推理 → 按 word2ph 展开
// 成"每个音素一行"的特征矩阵。没有 RoBERTa 时上游退化为全零，
// 本实现的降级路径见 reference.cpp / engine.cpp。

#ifndef GENIE_SRC_ENGINE_ROBERTA_HPP
#define GENIE_SRC_ENGINE_ROBERTA_HPP

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "genie/session.hpp"
#include "genie/tensor.hpp"
#include "text/roberta_tokenizer.hpp"

namespace genie::roberta {

/// 把 token 级特征展开成音素级特征（复刻上游 `_expand_roberta_output`）。
///
/// 三种可接受的输出布局（不同的 ONNX 导出变体）：
/// - T == num_phones        → 已经是音素级，直接用
/// - T == len(word2ph) + 2  → 去 [CLS]/[SEP] 后按 word2ph 重复
/// - T == len(word2ph)      → 直接按 word2ph 重复
///
/// @param token_bert token 级特征（元素个数必须是 T*1024）
/// @param word2ph    每个词占的音素数（总和必须等于 num_phones）
/// @param num_phones 音素总数
/// @return [num_phones, 1024] 的秩 2 张量
///
/// @throws genie::Error 布局不匹配或 word2ph 与音素数对不上
[[nodiscard]] Tensor expand_by_word2ph(std::span<const float> token_bert,
                                       const std::vector<std::int32_t>& word2ph,
                                       std::size_t num_phones);

/// 完整的中文韵律特征：规范化文本 → token → RoBERTa → 音素级特征。
///
/// @param tokenizer 分词器（跟随 RoBERTa 模型一起加载）
/// @param model     RoBERTa 会话
/// @param text      **规范化后**的文本（与上游 text_clean 同语义；
///                  本引擎里就是喂给文本前端的那串）
/// @param word2ph   文本前端对同一段文本产出的 word2ph
/// @param num_phones 音素总数
[[nodiscard]] Tensor compute_text_bert(const text::RobertaTokenizer& tokenizer, ModelSession& model,
                                       std::string_view text,
                                       const std::vector<std::int32_t>& word2ph,
                                       std::size_t num_phones);

}  // namespace genie::roberta

#endif  // GENIE_SRC_ENGINE_ROBERTA_HPP
