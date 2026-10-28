// src/text/frontend_factory_internal.hpp
//
// 前端工厂的内部入口：引擎在持有 g2pM 模型时用它把多音字消歧
// 接进中文前端。公开 API（include/genie/text.hpp 的 make_frontend）
// 不暴露 Polyphone，保持"音色包外的模型不进公开签名"。

#ifndef GENIE_SRC_TEXT_FRONTEND_FACTORY_INTERNAL_HPP
#define GENIE_SRC_TEXT_FRONTEND_FACTORY_INTERNAL_HPP

#include <memory>

#include "genie/text.hpp"
#include "text/jieba.hpp"
#include "text/polyphone.hpp"

namespace genie::text {

/// 中文前端（可带多音字消歧与 jieba 分词）。
///
/// @param polyphone g2pM 模型；为空时退化为单字拼音表最常用读音。
/// @param jieba     分词器；为空时变调边界退回词表贪心匹配。
/// @param english   英文前端；混读路径的英文块交给它。为空时英文块丢弃
///                  （与上游的混读路径一致，见 docs/upstream-audit.md §17）。
[[nodiscard]] std::unique_ptr<TextFrontend> make_chinese_frontend(
    const Polyphone* polyphone = nullptr, const jieba::Tokenizer* jieba = nullptr,
    const TextFrontend* english = nullptr);

/// 英文前端（词典是硬依赖，其余资源缺失时降级）。
[[nodiscard]] std::unique_ptr<TextFrontend> make_english_frontend(
    const EnglishG2pResources& resources);

/// 韩文前端（不需要外部资源 ✓）。
[[nodiscard]] std::unique_ptr<TextFrontend> make_korean_frontend();

}  // namespace genie::text

#endif  // GENIE_SRC_TEXT_FRONTEND_FACTORY_INTERNAL_HPP
