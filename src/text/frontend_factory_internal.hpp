// src/text/frontend_factory_internal.hpp
//
// 前端工厂的内部入口：引擎在持有 g2pM 模型时用它把多音字消歧
// 接进中文前端。公开 API（include/genie/text.hpp 的 make_frontend）
// 不暴露 Polyphone，保持"音色包外的模型不进公开签名"。

#ifndef GENIE_SRC_TEXT_FRONTEND_FACTORY_INTERNAL_HPP
#define GENIE_SRC_TEXT_FRONTEND_FACTORY_INTERNAL_HPP

#include <memory>

#include "text/polyphone.hpp"

namespace genie::text {

/// 中文前端（可带多音字消歧）。
///
/// @param polyphone g2pM 模型；为空时退化为单字拼音表最常用读音。
[[nodiscard]] std::unique_ptr<TextFrontend> make_chinese_frontend(
    const Polyphone* polyphone = nullptr);

}  // namespace genie::text

#endif  // GENIE_SRC_TEXT_FRONTEND_FACTORY_INTERNAL_HPP
