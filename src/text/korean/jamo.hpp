// src/text/korean/jamo.hpp —— 韩文字母的拆解与组装
//
// 移植 python-jamo（MIT）的 h2j / j2hcj，外加 g2pk2 的 compose
// （把组合字母重新组装成组合字）。

#ifndef GENIE_SRC_TEXT_KOREAN_JAMO_HPP
#define GENIE_SRC_TEXT_KOREAN_JAMO_HPP

#include <string>
#include <string_view>

namespace genie::text::korean {

/// 组合字（가）→ 组合字母（ᄀ ᅡ ᆫ）。非组合字的字符原样保留。
[[nodiscard]] std::string h2j(std::string_view text);

/// 组合字母（ᄀ ᅡ ᆫ）→ 兼容字母（ㄱ ㅏ ㄴ）。
[[nodiscard]] std::string j2hcj(std::string_view text);

/// 组合字母 → 组合字（g2pk2 的 assemble 步骤）。
[[nodiscard]] std::string compose(std::string_view text);

}  // namespace genie::text::korean

#endif  // GENIE_SRC_TEXT_KOREAN_JAMO_HPP
