// src/text/opencpop_table.hpp — opencpop 词表的内部声明。
//
// 词表由 tools/g2p/generate_opencpop_table.py 从上游 opencpop-strict.txt
// 生成，是"拼音键 → 音素符号对"转换的唯一权威（见 chinese_g2p.cpp 的
// syllable_to_phones）。

#ifndef GENIE_SRC_TEXT_OPENCPOP_TABLE_HPP
#define GENIE_SRC_TEXT_OPENCPOP_TABLE_HPP

#include <map>
#include <string>
#include <utility>

namespace genie::pinyin::detail {

/// 拼音键 → (声母符号, 韵母符号)。键按升序存放。
using OpencpopLexicon = std::map<std::string, std::pair<std::string, std::string>>;

/// 内嵌的原始词表文本（由生成器产出，定义在 opencpop_table.cpp）。
extern const char* const kOpencpopTable;

/// 惰性解析的词表单例。
[[nodiscard]] const OpencpopLexicon& opencpop_lexicon();

}  // namespace genie::pinyin::detail

#endif  // GENIE_SRC_TEXT_OPENCPOP_TABLE_HPP
