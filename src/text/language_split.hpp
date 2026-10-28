// src/text/language_split.hpp
//
// 中英混读的按语言切块（上游 GetPhonesAndBert.py 的 split_language）。
//
// 上游在 'hybrid-chinese-english' 模式下把文本按 `[a-zA-Z]+` 连续段切开，
// 交替产出中文块 / 英文块：字母串归英文，其余（汉字、数字、标点、空白）
// 全部留在中文块里。每个中文块各自走一遍完整的 TextNormalizer + G2P，
// 英文块走英文 G2P —— 本实现还没有英文 G2P（见 docs/upstream-audit.md
// §9.4），英文块目前由调用方丢弃。

#ifndef GENIE_SRC_TEXT_LANGUAGE_SPLIT_HPP
#define GENIE_SRC_TEXT_LANGUAGE_SPLIT_HPP

#include <string>
#include <string_view>
#include <vector>

namespace genie::text::lang_split {

/// 一个语言块。
struct Chunk {
    /// 块的语言。ASCII 字母连续段是英文块，其余都是中文块。
    enum class Language { Chinese, English };

    Language language = Language::Chinese;
    /// 块内容（未经规范化 —— 上游在切块之后才对中文块做规范化）。
    std::string content;
};

/// 按语言切块。
///
/// 移植自上游 `split_language`：`re.split(r'[a-zA-Z]+', text)` 与
/// `re.findall(r'[a-zA-Z]+', text)` 交错拼装。中文块只在
/// `content.strip()` 非空时保留（上游同样丢弃纯空白块），
/// 英文块必然非空（正则匹配到的字母串）。
[[nodiscard]] std::vector<Chunk> split_by_language(std::string_view text);

/// 文本里有没有 ASCII 字母 —— 上游据此决定走 'chinese' 还是
/// 'hybrid-chinese-english'（`re.search(r'[a-zA-Z]', prompt_text)`）。
[[nodiscard]] bool has_ascii_letter(std::string_view text);

}  // namespace genie::text::lang_split

#endif  // GENIE_SRC_TEXT_LANGUAGE_SPLIT_HPP
