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

/// 音素序列。
struct PhonemeSequence {
    /// 音素符号名（"zh"、"ong1"…）。
    std::vector<std::string> symbols;
    /// 音素对应的符号 ID —— 这才是模型要吃的东西。
    std::vector<std::int64_t> ids;
    /// 每个"词"占几个音素。上游按分词结果统计，用于韵律特征。
    /// 本实现按**单字**统计（没有分词），见 docs/upstream-audit.md。
    std::vector<std::int32_t> word2ph;

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

/// 创建指定语言的前端。
///
/// `Language::Auto` 会按文本内容自动判断（与 `detect_language()` 一致）。
/// `Language::HybridChineseEnglish` 目前回退到中文前端。
///
/// @return 前端实例；语言不受支持时返回 nullptr。
[[nodiscard]] std::unique_ptr<TextFrontend> make_frontend(Language language);

}  // namespace genie::text

#endif  // GENIE_TEXT_HPP
