// src/text/japanese/japanese.hpp —— 日语 G2P
//
// 上游 `G2P/Japanese/JapaneseG2P.py` 的链路：
//
//   1. 文本规范化（% → パーセント、折叠连续标点、转小写）
//   2. 按"非日语字符"切分：日语片段走 OpenJTalk，标点原样插回
//   3. OpenJTalk 的全上下文标签 → 音素 + 韵律符号（# [ ]）
//   4. 逐个音素过一遍替换表（全角标点 → 半角）
//
// 韵律符号（# [ ]）来自 OpenJTalk 标签里的 A/F 段 ✓ —— 上游把它们
// 一起送进模型 ✓，本移植照做 ✓。
//
// 分层：词典相关的部分（OpenJTalk 出标签）在 OpenJTalk 那侧 ✓；
// 这里的函数都是**纯逻辑** ✓ —— 不需要 103 MB 的词典就能测 ✓。

#ifndef GENIE_SRC_TEXT_JAPANESE_JAPANESE_HPP
#define GENIE_SRC_TEXT_JAPANESE_JAPANESE_HPP

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "text/japanese/openjtalk.hpp"

namespace genie::text::japanese {

/// 上游 `_text_normalize`（% → パーセント、折叠连续标点、转小写 ✓）。
[[nodiscard]] std::string text_normalize(std::string_view text);

/// 上游的切段结果 ✓。
struct Segmentation {
    /// 日语片段 ✓（非空 ✓ —— 空片段保留位置信息 ✓）。
    std::vector<std::string> segments;
    /// 夹在片段之间的非日语字符 ✓（与片段一一对应 ✓：
    /// marks[i] 在 segments[i] 之后 ✓）。
    std::vector<std::string> marks;
};

/// 上游 `_JAPANESE_MARKS_RE.split` + `findall` 的等价实现 ✓。
///
/// 上游的日语字符集是 `[A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff
/// \uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]` ✓ ——
/// 其中 `\d` 是 **Unicode 感知的** ✓（Python 3）✓，
/// 所以全角 '０'（U+FF10）也属于日语字符 ✓。
[[nodiscard]] Segmentation segment_japanese(std::string_view text);

/// 上游 `_pyopenjtalk_g2p_prosody` 的等价实现 ✓：标签 → 音素。
///
/// **包含**句首的 '^' 与句尾的 '$' / '?' ✓ —— 上游随后做 `[1:-1]` ✓，
/// 切片语义在 `text_to_phonemes` 里处理 ✓。
[[nodiscard]] std::vector<std::string> labels_to_phonemes(const std::vector<std::string>& labels);

/// 上游 `JapaneseG2P.g2p` 的纯逻辑部分 ✓：文本 + 每段的标签 → 音素。
///
/// @param segment_labels 与 `segment_japanese(text).segments` 一一对应 ✓
///                      （没有标签的片段给空表 ✓）。
[[nodiscard]] std::vector<std::string> text_to_phonemes(
    std::string_view text, const std::vector<std::vector<std::string>>& segment_labels);

/// 日语 G2P（需要一个已载入词典的 OpenJTalk ✓）。
class JapaneseG2p {
public:
    /// @param engine OpenJTalk 前端（调用方持有 ✓ —— 词典很大 ✓ 不复制 ✓）。
    explicit JapaneseG2p(OpenJTalk& engine) : engine_(&engine) {}

    /// 上游 `JapaneseG2P.g2p(text)`：文本 → 音素序列 ✓。
    [[nodiscard]] std::vector<std::string> g2p(std::string_view text) const;

private:
    OpenJTalk* engine_ = nullptr;
};

}  // namespace genie::text::japanese

#endif  // GENIE_SRC_TEXT_JAPANESE_JAPANESE_HPP
