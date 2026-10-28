// src/text/korean/korean.hpp —— 韩语 G2P
//
// 上游 `G2P/Korean/KoreanG2P.py` 的链路：
//
//   1. 拉丁字母 → 韩文读法（逐字母，"a" → "에이"）
//   2. 数字 → 韩文（汉字数词 / 固有数词，按后面的量词选）
//   3. g2pk2：惯用语表 → 英语转韩文 → 形态标注 → 数字展开 → 音节拆解
//      → 特殊规则 → 规则表 → 连音规则 → 重新组字
//   4. 组合字母拆分（j2hcj(h2j(...))）+ 复合元音拆分
//   5. g2pk2 的 "ㅇㅡㄹ ㄹ" 修正
//   6. 结尾的兼容字母后面补 "."
//   7. 逐字符过一遍符号表（不在表里的换成 "停"）
//
// 与上游的偏差见 docs/upstream-audit.md §18。

#ifndef GENIE_SRC_TEXT_KOREAN_KOREAN_HPP
#define GENIE_SRC_TEXT_KOREAN_KOREAN_HPP

#include <string>
#include <string_view>
#include <vector>

namespace genie::text::korean {

/// 上游 `KoreanG2P.g2p(text)`：韩语文本 → 音素（兼容字母）序列。
///
/// 返回的是**兼容字母**（U+3131–U+314E），不是组合字——
/// 上游最后一步把音节拆开了，音素表里存的就是兼容字母。
[[nodiscard]] std::vector<std::string> g2p(std::string_view text);

}  // namespace genie::text::korean

#endif  // GENIE_SRC_TEXT_KOREAN_KOREAN_HPP
