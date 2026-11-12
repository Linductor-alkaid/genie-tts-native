// src/text/segmenter.hpp
//
// 合成分句 —— M6 流式合成的分段基础。
//
// 上游 Genie-TTS 不做分句：长文本要求**调用方自己切句喂**（见
// docs/upstream-audit.md §24.1）。流式合成要"边合成边出音频"，就得
// 在引擎侧把整段文本切成可独立合成的句段，逐段走完
// 前端 → T2S → VITS 后立刻把音频推给消费者 —— 这正是上游
// "调用方分句多次合成再拼音频"用法的引擎内化（§24.3 已确认韵律等价）。
//
// 与 normalizer 内部的 split_sentences（src/text/normalizer.cpp，为规范化
// 粒度服务、会丢弃 ASCII '.'）是两回事：这里服务于合成边界，
// 规则见 split_for_synthesis 的注释。

#ifndef GENIE_SRC_TEXT_SEGMENTER_HPP
#define GENIE_SRC_TEXT_SEGMENTER_HPP

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace genie::text {

/// 把整段文本切成可独立合成的句段。
///
/// 规则：
///   - **硬终止符**（句界，立即落刀）：`。！？!?…` 与换行（\n / \r）。
///     连续终止符归并（`！！！` 不切成三段），紧随的收尾引号
///     （`” ’ 」』`）粘进前句；纯标点的前缀（如 `……你好` 的 `……`）丢弃。
///   - **软终止符**（`，,；;：:、`）：段长没到 max_chars 就一直攒着
///     （短句保持完整，韵律更好）；超长时回退到最近的软终止符落刀。
///   - ASCII `.` **不是**终止符 —— 小数（3.14）、时间（15:30:45）里的
///     点不能断句（与 normalizer 的决策一致）。代价是英文句尾的 `.`
///     不触发分句，长英文文本只能靠 max_chars 硬切或调用方自己分句
///     —— 上游本来就要求调用方自己分句，这个限制不比上游差。
///   - max_chars == 0 表示不限长（只按硬终止符切）。
///   - 每段去掉首尾空白；空白段丢弃。计数按字符（码点）而非字节。
///
/// 同步合成的 `synthesize()` **不走**这里 —— 上游一致性优先，
/// 整段文本仍然作为一次解码喂给模型；分句只用于流式路径。
[[nodiscard]] std::vector<std::string> split_for_synthesis(std::string_view text,
                                                           std::size_t max_chars);

}  // namespace genie::text

#endif  // GENIE_SRC_TEXT_SEGMENTER_HPP
