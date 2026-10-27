// include/genie/normalizer.hpp
//
// 中文文本规范化。
//
// 对齐上游 Genie-TTS 的
//     G2P/Chinese/Normalization/text_normlization.py
// 里的 `TextNormalizer`（源头 PaddlePaddle / WeTextProcessing）。
//
// 上游的规范化做这些事：
//   - 繁简转换
//   - 全角 → 半角（字母 / 数字 / 空格）
//   - 日期 / 时间 / 时间范围
//   - 小数 / 分数 / 百分比 / 区间 / 版本号 / 幂
//   - 手机号 / 固话 / 400 号
//   - 量词 / 温度 / 单位
//   - 数学运算符、希腊字母、带圈数字
//   - 标点映射与折叠
//
// 少了这一步，"2024年" 会被读成 "二千零二十四年"、
// 手机号会被读成 "一百三十八亿一千二百三十四万五千六百七十八" ——
// 上游分别读作 "二零二四年" 和 "幺三八幺二三四五六七八"。
//
// 本实现**逐条**照抄上游的规则与替换顺序（顺序会影响结果）。
// 行为验证见 docs/upstream-audit.md §15 与 tests/unit/test_normalizer.cpp。

#ifndef GENIE_NORMALIZER_HPP
#define GENIE_NORMALIZER_HPP

#include <string>
#include <string_view>

namespace genie::text {

/// 中文文本规范化。
///
/// @param text 原始文本
/// @return 规范化后的文本：只保留中文与 `!?…,.-` 六种标点
[[nodiscard]] std::string normalize_chinese(std::string_view text);

}  // namespace genie::text

#endif  // GENIE_NORMALIZER_HPP
