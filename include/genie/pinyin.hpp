// include/genie/pinyin.hpp
//
// 拼音 → 音素符号的转换。
//
// 规则与上游 `src/genie_tts/G2P/Chinese/ChineseG2P.py` 一致：
//   拼音（tone3）→ 声母 + 带调韵母 → 音素符号
//
// 其中"拼音 → 声母/韵母"的切分规则与 pypinyin 的
// `to_initials()` / `to_finals_tone3()` **逐字等价**（已对 41,923 个汉字
// 全量比对，见 tools/g2p/verify_pinyin_rules.py）。
//
// 最后一步"声母/韵母 → 音素符号"是上游自己的一层转换：
// 上游查的是 opencpop 词表，而本实现直接按等价规则推导 ——
// opencpop 的词素划分恰好就是"声母 + 收缩后的韵母"。

#ifndef GENIE_PINYIN_HPP
#define GENIE_PINYIN_HPP

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace genie::pinyin {

/// 一个拼音音节。
struct Syllable {
    /// 声母。零声母时为空。
    std::string initial;
    /// 带调韵母（tone3）。例如 "ong1"。无法切分时为空。
    std::string final_with_tone;

    [[nodiscard]] bool valid() const noexcept { return !final_with_tone.empty(); }
};

/// 切分一个 tone3 拼音（"zhong1"）。
///
/// @param tone3 形如 "zhong1" 的拼音；ü 写作 'v'（"lv4"）。
/// @return 切分结果。无法识别时 initial 与 final_with_tone 均为空。
[[nodiscard]] Syllable split(std::string_view tone3) noexcept;

/// 汉字 → tone3 拼音。词典里没有则返回空。
///
/// @param codepoint 汉字的 Unicode 码位。
[[nodiscard]] std::optional<std::string> lookup(std::uint32_t codepoint);

/// 汉字 → tone3 拼音。词典里没有则返回空。
///
/// @param utf8_char 单个汉字的 UTF-8 编码。
[[nodiscard]] std::optional<std::string> lookup(std::string_view utf8_char);

/// 音节 → 音素符号序列（尚未按符号表过滤）。
///
/// 例如 "zhong1" → {"zh", "ong1"}；"yi1" → {"y", "i1"}；
/// "a1" → {"a", "a1"}（其中 "a" 不在符号表里，会被过滤掉）。
[[nodiscard]] std::vector<std::string> syllable_to_phones(const Syllable& syllable);

}  // namespace genie::pinyin

#endif  // GENIE_PINYIN_HPP
