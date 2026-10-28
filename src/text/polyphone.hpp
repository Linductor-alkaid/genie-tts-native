// src/text/polyphone.hpp
//
// 多音字消歧的领域层接口。
//
// 上游用 g2pM（mozillazg/pypinyin-g2pM，单层双向 LSTM）做整句上下文的
// 多音字判定；引擎层持有模型与推理（src/engine/g2pm），本接口把
// "文本段 → 每字读音" 的能力倒置给文本前端，避免领域层反向依赖引擎层。
//
// 没有实现接进来时（音色包不带 g2pm 模型），前端退化为查单字拼音表的
// 最常用读音 —— 与 RoBERTa / HuBERT 缺失时的降级模式一致。

#ifndef GENIE_SRC_TEXT_POLYPHONE_HPP
#define GENIE_SRC_TEXT_POLYPHONE_HPP

#include <string>
#include <string_view>
#include <vector>

namespace genie::text {

/// 多音字读音提供者。
class Polyphone {
public:
    virtual ~Polyphone() = default;

    /// 给出 segment 中每个字符的读音（带调拼音，如 "zhong4"）。
    ///
    /// 语义对齐上游 `G2pM()(segment, char_split=True)`：
    ///   - 模型认识且单读音的字：返回该读音；
    ///   - 多音字：按整句上下文判定；
    ///   - 模型不认识的字（生僻字、标点）：原样返回该字符，
    ///     由调用方回退到单字拼音表。
    [[nodiscard]] virtual std::vector<std::string> pinyins_of(std::string_view segment) const = 0;

    /// 词级读音修正（上游 `CorrectPronunciation.correct_pronunciation`）。
    ///
    /// 上游在 g2pM 输出之后、变调之前按词表修正读音（g2pM 有已知的
    /// 误判，上游用一张 4.5 万词的表兜底，如「长大→zhang3」「分行→
    /// fen1 hang2」）。默认无修正 —— 不带词表的实现保持模型原样。
    ///
    /// @param chars   segment 的逐字列表（与 pinyins 一一对应）
    /// @param pinyins 就地修正
    virtual void correct_pronunciation(const std::vector<std::string>& chars,
                                       std::vector<std::string>& pinyins) const {
        (void)chars;
        (void)pinyins;
    }
};

}  // namespace genie::text

#endif  // GENIE_SRC_TEXT_POLYPHONE_HPP
