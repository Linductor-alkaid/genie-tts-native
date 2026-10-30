// src/text/tone_sandhi.hpp — 中文变调与儿化（内部实现，不进公开头）。
//
// 移植自上游 Genie-TTS 的 G2P/Chinese/ToneSandhi.py（源头 PaddlePaddle，
// Apache-2.0）与 G2P/Chinese/Erhua.py，基线 commit d347fd0f。
//
// 上游这两个处理器都作用在 **jieba 分词结果**（`(word, pos)` 流）上；
// 本项目已移植 jieba（见 text/jieba.hpp）—— 有 jieba 时词边界与上游
// 同源，缺失时退回"上游词表 + 单字"的贪心最长匹配分词。
// 依赖 POS 标注的分支换成记录在案的默认行为 —— 逐条差异与
// oracle 比对结果见 docs/upstream-audit.md §9.6。
//
// 管线位置：音素转换之前、拼音切分之后 —— 变调只改 final_with_tone
// 末尾的声调数字，不改声母与韵母主体，所以 opencpop 查表键不受影响。

#ifndef GENIE_SRC_TEXT_TONE_SANDHI_HPP
#define GENIE_SRC_TEXT_TONE_SANDHI_HPP

#include <string>
#include <vector>

namespace genie::text::sandhi {

/// 一个处理单元：上游的一个 jieba 词（词表命中）或单个汉字。
///
/// chars 与 finals 一一对应；finals 取 `pinyin::split()` 的
/// final_with_tone（tone3 声调数字结尾，'v' 表 ü）。
/// 拼音表查不到的字 final 为空串 —— 只参与不查韵母的合并，声调规则会跳过它。
struct Unit {
    /// UTF-8 字符序列（每元素一个字符）。
    std::vector<std::string> chars;
    /// 每字带调韵母（tone3），与 chars 等长；无拼音的字为空串。
    std::vector<std::string> finals;

    /// 词面（chars 拼接），词表查找用。
    [[nodiscard]] std::string word() const;
    /// 末尾至多两个字符（不足则原样返回）。上游 `word[-2:]` 的等价物。
    [[nodiscard]] std::string last_two() const;
};

/// 把一段纯汉字序列切成单元：词表命中取最长，否则按单字。
///
/// 这是**没有 jieba 时的降级路径**（音色包不带 jieba 角色）。
/// 有 jieba 时前端改用 `jieba::Tokenizer::cut` 的词边界
/// （见 chinese_frontend.cpp 的 process_han_run）。
///
/// 对应上游 `jieba_fast.posseg.lcut` —— 没有 POS 与统计分词，
/// 只有"上游四张词表里的词必须保持整词"这一条硬约束。
[[nodiscard]] std::vector<Unit> segment(const std::vector<std::string>& chars);

/// 上游 `ToneSandhi.pre_merge_for_modify`：在变调前重排单元边界。
///
/// 照搬五个合并（不/一/叠字/三声连读×2/儿）；"一 V 一 V"合并
/// 需要 POS 标注，未移植（见 audit §9.6）。
void pre_merge_for_modify(std::vector<Unit>& seg);

/// 上游 `ToneSandhi.modified_tone`：对单元内韵母做变调（原地修改）。
///
/// 含"不"变调、"一"变调、轻声、三声连读四类；依赖 POS 的分支取
/// 无 POS 默认行为（见 audit §9.6）。
void modified_tone(Unit& unit);

/// 上游 `ErhuaProcessor.merge_erhua`：儿化处理（原地修改）。
void merge_erhua(Unit& unit);

}  // namespace genie::text::sandhi

#endif  // GENIE_SRC_TEXT_TONE_SANDHI_HPP
