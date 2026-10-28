// src/engine/g2pm.hpp — g2pM 多音字模型的加载与推理（内部实现）
//
// 上游的中文多音字消歧用 g2pM（mozillazg/pypinyin-g2pM，Apache-2.0）：
// 单层双向 LSTM + 两层全连接，权重共约 1.6 MB，pip 包以 numpy 权重
// （np_ckpt.pkl）与三张词典（char2idx / class2idx / digest_cedict）发布，
// 推理在 numpy 里手写。本实现把权重转换成单一容器文件
// （tools/deploy/convert_g2pm.py），在 C++ 里复刻同一套前向计算。
//
// 数值精度：上游 numpy 以 float64 累积（float32 权重 × float64 状态），
// 本实现载入时把 float32 权重升为 double，保持一致的累积语义。
// 见 tools/g2p/verify_g2pm.py 的逐字 oracle 比对。

#ifndef GENIE_SRC_ENGINE_G2PM_HPP
#define GENIE_SRC_ENGINE_G2PM_HPP

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "text/polyphone.hpp"

namespace genie::g2pm {

/// g2pM 模型：容器加载 + 双向 LSTM 前向 + 多音字解码。
class Model final : public text::Polyphone {
public:
    /// 加载转换后的容器文件（convert_g2pm.py 的产物）。
    ///
    /// @throws genie::Error 文件不存在、magic/版本不符或校验不一致
    static Model load(const std::string& path);

    [[nodiscard]] std::vector<std::string> pinyins_of(std::string_view segment) const override;

private:
    // ---- 权重（载入时已升为 double）----
    std::vector<double> embeddings_;  // [vocab, 64]
    std::vector<double> w_ih_;        // [128, 64] 前向 input→gates
    std::vector<double> w_hh_;        // [128, 32] 前向 hidden→gates
    std::vector<double> b_ih_;        // [128]
    std::vector<double> b_hh_;        // [128]
    std::vector<double> w_ih_rev_;
    std::vector<double> w_hh_rev_;
    std::vector<double> b_ih_rev_;
    std::vector<double> b_hh_rev_;
    std::vector<double> fc0_w_;  // [32, 64]（输入是双向拼接的 kConcatDim）
    std::vector<double> fc0_b_;  // [32]
    std::vector<double> fc1_w_;  // [876, 32]
    std::vector<double> fc1_b_;  // [876]

    std::uint32_t vocab_size_ = 0;
    std::uint32_t class_count_ = 0;

    // ---- 词典 ----
    // 字 → 词表 id；查不到走 kUnkId。
    std::vector<std::pair<std::string, std::uint32_t>> char_ids_;
    // 类 id → 带调拼音（"bian3"）。
    std::vector<std::string> idx_class_;
    // 字 → 候选读音列表（>1 即多音字）。
    std::vector<std::pair<std::string, std::vector<std::string>>> cedict_;

    // ---- 词级读音修正表（上游 polyphonic.pickle，容器 v2 携带）----
    // 词 → 整词读音（值长度与词字数一致，载入时校验）。
    std::vector<std::pair<std::string, std::vector<std::string>>> poly_entries_;
    std::unordered_map<std::string, const std::vector<std::string>*> poly_lookup_;
    std::size_t poly_max_len_ = 0;  // 最长词（字数），扫描窗口上界

public:
    /// 词级读音修正（上游 `correct_pronunciation`）。
    ///
    /// 上游按 jieba 词调用：整词命中则整体替换，否则逐字兜底。本实现
    /// 没有 jieba，词边界用**词表最长匹配**近似（差异记录于
    /// docs/upstream-audit.md §9.7）。就地改写 pinyins。
    void correct_pronunciation(const std::vector<std::string>& chars,
                               std::vector<std::string>& pinyins) const override;
};

}  // namespace genie::g2pm

#endif  // GENIE_SRC_ENGINE_G2PM_HPP
