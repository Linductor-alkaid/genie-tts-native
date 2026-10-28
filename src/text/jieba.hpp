// src/text/jieba.hpp
//
// jieba 分词（数据驱动，MIT，见 third_party/jieba/）。
//
// 上游（Genie-TTS）的中文 G2P 用 jieba_fast.posseg.lcut 对每个标点段
// 做分词 + 词性标注；jieba_fast 与 jieba 结果一致（C 加速分支）。
// 本实现移植 jieba 0.42.1 的精确算法：
//   get_DAG（词典前向扫描）→ calc（逆序对数概率 DP，平局取较大 x）
//   → __cut_DAG（单字缓存块按词典存在性走 HMM 或逐字产出）
//   → finalseg viterbi（BMES，PrevStatus 约束，MIN_FLOAT 缺省）
//
// 数据由 tools/deploy/convert_jieba.py 打包为 third_party/jieba/jieba.bin。
// 平局语义、浮点运算顺序与上游逐一对照 —— oracle
// （tools/g2p/verify_jieba.py）以 jieba Python 为基准逐条锁定。

#ifndef GENIE_SRC_TEXT_JIEBA_HPP
#define GENIE_SRC_TEXT_JIEBA_HPP

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace genie::text::jieba {

/// 一个分词结果：词文本 + 词典词性（OOV 词为空；POS HMM 在后续切片接入）。
struct Word {
    std::string text;
    std::string tag;
};

/// jieba 分词器：容器加载 + DAG/DP/HMM 前向。
class Tokenizer {
public:
    /// 加载转换后的容器文件。
    ///
    /// @throws genie::Error 文件不存在、magic/版本不符或回读不一致
    static Tokenizer load(const std::string& path);

    /// 对**纯汉字段**分词（引擎的汉字连续段；上游对整句的正则分块
    /// 在我们的前端里由汉字段分组承担）。
    ///
    /// 语义对齐上游 `jieba.lcut(sentence)`（HMM=True 默认路径）。
    [[nodiscard]] std::vector<Word> cut(std::string_view han_text) const;

private:
    struct DictEntry {
        std::uint32_t freq = 0;
        std::string tag;
    };

    std::unordered_map<std::string, DictEntry> dict_;
    std::uint64_t total_ = 0;
    // finalseg HMM：状态序 B M E S（与转换器一致）；缺失概率已固化为
    // MIN_FLOAT，查表不再需要缺省分支。
    std::array<double, 4> start_{};
    std::array<std::array<double, 4>, 4> trans_{};
    std::unordered_map<std::string, std::array<double, 4>> emit_;
};

}  // namespace genie::text::jieba

#endif  // GENIE_SRC_TEXT_JIEBA_HPP
