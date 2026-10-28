// tools/g2p/dump_english.cpp
//
// 调试用：把英文文本的规范化结果 / 音素序列打出来，供
// tools/g2p/verify_english_g2p.py 与上游的 Python 实现逐条比对。
//
// 输入（stdin）：每行一句英文
// 输出（stdout）：每行 "<规范化结果>\t<音素|音素|...>"
//
// 环境变量：
//   GENIE_CMU_DICT          CMU 词典容器路径（third_party/english/cmudict.bin）
//   GENIE_ENGLISH_G2P_MODEL GRU 权重容器路径（third_party/english/english_g2p.bin）
//   GENIE_WORD_SEGMENT_DIR  ngram 分词词典目录（可选）

#include <cstdlib>
#include <iostream>
#include <string>

#include "text/english/english.hpp"

int main() {
    std::ios::sync_with_stdio(false);

    genie::text::english::CmuDict dict;
    const char* dict_path = std::getenv("GENIE_CMU_DICT");
    if (dict_path != nullptr) {
        try {
            dict = genie::text::english::CmuDict::load(dict_path);
        } catch (const std::exception& e) {
            std::cerr << "failed to load cmu dict: " << e.what() << '\n';
            return 2;
        }
    }

    genie::text::english::Seq2SeqG2p seq2seq;
    const char* model_path = std::getenv("GENIE_ENGLISH_G2P_MODEL");
    if (model_path != nullptr) {
        if (!seq2seq.load(model_path)) {
            std::cerr << "failed to load " << model_path << "\n";
            return 1;
        }
    }

    genie::text::english::PosTagger tagger;
    const char* tagger_path = std::getenv("GENIE_POS_TAGGER");
    if (tagger_path != nullptr) {
        try {
            tagger = genie::text::english::PosTagger::load(tagger_path);
        } catch (const std::exception& e) {
            std::cerr << "failed to load pos tagger: " << e.what() << '\n';
            return 2;
        }
    }

    genie::text::english::WordSegmenter segmenter;
    const char* segment_dir = std::getenv("GENIE_WORD_SEGMENT_DIR");
    if (segment_dir != nullptr) {
        segmenter.load(segment_dir);
    }

    std::string line;
    while (std::getline(std::cin, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        const std::string normalized = genie::text::english::normalize(line);
        const std::string text_normalized = genie::text::english::text_normalize(line);
        std::cout << normalized << '\t' << text_normalized << '\t';
        const auto phones = genie::text::english::phonemes(line, dict, seq2seq, segmenter, &tagger);
        for (std::size_t i = 0; i < phones.size(); ++i) {
            if (i != 0) {
                // 分隔符不能用逗号 —— 逗号本身就是音素（PUNCTUATION_SYMBOLS）。
                std::cout << '|';
            }
            std::cout << phones[i];
        }
        std::cout << '\n';
    }
    return 0;
}
