// tools/g2p/dump_japanese.cpp
//
// 调试用：把日语文本的音素序列打出来，供 tools/g2p/verify_japanese_g2p.py
// 与上游的 Python 实现（pyopenjtalk）逐条比对。
//
// 输入（stdin）：每行一句日语
// 输出（stdout）：每行 "<音素|音素|...>"（分隔符用 | —— 音素里没有 |）
//
// 需要 OpenJTalk 的词典（open_jtalk_dic_utf_8-1.11，约 103 MB ——
// 不在仓库里 ✓）。词典目录由 GENIE_JAPANESE_DICT 环境变量给出 ✓。

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "text/japanese/japanese.hpp"
#include "text/japanese/openjtalk.hpp"

int main() {
    std::ios::sync_with_stdio(false);

    const char* dict_env = std::getenv("GENIE_JAPANESE_DICT");
    if (dict_env == nullptr || *dict_env == '\0') {
        std::cerr << "GENIE_JAPANESE_DICT is not set (point it at open_jtalk_dic_utf_8-1.11)\n";
        return 2;
    }

    genie::text::japanese::OpenJTalk engine;
    if (!engine.load(dict_env)) {
        std::cerr << "failed to load OpenJTalk dictionary from " << dict_env << "\n";
        return 2;
    }
    const genie::text::japanese::JapaneseG2p g2p(engine);

    std::string line;
    while (std::getline(std::cin, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        const auto phonemes = g2p.g2p(line);
        for (std::size_t i = 0; i < phonemes.size(); ++i) {
            if (i != 0) {
                std::cout << '|';
            }
            std::cout << phonemes[i];
        }
        std::cout << '\n';
    }
    return 0;
}
