// tools/g2p/dump_pinyin.cpp
//
// 调试用：把 tone3 拼音的切分结果与音素序列打出来，供
// tools/g2p/verify_pinyin_rules.py 与 pypinyin 逐条比对。
//
// 输入（stdin）：每行 "<码点十六进制> <tone3 拼音>"
// 输出（stdout）：每行 "<码点十进制>\t<声母>\t<带调韵母>\t<音素,音素,...>"

#include <cstdint>
#include <iostream>
#include <string>

#include "genie/pinyin.hpp"

int main() {
    std::ios::sync_with_stdio(false);
    std::string line;
    while (std::getline(std::cin, line)) {
        const auto space = line.find(' ');
        if (space == std::string::npos) {
            continue;
        }
        const auto codepoint =
            static_cast<std::uint32_t>(std::stoul(line.substr(0, space), nullptr, 16));
        const auto syllable = genie::pinyin::split(line.substr(space + 1));
        const auto phones = genie::pinyin::syllable_to_phones(syllable);

        std::cout << codepoint << '\t' << syllable.initial << '\t' << syllable.final_with_tone
                  << '\t';
        for (std::size_t i = 0; i < phones.size(); ++i) {
            if (i != 0) {
                std::cout << ',';
            }
            std::cout << phones[i];
        }
        std::cout << '\n';
    }
    return 0;
}
