// tools/g2p/dump_korean.cpp
//
// 调试用：把韩语文本的音素序列打出来，供 tools/g2p/verify_korean_g2p.py
// 与上游的 Python 实现逐条比对。
//
// 输入（stdin）：每行一句韩语
// 输出（stdout）：每行 "<音素|音素|...>"（分隔符用 | —— 音素里没有 |）
//
// 上游的韩语链路不需要外部资源 ✓（词典、模型都是内置的）。

#include <iostream>
#include <string>

#include "text/korean/korean.hpp"

int main() {
    std::ios::sync_with_stdio(false);

    std::string line;
    while (std::getline(std::cin, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        const auto phonemes = genie::text::korean::g2p(line);
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
