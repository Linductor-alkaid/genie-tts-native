// tools/g2p/dump_jieba.cpp
//
// 调试用：把 jieba 分词结果打出来，供 tools/g2p/verify_jieba.py 与
// jieba Python（jieba.lcut，HMM=True）逐条比对。
//
// 输入（stdin）：每行一段**纯汉字**文本（调用方负责切好段，与引擎的
// 汉字连续段一致）。
// 输出（stdout）：每行一个 JSON 对象：
//   {"text": "...", "words": ["我们", "的", ...]}

#include <iostream>
#include <string>

#include "genie/json.hpp"
#include "genie/utf8.hpp"
#include "text/jieba.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: dump_jieba <jieba.bin>\n";
        return 1;
    }
    std::ios::sync_with_stdio(false);
    auto tokenizer = genie::text::jieba::Tokenizer::load(argv[1]);

    std::string line;
    while (std::getline(std::cin, line)) {
        // 契约与引擎一致：只喂纯汉字段（标点由前端先行切断）。
        std::string han;
        genie::utf8::Char ch;
        std::size_t offset = 0;
        while (genie::utf8::decode_next(line, offset, ch)) {
            if (ch.codepoint >= 0x4E00 && ch.codepoint <= 0x9FD5) {
                han.append(line, ch.offset, ch.length);
            }
        }
        if (han.empty()) {
            continue;
        }
        genie::json::ArrayType words;
        for (const auto& word : tokenizer.cut(han)) {
            words.emplace_back(genie::json::Value(word.text));
        }
        genie::json::ObjectType obj;
        obj.emplace("text", genie::json::Value(han));
        obj.emplace("words", genie::json::Value(std::move(words)));
        std::cout << genie::json::Value(std::move(obj)).dump() << '\n';
    }
    return 0;
}
