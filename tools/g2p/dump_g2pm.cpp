// tools/g2p/dump_g2pm.cpp
//
// 调试用：把 g2pM 模型的逐字读音打出来，供 tools/g2p/verify_g2pm.py
// 与上游 Python（pip 包 g2pM 的 G2pM()(seg, char_split=True)）逐条比对。
//
// 输入（stdin）：每行一段文本。
// 输出（stdout）：每行一个 JSON 对象：
//   {"text": "...", "pinyins": ["zhong4", "qing4", "，", ...]}

#include <iostream>
#include <string>

#include "engine/g2pm.hpp"
#include "genie/json.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: dump_g2pm <g2pm.bin>\n";
        return 1;
    }
    std::ios::sync_with_stdio(false);
    auto model = genie::g2pm::Model::load(argv[1]);

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) {
            continue;
        }
        genie::json::ArrayType pinyins;
        for (const auto& p : model.pinyins_of(line)) {
            pinyins.emplace_back(genie::json::Value(p));
        }
        genie::json::ObjectType obj;
        obj.emplace("text", genie::json::Value(line));
        obj.emplace("pinyins", genie::json::Value(std::move(pinyins)));
        std::cout << genie::json::Value(std::move(obj)).dump() << '\n';
    }
    return 0;
}
