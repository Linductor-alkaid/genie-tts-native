// tools/g2p/dump_sandhi.cpp
//
// 调试用：把变调/儿化流水线的中间结果打出来，供
// tools/verify_sandhi.py 与上游 Python（ToneSandhi.py / Erhua.py）
// 逐条比对。
//
// 输入（stdin）：每行一段**纯汉字**文本（调用方负责切好段）。
// 输出（stdout）：每行一个 JSON 对象：
//   {"text": "...", "units": [{"word": "...", "finals": ["i3", ...]}, ...]}
//
// 流水线与 src/text/chinese_frontend.cpp 的 process_han_run 保持一致
// （分词 → 填韵母 → pre_merge → modified_tone → merge_erhua）；
// 改了前端记得同步这里。
//
// 分词源由环境变量 GENIE_JIEBA_MODEL 切换（jieba 模型容器路径）：
//   未设置 → 词表贪心最长匹配（音色包不带 jieba 时的降级路径，
//             tests/golden/sandhi.json 锁的就是这条）
//   已设置 → jieba 统计分词（音色包带 jieba 时的生产路径）

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "genie/json.hpp"
#include "genie/pinyin.hpp"
#include "genie/utf8.hpp"
#include "text/jieba.hpp"
#include "text/tone_sandhi.hpp"

int main() {
    std::ios::sync_with_stdio(false);

    // 可选的 jieba 模型：与前端 engine.cpp 的装配同源（音色包角色 jieba）。
    std::unique_ptr<genie::text::jieba::Tokenizer> jieba;
    if (const char* model = std::getenv("GENIE_JIEBA_MODEL"); model != nullptr && *model != '\0') {
        jieba = std::make_unique<genie::text::jieba::Tokenizer>(
            genie::text::jieba::Tokenizer::load(model));
    }

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) {
            continue;
        }

        std::vector<std::string> chars;
        std::size_t offset = 0;
        genie::utf8::Char ch;
        while (genie::utf8::decode_next(line, offset, ch)) {
            chars.emplace_back(line.substr(ch.offset, ch.length));
        }

        std::vector<genie::text::sandhi::Unit> units;
        if (jieba) {
            // 与前端 process_han_run 的 jieba 分支一致。
            std::string run;
            for (const auto& c : chars) {
                run += c;
            }
            for (const auto& word : jieba->cut(run)) {
                genie::text::sandhi::Unit unit;
                std::size_t word_offset = 0;
                genie::utf8::Char word_ch;
                while (genie::utf8::decode_next(word.text, word_offset, word_ch)) {
                    unit.chars.emplace_back(word.text.substr(word_ch.offset, word_ch.length));
                }
                unit.finals.assign(unit.chars.size(), "");
                units.push_back(std::move(unit));
            }
            if (units.empty()) {
                units = genie::text::sandhi::segment(chars);
            }
        } else {
            units = genie::text::sandhi::segment(chars);
        }
        {
            std::size_t index = 0;
            for (auto& unit : units) {
                for (auto& final_with_tone : unit.finals) {
                    if (const auto pinyin = genie::pinyin::lookup(chars[index]); pinyin) {
                        if (const auto syllable = genie::pinyin::split(*pinyin); syllable.valid()) {
                            final_with_tone = syllable.final_with_tone;
                        }
                    }
                    ++index;
                }
            }
        }
        // 变调前的逐字韵母：oracle 比对要给上游喂**同一份**输入，
        // 隔离拼音表差异（见 tools/verify_sandhi.py）。
        genie::json::ArrayType finals_in;
        for (const auto& unit : units) {
            for (const auto& final_with_tone : unit.finals) {
                finals_in.emplace_back(genie::json::Value(final_with_tone));
            }
        }

        genie::text::sandhi::pre_merge_for_modify(units);
        for (auto& unit : units) {
            genie::text::sandhi::modified_tone(unit);
            genie::text::sandhi::merge_erhua(unit);
        }

        genie::json::ArrayType json_units;
        for (const auto& unit : units) {
            genie::json::ArrayType finals;
            for (const auto& final_with_tone : unit.finals) {
                finals.emplace_back(genie::json::Value(final_with_tone));
            }
            genie::json::ObjectType unit_obj;
            unit_obj.emplace("word", genie::json::Value(unit.word()));
            unit_obj.emplace("finals", genie::json::Value(std::move(finals)));
            json_units.emplace_back(genie::json::Value(std::move(unit_obj)));
        }
        genie::json::ObjectType obj;
        obj.emplace("text", genie::json::Value(line));
        obj.emplace("finals_in", genie::json::Value(std::move(finals_in)));
        obj.emplace("units", genie::json::Value(std::move(json_units)));
        std::cout << genie::json::Value(std::move(obj)).dump() << '\n';
    }
    return 0;
}
