// tools/g2p/dump_table_digest.cpp
//
// 调试用：算出整张拼音表的摘要，供 tests/unit/test_pinyin.cpp 的
// Pinyin.TableDigestIsStable 断言使用。
//
// 用法：
//   g++ -std=c++20 -Iinclude tools/g2p/dump_table_digest.cpp \
//       src/text/chinese_g2p.cpp src/text/pinyin_table.cpp \
//       src/runtime/sha256.cpp src/runtime/errors.cpp src/runtime/utf8.cpp \
//       -o /tmp/dump_table_digest && /tmp/dump_table_digest

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>

#include "genie/pinyin.hpp"
#include "genie/sha256.hpp"
#include "genie/utf8.hpp"

int main() {
    std::string blob;
    for (std::uint32_t codepoint = 0x4E00; codepoint <= 0x9FA5; ++codepoint) {
        const std::string character = genie::utf8::encode(codepoint);
        const auto pinyin = genie::pinyin::lookup(character);
        if (!pinyin) {
            continue;
        }
        const auto syllable = genie::pinyin::split(*pinyin);
        if (!syllable.valid()) {
            continue;
        }
        blob += character;
        blob += '\t';
        const auto phones = genie::pinyin::syllable_to_phones(syllable);
        for (std::size_t i = 0; i < phones.size(); ++i) {
            if (i != 0) {
                blob += ',';
            }
            blob += phones[i];
        }
        blob += '\n';
    }
    std::cout << "chars=" << std::count(blob.begin(), blob.end(), '\n') << "\n";
    std::cout << "sha256=" << genie::sha256_hex(blob) << "\n";
    return 0;
}
