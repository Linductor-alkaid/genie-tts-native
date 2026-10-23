// include/genie/pinyin_table.hpp
//
// 拼音词典的内部声明。数据文件由 tools/g2p/generate_pinyin_table.py 生成。

#ifndef GENIE_PINYIN_TABLE_HPP
#define GENIE_PINYIN_TABLE_HPP

namespace genie::pinyin::detail {

/// 每行 "<汉字> <tone3 拼音>"，按码位升序。
extern const char* const kPinyinTable;

}  // namespace genie::pinyin::detail

#endif  // GENIE_PINYIN_TABLE_HPP
