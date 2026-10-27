// src/text/char_convert_table.hpp
//
// 繁简转换表（自动生成的数据文件的接口）。
//
// 上游把繁简两个字表存成**两个等长的平行字符串** —— 下标 i 的简体字
// 对应下标 i 的繁体字。转换就是在这两个字符串之间查表。
//
// 数据来源：Genie-TTS 的 G2P/Chinese/Normalization/char_convert.py
// （源头 PaddlePaddle / WeTextProcessing，Apache-2.0），
// 逐字保留的声明见生成文件的头部，授权说明见仓库根目录 NOTICE。

#ifndef GENIE_TEXT_CHAR_CONVERT_TABLE_HPP
#define GENIE_TEXT_CHAR_CONVERT_TABLE_HPP

#include <string>
#include <string_view>

namespace genie::text {

/// 简体字表。
[[nodiscard]] std::string_view simplified_chars() noexcept;

/// 繁体字表。与 simplified_chars() **逐字对应**（等长）。
[[nodiscard]] std::string_view traditional_chars() noexcept;

}  // namespace genie::text

#endif  // GENIE_TEXT_CHAR_CONVERT_TABLE_HPP
