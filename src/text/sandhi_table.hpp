// src/text/sandhi_table.hpp — 变调 / 儿化词表的内部声明。
//
// 词表由 tools/g2p/generate_sandhi_tables.py 从上游
// ToneSandhi.py / Erhua.py 生成（用法见 tone_sandhi.cpp 顶部注释）。
//
// 词表来源：Genie-TTS 的 G2P/Chinese/ToneSandhi.py（源头 PaddlePaddle，
// Apache License 2.0）与 G2P/Chinese/Erhua.py。下面逐字保留原始声明：
//
//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0

#ifndef GENIE_SRC_TEXT_SANDHI_TABLE_HPP
#define GENIE_SRC_TEXT_SANDHI_TABLE_HPP

#include <string>
#include <unordered_set>

namespace genie::text::sandhi::detail {

/// 查找表：词 → 在表里。
using WordSet = std::unordered_set<std::string>;

/// 内嵌的原始词表文本（由生成器产出，定义在 sandhi_table.cpp）。
extern const char* const kMustNeuralToneWords;
extern const char* const kMustNotNeuralToneWords;
extern const char* const kMustErhua;
extern const char* const kNotErhua;

/// 四张词表（惰性解析，定义在 sandhi_table.cpp）。
[[nodiscard]] const WordSet& must_neural_tone_words();
[[nodiscard]] const WordSet& must_not_neural_tone_words();
[[nodiscard]] const WordSet& must_erhua();
[[nodiscard]] const WordSet& not_erhua();

/// 供分词用的并集（含全部四张表）。
[[nodiscard]] const WordSet& segmentation_lexicon();

}  // namespace genie::text::sandhi::detail

#endif  // GENIE_SRC_TEXT_SANDHI_TABLE_HPP
