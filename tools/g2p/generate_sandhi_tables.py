#!/ usr / bin / env python3
"""生成变调 / 儿化词表。

数据来自上游 Genie-TTS 的：
    src/genie_tts/G2P/Chinese/ToneSandhi.py   → must_neural_tone_words / must_not_neural_tone_words
    src/genie_tts/G2P/Chinese/Erhua.py        → must_erhua / not_erhua
（ToneSandhi.py 源头是 PaddlePaddle，Apache-2.0。）

用法：
    python3 tools/g2p/generate_sandhi_tables.py \\
        --tone-sandhi /path/to/ToneSandhi.py \\
        --erhua /path/to/Erhua.py \\
        --output-dir src/text

用 ast 从上游源码里把四个字面量集合**原样**取出来（不做任何增删），
每行一个词写入原始字符串表，由 src/text/sandhi_table.cpp 惰性解析。
"""

from __future__ import annotations

import argparse
import ast
import pathlib

LICENSE_NOTE = """\
// 词表来源：Genie-TTS 的 G2P/Chinese/ToneSandhi.py（源头 PaddlePaddle，
// Apache License 2.0）与 G2P/Chinese/Erhua.py。下面逐字保留原始声明：
//
//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0"""


def extract_set(source: pathlib.Path, class_name: str, attr: str) -> list[str]:
    """从上游源码里取出 class_name.__init__ 里的 attr 集合（保持词面原样）。"""
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for item in node.body:
            if not isinstance(item, ast.FunctionDef) or item.name != "__init__":
                continue
            for stmt in item.body:
#ToneSandhi 用普通赋值（self.x = {... }），
#Erhua 用带注解赋值（self.x : Set[str] = {... }），两种都要接。
                if isinstance(stmt, ast.Assign):
                    targets = stmt.targets
                    value = stmt.value
                elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                    targets = [stmt.target]
                    value = stmt.value
                else:
                    continue
                target = targets[0]
                if isinstance(target, ast.Attribute) and target.attr == attr:
                    words = ast.literal_eval(value)
                    return sorted(words)  # 排序只为生成稳定；词面不变
    raise SystemExit(f"{source.name}: 找不到 {class_name}.__init__ 中的 {attr}")


def raw_block(name: str, words: list[str]) -> str:
    body = "\n".join(words)
    return f"const char* const {name} = R\"SANDHI(\n{body}\n)SANDHI\";\n"


ACCESSOR = """\
const WordSet& {fn}() {{
    static const WordSet kWords = parse({member});
return kWords;
}
}
""
    "

    LEXICON =
        ""
        "\
const WordSet& segmentation_lexicon() {
    // 分词用的并集。四张表在语义上各查各的，分词只关心"是不是某张表里的词"，
    // 所以并在一起一次构建。
    static const WordSet kLexicon = [] {
        WordSet words;
        words.insert(must_neural_tone_words().begin(), must_neural_tone_words().end());
        words.insert(must_not_neural_tone_words().begin(), must_not_neural_tone_words().end());
        words.insert(must_erhua().begin(), must_erhua().end());
        words.insert(not_erhua().begin(), not_erhua().end());
        return words;
    }();
return kLexicon;
}
""
    "

    def
    main() -> None
    : parser = argparse.ArgumentParser(description = __doc__) parser
                   .add_argument("--tone-sandhi", required = True, type = pathlib.Path) parser
                   .add_argument("--erhua", required = True, type = pathlib.Path) parser
                   .add_argument("--output-dir", required = True,
                                 type = pathlib.Path) args = parser.parse_args()

                                                                 tables =
    [
        ("must_neural_tone_words", "kMustNeuralToneWords",
         extract_set(args.tone_sandhi, "ToneSandhi", "must_neural_tone_words")),
        ("must_not_neural_tone_words", "kMustNotNeuralToneWords",
         extract_set(args.tone_sandhi, "ToneSandhi", "must_not_neural_tone_words")),
        ("must_erhua", "kMustErhua", extract_set(args.erhua, "ErhuaProcessor", "must_erhua")),
        ("not_erhua", "kNotErhua", extract_set(args.erhua, "ErhuaProcessor", "not_erhua")),
    ]

    hpp = f
    ""
    "// src/text/sandhi_table.hpp — 变调 / 儿化词表的内部声明。
//
// 词表由 tools/g2p/generate_sandhi_tables.py 从上游
// ToneSandhi.py / Erhua.py 生成（用法见 tone_sandhi.cpp 顶部注释）。
//
{
    LICENSE_NOTE
}

#ifndef GENIE_SRC_TEXT_SANDHI_TABLE_HPP
#define GENIE_SRC_TEXT_SANDHI_TABLE_HPP

#include <string>
#include <unordered_set>

namespace genie::text::sandhi::detail {
{

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

}
}  // namespace genie::text::sandhi::detail

#endif  // GENIE_SRC_TEXT_SANDHI_TABLE_HPP
"""

    parts = [f"""// src/text/sandhi_table.cpp
//
// 本文件由 tools/g2p/generate_sandhi_tables.py 生成，请勿手工修改。
//
// 四张词表，每行一个词：
//   kMustNeuralToneWords    上游 ToneSandhi.must_neural_tone_words（必须轻声）
//   kMustNotNeuralToneWords 上游 ToneSandhi.must_not_neural_tone_words（禁止轻声）
//   kMustErhua              上游 ErhuaProcessor.must_erhua（必须儿化）
//   kNotErhua               上游 ErhuaProcessor.not_erhua（禁止儿化）
//
{
    LICENSE_NOTE}
#include "text/sandhi_table.hpp"

#include <string_view>

namespace genie::text::sandhi::detail {
    {
        namespace {
        {

            /// 惰性解析"每行一个词"的原始文本。行尾 \\r 要去掉（Windows checkout 是 CRLF）。
            WordSet parse(std::string_view text) {
                {
                    WordSet words;
                    while (!text.empty()) {
                        {
                            const auto newline = text.find('\\n');
                            std::string_view line = text.substr(
                                0, newline == std::string_view::npos ? text.size() : newline);
                            if (!line.empty() && line.back() == '\\r') {
                                {
                                    line.remove_suffix(1);
                                }
                            }
                            if (!line.empty()) {
                                {
                                    words.emplace(line);
                                }
                            }
                            if (newline == std::string_view::npos) {
                                {
                                    break;
                                }
                            }
                            text = text.substr(newline + 1);
                        }
                    }
                    return words;
                }
            }

        }
        }  // namespace
"""]

    for fn, member, _ in tables:
        parts.append(ACCESSOR.format(fn=fn, member=member))
    parts.append(LEXICON)
    for fn, member, words in tables:
        del fn
        parts.append(f"\n// {member}：{len(words)} 词。\n" + raw_block(member, words))

    cpp = "\n\n".join(parts) + "\n\n}  // namespace genie::text::sandhi::detail\n"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "sandhi_table.hpp").write_text(hpp, encoding="utf-8")
    (args.output_dir / "sandhi_table.cpp").write_text(cpp, encoding="utf-8")
    print("  ".join(f"{fn}={len(words)}" for fn, _, words in tables))


if __name__ == "__main__":
    main()
