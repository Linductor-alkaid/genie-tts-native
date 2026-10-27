#!/usr/bin/env python3
"""生成繁简转换表。

数据来自上游 Genie-TTS 的
    src/genie_tts/G2P/Chinese/Normalization/char_convert.py
（该文件源头是 PaddlePaddle / WeTextProcessing，Apache-2.0）。

上游把繁简两个字表存成**两个等长的平行字符串** —— 下标 i 的简体字
对应下标 i 的繁体字。转换就是在这两个字符串之间查表。

用法：
    python3 tools/g2p/generate_char_convert.py \
        --source /path/to/char_convert.py \
        --output src/text/char_convert_table.cpp
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

# 每个字符串字面量切多块：MSVC 单个字面量上限 65535 字节。
CHUNK_CHARS = 2000

HEADER = """// src/text/char_convert_table.cpp
//
// 繁简转换表 —— **自动生成**，不要手改。
//
// 生成方式：python3 tools/g2p/generate_char_convert.py
//
// 数据来源：Genie-TTS 的 G2P/Chinese/Normalization/char_convert.py
// （源头：PaddlePaddle / WeTextProcessing，Apache License 2.0）
//
// 下面是**逐字保留**的原始版权与许可声明（Apache-2.0 要求保留声明）：
//
//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
//
// 改动：仅把两个 Python 字符串常量改为 C++ 字符串字面量分块，
//       并提供查表接口。数据一字未改。
//
// 授权说明见仓库根目录 NOTICE。

#include <cstddef>
#include <string>

#include "text/char_convert_table.hpp"

namespace genie::text {
namespace {

/// 把分块的字符串字面量拼起来。
///
/// 为什么要分块：MSVC 对**单个**字符串字面量有 65535 字节上限，
/// 而这张表单个就有 32 KB —— 加上转义会超。分块后每块都远低于上限。
template <std::size_t N>
std::string join_chunks(const char* const (&chunks)[N]) {
    std::size_t total = 0;
    for (const char* chunk : chunks) {
        total += std::char_traits<char>::length(chunk);
    }
    std::string result;
    result.reserve(total);
    for (const char* chunk : chunks) {
        result.append(chunk);
    }
    return result;
}

/// 简体字表（与 kTraditionalChunks 逐字对应）。
constexpr const char* kSimplifiedChunks[] = {
"""

FOOTER = """};

/// 繁体字表（与 kSimplifiedChunks 逐字对应）。
constexpr const char* kTraditionalChunks[] = {
"""


def extract(source_text: str) -> tuple[str, str]:
    """从上游 char_convert.py 里抠出两个平行字符串。"""
    simplified = re.search(
        r'simplified_charcters\s*=\s*"(.*?)"\s*\n', source_text, re.S)
    traditional = re.search(
        r'traditional_characters\s*=\s*"(.*?)"\s*\n', source_text, re.S)
    if not simplified or not traditional:
        raise SystemExit("char_convert.py 结构变了：找不到两个字表字符串")
    s = simplified.group(1)
    t = traditional.group(1)
    if len(s) != len(t):
        raise SystemExit(f"两个字表长度不一致：{len(s)} vs {len(t)}")
    return s, t


def chunks(text: str) -> list[str]:
    """按 CHUNK_CHARS 个**字符**切块（不是字节）。"""
    return [text[i:i + CHUNK_CHARS] for i in range(0, len(text), CHUNK_CHARS)]


def emit_literal(value: str) -> str:
    """把一段文本变成 C++ 字符串字面量（转义引号与反斜杠）。"""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'    "{escaped}",\n'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="上游 char_convert.py 的路径")
    parser.add_argument("--output", required=True, help="生成的 C++ 文件路径")
    args = parser.parse_args()

    source = pathlib.Path(args.source).read_text(encoding="utf-8")
    simplified, traditional = extract(source)

    out: list[str] = [HEADER]
    for chunk in chunks(simplified):
        out.append(emit_literal(chunk))
    out.append(FOOTER)
    for chunk in chunks(traditional):
        out.append(emit_literal(chunk))
    out.append("""};

}  // namespace

std::string_view simplified_chars() noexcept {
    static const std::string joined = join_chunks(kSimplifiedChunks);
    return joined;
}

std::string_view traditional_chars() noexcept {
    static const std::string joined = join_chunks(kTraditionalChunks);
    return joined;
}

}  // namespace genie::text
""")

    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(out), encoding="utf-8")
    print(f"wrote {output} ({output.stat().st_size} bytes, "
          f"{len(simplified)} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
