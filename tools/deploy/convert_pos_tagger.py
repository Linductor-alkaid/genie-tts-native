#!/usr/bin/env python3
"""convert_pos_tagger.py —— 把 nltk 的英文词性标注模型打包成引擎容器。

上游（Genie-TTS）的英文 G2P 用 `nltk.pos_tag` 做词性标注，目的只有一个：
给 4 个同形异音词挑读音（read / complex / lead / presents）。
标注器是 nltk 的 averaged perceptron（Matthew Honnibal），权重来自
`averaged_perceptron_tagger_eng`（Apache-2.0）。

本脚本把三份 JSON 打包成单个容器：

  - `averaged_perceptron_tagger_eng.classes.json`  —— 49 个标签
  - `averaged_perceptron_tagger_eng.tagdict.json`  —— 1,549 个高频词的直接标注
  - `averaged_perceptron_tagger_eng.weights.json` —— 75,447 个特征 × 271,206 个权重

容器格式（`src/text/english/pos_tagger.cpp` 解析）：

    magic "GENIEPOS" | u32 version | u32 class_count
    | 标签名表：每条 u32 长度 + UTF-8 字节
    | u32 tagdict_count | 每条：u32 词长 + UTF-8 字节 + u8 标签下标
    | u32 feature_count
    | 每条：u32 特征长度 + UTF-8 字节 + u32 权重条数 + (u8 标签下标 + f32 权重) × n

用法：
    python3 tools/deploy/convert_pos_tagger.py \\
        --tagger-dir /path/to/nltk_data/taggers/averaged_perceptron_tagger_eng \\
        --output third_party/english/pos_tagger.bin
"""

import argparse
import json
import struct
from pathlib import Path

MAGIC = b"GENIEPOS"
VERSION = 1


def u32(v: int) -> bytes:
    return struct.pack("<I", v)


def u8(v: int) -> bytes:
    return struct.pack("<B", v)


def f32(v: float) -> bytes:
    return struct.pack("<f", v)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tagger-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    classes = json.loads(
        (args.tagger_dir / "averaged_perceptron_tagger_eng.classes.json").read_text(encoding="utf-8")
    )
    tagdict = json.loads(
        (args.tagger_dir / "averaged_perceptron_tagger_eng.tagdict.json").read_text(encoding="utf-8")
    )
    weights = json.loads(
        (args.tagger_dir / "averaged_perceptron_tagger_eng.weights.json").read_text(encoding="utf-8")
    )

    class_index = {name: i for i, name in enumerate(classes)}

    out = bytearray()
    out += MAGIC
    out += u32(VERSION)
    out += u32(len(classes))
    for name in classes:
        raw = name.encode("utf-8")
        out += u32(len(raw))
        out += raw

    out += u32(len(tagdict))
    for word in sorted(tagdict):
        raw = word.encode("utf-8")
        out += u32(len(raw))
        out += raw
        out += u8(class_index[tagdict[word]])

    out += u32(len(weights))
    for feature in sorted(weights):
        raw = feature.encode("utf-8")
        out += u32(len(raw))
        out += raw
        entries = weights[feature]
        out += u32(len(entries))
        for label in sorted(entries):
            out += u8(class_index[label])
            out += f32(entries[label])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(bytes(out))
    print(
        f"wrote {args.output} ({len(out)} bytes): {len(classes)} classes, "
        f"{len(tagdict)} tagdict entries, {len(weights)} features"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
