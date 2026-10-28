#!/usr/bin/env python3
"""convert_cmu_dict.py —— 把 CMU 发音词典打包成引擎容器。

上游（Genie-TTS）的英文 G2P 用 CMU 发音词典查词
（`G2P/English/EnglishG2P.py` 的 `_read_cmu_dict`）。上游那份词典随
GenieData 发布（`English_G2P/cmudict.rep`），本脚本从公开的 CMU
Pronouncing Dictionary（BSD 许可，见 third_party/english/LICENSE）打包出
等价的容器：

  - 每个词只保留**第一个**读音（与上游 `_read_cmu_dict` 一致：
    `if word not in g2p_dict: g2p_dict[word] = [pron]`）
  - 词尾的 `(2)` 序号被剥掉（上游 `re.sub(r'\\(\\d+\\)$', '', word)`）
  - 词统一小写

容器格式（`src/text/english/cmu_dict.cpp` 解析）：

    magic "GENIECMU" | u32 version | u32 word_count | u32 phoneme_count
    | u32 phoneme_name_count
    | 音素名表：每条 u32 长度 + UTF-8 字节
    | 词索引：每条 u32 word_offset / u32 word_length / u32 pron_offset / u32 pron_length
    | 词字节区：每个词后跟一个 NUL 分隔符
    | 音素 ID 数组：每条 u16

词按字节序排序，运行时二分查找。

数据来源（上游 Genie-TTS 的 `English_G2P` 目录，内容同源于 GPT-SoVITS）：

  - `engdict_cache.pickle` —— 上游直接加载的词典缓存
  - `engdict-hot.rep`     —— 热词表（上游 `_load_and_cache_dict` 会覆盖同名词）

也可以只给 `cmudict.rep`（现场读，行为与上游 `_read_cmu_dict` 一致）。

用法：
    python3 tools/deploy/convert_cmu_dict.py \\
        --cache /path/to/engdict_cache.pickle \\
        --hot /path/to/engdict-hot.rep \\
        --output third_party/english/cmudict.bin
"""

import argparse
import pickle
import re
import struct
from pathlib import Path

MAGIC = b"GENIECMU"
VERSION = 1
SUFFIX_RE = re.compile(r"\(\d+\)$")


def u32(v: int) -> bytes:
    return struct.pack("<I", v)


def u16(v: int) -> bytes:
    return struct.pack("<H", v)


def read_cmu_dict(path: Path) -> dict:
    """复刻上游 `_read_cmu_dict`。"""
    g2p: dict = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith(";;;"):
                continue
            parts = re.split(r"\s+", line, maxsplit=1)
            if len(parts) < 2:
                continue
            word = parts[0].lower()
            pron = parts[1].split(" ")
            word = SUFFIX_RE.sub("", word)
            if word not in g2p:
                g2p[word] = pron
    return g2p


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, help="engdict_cache.pickle 路径（上游的词典缓存）")
    parser.add_argument("--cmudict", type=Path, help="cmudict.rep 路径（现场读词典）")
    parser.add_argument("--hot", type=Path, help="engdict-hot.rep 路径（热词表）")
    parser.add_argument("--output", type=Path, required=True, help="输出容器路径")
    args = parser.parse_args()

    if args.cache:
        with args.cache.open("rb") as handle:
            raw = pickle.load(handle)
        # 缓存里的值是 `[[音素...]]`（上游 `_read_cmu_dict` 存的是 `[pron]`），
        # 而上游取的是 `self.cmu[word][0]` —— 也就是第一个读音。
        g2p = {}
        for word, prons in raw.items():
            if not prons:
                continue
            first = prons[0]
            g2p[word] = list(first) if isinstance(first, list) else list(prons)
    elif args.cmudict:
        g2p = read_cmu_dict(args.cmudict)
    else:
        print("需要 --cache 或 --cmudict", file=__import__("sys").stderr)
        return 1

    if args.hot:
        g2p.update(read_cmu_dict(args.hot))

    if not g2p:
        print("cmudict is empty", file=__import__("sys").stderr)
        return 1

    # 音素名表（按首次出现顺序编号）
    phoneme_ids: dict = {}
    phoneme_names: list = []
    pronunciations: list = []
    for word in sorted(g2p):
        ids = []
        for phoneme in g2p[word]:
            if phoneme not in phoneme_ids:
                phoneme_ids[phoneme] = len(phoneme_names)
                phoneme_names.append(phoneme)
            ids.append(phoneme_ids[phoneme])
        pronunciations.append((word, ids))

    # 音素名表改成排序后的稳定编号（容器里固定下来，便于回读校验）
    phoneme_names = sorted(phoneme_names)
    remap = {name: i for i, name in enumerate(phoneme_names)}

    word_blob = bytearray()
    entries = []
    pron_blob = bytearray()
    pron_cursor = 0
    for word, ids in pronunciations:
        raw = word.encode("utf-8")
        entries.append((len(word_blob), len(raw), pron_cursor, len(ids)))
        word_blob += raw
        # 词之间放一个 NUL：否则相邻的词会拼成一条长串，
        # 像 "sl." + 后面 140 个字符正好撞上 GitHub 的
        # "Dropbox 短效访问令牌" 特征 —— secret scanning 会把词典误判成密钥，
        # 直接把 push 挡下来。
        word_blob += b"\x00"
        for phoneme in g2p[word]:
            pron_blob += u16(remap[phoneme])
        pron_cursor += len(ids)

    out = bytearray()
    out += MAGIC
    out += u32(VERSION)
    out += u32(len(entries))
    out += u32(len(phoneme_names))
    out += u32(len(pron_blob) // 2)
    for name in phoneme_names:
        raw = name.encode("utf-8")
        out += u32(len(raw))
        out += raw
    for word_offset, word_length, pron_offset, pron_length in entries:
        out += u32(word_offset)
        out += u32(word_length)
        out += u32(pron_offset)
        out += u32(pron_length)
    out += word_blob
    out += pron_blob

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(bytes(out))
    print(
        f"wrote {args.output} ({len(out)} bytes): {len(entries)} words, "
        f"{len(phoneme_names)} phonemes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
