#!/usr/bin/env python3
"""convert_english_g2p.py —— 把 g2p_en 的 seq2seq 权重转成引擎容器。

上游（Genie-TTS）的英文 G2P 在 CMU 词典查不到词时，用一个 GRU seq2seq
猜发音（`G2P/English/EnglishG2P.py` 的 `CleanG2p`）。权重是 g2p_en 的
`checkpoint20.npz`（MIT，见 third_party/english/LICENSE-g2p_en）。

.npz 是 zip 容器，C++ 侧解压很麻烦 —— 这里把 12 个数组原样转成裸
float32，顺序固定：

    enc_emb enc_w_ih enc_w_hh enc_b_ih enc_b_hh
    dec_emb dec_w_ih dec_w_hh dec_b_ih dec_b_hh
    fc_w fc_b

容器格式（`src/text/english/seq2seq.cpp` 解析）：

    magic "GENIEG2P" | u32 version | u32 phoneme_count
    | 音素名表：每条 u32 长度 + UTF-8 字节
    | 12 个数组：每条 u32 rows | u32 cols | rows*cols 个 f32

用法：
    python3 tools/deploy/convert_english_g2p.py \\
        --checkpoint /tmp/g2p_en/checkpoint20.npz \\
        --output third_party/english/english_g2p.bin
"""

import argparse
import struct
from pathlib import Path

import numpy as np

MAGIC = b"GENIEG2P"
VERSION = 1

# 与上游 `CleanG2p._setup_nn_components` 的音素表顺序一致
PHONEMES = [
    "<pad>", "<unk>", "<s>", "</s>", "AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0",
    "AH1", "AH2", "AO0", "AO1", "AO2", "AW0", "AW1", "AW2", "AY0", "AY1", "AY2", "B",
    "CH", "D", "DH", "EH0", "EH1", "EH2", "ER0", "ER1", "ER2", "EY0", "EY1", "EY2",
    "F", "G", "HH", "IH0", "IH1", "IH2", "IY0", "IY1", "IY2", "JH", "K", "L", "M",
    "N", "NG", "OW0", "OW1", "OW2", "OY0", "OY1", "OY2", "P", "R", "S", "SH", "T",
    "TH", "UH0", "UH1", "UH2", "UW", "UW0", "UW1", "UW2", "V", "W", "Y", "Z", "ZH",
]

ARRAYS = [
    "enc_emb", "enc_w_ih", "enc_w_hh", "enc_b_ih", "enc_b_hh",
    "dec_emb", "dec_w_ih", "dec_w_hh", "dec_b_ih", "dec_b_hh",
    "fc_w", "fc_b",
]


def u32(v: int) -> bytes:
    return struct.pack("<I", v)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="checkpoint20.npz 路径")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    weights = np.load(args.checkpoint)
    out = bytearray()
    out += MAGIC
    out += u32(VERSION)
    out += u32(len(PHONEMES))
    for name in PHONEMES:
        raw = name.encode("utf-8")
        out += u32(len(raw))
        out += raw
    for name in ARRAYS:
        array = np.ascontiguousarray(weights[name], dtype=np.float32)
        if array.ndim == 1:
            rows, cols = 1, array.shape[0]
        else:
            rows, cols = array.shape
        out += u32(rows)
        out += u32(cols)
        out += array.astype("<f4").tobytes()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(bytes(out))
    print(f"wrote {args.output} ({len(out)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
