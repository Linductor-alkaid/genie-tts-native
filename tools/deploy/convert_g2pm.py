#!/usr/bin/env python3
"""convert_g2pm.py —— 把 pip 包 g2pM 的 numpy 权重与上游词级修正表转成引擎容器。

上游（Genie-TTS）的多音字消歧用 pip 包 g2pM（kakaobrain/g2pM 训练的单层
双向 LSTM，mozillazg 打包发布，Apache-2.0）：权重 np_ckpt.pkl + 词典
char2idx.pkl / class2idx.pkl / digest_cedict.pkl。g2pM 有已知误判，上游用
polyphonic.pickle（HF GenieData/G2P/ChineseG2P/，4.5 万词）做词级读音修正
（CorrectPronunciation.py）—— 本脚本把它一并打进容器（v2）。

转换后**逐字节回读校验**，保证权重与词典零失真。

用法：
    pip download g2pM --no-deps -d /tmp/g2pm && unzip g2pM-*.whl -d /tmp/g2pm
    python3 tools/deploy/convert_g2pm.py --package-dir /tmp/g2pm/g2pM \
        --polyphonic /tmp/polyphonic.pickle --output third_party/g2pm/g2pm.bin
"""

import argparse
import pickle
import struct
from pathlib import Path

MAGIC = b"G2PM"
VERSION = 2  # v2: 附带词级读音修正表（polyphonic.pickle）

# 权重的写出顺序 —— 必须与 src/engine/g2pm.cpp 的切分顺序逐项一致。
WEIGHT_ORDER = [
    "embedding.weight",
    "lstm.weight_ih_l0",
    "lstm.weight_hh_l0",
    "lstm.bias_ih_l0",
    "lstm.bias_hh_l0",
    "lstm.weight_ih_l0_reverse",
    "lstm.weight_hh_l0_reverse",
    "lstm.bias_ih_l0_reverse",
    "lstm.bias_hh_l0_reverse",
    "logit_layer.0.weight",
    "logit_layer.0.bias",
    "logit_layer.2.weight",
    "logit_layer.2.bias",
]


def u32(v: int) -> bytes:
    return struct.pack("<I", v)


def sized(s: str) -> bytes:
    raw = s.encode("utf-8")
    assert len(raw) < 256, f"entry too long: {s!r}"
    return struct.pack("<B", len(raw)) + raw


def idx2class_names(class2idx):
    return [pron for pron, _ in sorted(class2idx.items(), key=lambda kv: kv[1])]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", required=True, help="g2pM pip 包的数据目录")
    parser.add_argument("--polyphonic", required=True,
                        help="上游 polyphonic.pickle（HF GenieData/G2P/ChineseG2P/）")
    parser.add_argument("--output", required=True, help="输出容器路径")
    args = parser.parse_args()

    base = Path(args.package_dir)
    char2idx = pickle.load(open(base / "char2idx.pkl", "rb"))
    class2idx = pickle.load(open(base / "class2idx.pkl", "rb"))
    cedict = pickle.load(open(base / "digest_cedict.pkl", "rb"))
    state = pickle.load(open(base / "np_ckpt.pkl", "rb"))
    poly = pickle.load(open(args.polyphonic, "rb"))
    # 值长度必须与词字数一致（C++ 侧按此对齐，转换期校验拒绝脏数据）。
    bad = [k for k, v in poly.items() if len(v) != len(k)]
    assert not bad, f"polyphonic entries with length mismatch: {bad[:5]}"

    import numpy as np

    out = bytearray()
    out += MAGIC
    out += u32(VERSION)
    out += u32(len(char2idx))
    out += u32(len(class2idx))
    out += u32(len(cedict))

    for name in WEIGHT_ORDER:
        arr = np.ascontiguousarray(state[name], dtype="<f4")
        out += arr.tobytes()

    for ch in sorted(char2idx, key=char2idx.get):
        out += sized(ch) + u32(char2idx[ch])
    # 类 id → 拼音：按下标写出，顺序即 0..N-1。
    for pron in idx2class_names(class2idx):
        out += sized(pron)
    for ch in sorted(cedict):
        prons = cedict[ch]
        out += sized(ch) + u32(len(prons))
        for pron in prons:
            out += sized(pron)

    out += u32(len(poly))
    for word in sorted(poly):
        prons = poly[word]
        out += sized(word) + u32(len(prons))
        for pron in prons:
            out += sized(pron)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_bytes(out)
    print(f"wrote {args.output}: {len(out)} bytes")

    # ---- 回读校验：容器内容必须与源 pkl 逐项一致 ----
    parsed = parse_container(out)
    assert parsed["char2idx"] == char2idx, "char2idx round-trip mismatch"
    assert parsed["idx2class"] == idx2class_names(class2idx), "class round-trip mismatch"
    assert parsed["cedict"] == {k: list(v) for k, v in cedict.items()}, "cedict round-trip mismatch"
    assert parsed["poly"] == {k: list(v) for k, v in poly.items()}, "polyphonic round-trip mismatch"
    for name in WEIGHT_ORDER:
        a = np.frombuffer(parsed["weights"][name], dtype="<f4")
        assert np.array_equal(a, state[name].reshape(-1)), f"{name} round-trip mismatch"
    print("round-trip verified: weights + 3 dicts + polyphonic table exact")
    return 0


def parse_container(data: bytes):
    pos = 0

    def u32():
        nonlocal pos
        v = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        return v

    def sized():
        nonlocal pos
        n = data[pos]
        pos += 1
        s = data[pos:pos + n].decode("utf-8")
        pos += n
        return s

    assert data[:4] == MAGIC
    pos = 4
    assert u32() == VERSION
    vocab, nclass, nced = u32(), u32(), u32()

    counts = {
        "embedding.weight": vocab * 64,
        "lstm.weight_ih_l0": 128 * 64, "lstm.weight_hh_l0": 128 * 32,
        "lstm.bias_ih_l0": 128, "lstm.bias_hh_l0": 128,
        "lstm.weight_ih_l0_reverse": 128 * 64, "lstm.weight_hh_l0_reverse": 128 * 32,
        "lstm.bias_ih_l0_reverse": 128, "lstm.bias_hh_l0_reverse": 128,
        "logit_layer.0.weight": 32 * 64, "logit_layer.0.bias": 32,
        "logit_layer.2.weight": nclass * 32, "logit_layer.2.bias": nclass,
    }
    weights = {}
    for name, n in counts.items():
        weights[name] = data[pos:pos + n * 4]
        pos += n * 4

    char2idx = {sized(): u32() for _ in range(vocab)}
    idx2class = [sized() for _ in range(nclass)]
    cedict = {}
    for _ in range(nced):
        ch = sized()
        n = u32()
        cedict[ch] = [sized() for _ in range(n)]
    npoly = u32()
    poly = {}
    for _ in range(npoly):
        word = sized()
        n = u32()
        poly[word] = [sized() for _ in range(n)]
    assert pos == len(data), "trailing bytes"
    return {"char2idx": char2idx, "idx2class": idx2class, "cedict": cedict,
            "poly": poly, "weights": weights}


if __name__ == "__main__":
    raise SystemExit(main())
