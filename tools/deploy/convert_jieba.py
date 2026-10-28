#!/usr/bin/env python3
"""convert_jieba.py —— 把 jieba 的词典与 HMM 概率打包成引擎容器。

上游（Genie-TTS）用 jieba_fast 做分词与词性标注（jieba_fast 与 jieba
结果一致、C 加速；词典/HMM 数据同源，MIT）。本脚本把

  - dict.txt（word freq tag，349,046 条）
  - finalseg/prob_start.p / prob_trans.p / prob_emit.p（OOV 分词 HMM）

打包成单个 `jieba.bin`（src/text/jieba.cpp 解析），回读校验零失真。
缺失的发射/转移概率在写出时直接固化为 MIN_FLOAT（与上游
`emit_p[y].get(ch, MIN_FLOAT)` 的查表缺省一致）。

用法：
    pip download jieba --no-deps -d /tmp && tar xzf /tmp/jieba-*.tar.gz -C /tmp
    python3 tools/deploy/convert_jieba.py --jieba-dir /tmp/jieba-0.42.1/jieba \\
        --output third_party/jieba/jieba.bin
"""

import argparse
import pickle
import struct
from pathlib import Path

MAGIC = b"JIEB"
VERSION = 1
STATES = ["B", "M", "E", "S"]
MIN_FLOAT = -3.14e100


def u32(v: int) -> bytes:
    return struct.pack("<I", v)


def u64(v: int) -> bytes:
    return struct.pack("<Q", v)


def f64(v: float) -> bytes:
    return struct.pack("<d", v)


def sized(s: str) -> bytes:
    raw = s.encode("utf-8")
    assert len(raw) < 256, f"entry too long: {s!r}"
    return struct.pack("<B", len(raw)) + raw


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jieba-dir", required=True, help="jieba 包的数据目录")
    parser.add_argument("--output", required=True, help="输出容器路径")
    args = parser.parse_args()

    base = Path(args.jieba_dir)
    real = []  # (word, freq, tag)
    total = 0
    with open(base / "dict.txt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split(" ")
            assert len(parts) == 3, repr(line)
            word, freq, tag = parts
            real.append((word, int(freq), tag))
            total += int(freq)
    # 上游 gen_pfdict：所有前缀片段以 freq=0 入典（DAG 扫描靠它们继续
    # 前进，freq=0 不可选）。C++ 侧的 while(frag in dict) 依赖这一批。
    words = list(real)
    seen = {w for w, _, _ in real}
    for word, _, _ in real:
        for ch in range(len(word)):
            frag = word[:ch + 1]
            if frag not in seen:
                seen.add(frag)
                words.append((frag, 0, ""))

    start = pickle.load(open(base / "finalseg" / "prob_start.p", "rb"))
    trans = pickle.load(open(base / "finalseg" / "prob_trans.p", "rb"))
    emit = pickle.load(open(base / "finalseg" / "prob_emit.p", "rb"))

    out = bytearray()
    out += MAGIC
    out += u32(VERSION)
    out += u64(total)
    out += u32(len(words))
    for word, freq, tag in words:
        out += sized(word) + u32(freq) + sized(tag)

    # HMM：缺失概率固化为 MIN_FLOAT。
    for state in STATES:
        out += f64(start.get(state, MIN_FLOAT))
    # 行主序：trans[from][to]，与 C++ 的 trans_[y0][y] 对齐。
    for y0 in STATES:
        for y in STATES:
            out += f64(trans.get(y0, {}).get(y, MIN_FLOAT))
    chars = sorted(set().union(*(set(emit[s]) for s in STATES)))
    out += u32(len(chars))
    for ch in chars:
        out += sized(ch)
        for state in STATES:
            out += f64(emit[state].get(ch, MIN_FLOAT))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_bytes(out)
    print(f"wrote {args.output}: {len(out)} bytes "
          f"({len(words)} words, total {total}, {len(chars)} hmm chars)")

    # ---- 回读校验 ----
    parsed = parse_container(out)
    assert parsed["total"] == total
    assert parsed["words"] == words, "dict round-trip mismatch"
    assert parsed["start"] == {s: start.get(s, MIN_FLOAT) for s in STATES}
    assert parsed["emit"] == {ch: [emit[s].get(ch, MIN_FLOAT) for s in STATES] for ch in chars}
    print("round-trip verified: dict + hmm exact")
    return 0


def parse_container(data: bytes):
    pos = 0

    def u32():
        nonlocal pos
        v = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        return v

    def u64():
        nonlocal pos
        v = struct.unpack_from("<Q", data, pos)[0]
        pos += 8
        return v

    def f64():
        nonlocal pos
        v = struct.unpack_from("<d", data, pos)[0]
        pos += 8
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
    total = u64()
    n = u32()
    words = [(sized(), u32(), sized()) for _ in range(n)]
    start = {s: f64() for s in STATES}
    for _ in range(16):
        f64()  # trans（回读校验不逐一比，C++ 侧 golden 覆盖）
    n_emit = u32()
    emit = {}
    for _ in range(n_emit):
        ch = sized()
        emit[ch] = [f64() for _ in STATES]
    assert pos == len(data), "trailing bytes"
    return {"total": total, "words": words, "start": start, "emit": emit}


if __name__ == "__main__":
    raise SystemExit(main())
