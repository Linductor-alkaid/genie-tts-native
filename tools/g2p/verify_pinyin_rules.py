#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""校验 C++ 的拼音切分与音素推导规则。

分两层比对：

1. **切分**：C++ 的 `pinyin::split()` vs pypinyin 的
   `to_initials()` / `to_finals_tone3()`。
2. **音素**：C++ 的 `pinyin::syllable_to_phones()` vs 本文件里按上游
   `ChineseG2P._pinyin_to_opencpop_phones()` 复刻的参考实现。

第 2 层的参考实现需要 opencpop 词表（上游从 HuggingFace 资源包下载，
沙箱里拿不到）。这里按 opencpop 的划分规则重建：
声母 + 收缩后的韵母；零声母的字按 y/w/元音三分。

用法::

    g++ -std=c++20 -Iinclude tools/g2p/dump_pinyin.cpp \\
        src/text/chinese_g2p.cpp src/text/pinyin_table.cpp -o /tmp/dump_pinyin
    python3 tools/g2p/verify_pinyin_rules.py /tmp/dump_pinyin

退出码非 0 表示有不一致。
"""

import json
import os
import subprocess
import sys

# 上游 ChineseG2P.py 里的三张替换表
V_REP_MAP = {"uei": "ui", "iou": "iu", "uen": "un"}
PINYIN_REP_MAP = {"ing": "ying", "i": "yi", "in": "yin", "u": "wu"}
SINGLE_REP_MAP = {"v": "yu", "e": "e", "i": "y", "u": "w"}

# opencpop 的声母表（顺序即前缀匹配顺序）
OPENCPOP_INITIALS = [
    "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
    "j", "q", "x", "zh", "ch", "sh", "r", "z", "c", "s",
]


def opencpop_split(key):
    """opencpop 词表里一个音节的 (声母, 韵母) 划分。"""
    if key.startswith("y"):
        return "y", key[1:]
    if key.startswith("w"):
        return "w", key[1:]
    for initial in OPENCPOP_INITIALS:
        if key.startswith(initial):
            return initial, key[len(initial):]
    # 零声母：元音本身当声母，韵母是整个音节（opencpop 词表的写法）
    if key.startswith("er"):
        return "er", "er"
    return key[0], key


def reference_phones(initial, final_with_tone):
    """复刻上游 ChineseG2P._pinyin_to_opencpop_phones()。"""
    if initial == final_with_tone:
        # 标点：上游直接把它当音素
        return [initial] if initial else []
    if not final_with_tone:
        return []

    without_tone = final_with_tone[:-1]
    tone = final_with_tone[-1]

    if initial:
        final = V_REP_MAP.get(without_tone, without_tone)
        key = initial + final
    else:
        temp = without_tone
        if temp in PINYIN_REP_MAP:
            key = PINYIN_REP_MAP[temp]
        elif temp and temp[0] in SINGLE_REP_MAP:
            key = SINGLE_REP_MAP[temp[0]] + temp[1:]
        else:
            key = temp

    new_c, new_v = opencpop_split(key)
    return [new_c, new_v + tone]


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    dump = sys.argv[1]

    try:
        import pypinyin
        from pypinyin import Style, lazy_pinyin
        from pypinyin.contrib.tone_convert import to_finals_tone3, to_initials
    except ImportError:
        print("缺少 pypinyin（pip install pypinyin）", file=sys.stderr)
        return 2

    base = os.path.dirname(pypinyin.__file__)
    with open(os.path.join(base, "pinyin_dict.json"), encoding="utf-8") as handle:
        table = json.load(handle)

    # 1) 生成输入与期望值
    payload = []
    expected_split = {}
    expected_phones = {}
    for key in table:
        ch = chr(int(key))
        pinyin = lazy_pinyin(
            ch, style=Style.TONE3, strict=True, neutral_tone_with_five=True
        )[0]
        if not pinyin:
            continue
        payload.append(f"{int(key):x} {pinyin}")
        initial = to_initials(pinyin)
        final = to_finals_tone3(pinyin, neutral_tone_with_five=True)
        expected_split[int(key)] = (initial, final)
        expected_phones[int(key)] = reference_phones(initial, final)

    # 2) 跑 C++ 实现
    result = subprocess.run(
        [dump], input="\n".join(payload) + "\n", capture_output=True, text=True
    )
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        return 1

    # 3) 比对
    split_mismatch = []
    phone_mismatch = []
    nasal_only = []
    seen = 0
    for line in result.stdout.splitlines():
        parts = (line.split("\t") + ["", "", "", ""])[:4]
        cp = int(parts[0])
        seen += 1
        actual_split = (parts[1], parts[2])
        actual_phones = parts[3].split(",") if parts[3] else []

        if actual_split != expected_split[cp]:
            # 纯鼻音字（嗯/呣/噷…）：没有韵母，音素一定为空。
            # 上游用带调拼音，to_initials 得到空；本实现用 tone3，得到声母。
            # 两者最终都不产生音素，所以只记为「鼻音特例」不算不一致。
            if not expected_split[cp][1] and not actual_split[1]:
                nasal_only.append(chr(cp))
            else:
                split_mismatch.append((chr(cp), expected_split[cp], actual_split))

        if actual_phones != expected_phones[cp]:
            phone_mismatch.append(
                (chr(cp), expected_phones[cp], actual_phones, expected_split[cp])
            )

    print(f"比对字符数: {seen} / {len(expected_split)}")
    print(f"纯鼻音特例（无韵母，不产生音素）: {len(nasal_only)} 个")
    if split_mismatch:
        print(f"切分不一致: {len(split_mismatch)} 处")
        for item in split_mismatch[:20]:
            print(f"  {item[0]}  期望 {item[1]}  实际 {item[2]}")
    if phone_mismatch:
        print(f"音素不一致: {len(phone_mismatch)} 处")
        for item in phone_mismatch[:20]:
            print(f"  {item[0]}  期望 {item[1]}  实际 {item[2]}")
    if split_mismatch or phone_mismatch:
        return 1
    print("切分与音素全部一致 ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
