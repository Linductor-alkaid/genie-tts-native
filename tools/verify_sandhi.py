#!/usr/bin/env python3
"""把上游的变调/儿化当 oracle，与本项目的 C++ 移植逐条比对。

上游：
    Genie-TTS 的 G2P/Chinese/ToneSandhi.py 与 G2P/Chinese/Erhua.py。
    ToneSandhi 依赖 jieba_fast —— 这里用纯 Python 的 jieba 顶替
    （两者对 cut/posseg/cut_for_search 的 API 兼容）。

本项目：
    build 出的 tools/g2p/dump_sandhi（分词 → 填韵母 → pre_merge →
    变调 → 儿化），输出每段的处理单元与韵母。

比对方法：**两侧喂同一份逐字韵母**（取自本项目拼音表的读音），
隔离 g2pM 与读音表的差异，只比"分词边界 + 变调/儿化逻辑"。

用法：
    python3 tools/verify_sandhi.py \
        --tone-sandhi /path/to/ToneSandhi.py \
        --erhua /path/to/Erhua.py \
        --dump /path/to/dump_sandhi \
        [--corpus tools/g2p/sandhi_corpus.txt] \
        [--output tests/golden/sandhi.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import tempfile

HAN_RE = re.compile(r"[\u4e00-\u9fa5]+")


def load_upstream(tone_sandhi: pathlib.Path, erhua: pathlib.Path):
    """把上游两个模块加载进来（jieba_fast → jieba 顶替）。"""
    import jieba
    import jieba.posseg

    sys.modules["jieba_fast"] = jieba
    sys.modules["jieba_fast.posseg"] = jieba.posseg

    for path in (tone_sandhi, erhua):
        # sys.path 直接指到上游文件目录，import 文件名。
        if str(path.parent) not in sys.path:
            sys.path.insert(0, str(path.parent))
    import Erhua  # type: ignore
    import ToneSandhi  # type: ignore

    return ToneSandhi.ToneSandhi(), Erhua.ErhuaProcessor()


def han_runs(sentence: str) -> list[str]:
    return HAN_RE.findall(sentence)


def run_dump(binary: str, runs: list[str]) -> list[dict]:
    text = "\n".join(runs) + "\n"
    proc = subprocess.run([binary], input=text, capture_output=True, text=True, check=True)
    return [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]


def flatten_units(units: list[dict]) -> tuple[str, list[str]]:
    """单元列表 → (词面串, 逐字韵母列表)。"""
    word = "".join(u["word"] for u in units)
    finals: list[str] = []
    for u in units:
        finals.extend(u["finals"])
    return word, finals


def upstream_units(tone, erhua, run: str, per_char_finals: list[str]) -> list[dict] | None:
    """上游管线：posseg 分词 → pre_merge → 逐词 modified_tone / merge_erhua。

    韵母取 per_char_finals（与本项目同源），不跑 g2pM / pypinyin。
    词长拼不出逐字韵母时返回 None（跳过该条）。
    """
    seg_cut = list(jieba_posseg.lcut(run))
    seg_cut = tone.pre_merge_for_modify(seg_cut)
    units: list[dict] = []
    pos = 0
    for word, tag in seg_cut:
        finals = per_char_finals[pos : pos + len(word)]
        pos += len(word)
        if len(finals) != len(word):
            return None
        finals = tone.modified_tone(word, tag, list(finals))
        initials = ["" for _ in word]
        initials, finals = erhua.merge_erhua(initials, finals, word, tag)
        units.append({"word": word, "finals": list(finals)})
    return units


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tone-sandhi", required=True, type=pathlib.Path)
    parser.add_argument("--erhua", required=True, type=pathlib.Path)
    parser.add_argument("--dump", required=True, help="dump_sandhi 可执行文件")
    parser.add_argument("--corpus", type=pathlib.Path,
                        default=pathlib.Path(__file__).parent / "g2p" / "sandhi_corpus.txt")
    parser.add_argument("--output", type=pathlib.Path, help="黄金语料输出（可选）")
    parser.add_argument("--max-examples", type=int, default=8)
    args = parser.parse_args()

    global jieba_posseg
    import jieba.posseg as jieba_posseg  # noqa: F401  (供 upstream_units 使用)

    tone, erhua = load_upstream(args.tone_sandhi, args.erhua)

    sentences = [line.strip() for line in args.corpus.read_text(encoding="utf-8").splitlines()
                 if line.strip() and not line.startswith("#")]
    runs = [r for s in sentences for r in han_runs(s)]

    # 本项目侧。
    ours = run_dump(args.dump, runs)
    assert len(ours) == len(runs)

    # 上游侧 + 比对。
    total = same_finals = same_words = skipped = 0
    examples: list[dict] = []
    for run, our in zip(runs, ours):
        # 两侧喂**同一份变调前**的逐字韵母（finals_in）——比的是
        # 分词边界 + 变调/儿化逻辑，不是读音表。
        input_finals = our["finals_in"]
        # 拼音表缺字的段没有比对意义（上游 g2pM 永远有读音）。
        if any(f == "" for f in input_finals):
            skipped += 1
            continue
        their = upstream_units(tone, erhua, run, input_finals)
        if their is None:
            skipped += 1
            continue
        their_word, their_finals = flatten_units(their)
        our_word, our_finals = flatten_units(our["units"])
        total += 1
        finals_ok = their_finals == our_finals
        words_ok = their_word == our_word
        same_finals += finals_ok
        same_words += words_ok
        if not finals_ok and len(examples) < args.max_examples:
            examples.append({
                "text": run,
                "ours": our_finals,
                "upstream": their_finals,
            })

    print(f"段数：{len(runs)}（跳过 {skipped}：拼音表缺字）")
    if total:
        print(f"韵母序列一致：{same_finals}/{total} = {same_finals / total:.1%}")
        print(f"分词边界一致：{same_words}/{total} = {same_words / total:.1%}")
    for ex in examples:
        print(f"  差异 [{ex['text']}]")
        print(f"    本项目: {' '.join(ex['ours'])}")
        print(f"    上游:   {' '.join(ex['upstream'])}")

    if args.output:
        args.output.write_text(
            json.dumps({"cases": ours}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        print(f"黄金语料已写入 {args.output}")


if __name__ == "__main__":
    main()
