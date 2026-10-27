#!/usr/bin/env python3
"""中文文本规范化的**参考实现**与黄金语料生成器。

上游 Genie-TTS 的 `G2P/Chinese/Normalization` 是一个自包含的 Python 包
（只依赖 `pypinyin.compat.SUPPORT_UCS4` 这一个常量）。本脚本把它当**oracle**：

    python3 tools/verify_normalizer.py --source <上游 Normalization 目录> \\
        --output tests/golden/normalizer.json

生成黄金语料：每个用例的输入 → 上游的规范化结果。

    python3 tools/verify_normalizer.py --source ... --output ... --check

`--check` 模式会把黄金语料与**当前 C++ 实现**的输出比对
（需要先构建出 `genie-tts` CLI，用 `genie-tts normalize` 逐条跑）。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys


def install_pypinyin_stub(target: pathlib.Path) -> None:
    """上游的 constants.py 只用到 pypinyin.compat.SUPPORT_UCS4。

    造一个最小的 stub，避免为了一个常量去装整个 pypinyin。
    """
    pkg = target / "pypinyin"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    compat = pkg / "compat"
    compat.mkdir(parents=True, exist_ok=True)
    (compat / "__init__.py").write_text("SUPPORT_UCS4 = True\n", encoding="utf-8")


def build_oracle(source: pathlib.Path, workdir: pathlib.Path):
    """把上游 Normalization 拷到工作目录，返回可调用的 TextNormalizer。"""
    import shutil

    workdir.mkdir(parents=True, exist_ok=True)
    install_pypinyin_stub(workdir)
    target = workdir / "Normalization"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    # 包内的相对 import 需要 __init__.py
    init = target / "__init__.py"
    if not init.exists():
        init.write_text("", encoding="utf-8")

    sys.path.insert(0, str(workdir))
    from Normalization.text_normlization import TextNormalizer  # type: ignore

    return TextNormalizer()


# ---------------------------------------------------------------------------
# 上游 ChineseG2P 的标点处理
# ---------------------------------------------------------------------------

PUNCTUATION_REPLACEMENTS = {
    "：": ",", "；": ",", "，": ",", "。": ".", "！": "!",
    "？": "?", "\n": ".", "·": ",", "、": ",", "$": ".",
    "/": ",", "—": "-", "~": "…", "～": "…",
}
SPECIAL_REPLACEMENTS = {"...": "…"}

ALLOWED = "!?…,.-"
PUNCTUATION = ["!", "?", "…", ",", ".", "-"]


def replace_punctuation(text: str) -> str:
    """上游 ChineseG2P._replace_punctuation。"""
    for k, v in SPECIAL_REPLACEMENTS.items():
        text = text.replace(k, v)
    for k, v in PUNCTUATION_REPLACEMENTS.items():
        text = text.replace(k, v)
    # 上游用正则一次性替换；逐个 replace 等价（替换结果不再含原字符）
    text = re.sub(r"[^\u4e00-\u9fa5" + re.escape(ALLOWED) + r"]+", "", text)
    return text


def collapse(text: str) -> str:
    """上游 pattern_consecutive。"""
    return re.sub(r"([" + re.escape(ALLOWED) + r"])\1+", r"\1", text)


def upstream_normalize(normalizer, text: str) -> str:
    """上游 ChineseG2P.normalize_text 的完整流程。"""
    sentences = normalizer.normalize(text)
    parts = [replace_punctuation(s) for s in sentences]
    return collapse("".join(parts))


# ---------------------------------------------------------------------------
# 语料
# ---------------------------------------------------------------------------

CASES: list[tuple[str, str]] = [
    # 基础
    ("plain", "你好，世界。"),
    ("simple", "今天天气不错。"),
    ("no_punct", "没有标点的文本"),
    ("english", "hello world"),
    ("mixed", "我有12个苹果和3.5个橘子"),
    # 数字
    ("int_small", "我有3个苹果"),
    ("int_two_digit", "一共24个人"),
    ("int_three_digit", "编号00078"),
    ("year", "2024年"),
    ("year_short", "24年"),
    ("big_number", "1234567"),
    ("huge_number", "123456789012"),
    ("decimal", "3.14的圆周率"),
    ("decimal_trailing_zero", "3.20元"),
    ("decimal_leading_dot", ".22秒"),
    ("negative", "温度是-5度"),
    ("percent", "增长了50%"),
    ("percent_decimal", "增长了12.5%"),
    ("frac", "3/4的圆"),
    ("frac_negative", "-3/4的圆"),
    ("range", "3-5个"),
    ("range_decimal", "3.5-5.5个"),
    ("to_range", "20~30cm"),
    ("power", "x²"),
    ("version", "版本1.2.3"),
    ("asmd", "1+2=3"),
    ("asmd_chain", "1+2+3=6"),
    # 日期时间
    ("date_full", "2024年3月15日"),
    ("date_month", "2024年3月"),
    ("date_year_only", "2024年"),
    ("date_slash", "2024/03/15"),
    ("date_dash", "2024-03-15"),
    ("time", "下午3点30分"),
    ("time_full", "15:30:45"),
    ("time_range", "8:30-12:30"),
    ("time_half", "8:30"),
    # 电话
    ("mobile", "13812345678"),
    ("mobile_86", "+8613812345678"),
    ("telephone", "010-12345678"),
    ("national_400", "400-123-4567"),
    # 单位与量词
    ("measure_cm", "长5cm"),
    ("measure_km", "距离3km"),
    ("measure_m2", "面积10m2"),
    ("temperature", "25度"),
    ("temperature_c", "25°C"),
    ("temperature_negative", "-3摄氏度"),
    ("quantifier", "3个人"),
    ("quantifier_multi", "5个苹果和2本书"),
    # 繁简
    ("traditional", "這個時間點我們需要重新評估"),
    ("traditional_mixed", "繁體中文和简体中文混排"),
    # 特殊字符
    ("circled", "①②③"),
    ("greek", "α和β"),
    ("math", "1+1"),
    ("slash", "每/天"),
    ("ellipsis", "等等..."),
    ("wave", "a~b"),
    # 空白与标点
    ("spaces", "有 空格 的 文本"),
    ("ideographic_space", "有　全角空格的文本"),
    ("fullwidth_digits", "２０２４年"),
    ("fullwidth_letters", "ＡＢＣ"),
    ("repeated_punct", "你好！！！世界？？？"),
    ("quotes", "他说：“你好。”然后走了。"),
    ("english_mix", "这是 test 123 测试"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="上游 Normalization 目录")
    parser.add_argument("--output", required=True, help="黄金语料输出路径")
    parser.add_argument("--workdir", default="/tmp/genie_normalizer_oracle")
    parser.add_argument("--check", action="store_true",
                        help="与 C++ 实现的输出比对（需要 --binary）")
    parser.add_argument("--binary", help="genie-tts 可执行文件路径")
    args = parser.parse_args()

    normalizer = build_oracle(
        pathlib.Path(args.source).resolve(), pathlib.Path(args.workdir).resolve())

    results = []
    for name, text in CASES:
        expected = upstream_normalize(normalizer, text)
        results.append({"name": name, "input": text, "expected": expected})

    payload = {
        "schema": 1,
        "source": "Genie-TTS G2P/Chinese/Normalization (PaddlePaddle / WeTextProcessing)",
        "cases": results,
    }
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output} ({len(results)} cases)")

    if args.check:
        if not args.binary:
            print("--check 需要 --binary", file=sys.stderr)
            return 2
        return check(args.binary, results)
    return 0


def check(binary: str, results: list[dict]) -> int:
    failures = 0
    for case in results:
        # 强制中文：上游的 TextNormalizer 是**中文专用**的，
        # 自动语言识别会把纯数字 / 纯英文的用例判成别的语言。
        proc = subprocess.run(
            [binary, "normalize", "--language", "chinese", "--text", case["input"]],
            capture_output=True, text=True, check=False)
        actual = proc.stdout.strip()
        if actual != case["expected"]:
            failures += 1
            print(f"FAIL {case['name']}")
            print(f"  input    : {case['input']}")
            print(f"  expected : {case['expected']}")
            print(f"  actual   : {actual}")
    if failures:
        print(f"\n{failures}/{len(results)} 个用例不一致")
        return 1
    print(f"\n全部 {len(results)} 个用例与上游一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
