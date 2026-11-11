#!/usr/bin/env python3
# tools/bench/summarize_profile.py —— ORT profiler trace 聚合（M5 性能分析）
#
# 读 genie-bench --profile 落下的 ORT Chrome trace JSON（每个会话/角色一份，
# 每次 Run 追加事件），聚合成算子级耗时表：哪个算子在整段合成里吃掉了
# 多少时间 —— roadmap M5"子模型级性能分析"的算子维度，为热路径优化
# （IOBinding / 算子替换）定预期与验收线。
#
# 用法：
#   python3 tools/bench/summarize_profile.py --trace <trace.json 或目录> [--top 20]
#       [--markdown] [--op-name]
#
# 目录 = genie-bench --profile 的输出目录，glob 全部 *.json 逐文件（角色）
# 出表。OR 的 trace 里节点事件带 args.op_name（CPU EP）——缺了就退回
# 用事件名的前导段。时间单位统一换算成毫秒。

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_events(path):
    text = Path(path).read_text(encoding="utf-8")
    # ORT 的 trace 文件以 "\n" 结尾且是合法 JSON 数组；个别版本会在结尾留
    # 逗号（长 trace 截断）——这里宽容解析，截断的尾巴丢弃不致命。
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        trimmed = text.rstrip().rstrip(",")
        return json.loads(trimmed + "]")


def op_name_of(event):
    args = event.get("args") or {}
    op = args.get("op_name")
    if op:
        return op
    name = event.get("name") or ""
    return name.split("_", 1)[0] or "(unknown)"


def summarize_file(path):
    events = load_events(path)
    if not isinstance(events, list):
        # 目录里可能混着 bench 自己的 --json 输出（对象不是数组）——明确拒绝
        # 而不是迭代出字符串键。
        raise ValueError("不是 ORT trace（顶层不是事件数组）")
    total_us = defaultdict(int)   # op → 累计 kernel 自耗时（µs）
    counts = defaultdict(int)     # op → 调用次数
    for event in events:
        if (event.get("cat") or "").lower() == "node":
            total_us[op_name_of(event)] += int(event.get("dur", 0))
            counts[op_name_of(event)] += 1

    ranked = sorted(total_us.items(), key=lambda kv: kv[1], reverse=True)
    return ranked, counts, len(events)


def print_table(title, ranked, counts, top, markdown):
    if markdown:
        print(f"\n### {title}\n")
        print("| 算子 | 总耗时 (ms) | 调用次数 | 均值 (µs) | 占比 |")
        print("|---|---|---|---|---|")
    else:
        print(f"\n== {title}")
        print(f"{'算子':<28s} {'总耗时(ms)':>10s} {'次数':>8s} {'均值(µs)':>10s} {'占比':>7s}")

    shown = ranked[:top]
    grand = sum(us for _, us in ranked) or 1
    for op, us in shown:
        share = us / grand * 100
        if markdown:
            print(f"| {op} | {us / 1000:.2f} | {counts[op]} | {us / counts[op]:.1f} | {share:.1f}% |")
        else:
            print(f"{op:<28s} {us / 1000:>10.2f} {counts[op]:>8d} {us / counts[op]:>10.1f} {share:>6.1f}%")


def main():
    parser = argparse.ArgumentParser(description="ORT profiler trace 聚合（算子级耗时表）")
    parser.add_argument("--trace", required=True, help="trace JSON 文件或目录（目录则逐文件出表）")
    parser.add_argument("--top", type=int, default=20, help="每个文件显示前 N 个算子（默认 20）")
    parser.add_argument("--markdown", action="store_true", help="输出 markdown 表格（进文档用）")
    args = parser.parse_args()

    root = Path(args.trace)
    if root.is_dir():
        traces = sorted(root.glob("*.json"))
    elif root.is_file():
        traces = [root]
    else:
        sys.exit(f"trace 不存在: {root}")
    if not traces:
        sys.exit(f"目录里没有 trace JSON: {root}")

    for path in traces:
        # 文件名形如 <模型 stem>_<时间戳>_<pid>.json，stem 就是角色名。
        role = path.name.split("_", 1)[0]
        try:
            ranked, counts, event_count = summarize_file(path)
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            print(f"!! 跳过损坏的 trace {path.name}: {exc}", file=sys.stderr)
            continue
        title = f"{role}（{path.name}，{event_count} 事件）"
        print_table(title, ranked, counts, args.top, args.markdown)

    print("")
    return 0


if __name__ == "__main__":
    sys.exit(main())
