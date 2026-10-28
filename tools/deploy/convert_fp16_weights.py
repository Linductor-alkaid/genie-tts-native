#!/usr/bin/env python3
"""把上游发布的 fp16 外部权重转换成 fp32。

背景：HF 仓库 High-Logic/Genie 里，t2s_shared / vits / prompt_encoder /
chinese-hubert-base 的外部权重文件是 **fp16 原始数组**（`*_fp16.bin`），
而随附的 `.onnx` 图把张量声明为 **fp32 + external data**。直接加载会
把 fp16 字节当 fp32 读，输出乱码。

转换 = 整文件按 `<f2` 读入、cast 成 `<f4` 写出（大小翻倍）。整文件
转换成立的前提是图里所有外部张量的存储区间**连续且无缝**（offset 从
0 开始、按序排满整个文件）—— 本项目在审计时对随附模型逐个核实过；
`--verify` 会在转换前后再自动核一遍。

用法：
    python3 tools/deploy/convert_fp16_weights.py \
        --weights path/to/vits_fp16.bin \
        --model   path/to/vits_fp32.onnx \
        --output  path/to/vits_fp32.bin

    # 只转换、不做图校验（没有 onnx 包时）：
    python3 tools/deploy/convert_fp16_weights.py \
        --weights path/to/t2s_shared_fp16.bin --output path/to/t2s_shared_fp32.bin
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np


def verify_against_graph(model_path: pathlib.Path, fp32_size: int) -> None:
    """核对 .onnx 图里外部数据的区间：连续、无缝、总长等于 fp32 文件大小。

    需要 `onnx` 包（`pip install onnx`）；没有就跳过并给出提示 ——
    转换本身只依赖 numpy。
    """
    try:
        import onnx
    except ImportError:
        print("  [提示] 未安装 onnx 包，跳过图校验（pip install onnx 可启用）")
        return
    model = onnx.load(str(model_path), load_external_data=False)
    spans: list[tuple[int, int, str]] = []
    for tensor in model.graph.initializer:
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        info = {e.key: e.value for e in tensor.external_data}
        offset = int(info.get("offset", "0"))
        length = int(info["length"])  # external_data 的 length 单位是**字节**
        spans.append((offset, offset + length, tensor.name))
    if not spans:
        raise SystemExit(f"{model_path}: 图里没有外部权重，无需转换")
    spans.sort()
    pos = 0
    for begin, end, name in spans:
        if begin != pos:
            raise SystemExit(
                f"{model_path}: 外部权重区间不连续（{name} 期望 offset={pos}，实际 {begin}）"
                " —— 整文件转换不安全，需要逐张量转换")
        pos = end
    if pos != fp32_size:
        raise SystemExit(f"{model_path}: 图期望外部权重 {pos} 字节（fp32），"
                         f"转换产物 {fp32_size} 字节，不一致")
    print(f"  图校验通过：{len(spans)} 个外部张量，区间连续，总长 {pos} 字节")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, type=pathlib.Path,
                        help="上游发布的 fp16 外部权重（*_fp16.bin）")
    parser.add_argument("--model", type=pathlib.Path,
                        help="对应的 .onnx 图（可选；提供时做外部区间校验）")
    parser.add_argument("--output", required=True, type=pathlib.Path,
                        help="输出的 fp32 权重路径（一般命名 *_fp32.bin）")
    args = parser.parse_args()

    raw = np.fromfile(args.weights, dtype="<f2")
    if raw.size == 0:
        raise SystemExit(f"{args.weights}: 空文件")
    out = raw.astype("<f4")
    if args.model:
        verify_against_graph(args.model, out.nbytes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.tofile(args.output)
    print(f"{args.weights.name}（{raw.nbytes} 字节 fp16）→ "
          f"{args.output}（{out.nbytes} 字节 fp32，{raw.size} 个值）")


if __name__ == "__main__":
    main()
