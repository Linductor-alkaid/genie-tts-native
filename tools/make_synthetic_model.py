#!/usr/bin/env python3
"""make_synthetic_model.py — 生成用于测试的合成 ONNX 模型。

为什么需要它
------------
上游 Genie-TTS 随仓库发布的六个 ONNX 模型是**图壳**：权重在外部 .bin 文件里
（合计约 1.1 GB），由上游的模型转换流程生成，不在仓库中。因此这些模型
无法被 ONNX Runtime 直接加载 —— 只能用来锁定 I/O 契约。

要真正跑通「加载 → 按名字绑定 → 推理 → 解析输出」这条路径，需要一个
**自包含**的模型。这里生成一个刻意设计的模型：

  * 三个输入，**声明顺序**是 alpha / beta / gamma
  * 三个输出，**声明顺序**是 out_i / out_f / out_scalar
  * 元素类型各不相同：int64 / float32 / bool（标量）

测试会**按与声明顺序不同的次序**喂输入与取输出，因此如果实现是按位置绑定
而不是按名字绑定，测试立刻失败 —— 这正是上游 Python 用
`zip(input_names, [...])` 时埋下的那类坑。

用法：
    python3 tools/make_synthetic_model.py <输出路径>
    python3 tools/make_synthetic_model.py --external-fp16 <输出目录>

--external-fp16 额外生成一对文件：echo_fp16.onnx + echo_fp16_data.bin。
图与 echo.onnx 同 I/O 契约，但 out_f = beta * w，其中 w 是 [1024] 的
FLOAT 初始化器、external data 指向 echo_fp16_data.bin —— 文件里是
**fp16 连续打包**的全 1.0（HF 上游 `*_fp16.bin` 的布局：图声明 fp32、
offset/length 为 fp32 字节单位，文件按一半偏移存 fp16）。引擎的 fp16
上转路径（manifest.weight_format = "fp16"）用它做端到端回归：
w 全 1.0 ⇒ out_f == beta，转码错一个字节测试就会失败。
"""

import argparse
import pathlib
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto


def build() -> onnx.ModelProto:
    # ---- 输入 ----
    alpha = helper.make_tensor_value_info("alpha", TensorProto.INT64, ["batch", "n"])
    beta = helper.make_tensor_value_info("beta", TensorProto.FLOAT, ["n", 1024])
    gamma = helper.make_tensor_value_info("gamma", TensorProto.FLOAT, ["batch", 768, "s"])

    # ---- 输出 ----
    out_i = helper.make_tensor_value_info("out_i", TensorProto.INT64, ["batch", "n"])
    out_f = helper.make_tensor_value_info("out_f", TensorProto.FLOAT, ["n", 1024])
    out_scalar = helper.make_tensor_value_info("out_scalar", TensorProto.BOOL, [])

    nodes = [
        # out_i = Identity(alpha)
        helper.make_node("Identity", ["alpha"], ["out_i"], name="id_alpha"),
        # out_f = Identity(beta)
        helper.make_node("Identity", ["beta"], ["out_f"], name="id_beta"),
        # out_scalar = 常量 false（rank-0 bool，对应 stage_decoder 的停止条件）
        helper.make_node(
            "Constant",
            [],
            ["out_scalar"],
            name="stop_const",
            value=numpy_helper.from_array(np.array(False), name="stop_value"),
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "genie_synthetic",
        [alpha, beta, gamma],
        [out_i, out_f, out_scalar],
        doc_string="synthetic model for the native runtime's binding tests",
    )

    model = helper.make_model(
        graph,
        producer_name="genie-tts-native",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def build_external_fp16(out_dir: pathlib.Path) -> None:
    """生成 echo_fp16.onnx + echo_fp16_data.bin（HF fp16 打包布局的回归资产）。

    w 是 [1024] FLOAT 初始化器、external data 指向 echo_fp16_data.bin：
    图里的 offset/length 按 fp32 字节写（0 / 4096），文件按 HF fp16 打包
    存 2048 字节的全 1.0（half 0x3C00 × 1024）。out_f = beta * w，
    w 全 1.0 ⇒ out_f == beta。
    """
    dim = 1024
    alpha = helper.make_tensor_value_info("alpha", TensorProto.INT64, ["batch", "n"])
    beta = helper.make_tensor_value_info("beta", TensorProto.FLOAT, ["n", dim])
    gamma = helper.make_tensor_value_info("gamma", TensorProto.FLOAT, ["batch", 768, "s"])
    out_i = helper.make_tensor_value_info("out_i", TensorProto.INT64, ["batch", "n"])
    out_f = helper.make_tensor_value_info("out_f", TensorProto.FLOAT, ["n", dim])
    out_scalar = helper.make_tensor_value_info("out_scalar", TensorProto.BOOL, [])

    # 外部权重：图声明 fp32（offset=0, length=4096 字节）；文件存 fp16 打包。
    tensor = onnx.TensorProto()
    tensor.name = "w"
    tensor.data_type = TensorProto.FLOAT
    tensor.dims.extend([dim])
    tensor.data_location = onnx.TensorProto.EXTERNAL
    for k, v in (("location", "echo_fp16_data.bin"), ("offset", "0"), ("length", str(dim * 4))):
        entry = tensor.external_data.add()
        entry.key = k
        entry.value = v

    nodes = [
        helper.make_node("Identity", ["alpha"], ["out_i"], name="id_alpha"),
        helper.make_node("Mul", ["beta", "w"], ["out_f"], name="scale_beta"),
        helper.make_node(
            "Constant",
            [],
            ["out_scalar"],
            name="stop_const",
            value=numpy_helper.from_array(np.array(False), name="stop_value"),
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "genie_synthetic_fp16",
        [alpha, beta, gamma],
        [out_i, out_f, out_scalar],
        [tensor],
        doc_string="synthetic model with fp16-packed external weights",
    )
    model = helper.make_model(
        graph,
        producer_name="genie-tts-native",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8
    # 不跑 onnx.checker：fp16 打包（声明 4096 字节 / 文件 2048 字节）故意
    # 不符合 ONNX 规范 —— 这是 HF 上游 `*_fp16.bin` 的真实布局，本引擎的
    # fp16 上转路径就是为它设计的。
    onnx.save(model, str(out_dir / "echo_fp16.onnx"))

    # fp16 打包数据：1024 个 1.0 的 half = 2048 字节全 0x3C00。
    halves = np.full(dim, 0x3C00, dtype="<u2")
    (out_dir / "echo_fp16_data.bin").write_bytes(halves.tobytes())
    print(f"wrote {out_dir / 'echo_fp16.onnx'} + echo_fp16_data.bin "
          f"({dim * 2} 字节 fp16 打包数据)")


def main() -> int:
    # 进度输出带中文；Windows 控制台默认 cp1252，print 直接 UnicodeEncodeError
    # （首个 Windows 发布构建实测 —— 模型已写完，只在最后的 print 上炸）。
    if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_path", nargs="?", help="自包含模型的输出路径")
    parser.add_argument("--external-fp16", type=pathlib.Path,
                        help="输出目录：额外生成 echo_fp16.onnx + echo_fp16_data.bin")
    args = parser.parse_args()
    if args.out_path:
        model = build()
        # 干净 checkout 里 tests/models/ 不存在（.gitignore 排除）—— CI 直接
        # 往里写，父目录要自己建。
        pathlib.Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, args.out_path)
        print(f"wrote {args.out_path} ({len(model.graph.node)} nodes)")
    if args.external_fp16:
        args.external_fp16.mkdir(parents=True, exist_ok=True)
        build_external_fp16(args.external_fp16)
    if not args.out_path and not args.external_fp16:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
