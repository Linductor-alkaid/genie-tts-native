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
"""

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


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    out_path = sys.argv[1]
    model = build()
    onnx.save(model, out_path)
    print(f"wrote {out_path} ({len(model.graph.node)} nodes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
