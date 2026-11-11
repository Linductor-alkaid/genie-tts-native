#!/usr/bin/env python3
# tools/deploy/quantize_package.py —— INT8 混合精度音色包生成（M5 量化阶梯）
#
# 读一个现有音色包，把指定角色的 ONNX 模型转成 int8 权重，产出一个**混合精度**新包：
#   - 被量化的角色：新 .onnx（权重 int8）+ 紧凑外部 bin
#   - 其余角色：原样拷贝（fp32），manifest 用逐模型 weight_format 覆盖
#
# 两档量化（--static 切换）：
#   - 动态 INT8（默认）：quantize_dynamic，只量化 MatMul/Gemm 的常量 B 侧，
#     不需要校准集。第一档（2026-11-11）：roberta / hubert / prompt_encoder
#     的收益全部来自这里；但 Conv 系（vits / t2s_encoder / SV）收益为零。
#   - 静态 INT8：quantize_static + QDQ，图内插 QuantizeLinear / DequantizeLinear，
#     **Conv 从此可量化** —— 校准集用 genie-gate 捕获的真实流水线输入
#     （--calibration-dir，布局见 tools/gate/capture.hpp）。
#
# 为什么 int8 而不是 fp16 常驻：本机 CPU 有 AVX_VNNI，MLAS 的 u8s8 内核有
# 真实加速；fp16 常驻在无 AVX512-FP16 的机器上会被 ORT 逐算子 cast，速度倒退。
#
# 内存账：权重常驻 4 字节/元素 → 1 字节/元素。vits / SV 这类听感敏感模型
# 是否量化交给精度门控（tools/gate/）裁决，本工具只负责"能产什么"。
#
# 用法（动态档）：
#   python3 tools/deploy/quantize_package.py \
#       --src /path/voice-package-multilang \
#       --dst /path/voice-package-int8 \
#       --roles roberta,chinese_hubert \
#       [--ops MatMul,Gemm] [--workers 4]
#
# 用法（静态档）：先 genie-gate capture 落一份捕获目录，再
#   python3 tools/deploy/quantize_package.py --static \
#       --src /path/voice-package-multilang \
#       --dst /path/voice-package-static \
#       --calibration-dir /path/capture \
#       --roles vits,t2s_encoder \
#       [--ops Conv,MatMul,Gemm] [--calibration-method minmax]
#
# 量化完成后，用精度门控验证（见 docs/regression.md 精度门控一节）：
#   genie-gate capture --model <fp32 包> ...
#   python3 tools/gate/precision_gate.py gate --capture <capture目录> --package <int8包>

import argparse
import hashlib
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    sys.exit(f"需要 numpy（静态档校准集重建）：pip install numpy\n{exc}")

try:
    import onnxruntime.quantization as ort_quant
except ImportError as exc:  # pragma: no cover
    sys.exit(f"需要 onnxruntime（量化工具链）：pip install onnxruntime\n{exc}")

try:
    import onnx
    from onnx.external_data_helper import convert_model_to_external_data
except ImportError as exc:  # pragma: no cover
    sys.exit(f"需要 onnx（外部数据重整）：pip install onnx\n{exc}")

# 上游 HF dump 的文件名与角色的对应关系走 manifest，这里只列默认建议：
# 听感敏感（vits / speaker_verification）与重采样敏感（chinese_hubert）不进
# 默认清单 —— 由实验门控决定，门控过了再加。
DEFAULT_ROLES = ["roberta", "prompt_encoder"]
DEFAULT_OPS_DYNAMIC = "MatMul,Gemm"
# 静态档默认带上 Conv：这正是第二档存在的意义 —— 动态档对 Conv 系收益为零。
# **显式列出而不依赖量化器默认集**：默认集若含 ConvTranspose，CPU EP 没有
# int8 内核，会话初始化直接 NOT_IMPLEMENTED；显式列表把风险挡在量化期。
DEFAULT_OPS_STATIC = "Conv,MatMul,Gemm"

# 捕获张量的 dtype 后缀 → numpy（与 tools/gate/precision_gate.py 的映射一致）
CAPTURE_DTYPES = {"float32": np.float32, "int64": np.int64, "bool": np.bool_}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CaptureCalibrationReader(ort_quant.CalibrationDataReader):
    """从 genie-gate 捕获目录构建静态量化校准集。

    捕获目录里存的是真实流水线逐模型收到的输入（裸二进制 + capture.json
    元数据）。只取 full call —— partial 步（stage_decoder 的后续 token 步）
    喂不出完整 KV cache，校准推理会直接失败。逐 call 惰性重建 numpy：
    t2s 解码器一步的输入 ~17 MB，不必整集驻留。
    """

    def __init__(self, capture_dir, role):
        meta = json.loads((Path(capture_dir) / "capture.json").read_text(encoding="utf-8"))
        self._root = Path(capture_dir)
        self._calls = [
            c for c in meta.get("calls", [])
            if c.get("role") == role and c.get("inputs") and not c.get("partial", False)
        ]
        self._index = 0
        if not self._calls:
            raise ValueError(
                f"捕获目录里没有 {role} 的全量调用（全是 partial 或没有输入）：{capture_dir}"
                " —— 用 genie-gate capture 补采后再量化该角色")

    def get_next(self):
        if self._index >= len(self._calls):
            return None
        feed = {}
        for tensor in self._calls[self._index]["inputs"]:
            data = np.fromfile(self._root / tensor["file"], dtype=CAPTURE_DTYPES[tensor["dtype"]])
            feed[tensor["name"]] = data.reshape(tensor["shape"])
        self._index += 1
        return feed


def _externalize_quantized(dst_onnx):
    """量化产物外置收尾。成功返回 None，失败返回错误串。

    量化产物默认把 int8 权重**内联**进 .onnx（t2s_stage 会变成 288 MB 的
    大 protobuf），而旧的 fp32 外部 bin 又因少量未量化张量（LayerNorm /
    bias）仍被引用而整份留在包里 —— 包体积不降反升。这里把全部初始化器
    重新外置到**新的紧凑 int8 bin**：protobuf 回到壳大小，磁盘读走顺序
    大块，旧 bin 在引用清点后删除。
    """
    try:
        stem = Path(dst_onnx).stem
        weights_name = f"{stem}_int8.bin"
        model = onnx.load(str(dst_onnx))  # 外部引用在镜像目录里解析
        convert_model_to_external_data(
            model, all_tensors_to_one_file=True, location=weights_name, size_threshold=1024)
        onnx.save(model, str(dst_onnx))
    except Exception as exc:  # noqa: BLE001
        return f"externalize failed: {type(exc).__name__}: {exc}"
    return None


def quantize_one(src_onnx, dst_onnx, op_types):
    """动态量化单个模型。返回 (成功, 信息)。

    staging 原则：quantize_dynamic 会 onnx.load 整个模型（含外部数据），
    保存时**未量化**的初始化器原样带回（外部引用路径不变）——所以必须在
    "整包镜像"目录里原地转换，否则外部 .bin 的相对路径会断。静态档同理。
    """
    try:
        ort_quant.quantize_dynamic(
            model_input=str(src_onnx),
            model_output=str(dst_onnx),
            op_types_to_quantize=list(op_types),
            per_channel=True,  # 每通道 scale：transformer 权重列分布差异大，per-tensor 掉点明显
            weight_type=ort_quant.QuantType.QInt8,
            extra_options={"MatMulConstBOnly": True},  # 只量化常量 B 侧：transformer 的 QK^V 动态乘不动
        )
    except Exception as exc:  # noqa: BLE001 —— 单模型失败不应中止整包，报告里给结论
        return False, f"{type(exc).__name__}: {exc}"
    error = _externalize_quantized(dst_onnx)
    if error:
        return False, error
    return True, "ok"


def quantize_one_static(role, src_onnx, dst_onnx, op_types, capture_dir, calibrate_method):
    """静态量化单个模型（QDQ：图内插 QuantizeLinear / DequantizeLinear）。

    u8s8（激活 QUInt8 / 权重 QInt8）+ per-channel，对齐动态档的内核选择：
    AVX_VNNI 的 MLAS u8s8 快路径在静态档同样生效。区别在激活 scale 来自
    校准集（捕获的真实输入分布），运行期不再逐次动态求 scale。
    """
    try:
        ort_quant.quantize_static(
            model_input=str(src_onnx),
            model_output=str(dst_onnx),
            calibration_data_reader=CaptureCalibrationReader(capture_dir, role),
            quant_format=ort_quant.QuantFormat.QDQ,
            op_types_to_quantize=list(op_types),
            per_channel=True,
            activation_type=ort_quant.QuantType.QUInt8,
            weight_type=ort_quant.QuantType.QInt8,
            calibrate_method=calibrate_method,
        )
    except Exception as exc:  # noqa: BLE001 —— 单模型失败不应中止整包，报告里给结论
        return False, f"{type(exc).__name__}: {exc}"
    error = _externalize_quantized(dst_onnx)
    if error:
        return False, error
    return True, "ok"


def dedupe_identical_bins(dst, manifest):
    """量化后内容相同的权重 bin 只留一份，其余模型的 .onnx 引用改指向它。

    实测回归（2026-11-11 动态档）：t2s 双解码器在 fp32 包里共享同一份
    t2s_shared bin，量化后各自产出**逐字节相同**的 int8 bin（量化是权重的
    确定性函数，sha256 实测一致），包里却存了两份（306 MB -> 576 MB）。
    fp32 包的约定本来就是"一份 bin、role 表双引用"，这里恢复该约定：
    改 external_data.location 指向保留的那份，删除重复文件。静态档同样
    受益（两个解码器的量化产物同样逐字节相同）。返回 (删除份数, 字节数)。
    """
    owners = {}  # bin 绝对路径 -> [(onnx 路径, location 名)]
    for entry in manifest["models"]:
        if entry.get("weight_format") != "int8":
            continue
        onnx_path = dst / entry["file"]
        model = onnx.load(str(onnx_path), load_external_data=False)
        for initializer in model.graph.initializer:
            if initializer.data_location != onnx.TensorProto.EXTERNAL:
                continue
            for ext in initializer.external_data:
                if ext.key == "location":
                    bin_path = (onnx_path.parent / ext.value).resolve()
                    owners.setdefault(bin_path, []).append((onnx_path, ext.value))

    by_hash = {}
    for bin_path in owners:
        by_hash.setdefault(sha256_file(bin_path), []).append(bin_path)

    removed, removed_bytes = 0, 0
    for paths in by_hash.values():
        if len(paths) < 2:
            continue
        keep = paths[0]  # manifest 顺序里第一个角色保留；其余改指过来
        for dup in paths[1:]:
            for onnx_path, _ in owners[dup]:
                model = onnx.load(str(onnx_path), load_external_data=False)
                changed = False
                for initializer in model.graph.initializer:
                    if initializer.data_location != onnx.TensorProto.EXTERNAL:
                        continue
                    for ext in initializer.external_data:
                        if ext.key == "location" and \
                                (onnx_path.parent / ext.value).resolve() == dup:
                            ext.value = keep.name
                            changed = True
                if changed:
                    onnx.save(model, str(onnx_path))
            removed_bytes += dup.stat().st_size
            dup.unlink()
            removed += 1
    return removed, removed_bytes


def referenced_external_files(package_dir):
    """清点包内所有 .onnx 引用的外部数据文件名（相对 models/ 的位置）。"""
    referenced = set()
    for onnx_file in Path(package_dir).rglob("*.onnx"):
        model = onnx.load(str(onnx_file), load_external_data=False)
        for initializer in model.graph.initializer:
            if initializer.data_location == onnx.TensorProto.EXTERNAL:
                for entry in initializer.external_data:
                    if entry.key == "location":
                        referenced.add(Path(onnx_file).parent / entry.value)
    return referenced


def main():
    parser = argparse.ArgumentParser(description="INT8 混合精度音色包生成（动态 / 静态两档）")
    parser.add_argument("--src", required=True, help="源音色包目录（fp32）")
    parser.add_argument("--dst", required=True, help="目标音色包目录（不存在的目录会创建）")
    parser.add_argument("--roles", default=",".join(DEFAULT_ROLES),
                        help=f"要量化的角色，逗号分隔（默认 {','.join(DEFAULT_ROLES)}；"
                             "听感敏感的 vits / speaker_verification 必须显式指定并过门控）")
    parser.add_argument("--ops", default=None,
                        help=f"量化的算子类型（动态档默认 {DEFAULT_OPS_DYNAMIC}；"
                             f"静态档默认 {DEFAULT_OPS_STATIC}）")
    parser.add_argument("--workers", type=int, default=2,
                        help="并行量化进程数（默认 2；仅动态档生效，静态档恒为串行）")
    parser.add_argument("--static", action="store_true",
                        help="静态 INT8（QDQ，Conv 可量化）：需要 --calibration-dir 提供校准集")
    parser.add_argument("--calibration-dir",
                        help="genie-gate capture 的输出目录（静态档校准集）")
    parser.add_argument("--calibration-method", choices=("minmax", "percentile"),
                        default="minmax",
                        help="静态档校准法（默认 minmax；percentile 用于门控 FAIL 时的兜底）")
    parser.add_argument("--no-verify", action="store_true",
                        help="跳过量化后用 onnx.load 复核（大模型复核要整包解析，较慢）")
    args = parser.parse_args()

    if args.static and not args.calibration_dir:
        sys.exit("--static 需要 --calibration-dir（genie-gate capture 的输出目录）")
    if not args.static and args.calibration_dir:
        sys.exit("--calibration-dir 只在 --static 下有意义")
    ops = args.ops or (DEFAULT_OPS_STATIC if args.static else DEFAULT_OPS_DYNAMIC)
    calibrate_method = {
        "minmax": ort_quant.CalibrationMethod.MinMax,
        "percentile": ort_quant.CalibrationMethod.Percentile,
    }[args.calibration_method]

    src, dst = Path(args.src), Path(args.dst)
    # 静态档强制串行：并发 quantize_static（校准期建 ORT 会话 + shape 推理）
    # 实测过一次 SIGSEGV（vits + roberta 双任务；两个模型单跑、小对子并发
    # 都不复现，动态档同参数从没出过事），量化本来就是离线重活，不值得赌。
    workers = 1 if args.static else max(1, args.workers)
    manifest_path = src / "manifest.json"
    if not manifest_path.exists():
        sys.exit(f"源包没有 manifest.json: {src}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("weight_format", "fp32") == "fp16":
        sys.exit("源包是 fp16 分发包：量化要在 fp32 包上做（fp16 外部数据与"
                 "量化器声明的 fp32 字节数不符，字节级裁剪会错位）")

    roles = [r.strip() for r in args.roles.split(",") if r.strip()]
    entries = {m["role"]: m for m in manifest["models"]}
    missing = [r for r in roles if r not in entries]
    if missing:
        sys.exit(f"manifest 里没有这些角色: {', '.join(missing)}")

    # 1) 整包镜像拷贝 —— 保留外部数据相对路径与 reference / 词典等资源。
    print(f"mirror {src} -> {dst}")
    if dst.exists():
        sys.exit(f"目标目录已存在（量化是重活，不覆盖）: {dst}")
    shutil.copytree(src, dst, symlinks=True)

    # 2) 逐角色量化（原地覆盖镜像里的 .onnx）。
    targets = [(role, entries[role]["file"]) for role in roles]
    results = {}

    def run(item):
        role, rel_file = item
        src_onnx = src / rel_file   # onnx.load 从**源**读（外部数据在源包解析）
        dst_onnx = dst / rel_file   # 输出写到镜像；未动的外部引用仍指向镜像内相对路径
        if args.static:
            ok, info = quantize_one_static(role, src_onnx, dst_onnx,
                                           set(ops.split(",")), args.calibration_dir,
                                           calibrate_method)
        else:
            ok, info = quantize_one(src_onnx, dst_onnx, set(ops.split(",")))
        return role, ok, info

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for role, ok, info in pool.map(run, targets):
            results[role] = (ok, info)
            print(f"  quantize {role:24s} {'OK' if ok else 'FAIL  ' + info}")

    # 3) 重写 manifest：包级标 int8（描述"这是一个量化包"），每个量化成功
    #    的角色显式写 weight_format=int8，没量化的角色写 fp32 —— 逐模型
    #    覆盖才是真正的裁决字段（src/runtime/voice_package.cpp）。静态 /
    #    动态在引擎侧是同一个标签：loader 只用它跳过 fp16 上转扫描。
    for entry in manifest["models"]:
        role = entry["role"]
        if role in results:
            if not results[role][0]:
                entry["weight_format"] = "fp32"  # 量化失败的角色保持 fp32
            else:
                entry["weight_format"] = "int8"
        else:
            entry["weight_format"] = "fp32"
    manifest["weight_format"] = "int8"

    # 3.5) 内容相同的量化 bin 去重（t2s 双解码器共享权重，见函数注释）。
    deduped, dedup_bytes = dedupe_identical_bins(dst, manifest)
    if deduped:
        print(f"deduped {deduped} identical weight bin(s), saved {dedup_bytes / 1e6:.0f} MB")

    # 4) 删除不再被任何 .onnx 引用的旧外部 bin（量化把大权重挪进了新的
    #    int8 bin；旧 fp32 bin 只剩零星未量化张量引用，重整后引用清零）。
    #    **manifest 登记的角色文件不在此列** —— g2pm.bin / jieba.bin 这些
    #    前端容器不是 ONNX 外部权重，图引用清点看不见它们。
    referenced = referenced_external_files(dst)
    role_files = {(dst / entry["file"]).resolve() for entry in manifest["models"]}
    removed = 0
    for bin_file in (dst / "models").glob("*.bin"):
        resolved = bin_file.resolve()
        if resolved not in referenced and resolved not in role_files:
            bin_file.unlink()
            removed += 1
    print(f"removed {removed} stale weight bin(s)")

    # 5) 重算被量化模型与所有文件的 SHA-256（校验值是部署契约，见 §7）。
    #    去重改写过 .onnx 的引用字段，必须在这一步之后取值。
    for entry in manifest["models"]:
        file_path = dst / entry["file"]
        if file_path.exists():
            entry["sha256"] = sha256_file(file_path)

    (dst / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 6) 体积账 + 复核。
    def tree_bytes(root):
        return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())

    src_size, dst_size = tree_bytes(src), tree_bytes(dst)
    print(f"package size: {src_size / 1e6:.0f} MB -> {dst_size / 1e6:.0f} MB "
          f"({(1 - dst_size / src_size) * 100:.1f}% smaller)")

    if not args.no_verify:
        # 复核 = 变体包的每个量化模型能被 ORT 建会话。跑在**镜像目录**里，
        # 外部数据按相对路径解析 —— 与 C++ 加载路径一致；QDQ 模型若点了
        # CPU EP 没有 int8 内核的算子，这一步会当场报出来。
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1  # 只验证可加载性，不需要并行
        failed = []
        for entry in manifest["models"]:
            if entry.get("weight_format") != "int8":
                continue
            model_path = dst / entry["file"]
            try:
                ort.InferenceSession(str(model_path), so,
                                     providers=["CPUExecutionProvider"])
                print(f"  verify  {entry['role']:24s} OK")
            except Exception as exc:  # noqa: BLE001
                failed.append(entry["role"])
                print(f"  verify  {entry['role']:24s} FAIL: {exc}")
        if failed:
            sys.exit(f"这些量化模型无法被 ORT 加载: {', '.join(failed)}")

    failed = [role for role, (ok, _) in results.items() if not ok]
    print("quantized OK" if not failed else f"quantization failed for: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
