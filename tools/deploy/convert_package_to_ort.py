#!/usr/bin/env python3
# tools/deploy/convert_package_to_ort.py —— ORT Format 音色包生成（M5 冷启动）
#
# 读一个现有音色包，把每个角色的 .onnx 用 ORT 离线图优化后转成 **.ort**
# （ORT 自有格式，自包含权重），产出加载更快的新包：
#   - 图优化（融合 / 常量折叠 / NCHWc）在离线完成，运行期只做执行准备
#     —— 静态量化（QDQ）包尤其受益：vits 的 QDQ 图在加载期优化要 ~0.9 s，
#     .ort 把这笔账挪到离线（docs/performance.md §静态 INT8）。
#   - **零引擎改动**：manifest 的 file 字段直接改指 .ort，ORT 按后缀嗅探
#     格式，加载路径与 .onnx 完全同构（src/engine/model_loader.cpp 对
#     图内容零假设）。
#
# 用法与约束（沿用 quantize_package.py 的惯例）：
#   python3 tools/deploy/convert_package_to_ort.py \
#       --src /path/voice-package-int8 --dst /path/voice-package-ort
#   - 转换要**链在最后**：fp32 → (int8 量化) → ort。.ort 是已优化格式，
#     onnx 量化工具解不了它， Converted 包不能再进 quantize_package.py。
#   - .ort 与转换侧的 ORT 版本 + 图优化级别 + 硬件类绑定（level-ALL 的
#     NCHWc 变换是硬件相关的，ORT 会显式警告）：升级 ORT / 换硬件类部署
#     需要重新转换。manifest 顶层 `model_format: "ort"` 是给部署侧的标记。
#
# 转换完成后照旧过精度门控抽查 + genie-bench 冷加载对比。

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

try:
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover
    sys.exit(f"需要 onnxruntime（ORT Format 转换）：pip install onnxruntime\n{exc}")

try:
    import onnx
except ImportError as exc:  # pragma: no cover
    sys.exit(f"需要 onnx（外部数据引用清点）：pip install onnx\n{exc}")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_one(src_onnx, dst_ort, graph_opt_level):
    """单个模型：离线图优化 + ORT Format 落盘。返回 (成功, 信息)。

    序列化发生在 InferenceSession 构造期（optimized_model_filepath）；自包含
    （外部权重按引用收编进 .ort）。level > EXTENDED 时 NCHWc 变换是硬件
    相关的 —— 这是文档里"同硬件类部署"警告的来源，ORT 自己也会打日志。
    """
    try:
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel(graph_opt_level)
        options.optimized_model_filepath = str(dst_ort)
        options.add_session_config_entry("session.save_model_format", "ORT")
        # 单线程建会话：转换是离线一次性成本，并行不值得引入线程安全隐患
        # （quantize_static 校准并发实测过一次 SIGSEGV，同类风险不赌）。
        options.intra_op_num_threads = 1
        ort.InferenceSession(str(src_onnx), options, providers=["CPUExecutionProvider"])
    except Exception as exc:  # noqa: BLE001 —— 单模型失败不应中止整包
        return False, f"{type(exc).__name__}: {exc}"
    if not dst_ort.exists() or dst_ort.stat().st_size == 0:
        return False, ".ort 未产出（save_model_format 可能不被此 ORT 版本支持）"
    return True, "ok"


def referenced_external_files(package_dir):
    """清点包内所有 .onnx 引用的外部数据文件（相对 models/ 的位置）。"""
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
    parser = argparse.ArgumentParser(description="ORT Format 音色包生成（离线图优化）")
    parser.add_argument("--src", required=True, help="源音色包目录（fp32 / int8；fp16 不支持）")
    parser.add_argument("--dst", required=True, help="目标音色包目录（不存在的目录会创建）")
    parser.add_argument("--roles", default=None,
                        help="要转换的角色，逗号分隔（默认全部 .onnx 角色）")
    parser.add_argument("--graph-opt", type=int, default=3, choices=(0, 1, 2, 3),
                        help="离线图优化级别（默认 3，与部署默认一致）")
    parser.add_argument("--no-verify", action="store_true",
                        help="跳过转换后用 ORT 逐角色复核")
    args = parser.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    manifest_path = src / "manifest.json"
    if not manifest_path.exists():
        sys.exit(f"源包没有 manifest.json: {src}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("weight_format", "fp32") == "fp16":
        sys.exit("源包是 fp16 分发包：fp16 的上转扫描按 ONNX wire format 解析"
                 "模型文件，对 .ort 会直接报错 —— ORT Format 包不做 fp16 分发")
    if manifest.get("model_format") == "ort":
        sys.exit("源包已经是 ORT Format：转换链是 fp32 → (int8) → ort，不回头")

    entries = {m["role"]: m for m in manifest["models"]}
    if args.roles:
        roles = [r.strip() for r in args.roles.split(",") if r.strip()]
        missing = [r for r in roles if r not in entries]
        if missing:
            sys.exit(f"manifest 里没有这些角色: {', '.join(missing)}")
    else:
        roles = [role for role, entry in entries.items() if entry["file"].endswith(".onnx")]
    targets = [(role, entries[role]["file"]) for role in roles if entries[role]["file"].endswith(".onnx")]
    if not targets:
        sys.exit("没有可转换的 .onnx 角色")

    # 1) 整包镜像拷贝 —— 词典 / reference 等资源原样带走。
    print(f"mirror {src} -> {dst}")
    if dst.exists():
        sys.exit(f"目标目录已存在（转换是重活，不覆盖）: {dst}")
    shutil.copytree(src, dst, symlinks=True)

    # 2) 逐角色转换：models/<stem>.onnx -> models/<stem>.ort（原地，自包含）。
    converted = {}  # role -> (旧 onnx 相对路径, 成功?)
    for role, rel_file in targets:
        src_onnx = dst / rel_file          # 镜像内原地转换：外部数据相对路径可达
        dst_ort = src_onnx.with_suffix(".ort")
        ok, info = convert_one(src_onnx, dst_ort, args.graph_opt)
        converted[role] = (rel_file, ok)
        print(f"  convert {role:24s} {'OK' if ok else 'FAIL  ' + info}")

    # 3) 重写 manifest：转换成功的角色 file 改指 .ort；顶层加 model_format
    #    标记（引擎忽略未知字段，纯部署契约）。转换失败的角色保持 .onnx 原样。
    any_ok = False
    for role, (rel_file, ok) in converted.items():
        if ok:
            entries[role]["file"] = str(Path(rel_file).with_suffix(".ort"))
            any_ok = True
    if any_ok:
        manifest["model_format"] = "ort"

    # 4) 清理：**先删**转换成功的角色的旧 .onnx，再清点引用 —— 顺序反了
    #    的话 shell 还在、其外部 bin 全被"引用"保护（实测踩过：removed 0，
    #    包体虚胖 1 GB）。**manifest 登记的文件**（.ort 与词典类 .bin）必须
    #    保留；部分转换时剩下的 .onnx 还引用着自己的 bin，一并保护。
    for role, (rel_file, ok) in converted.items():
        if ok:
            (dst / rel_file).unlink()
    referenced = referenced_external_files(dst)
    role_files = {(dst / entry["file"]).resolve() for entry in manifest["models"]}
    removed = 0
    for bin_file in (dst / "models").glob("*.bin"):
        resolved = bin_file.resolve()
        if resolved not in referenced and resolved not in role_files:
            bin_file.unlink()
            removed += 1
    print(f"removed {removed} stale weight bin(s)")

    # 5) 全量重算 SHA-256（部署契约；file 字段已改，校验值必须跟着文件走）。
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
        # 复核 = 每个转换后的 .ort 能被 ORT 建会话（后缀嗅探路径，与 C++
        # 加载路径一致）。单线程，只验证可加载性。
        failed = []
        for entry in manifest["models"]:
            if not entry["file"].endswith(".ort"):
                continue
            model_path = dst / entry["file"]
            try:
                ort.InferenceSession(str(model_path),
                                     providers=["CPUExecutionProvider"])
                print(f"  verify  {entry['role']:24s} OK")
            except Exception as exc:  # noqa: BLE001
                failed.append(entry["role"])
                print(f"  verify  {entry['role']:24s} FAIL: {exc}")
        if failed:
            sys.exit(f"这些 .ort 无法被 ORT 加载: {', '.join(failed)}")

    failed = [role for role, (_, ok) in converted.items() if not ok]
    print("converted OK" if not failed else f"conversion failed for: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
