#!/usr/bin/env python3
# tools/gate/precision_gate.py —— 精度门控的数值比对端（roadmap M0.2 / M5 量化）
#
# 消费 genie-gate capture 的产物（genie-gate/capture.json + 裸张量文件），
# 把**变体模型**（int8 量化 / fp16 常驻等）放在捕获的真实输入上重放，
# 与捕获的 C++ fp32 输出逐张量比对，按逐角色容差裁决 PASS / FAIL。
#
# 为什么基线是"C++ fp32 自基线"而不是 Python 上游：上游权重在本环境不可达
# （roadmap M0.2），但门控回答的问题是"变体相对 fp32 掉了多少精度"——
# 这与上游无关，fp32 基线就够。与 Python 上游的黄金比对是另一件事。
#
# 容差哲学（AGENTS.md §5.3）：文本/token 必须逐位一致；连续张量用
# 绝对+相对容差；最终 PCM 不做逐样本裁决（交给 e2e 的 token 一致率 + 听测），
# 这里只输出 advisory 指标。
#
# 用法：
#   python3 tools/gate/precision_gate.py gate \
#       --capture <genie-gate capture 输出目录> \
#       --package <变体音色包目录> \
#       [--tol tools/gate/tolerances.json] [--report <out.json>] [--roles r1,r2]
#
#   python3 tools/gate/precision_gate.py e2e \
#       --floor <基线自比 e2e.json> --variant <变体 e2e.json> [--report <out.json>]

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover
    sys.exit(f"需要 onnxruntime（量化实验环境）：pip install onnxruntime\n{exc}")

DTYPE_TO_NUMPY = {"float32": np.float32, "int64": np.int64, "bool": np.uint8}

# advisory：只报告、不判死。听觉效果的终审是 e2e WAV 听测。
ADVISORY_OUTPUTS = {"audio"}

DEFAULT_TOLERANCES = {
    # 连续张量：相对 L2 与余弦双门槛。默认值是"量化可感知但不至于明显失真"
    # 的经验起点，实验后按模型逐个收紧/放宽（结果写进 performance.md）。
    "float": {"max_rel_l2": 0.05, "min_cosine": 0.99},
    # token 类输出：量化若翻了一个采样 token，语义就可能走岔 —— 必须逐位一致。
    "int64": {"min_exact": 1.0},
    "bool": {"min_exact": 1.0},
}


def load_tolerances(path):
    merged = json.loads(json.dumps(DEFAULT_TOLERANCES))
    if path and Path(path).exists():
        user = json.loads(Path(path).read_text(encoding="utf-8"))
        for kind, limits in user.get("float", {}).items():
            merged["float"][kind] = limits
        for kind, limits in user.get("int64", {}).items():
            merged["int64"][kind] = limits
        for role, spec in user.get("roles", {}).items():
            merged.setdefault("roles", {}).setdefault(role, {})
            merged["roles"][role] = spec
    return merged


def load_tensor(root, meta):
    dtype = DTYPE_TO_NUMPY[meta["dtype"]]
    data = np.fromfile(root / meta["file"], dtype=dtype)
    shape = [int(d) for d in meta["shape"]]
    return data.reshape(shape if shape else (1,))


# ---------------------------------------------------------------- 指标

def float_metrics(baseline, actual):
    """连续张量：相对 L2 + 余弦。二维输出（如 RoBERTa [T,1024]）逐行算
    余弦再取均值 —— 全局余弦会被个别大值行淹没，行均值更能反映
    "每一帧特征都像不像"。"""
    b = baseline.astype(np.float64).ravel()
    a = actual.astype(np.float64).ravel()
    metrics = {
        "max_abs": float(np.max(np.abs(b - a))) if b.size else 0.0,
        "rel_l2": float(np.linalg.norm(b - a) / max(np.linalg.norm(b), 1e-30)),
    }
    if baseline.ndim == 2 and baseline.shape[0] > 1:
        bn = b.reshape(baseline.shape)
        an = a.reshape(actual.shape)
        denom = np.linalg.norm(bn, axis=1) * np.linalg.norm(an, axis=1)
        row_cos = (bn * an).sum(axis=1) / np.maximum(denom, 1e-30)
        metrics["cosine_mean"] = float(row_cos.mean())
        metrics["cosine_min"] = float(row_cos.min())
        metrics["cosine"] = metrics["cosine_mean"]
    else:
        denom = np.linalg.norm(b) * np.linalg.norm(a)
        cos = float((b * a).sum() / max(denom, 1e-30))
        metrics["cosine"] = cos
        metrics["cosine_mean"] = cos
    return metrics


def exact_metrics(baseline, actual):
    b = baseline.ravel()
    a = actual.ravel()
    if b.shape != a.shape:
        return {"exact": 0.0, "shape_mismatch": [list(b.shape), list(a.shape)]}
    return {"exact": float((b == a).mean()) if b.size else 1.0}


def mel_spectral_distance(ref, other, sample_rate=32000):
    """对齐长度的 log-mel 谱平均绝对差（advisory）。

    不引第三方：手写 STFT + HTK mel 滤波器组。长度不同时截到较短者 ——
    token 分岔会让时长不同，此时这个指标只在公共前缀内有意义，
    解读时必须结合 token 一致率。
    """
    n_fft, hop, n_mels = 1024, 256, 80
    n = min(len(ref), len(other))
    if n < n_fft:
        return None
    ref, other = ref[:n].astype(np.float64), other[:n].astype(np.float64)
    window = np.hanning(n_fft)

    def logmel(x):
        frames = 1 + (n - n_fft) // hop
        idx = np.arange(n_fft)[None, :] + hop * np.arange(frames)[:, None]
        spec = np.abs(np.fft.rfft(x[idx] * window, axis=1)) ** 2
        f_max = sample_rate / 2
        mel_pts = np.linspace(0, 2595 * np.log10(1 + f_max / 700), n_mels + 2)
        hz = 700 * (10 ** (mel_pts / 2595) - 1)
        bins = np.floor((n_fft + 1) * hz / sample_rate).astype(int)
        fb = np.zeros((n_mels, n_fft // 2 + 1))
        for m in range(n_mels):
            left, center, right = bins[m], bins[m + 1], bins[m + 2]
            if center > left:
                fb[m, left:center] = (np.arange(left, center) - left) / (center - left)
            if right > center:
                fb[m, center:right] = (right - np.arange(center, right)) / (right - center)
        mel = np.log(spec @ fb.T + 1e-10)
        return mel

    return float(np.abs(logmel(ref) - logmel(other)).mean())


# ---------------------------------------------------------------- 门控

def check_output(role, name, baseline, actual, tol, floor=None, exact_margin=0.0,
                 sample_rate=32000):
    """裁决单个输出张量。

    floor：同输出在"fp32 基线包自比回放"下的指标（噪声地板）。采样在图内，
    fp32 重放也会翻近平局的 token（地板 exact<1）——token 类输出的裁决因此是
    **地板相对**的：变体不得比 fp32 自己更差（exact_margin 允许再多翻一点）；
    连续张量保持绝对容差（地板通常在 1e-6 量级，绝对容差天然覆盖它）。
    """
    entry = {"name": name, "shape": list(baseline.shape)}
    role_tol = tol.get("roles", {}).get(role, {})
    if baseline.dtype == np.float32:
        entry["metrics"] = float_metrics(baseline, actual)
        if name in ADVISORY_OUTPUTS:
            entry["metrics"]["mel_distance"] = mel_spectral_distance(
                baseline.ravel(), actual.ravel(), sample_rate)
        if floor is not None:
            entry["floor"] = floor
        limits = {**tol["float"], **role_tol}
        passed = entry["metrics"]["rel_l2"] <= limits["max_rel_l2"] and (
            entry["metrics"].get("cosine", 1.0) >= limits.get("min_cosine", 0.0)
        )
    else:
        entry["metrics"] = exact_metrics(baseline, actual)
        limits = {**(tol["int64"] if baseline.dtype == np.int64 else tol["bool"]), **role_tol}
        if floor is not None and "exact" in floor:
            # 地板相对：有地板时以地板为准（变体不得比 fp32 自身重放更差），
            # 容差里的 min_exact 只在无地板时生效。
            threshold = floor["exact"] - exact_margin
            entry["threshold"] = threshold
            passed = entry["metrics"].get("exact", 0.0) >= threshold
        else:
            passed = entry["metrics"].get("exact", 0.0) >= limits.get("min_exact", 1.0)
    entry["advisory"] = name in ADVISORY_OUTPUTS
    entry["verdict"] = "advisory" if entry["advisory"] else ("PASS" if passed else "FAIL")
    return entry


def replay_role(session, capture_root, calls, role, tol, floor_by_output=None,
                exact_margin=0.0):
    """把捕获的输入喂给变体会话，返回与捕获输出的比对记录。

    floor_by_output：(call, 输出名) → 地板指标，来自 fp32 基线包的自比回放。
    """
    results = []
    for call in calls:
        if not call["outputs"]:
            continue  # partial 捕获（stage_decoder 后续步）没有完整输出，跳过
        feeds = {}
        for meta in call["inputs"]:
            feeds[meta["name"]] = load_tensor(capture_root, meta)
        try:
            outputs = session.run(None, feeds)
        except Exception as exc:
            results.append({
                "call": call["call"], "verdict": "ERROR",
                "error": f"replay failed: {exc}",
            })
            continue
        # session.run 返回顺序与图输出一致；用名字映射避免顺序假设。
        out_names = [o.name for o in session.get_outputs()]
        for meta in call["outputs"]:
            name = meta["name"]
            if name not in out_names:
                results.append({"call": call["call"], "name": name,
                                "verdict": "ERROR", "error": "output missing in variant graph"})
                continue
            baseline = load_tensor(capture_root, meta)
            actual = np.asarray(outputs[out_names.index(name)])
            floor = None
            if floor_by_output is not None:
                floor = floor_by_output.get((call["call"], name))
            results.append({"call": call["call"],
                            **check_output(role, name, baseline, actual, tol,
                                           floor=floor, exact_margin=exact_margin)})
    return results


def gate_command(args):
    capture_root = Path(args.capture)
    capture = json.loads((capture_root / "capture.json").read_text(encoding="utf-8"))
    tol = load_tolerances(args.tol)

    manifest = json.loads((Path(args.package) / "manifest.json").read_text(encoding="utf-8"))
    role_to_file = {m["role"]: m["file"] for m in manifest["models"]}

    roles = args.roles.split(",") if args.roles else None
    by_role = {}
    for call in capture["calls"]:
        if roles and call["role"] not in roles:
            continue
        by_role.setdefault(call["role"], []).append(call)

    report = {
        "schema": 1,
        "capture": str(capture_root),
        "package": str(Path(args.package)),
        "package_weight_format": manifest.get("weight_format", "fp32"),
        "roles": {},
    }
    # 地板报告：fp32 基线包用**同一套回放流程**跑出来的指标。token 类输出
    # 的裁决以它为基准（变体不得比 fp32 自己更差）。
    floor_by_role = {}
    if args.floor_report and Path(args.floor_report).exists():
        floor_report = json.loads(Path(args.floor_report).read_text(encoding="utf-8"))
        for role, entry in floor_report.get("roles", {}).items():
            floor_by_role[role] = {
                (o["call"], o["name"]): o.get("metrics", {})
                for o in entry.get("outputs", [])
            }

    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    all_pass = True
    for role, calls in sorted(by_role.items()):
        model_path = role_to_file.get(role)
        if model_path is None:
            report["roles"][role] = {"verdict": "SKIP", "error": "role missing in variant package"}
            all_pass = False
            continue
        full_calls = [c for c in calls if c["outputs"]]
        if not full_calls:
            report["roles"][role] = {"verdict": "SKIP", "error": "no full captures"}
            continue
        session = ort.InferenceSession(str(Path(args.package) / model_path), so, providers=["CPUExecutionProvider"])
        outputs = replay_role(session, capture_root, full_calls, role, tol,
                              floor_by_output=floor_by_role.get(role),
                              exact_margin=args.exact_margin)
        # advisory 输出（vits 音频）不参与裁决 —— 听觉终审在 e2e 听测。
        # ERROR 属于硬失败（重放都跑不起来，谈何一致）。
        role_pass = not any(o.get("verdict") in ("FAIL", "ERROR") for o in outputs)
        report["roles"][role] = {
            "model": model_path,
            "calls": len(full_calls),
            "verdict": "PASS" if role_pass else "FAIL",
            "outputs": outputs,
        }
        all_pass = all_pass and role_pass

    report["verdict"] = "PASS" if all_pass else "FAIL"
    text = json.dumps(report, ensure_ascii=False)
    if args.report:
        Path(args.report).write_text(text + "\n", encoding="utf-8")

    for role, entry in report["roles"].items():
        line = f"{role:28s} {entry.get('verdict', '?'):4s}"
        for out in entry.get("outputs", []):
            m = out.get("metrics", {})
            if "rel_l2" in m:
                line += f"\n    {out['name']:24s} rel_l2={m['rel_l2']:.5f} cosine={m.get('cosine_mean', float('nan')):.5f} {out['verdict']}"
            elif "exact" in m:
                line += f"\n    {out['name']:24s} exact={m['exact']:.4f} {out['verdict']}"
            elif "error" in out:
                line += f"\n    {out['name']}: {out['error']}"
        print(line)
    print(f"\nGATE: {report['verdict']}")
    return 0 if all_pass else 1


def e2e_command(args):
    floor = json.loads(Path(args.floor).read_text(encoding="utf-8"))
    variant = json.loads(Path(args.variant).read_text(encoding="utf-8"))

    def same_index(entries):
        return [p["agreement"] for p in entries["pairs"] if p["baseline_run"] == p["variant_run"]]

    floor_agr = same_index(floor)
    variant_agr = same_index(variant)
    floor_min = min(floor_agr) if floor_agr else 0.0
    variant_min = min(variant_agr) if variant_agr else 0.0
    # 变体与其自身噪声地板相比：同 run 序号配对下，确定性引擎的基线自比
    # 应当是 1.0（同包双引擎逐字节一致，见 docs/performance.md 量化一节）。
    # 变体掉得比地板多，才算是量化引入的语义分岔。
    margin = args.margin
    passed = variant_min >= max(floor_min - margin, 0.0)
    report = {
        "schema": 1,
        "floor_min_agreement": floor_min,
        "variant_min_agreement": variant_min,
        "margin": margin,
        "verdict": "PASS" if passed else "FAIL",
        "floor_pairs": len(floor_agr),
        "variant_pairs": len(variant_agr),
    }
    if args.report:
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"floor min agreement:    {floor_min:.4f} ({len(floor_agr)} pairs)")
    print(f"variant min agreement:  {variant_min:.4f} ({len(variant_agr)} pairs)")
    print(f"\nE2E GATE: {report['verdict']}")
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description="精度门控：变体模型 vs fp32 捕获基线")
    sub = parser.add_subparsers(dest="command", required=True)

    gate = sub.add_parser("gate", help="张量级门控（capture 产物 + 变体包）")
    gate.add_argument("--capture", required=True, help="genie-gate capture 输出目录")
    gate.add_argument("--package", required=True, help="变体音色包目录")
    gate.add_argument("--tol", default=str(Path(__file__).parent / "tolerances.json"))
    gate.add_argument("--roles", help="只测这些角色（逗号分隔），默认全部")
    gate.add_argument("--threads", type=int, default=8)
    gate.add_argument("--floor-report", help="fp32 基线包自比回放的门控报告"
                                             "（token 类输出按地板相对裁决）")
    gate.add_argument("--exact-margin", type=float, default=0.0,
                      help="token 一致率相对地板的容忍余量（默认 0：不得比地板差）")
    gate.add_argument("--report", help="JSON 报告输出路径")

    e2e = sub.add_parser("e2e", help="端到端 token 一致率门控（两个 genie-gate e2e 报告）")
    e2e.add_argument("--floor", required=True, help="基线自比的 e2e.json（噪声地板）")
    e2e.add_argument("--variant", required=True, help="变体 vs 基线的 e2e.json")
    e2e.add_argument("--margin", type=float, default=0.02, help="相对地板的容忍余量")
    e2e.add_argument("--report", help="JSON 报告输出路径")

    args = parser.parse_args()
    sys.exit(gate_command(args) if args.command == "gate" else e2e_command(args))


if __name__ == "__main__":
    main()
