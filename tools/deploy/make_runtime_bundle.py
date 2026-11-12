#!/usr/bin/env python3
"""把构建产物组装成可分发的运行时 bundle（tar.gz / zip + 体积清单）。

输入是一个**已构建**的 build 目录与 ONNX Runtime 预编译包根目录；脚本用
`cmake --install` 复用既有 install 规则（bin / lib / include / cmake config）
staging 出目录树，补上 ORT 共享库，strip、写清单、压缩。

产物布局（Linux）：

    genie-tts-runtime-<版本>-<tag>.tar.gz
    ├── bin/        genie-tts、genie-bench、genie-gate（Windows 另含 DLL）
    ├── lib/        libgenie_tts.so（C ABI）+ libonnxruntime.so*（版本化 + 软链）
    │               （SDK 部分照常安装：include/、lib/cmake/genie、静态库）
    ├── packages/   可选：--voice-package 嵌入的音色包（本地端侧交付用）
    └── BUNDLE_MANIFEST.txt   逐文件大小 + sha256 + 汇总（体积清单）

为什么用 `cmake --install` 而不是手工挑文件：install 规则（src/CMakeLists.txt
的 install(TARGETS ...) 与 include 安装）是"装什么"的唯一权威；bundle 侧只
做 install 覆盖不到的三件事 —— ORT 共享库、strip、清单与压缩。

用法：

    python3 tools/deploy/make_runtime_bundle.py \
        --build-dir build/release \
        --ort-root ~/tools/onnxruntime \
        --tag linux-x64 --ort-version 1.31.0 \
        --output-dir dist/

交叉构建（如 aarch64）：产物无法在本机运行，脚本自动跳过运行冒烟，
改用 `file` 校验架构；版本号从安装树的 include/genie/version.hpp 解析。
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import tarfile
import zipfile


def sha256_of(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if proc.returncode != 0:
        raise SystemExit(f"命令失败（{proc.returncode}）: {' '.join(cmd)}\n"
                         f"{proc.stdout}\n{proc.stderr}")
    return proc


def elf_arch(path: pathlib.Path) -> str:
    """file 输出里的架构段（交叉产物校验用；file 缺失时返回空）。"""
    try:
        out = subprocess.run(["file", "-b", str(path)],
                             capture_output=True, text=True).stdout
    except FileNotFoundError:
        return ""
    for key in ("x86-64", "ARM aarch64"):
        if key in out:
            return key
    return ""


def version_from_install_tree(stage: pathlib.Path) -> str:
    """从安装树的生成头文件解析版本 —— 交叉构建时产物跑不了，
    version.hpp 是唯一与产物同源且可读的版本来源。"""
    header = stage / "include" / "genie" / "version.hpp"
    if not header.exists():
        raise SystemExit(f"安装树里没有 {header}（install 规则被跳过？）")
    for line in header.read_text(encoding="utf-8").splitlines():
        if "GENIE_VERSION_STRING" in line and "@PROJECT_VERSION@" not in line:
            # 形如 #define GENIE_VERSION_STRING "0.5.0"
            return line.rsplit('"', 2)[-2]
    raise SystemExit(f"无法从 {header} 解析版本号")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build-dir", required=True, type=pathlib.Path,
                        help="已构建的 CMake build 目录（含 CMakeCache.txt）")
    parser.add_argument("--ort-root", required=True, type=pathlib.Path,
                        help="ONNX Runtime 预编译包根目录（与构建时同一份）")
    parser.add_argument("--tag", required=True,
                        help="平台标签：linux-x64 / linux-aarch64 / windows-x64")
    parser.add_argument("--ort-version", default=None,
                        help="写进清单的 ORT 版本（缺省从库名探测）")
    parser.add_argument("--voice-package", type=pathlib.Path, default=None,
                        help="可选：音色包目录，嵌入 packages/（本地端侧交付）")
    parser.add_argument("--no-strip", action="store_true",
                        help="跳过 strip（调试产物 / 复现符号用）")
    parser.add_argument("--output-dir", required=True, type=pathlib.Path,
                        help="bundle 输出目录（stage 树与压缩包都放这里）")
    args = parser.parse_args()

    build_dir = args.build_dir.resolve()
    if not (build_dir / "CMakeCache.txt").exists():
        raise SystemExit(f"{build_dir} 不是 CMake build 目录（缺 CMakeCache.txt）")
    bin_name = "genie-tts.exe" if args.tag.startswith("windows") else "genie-tts"
    built_bin = build_dir / "tools" / bin_name
    if not built_bin.exists():
        raise SystemExit(f"{built_bin} 不存在 —— 先 cmake --build")

    ort_root = args.ort_root.resolve()
    is_windows = args.tag.startswith("windows")
    if is_windows:
        # win-x64 预编译包把 DLL 与 .lib 一起放在 lib\（没有 bin\）；老布局
        # 可能拆在 bin\ —— 两处都找，取真有 DLL 的那份。
        candidates = [ort_root / "lib", ort_root / "bin"]
        ort_libdir = next((d for d in candidates
                           if d.is_dir() and any(d.glob("onnxruntime*.dll"))),
                          ort_root / "lib")
    else:
        ort_libdir = ort_root / "lib"

    stage = args.output_dir.resolve() / "stage"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    # ---- 1) install 规则 staging（"装什么"的唯一权威）----
    print(f"[1/5] cmake --install {build_dir} -> {stage}")
    run(["cmake", "--install", str(build_dir), "--prefix", str(stage)])

    # ---- 2) ORT 共享库：Windows 进 bin/（与 exe 同目录），其余进 lib/ ----
    print(f"[2/5] 复制 ONNX Runtime 共享库 <- {ort_libdir}")
    dest_dir = stage / "bin" if is_windows else stage / "lib"
    dest_dir.mkdir(parents=True, exist_ok=True)
    pattern = "onnxruntime*.dll" if is_windows else "libonnxruntime*"
    copied_ort = sorted(p.name for p in ort_libdir.glob(pattern))
    if not copied_ort:
        raise SystemExit(f"{ort_libdir} 里没有 {pattern}（ORT 包不完整？）")
    for name in copied_ort:
        src = ort_libdir / name
        # follow_symlinks=False：libonnxruntime.so -> .so.1 -> .so.x.y.z 的
        # 软链结构原样保留，动态 loader 的版本解析才成立。
        shutil.copy2(src, dest_dir / name, follow_symlinks=False)
        print(f"    {name}")
    ort_version = args.ort_version
    if not ort_version:
        # 从实际复制的主文件名探测（libonnxruntime.so.1.31.0 → 1.31.0）
        for name in copied_ort:
            parts = name.rsplit(".so.", 1)
            if len(parts) == 2 and parts[1].count(".") >= 2:
                ort_version = parts[1]
                break

    # ---- 3) strip（仅自有 ELF；ORT 上游已 strip，静态库不动）----
    if args.no_strip:
        print("[3/5] 跳过 strip（--no-strip）")
    elif is_windows:
        print("[3/5] Windows 产物不 strip（MSVC 无 strip；PDB 不在产物内）")
    else:
        print("[3/5] strip 自有 ELF")
        # 交叉产物：宿主 binutils 的 strip 不识别其他架构的 ELF
        # （"Unable to recognise the architecture"），有对应交叉 strip 就用它。
        arch = elf_arch(stage / "bin" / bin_name)
        strip_tool = "strip"
        if arch == "ARM aarch64" and shutil.which("aarch64-linux-gnu-strip"):
            strip_tool = "aarch64-linux-gnu-strip"
            print(f"    （aarch64 产物 → {strip_tool}）")
        targets = sorted((stage / "bin").glob("*")) + \
            sorted(p for p in (stage / "lib").glob("libgenie*")
                   if p.is_file() and not p.name.endswith(".a"))
        for t in targets:
            if t.is_file() and not t.is_symlink():
                run([strip_tool, "--strip-unneeded", str(t)])

    # ---- 4) 版本 + 可运行性冒烟 ----
    version = version_from_install_tree(stage)
    binary_arch = elf_arch(stage / "bin" / bin_name)
    if is_windows:
        runnable = False  # 在 Windows runner 上另行冒烟
    elif binary_arch in ("x86-64", "ARM aarch64"):
        # 产物架构与构建机架构一致才运行（交叉构建在 x86 宿主机上跑不了）。
        elf_is_arm = binary_arch == "ARM aarch64"
        runnable = elf_is_arm == (platform.machine() in ("aarch64", "arm64"))
    else:
        runnable = False
        print(f"    警告：无法识别产物架构（file 输出：{binary_arch!r}）")
    if runnable:
        # 从 stage 树里跑 —— INSTALL_RPATH($ORIGIN/../lib) 是否生效在这里
        # 见分晓；不设 LD_LIBRARY_PATH，故意让环境里没有别的 ORT 可借。
        out = run([str(stage / "bin" / bin_name), "version"]).stdout.strip()
        print(f"[4/5] 运行冒烟：{out}（rpath 解析 OK）")
        if version not in out:
            raise SystemExit(f"版本不一致：install 树 {version} vs 产物输出 {out}")
    elif is_windows:
        print("[4/5] Windows bundle：跳过本机运行冒烟（在 Windows runner 上验证）")
    else:
        print(f"[4/5] 交叉产物（{binary_arch}）无法在本机运行，跳过冒烟"
              f"（file 架构校验在 ARM CI job 另有断言）")

    if args.voice_package:
        dest = stage / "packages" / args.voice_package.name
        shutil.copytree(args.voice_package, dest)
        print(f"    音色包已嵌入 packages/{args.voice_package.name}")

    # ---- 清单 ----
    manifest = stage / "BUNDLE_MANIFEST.txt"
    entries = sorted(stage.rglob("*"))
    files = [p for p in entries if p.is_file() and not p.is_symlink()]
    links = [p for p in entries if p.is_symlink()]
    total = sum(p.lstat().st_size for p in files)
    lines = [
        "genie-tts-native runtime bundle",
        f"tag: {args.tag}",
        f"version: {version}",
        f"ort: {ort_version or '(from filenames)'}",
        f"ort_files: {', '.join(copied_ort)}",
        f"generated: {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"runnable_on_build_host: {bool(runnable)}",
        "",
        f"total: {len(files)} files + {len(links)} links, {human(total)}",
        "",
        "sha256  size  path",
    ]
    for p in files:
        rel = p.relative_to(stage)
        lines.append(f"{sha256_of(p)}  {p.lstat().st_size:>12}  {rel}")
    for p in links:
        rel = p.relative_to(stage)
        lines.append(f"(link -> {os.readlink(p)})  "
                     f"{p.lstat().st_size:>12}  {rel}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"    清单：{manifest}（{len(files)} 文件 + {len(links)} 软链，"
          f"解包 {human(total)}）")

    # 清单写完后重新采集 —— 否则压缩包里没有 BUNDLE_MANIFEST.txt 自己。
    entries = sorted(stage.rglob("*"))

    # ---- 5) 压缩 + 校验和 ----
    print("[5/5] 压缩")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if is_windows:
        arc_path = args.output_dir / f"genie-tts-runtime-{version}-{args.tag}.zip"
        with zipfile.ZipFile(arc_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in entries:
                zf.write(p, str(p.relative_to(stage)))
    else:
        arc_path = args.output_dir / f"genie-tts-runtime-{version}-{args.tag}.tar.gz"
        with tarfile.open(arc_path, "w:gz") as tf:
            for p in entries:
                # 逐条加（recursive=False）：目录/软链/文件都按 stage 内相对路径
                # 落位，软链的链接关系由 tarfile 从文件系统自动读取。
                tf.add(p, arcname=str(p.relative_to(stage)), recursive=False)

    arc_digest = sha256_of(arc_path)
    (arc_path.with_suffix(arc_path.suffix + ".sha256")
     ).write_text(f"{arc_digest}  {arc_path.name}\n", encoding="utf-8")
    print(f"\nbundle：{arc_path}（{human(arc_path.stat().st_size)}）")
    print(f"sha256：{arc_digest}")


if __name__ == "__main__":
    sys.exit(main())
