#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""取 OpenJTalk 的日语词典（open_jtalk_dic_utf_8-1.11）。

为什么需要单独一个脚本：词典约 103 MB ✓ —— **不进仓库** ✓
（与英文的 wordsegment 词典同理 ✓），由音色包在运行时给出路径 ✓。

词典的来源：pypi 上的 `openjtalk` wheel ✓ —— 那是 pyopenjtalk 的发行包 ✓
（上游 Genie-TTS 的日文 G2P 就是 pyopenjtalk ✓），
wheel 里带着 `pyopenjtalk/open_jtalk_dic_utf_8-1.11/` ✓。

用法::

    python3 tools/deploy/fetch_openjtalk_dictionary.py --output /tmp/openjtalk-dict

    # 然后
    python3 tools/deploy/make_voice_package.py ... \\
        --japanese-dictionary /tmp/openjtalk-dict/open_jtalk_dic_utf_8-1.11

词典本身的授权见同目录的 COPYING（BSD 3-Clause ✓）。
"""

import argparse
import hashlib
import json
import pathlib
import shutil
import sys
import tempfile
import urllib.request
import zipfile

DEFAULT_INDEX = "https://mirrors.aliyun.com/pypi/simple/openjtalk/"

# open_jtalk_dic_utf_8-1.11 各文件的 SHA-256 ✓（防止拿到被改过的包 ✓）。
EXPECTED = {
    "COPYING": "f4eca42ebd930e2c6e57fca58319d989bebcd1510cb7714b149c50f5425135ea",
    "char.bin": "888ee94c5a8a7a26d24ab3f1b7155441351954fd51ea06b4a2f78bd742492b2f",
    "left-id.def": "db1adac8a7f9e5854cd82ea044c85115249206c8181b9d88cf92ae2ee5e87b84",
    "matrix.bin": "62fd16b4f64c851d5dc352ef0d5740c5fc83ddc7c203b2b0b1fc5271969a14ce",
    "pos-id.def": "3460aa742053085af47cdfc889a1e0e6f557e89b406e501ba81c9ccc286de0c7",
    "rewrite.def": "7f7c8dfbfe24092e8a149a9b6e0a3a7f1c2cf37d6c3dc29d1cccc6c004da9c1c",
    "right-id.def": "db1adac8a7f9e5854cd82ea044c85115249206c8181b9d88cf92ae2ee5e87b84",
    "sys.dic": "ca57d9029691a70a5dfb99afc2844180256161d7130da65b1a867510e129b9a6",
    "unk.dic": "ce97851ecda075914fa3ffe7294a1ab34ee4f6d56ba6bf9197d74143b5dffbfe",
}

DICT_PREFIX = "pyopenjtalk/open_jtalk_dic_utf_8-1.11/"
DICT_NAME = "open_jtalk_dic_utf_8-1.11"


def sha256_of(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pick_wheel(index_url: str) -> str:
    """从 simple index 里挑一个 manylinux x86_64 的 wheel ✓。"""
    with urllib.request.urlopen(index_url) as response:
        body = response.read().decode("utf-8", errors="replace")
    candidates = []
    for chunk in body.split('"'):
        # href 后面可能跟着 #sha256=... ✓ —— 取 .whl 那一段 ✓。
        link = chunk.split("#")[0]
        if not link.endswith(".whl"):
            continue
        if "manylinux" in link and "x86_64" in link:
            candidates.append(link)
    if not candidates:
        # 退一步：任何 cp311 的 wheel ✓（本机 python 版本要对得上 ✓）。
        for chunk in body.split('"'):
            link = chunk.split("#")[0]
            if link.endswith(".whl") and "cp311" in link:
                candidates.append(link)
    if not candidates:
        raise SystemExit("no suitable wheel found in " + index_url)
    return sorted(candidates)[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=pathlib.Path,
                        help="输出目录（会在里面建 open_jtalk_dic_utf_8-1.11/）")
    parser.add_argument("--index", default=DEFAULT_INDEX, help="pypi simple index 的地址")
    parser.add_argument("--url", help="直接指定 wheel 的 URL（跳过 index 查询）")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = pathlib.Path(tmp)
        url = args.url or pick_wheel(args.index)
        if not url.startswith("http"):
            url = urljoin(args.index, url)
        print(f"downloading {url}")
        wheel = tmpdir / "openjtalk.whl"
        with urllib.request.urlopen(url) as response, wheel.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        print(f"  {wheel.stat().st_size} bytes")

        print("extracting dictionary")
        dest = args.output / DICT_NAME
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        with zipfile.ZipFile(wheel) as archive:
            for name in archive.namelist():
                if not name.startswith(DICT_PREFIX):
                    continue
                relative = name[len(DICT_PREFIX):]
                if not relative or relative.endswith("/"):
                    continue
                target = dest / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)

    print("verifying checksums")
    failed = 0
    for name, expected in EXPECTED.items():
        path = dest / name
        if not path.exists():
            print(f"  MISSING {name}")
            failed += 1
            continue
        actual = sha256_of(path)
        if actual != expected:
            print(f"  BAD {name}\n    expected {expected}\n    actual   {actual}")
            failed += 1
        else:
            print(f"  ok   {name}")
    if failed:
        print(f"{failed} file(s) failed verification", file=sys.stderr)
        return 1

    print(f"\ndictionary ready: {dest}")
    print(json.dumps({"path": str(dest), "files": len(EXPECTED)}, ensure_ascii=False))
    return 0


def urljoin(base: str, link: str) -> str:
    from urllib.parse import urljoin as _urljoin

    return _urljoin(base, link)


if __name__ == "__main__":
    sys.exit(main())
