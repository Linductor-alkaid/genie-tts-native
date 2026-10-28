#!/usr/bin/env python3
"""把上游发布的模型文件组装成 genie-tts-native 的音色包（voice package）。

输入是按 HF 仓库 High-Logic/Genie 原始结构下载（并用
convert_fp16_weights.py 转好 fp32）的目录；脚本负责改名、算 SHA-256、
写 manifest.json。产物可以直接给 `genie-tts inspect` / `synth` 用。

上游目录约定（与本脚本参数的对应关系）：

    <dl>/CharacterModels/v2ProPlus/<voice>/tts_models/   → --tts-models
        t2s_encoder_fp32.onnx / t2s_encoder_fp32.bin（fp32，免转换）
        t2s_first_stage_decoder_fp32.onnx
        t2s_stage_decoder_fp32.onnx
        t2s_shared_fp16.bin → t2s_shared_fp32.bin
        vits_fp32.onnx / vits_fp16.bin → vits_fp32.bin
        prompt_encoder_fp32.onnx / prompt_encoder_fp16.bin → prompt_encoder_fp32.bin
    <dl>/GenieData/chinese-hubert-base/                  → --hubert
        chinese-hubert-base.onnx / chinese-hubert-base_weights_fp16.bin → *_weights.bin
    <dl>/GenieData/speaker_encoder.onnx                  → --speaker-encoder
    <dl>/GenieData(Optional)/RoBERTa/                    → --roberta
        RoBERTa.onnx / roberta_tokenizer/vocab.txt
    third_party/g2pm/g2pm.bin（仓库自带，可选）          → --g2pm
        g2pM 多音字消歧模型；缺省时多音字取最常用读音
    <voice>/prompt_wav/*.wav                             → --reference（任选一段）

用法（feibi 音色完整示例见 docs/deployment.md）：

    python3 tools/deploy/make_voice_package.py \
        --name feibi --tts-models <dl>/tts_models \
        --hubert <dl>/chinese-hubert-base \
        --speaker-encoder <dl>/speaker_encoder.onnx \
        --roberta <dl>/RoBERTa \
        --reference <voice>/prompt_wav/xxx.wav \
        --reference-text ref.txt \
        --output ~/voice-packages/feibi
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil

# 角色 → (源文件相对 tts-models/hubert/roberta 目录的名字, 是否必需)
TTS_ROLES: list[tuple[str, str]] = [
    ("t2s_encoder", "t2s_encoder_fp32.onnx"),
    ("t2s_first_stage_decoder", "t2s_first_stage_decoder_fp32.onnx"),
    ("t2s_stage_decoder", "t2s_stage_decoder_fp32.onnx"),
    ("vits", "vits_fp32.onnx"),
    ("prompt_encoder", "prompt_encoder_fp32.onnx"),
]
# fp16 需转换的权重：角色 → 转换后的 .bin 名（与 .onnx 同目录）。
TTS_WEIGHTS: list[tuple[str, str]] = [
    ("t2s_encoder", "t2s_encoder_fp32.bin"),
    ("vits", "vits_fp32.bin"),
    ("prompt_encoder", "prompt_encoder_fp32.bin"),
]


def sha256_of(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="音色名（写进 manifest.name）")
    parser.add_argument("--tts-models", required=True, type=pathlib.Path,
                        help="<voice>/tts_models 目录（已含 fp32 图与 fp32 权重）")
    parser.add_argument("--hubert", type=pathlib.Path, help="chinese-hubert-base 目录（可选）")
    parser.add_argument("--speaker-encoder", type=pathlib.Path,
                        help="speaker_encoder.onnx 路径（可选；hubert 同给时启用 V2ProPlus）")
    parser.add_argument("--roberta", type=pathlib.Path,
                        help="RoBERTa 目录（含 RoBERTa.onnx 与 roberta_tokenizer/vocab.txt，可选）")
    parser.add_argument("--g2pm", type=pathlib.Path,
                        help="g2pM 容器文件（仓库 third_party/g2pm/g2pm.bin，可选；"
                             "缺省时多音字取最常用读音）")
    parser.add_argument("--reference", required=True, type=pathlib.Path,
                        help="参考音频 wav（任意采样率，引擎会重采样到 32 kHz）")
    parser.add_argument("--reference-text", type=pathlib.Path,
                        help="参考文本（可选；给了才有 ref_seq）")
    parser.add_argument("--language", default="chinese", help="manifest.language（默认 chinese）")
    parser.add_argument("--output", required=True, type=pathlib.Path, help="音色包输出目录")
    args = parser.parse_args()

    models_dir = args.output / "models"
    ref_dir = args.output / "reference"
    models_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []

    def place(role: str, src: pathlib.Path, dest_name: str) -> None:
        dest = models_dir / dest_name
        shutil.copyfile(src, dest)
        entries.append({"role": role, "file": f"models/{dest_name}",
                        "sha256": sha256_of(dest)})
        print(f"  {role:<22} <- {src.name}")

    def place_weights(src: pathlib.Path, dest_name: str) -> None:
        """外部权重**不写 manifest** —— 清单只登记 .onnx 角色；
        ORT 按图里的 external_data 描述自动加载同目录的 .bin。"""
        dest = models_dir / dest_name
        shutil.copyfile(src, dest)
        print(f"  {'(外部权重)':<22} <- {src.name}")

    # 核心 TTS 图（tts_models 目录内平铺）。
    for role, filename in TTS_ROLES:
        src = args.tts_models / filename
        if not src.exists():
            raise SystemExit(f"缺少 {src}（fp32 图必须存在）")
        place(role, src, filename)
    for _, filename in TTS_WEIGHTS:
        src = args.tts_models / filename
        if src.exists():
            place_weights(src, filename)

    # V2ProPlus：prompt_encoder（上面）+ 说话人验证 + HuBERT 内容特征。
    v2proplus = args.speaker_encoder is not None
    if v2proplus:
        place("speaker_verification", args.speaker_encoder, "speaker_encoder.onnx")
    if args.hubert:
        place("chinese_hubert", args.hubert / "chinese-hubert-base.onnx",
              "chinese-hubert-base.onnx")
        weights = args.hubert / "chinese-hubert-base_weights.bin"
        if weights.exists():
            place_weights(weights, "chinese-hubert-base_weights.bin")

    # 中文韵律（RoBERTa）：vocab.txt 必须与 roberta.onnx 同目录（引擎约定）。
    if args.roberta:
        place("roberta", args.roberta / "RoBERTa.onnx", "roberta.onnx")
        vocab = args.roberta / "roberta_tokenizer" / "vocab.txt"
        if not vocab.exists():
            raise SystemExit(f"缺少 {vocab}（RoBERTa 分词表必须与 onnx 同目录）")
        shutil.copyfile(vocab, models_dir / "vocab.txt")
        print(f"  {'roberta vocab':<22} <- vocab.txt")

    # 多音字消歧（g2pM）：单文件容器，sha256 照常进 manifest。
    if args.g2pm:
        place("g2pm", args.g2pm, "g2pm.bin")
        print(f"  {'g2pm':<22} <- {args.g2pm.name}")

    # 参考音频与文本。
    shutil.copyfile(args.reference, ref_dir / "audio.wav")
    if args.reference_text:
        shutil.copyfile(args.reference_text, ref_dir / "text.txt")

    manifest = {
        "name": args.name,
        "format_version": 1,
        "model_version": "v2ProPlus" if v2proplus else "v2",
        "language": args.language,
        "sample_rate": 32000,
        "capabilities": {"v2proplus": v2proplus,
                         "roberta": args.roberta is not None},
        "models": entries,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    print(f"\n音色包已生成：{args.output}（{len(entries)} 个模型条目）")
    print(f"下一步：genie-tts inspect {args.output}")


if __name__ == "__main__":
    main()
