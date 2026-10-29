# 部署与推理完整教程

从零开始：下载上游发布的权重 → 转换 fp16 → 组装音色包 → 合成第一句话。
全程**只需要**：git + C++20 编译器 + CMake + ONNX Runtime 预编译包；
转换与组装两步各要一个 `python3`（`numpy`，校验另需 `onnx`），
推理本身**不依赖 Python**。

> 本文以 Linux x86-64 + [feibi](https://hf-mirror.com/High-Logic/Genie) 音色
> （V2ProPlus）为例。其他音色（`mika` / `thirtyseven`）只换路径。
> 所有命令都在仓库根目录执行。

---

## 0. 你需要拿到哪些东西

| 类别 | 文件 | 来源 | 说明 |
|---|---|---|---|
| 推理引擎 | `genie-tts` 可执行文件 / `libgenie.so` | 本仓库构建（§2） | 运行时不依赖 Python |
| TTS 模型 ×5 | `t2s_encoder_fp32.onnx` + `.bin`、`t2s_first_stage_decoder_fp32.onnx`、`t2s_stage_decoder_fp32.onnx`、`t2s_shared_fp16.bin`、`vits_fp32.onnx` + `vits_fp16.bin` | HF `High-Logic/Genie`，按音色发布（§1） | V2 与 V2ProPlus 都要 |
| V2ProPlus 增件 | `prompt_encoder_fp32.onnx` + `prompt_encoder_fp16.bin`、`speaker_encoder.onnx` | 同上（`prompt_encoder` 在音色目录，`speaker_encoder` 是公共文件） | 可选：不给则按 V2 路径合成 |
| 内容特征 | `chinese-hubert-base.onnx` + `chinese-hubert-base_weights_fp16.bin` | 同上 `GenieData/chinese-hubert-base/` | 可选：参考音频的 HuBERT 特征，缺了退化为全零 |
| 韵律特征 | `RoBERTa.onnx` + `roberta_tokenizer/vocab.txt` | 同上 `GenieData(Optional)/RoBERTa/` | 可选：中文韵律，缺了退化为全零 |
| 多音字 | `third_party/g2pm/g2pm.bin`（**仓库自带**，3.1 MB） | 无需下载 | 可选：g2pM 多音字消歧（含 4.5 万词读音修正表）；`make_voice_package.py --g2pm third_party/g2pm/g2pm.bin` 接入，缺了回退单字最常用读音 |
| 分词 | `third_party/jieba/jieba.bin`（**仓库自带**，8.3 MB） | 无需下载 | 可选：jieba 分词（变调边界 + 词级修正的词边界）；`make_voice_package.py --jieba third_party/jieba/jieba.bin` 接入，缺了退回词表匹配 |
| 英文 G2P | `third_party/english/cmudict.bin`（5.0 MB）+ `pos_tagger.bin`（3.3 MB）+ `english_g2p.bin`（3.2 MB） | **仓库自带** | 可选：`make_voice_package.py --english-cmu-dict ... --english-pos-tagger ... --english-g2p-model ...` 接入；词典是英文合成的硬依赖，其余缺失时降级（OOV 词走 GRU / 同形异音词取默认读音） |
| 日语 G2P | OpenJTalk 词典 `open_jtalk_dic_utf_8-1.11/`（约 103 MB，**仓库不带**） | `python3 tools/deploy/fetch_openjtalk_dictionary.py --output /tmp/oj-dict`（从 pypi 的 openjtalk wheel 取 ✓，附 SHA-256 校验 ✓） | 可选：`make_voice_package.py --japanese-dictionary /tmp/oj-dict/open_jtalk_dic_utf_8-1.11` 接入；词典是日文合成的硬依赖，缺了日文文本没有音素。OpenJTalk 的**运行时源码已内置**（`third_party/openjtalk/`，BSD ✓），不需要额外编译 |
| 参考音频 | 一段 wav + 参考文本 | 音色目录 `prompt_wav/`（文本随 `prompt_wav.json` 发布） | 音色包的"音色本体"，**必需** |
| ONNX Runtime | 预编译包（CPU） | [microsoft/onnxruntime releases](https://github.com/microsoft/onnxruntime/releases) | CI 基线 1.22.0；本项目验证到 1.31.0 |

**体积预算**（feibi，全量 V2ProPlus + RoBERTa）：
下载约 1.1 GB（fp16 权重），转换后磁盘约 1.6 GB。

---

## 1. 下载上游发布文件

官方源 `https://huggingface.co/High-Logic/Genie`；国内镜像把域名换成
`https://hf-mirror.com` 即可。

**方式 A：huggingface-cli（推荐，可断点续传）**

```bash
pip install -U "huggingface_hub[cli]"
export HF_ENDPOINT=https://hf-mirror.com          # 国内镜像；官方源不用设

# 一个音色的全部 TTS 模型 + 公共模型 + 参考音频
hf download High-Logic/Genie \
    --include "CharacterModels/v2ProPlus/feibi/*" \
              "GenieData/*" \
              "GenieData(Optional)/RoBERTa/*" \
    --local-dir ~/genie-dl
```

**方式 B：逐文件 curl**

```bash
DL=~/genie-dl
BASE=https://hf-mirror.com/High-Logic/Genie/resolve/main
mkdir -p $DL/tts_models $DL/hubert $DL/roberta/roberta_tokenizer

# —— feibi 音色的 TTS 模型 ——
for f in t2s_encoder_fp32.onnx t2s_encoder_fp32.bin \
         t2s_first_stage_decoder_fp32.onnx t2s_stage_decoder_fp32.onnx \
         t2s_shared_fp16.bin vits_fp32.onnx vits_fp16.bin \
         prompt_encoder_fp32.onnx prompt_encoder_fp16.bin; do
    curl -L --fail -o $DL/tts_models/$f \
        "$BASE/CharacterModels/v2ProPlus/feibi/tts_models/$f"
done

# —— 公共：HuBERT 与说话人验证 ——
curl -L --fail -o $DL/hubert/chinese-hubert-base.onnx \
    "$BASE/GenieData/chinese-hubert-base/chinese-hubert-base.onnx"
curl -L --fail -o $DL/hubert/chinese-hubert-base_weights_fp16.bin \
    "$BASE/GenieData/chinese-hubert-base/chinese-hubert-base_weights_fp16.bin"
curl -L --fail -o $DL/speaker_encoder.onnx \
    "$BASE/GenieData/speaker_encoder.onnx"

# —— 可选：中文韵律 RoBERTa（599 MB，不要可跳过） ——
curl -L --fail -o $DL/roberta/RoBERTa.onnx \
    "$BASE/GenieData(Optional)/RoBERTa/RoBERTa.onnx"
curl -L --fail -o $DL/roberta/roberta_tokenizer/vocab.txt \
    "$BASE/GenieData(Optional)/RoBERTa/roberta_tokenizer/vocab.txt"

# —— 参考音频（任选一段；文本在 prompt_wav.json 里） ——
curl -L --fail -o $DL/reference.wav \
    "$BASE/CharacterModels/v2ProPlus/feibi/prompt_wav/zh_vo_Main_Linaxita_2_1_10_26.wav"
curl -L --fail -o $DL/prompt_wav.json \
    "$BASE/CharacterModels/v2ProPlus/feibi/prompt_wav.json"
```

---

## 2. 构建引擎

```bash
# ONNX Runtime：解压官方预编译包即可（1.17+ 自带 CMake config）
#   wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz
#   tar xf onnxruntime-linux-x64-1.22.0.tgz -C ~/tools/

cmake --preset release -DGENIE_ONNXRUNTIME_ROOT=~/tools/onnxruntime-linux-x64-1.22.0
cmake --build --preset release
ctest --preset release --output-on-failure     # 应当 5/5 通过
```

产物：`build/release/tools/genie-tts`（CLI）。
不带 ONNX Runtime 也能构建（`GENIE_ENABLE_ORT=OFF`），但只能跑文本前端。

---

## 3. fp16 → fp32 权重转换

四个外部权重是 **fp16 原始数组**（`*_fp16.bin`），而 `.onnx` 图把张量
声明为 **fp32 + external data**，必须转一次才能加载：

| 上游文件 | 转换产物 | 图 |
|---|---|---|
| `t2s_shared_fp16.bin` | `t2s_shared_fp32.bin` | `t2s_first_stage_decoder_fp32.onnx` 与 `t2s_stage_decoder_fp32.onnx` **共用** |
| `vits_fp16.bin` | `vits_fp32.bin` | `vits_fp32.onnx` |
| `prompt_encoder_fp16.bin` | `prompt_encoder_fp32.bin` | `prompt_encoder_fp32.onnx` |
| `chinese-hubert-base_weights_fp16.bin` | `chinese-hubert-base_weights.bin` | `chinese-hubert-base.onnx` |

`t2s_encoder_fp32.bin` 本来就是 fp32，**不要转**。

```bash
pip install numpy onnx            # onnx 用于转换前的图校验（可选但强烈建议）

python3 tools/deploy/convert_fp16_weights.py \
    --weights $DL/tts_models/t2s_shared_fp16.bin \
    --model   $DL/tts_models/t2s_stage_decoder_fp32.onnx \
    --output  $DL/tts_models/t2s_shared_fp32.bin
#   图校验通过：291 个外部张量，区间连续，总长 306827268 字节
#   t2s_shared_fp16.bin（153413634 字节 fp16）→ …fp32.bin（306827268 字节）

# 其余三个同理：vits / prompt_encoder / chinese-hubert-base
```

脚本做的事：整文件按 `<f2` 读入 → cast 成 `<f4` 写出（大小翻倍）。
`--model` 提供时会先核对图里所有外部张量的存储区间**连续且总长恰好等于
转换产物**，不一致会拒绝转换 —— 这一步能抓住"转错图、下错文件"。

---

## 4. 组装音色包

音色包 = `manifest.json` + `models/`（全部 onnx 与权重）+ `reference/`
（参考音频 wav + 参考文本 txt）。**参考音频就是"音色"本身**：合成出
来的声音像谁，全由它决定。

```bash
python3 tools/deploy/make_voice_package.py \
    --name feibi \
    --tts-models      $DL/tts_models \
    --hubert          $DL/hubert \
    --speaker-encoder $DL/speaker_encoder.onnx \
    --roberta         $DL/roberta \
    --g2pm            <repo>/third_party/g2pm/g2pm.bin \
    --jieba           <repo>/third_party/jieba/jieba.bin \
    --english-cmu-dict   <repo>/third_party/english/cmudict.bin \
    --english-pos-tagger <repo>/third_party/english/pos_tagger.bin \
    --english-g2p-model  <repo>/third_party/english/english_g2p.bin \
    --japanese-dictionary /tmp/oj-dict/open_jtalk_dic_utf_8-1.11 \
    --reference       $DL/reference.wav \
    --reference-text  ref.txt \
    --output          ~/voice-packages/feibi
```

脚本自动：改名、计算每个模型的 SHA-256、写 `manifest.json`
（`capabilities` 按你给了什么自动置位：上面这条全量命令得到
`v2proplus: true, roberta: true`）。缺 RoBERTa / HuBERT /
speaker-encoder 就**不传对应参数**，引擎按上游的降级路径跑
（韵律/内容特征置零）。`--g2pm` 同理可选：不给则多音字按单字表
最常用读音（上游 g2pM 缺失时无法运行，这是本项目的扩展降级路径）。

多音字的效果：`重庆` 读 chóng、"银行分行"两个行分别读 háng/xíng、
"音乐"的乐读 yuè —— 模型按整句上下文判定，而不是每个字取最常用读音。
g2pM 也有自己的误判（例如"行长在银行上班"里给"银行"读 xíng），
与上游一致地保留，见 `docs/upstream-audit.md` §9.7。

目录长这样：

```
~/voice-packages/feibi/
├── manifest.json                 # 角色 → 文件 + sha256
├── models/
│   ├── t2s_encoder_fp32.onnx     # 4 个必需角色
│   ├── t2s_encoder_fp32.bin
│   ├── t2s_first_stage_decoder_fp32.onnx
│   ├── t2s_stage_decoder_fp32.onnx
│   ├── t2s_shared_fp32.bin       # 两张解码器图共用（external data 'location'）
│   ├── vits_fp32.onnx / vits_fp32.bin
│   ├── prompt_encoder_fp32.onnx / prompt_encoder_fp32.bin   # V2ProPlus
│   ├── speaker_encoder.onnx
│   ├── chinese-hubert-base.onnx / chinese-hubert-base_weights.bin
│   └── roberta.onnx + vocab.txt  # vocab.txt 必须与 onnx 同目录（引擎约定）
└── reference/
    ├── audio.wav                 # 参考音频（任意采样率，引擎重采样到 32 kHz）
    └── text.txt                  # 参考文本（可选；有了才有 ref_seq）
```

---

## 5. 校验与第一次合成

```bash
G=build/release/tools/genie-tts
PKG=~/voice-packages/feibi

$G inspect $PKG        # 清单 + SHA-256 逐个校验，全部 ok 才往下走
$G normalize --language chinese --text "2024年10月2日，发布第3版！"
#   → 二零二四年十月二日,发布第三版!

$G synth --model $PKG --text "你好，我是 feibi。" --output hello.wav
#   wrote hello.wav (… frames, 32000 Hz, … s)
#     phonemes: …
#     semantic tokens: …
```

听见 `hello.wav` 就部署完成了。换文本直接改 `--text`；
`--language auto`（默认）会自动检测，纯中文无需指定。

**C++ SDK / C ABI**：同一个音色包目录直接喂给
`genie::Engine engine(pkg_path)` 或 `genie_engine_create(pkg_path, …)`，
用法见 [`docs/usage.md`](usage.md)。

---

## 6. 部署注意事项

- **线程数**：Engine 一次加载 4–8 个 ONNX 会话，每个会话默认 8 个
  intra-op 线程（CLI `--threads`）。低配机器可以把 `--threads` 调小；
  核心充足想压低延迟可调大。并发请求请用多个 Engine 实例
  （Engine 非线程安全）。
- **结果不可复现**：采样（TopK/温度）在 ONNX 图内，ORT 未公开图内
  采样种子；`--seed` 目前只是预留字段。同一文本多跑几次，时长/韵律
  会有小方差。
- **句尾偶发"说一半"**：同上方差的一种表现 —— AR 解码提前采样到
  EOS 时最后一个音节会被挤压。重合成一次即可，机制见
  [`docs/upstream-audit.md`](upstream-audit.md) §10.5。
- **中文读音质量边界**：多音字取最常用读音（上游 g2pM 是上下文判定）、
  变调/儿化按词表近似（oracle 比对差异见 audit §9.6）、
  英/日/韩文本暂不支持。
- **换音色** = 换一个音色包目录；引擎无状态。

## 7. 常见问题

| 现象 | 原因与处理 |
|---|---|
| `inspect` 报 sha256 不匹配 | 文件下载不完整或转错了图 —— 重下/重转，别手改 manifest |
| 推理输出乱码式噪声 / 崩溃 | fp16 权重没转 fp32（§3），或权重与图不配对 |
| 加载报 "no such file: t2s_shared_fp32.bin" | 两张解码器图共用该权重（external data `location`），文件名不能改、必须与 onnx 同目录 |
| 合成出来是复读参考音频 | 参考音频缺 0.3 s 静音尾巴是引擎内部自动补的，确认没有手工裁剪 reference/audio.wav；另见 audit §11 |
| CMake 找不到 ONNX Runtime | `-DGENIE_ONNXRUNTIME_ROOT` 指到解压根（含 `include/` 与 `lib/`） |
| 内存占用高 | 模型常驻约 1.6 GB（fp32），另加 ORT arena；多实例部署按实例数估算 |
