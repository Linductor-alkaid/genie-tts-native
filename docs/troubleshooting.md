# 故障排查

按症状组织：先看现象，再对原因，最后是处理。部署全流程见
[`deployment.md`](deployment.md)；与上游行为的差异及依据见
[`upstream-audit.md`](upstream-audit.md)。

---

## 1. 构建 / 加载

### CMake 找不到 ONNX Runtime

- **现象**：`GENIE_ORT_STATUS: OFF (not found)`，构建能过但没有推理能力
  （`genie::build_features().onnxruntime == false`）。
- **原因**：`Dependencies.cmake` 找不到 ORT 时**强制**关闭后端（哪怕你显式
  传了 `GENIE_ENABLE_ORT=ON`）—— 防的是"以为在编真后端，其实编了个桩"。
- **处理**：`-DGENIE_ONNXRUNTIME_ROOT=<预编译包根目录>`，目录下要有
  `include/` 与 `lib/`。交叉编译用对应架构的包（aarch64 见 CI 的
  `arm64` job 命令）。

### 推理输出乱码式噪声 / 崩溃

- **原因**：fp16 权重没转 fp32（`tools/deploy/convert_fp16_weights.py`），
  或权重与图不配对。
- **处理**：核对 §3 的转换表；`genie-tts inspect` 先过一遍 sha256。

### 加载报 "no such file: t2s_shared_fp32.bin"

- **原因**：两张解码器图**共用**这份外部权重（external data `location`
  写死在图里），文件名不能改、必须与 .onnx 同目录。
- **处理**：确认组包用了 `make_voice_package.py`（它会摆对位置）；手工组包
  时别改名。

### `inspect` 报 sha256 不匹配

- **原因**：下载不完整或转错了图。
- **处理**：重下 / 重转；不要手改 manifest。

---

## 2. 合成行为

### 同一文本两次合成结果不同（时长 / 韵律有方差）

- **原因**：采样在 ONNX 图内（TopK + 随机算子），ORT 未公开图内采样种子，
  跨进程**不可复现**（同进程逐 run 可复现）。`--seed` / `seed` 字段是预留，
  当前不生效（audit §14.3 勘误）。
- **处理**：接受方差（上游同款）；对质量评估用多次采样取代表。

### 句尾偶发"说一半"（末音节没有衰减尾巴）

- **原因**：AR 解码提前采样到 EOS 时最后一个音节被挤压 —— 上游固有现象
  （机制与证据见 audit §10.5），不是引擎缺陷。
- **处理**：重合成一次；或在调用方对结果末尾补静音。

### 合成出来是复读参考音频

- **原因**：参考音频缺 0.3 s 静音尾巴会引导 T2S 第一轮就采样到 EOS
  （audit §11 / §14.4 的踩坑记录）。
- **处理**：引擎内部会自动拼接静音尾巴 —— 别手工裁剪
  `reference/audio.wav`；仍复现时检查音色包是不是手工组的老结构。

### 中文读音不对（多音字 / 变调）

- **原因与处理**：读音质量取决于音色包角色 —— `g2pm` 缺失回退单字最常用
  读音；`jieba` 缺失退回词表近似（变调韵母一致度 95.5% → ~68%，audit
  §9.6）。`--preset min-zh` 会自动带上两者；全量组包时别漏
  `--g2pm / --jieba`。

### 英 / 日 / 韩文本报"缺资源"

- **原因**：英文要 `english_cmu_dict` 等 4 角色（词典是硬依赖）、日文要
  `japanese_dictionary`（103 MB OpenJTalk 词典）、缺资源时**明确报错**
  而不是静默退回中文前端。
- **处理**：按 `docs/deployment.md` §0 的表把对应角色组进音色包；
  min-zh 预设有意不带英日资源（只做中文）。Windows 上 MSVC 默认不编
  OpenJTalk（POSIX 系代码），日文需要显式 `-DGENIE_ENABLE_OPENJTALK=ON`。
  韩文的 mecab-ko 形态标注未移植 —— 依赖标注的四条音变规则不触发
  （audit §18.3），其余规则一致。

---

## 3. 内存 / 性能

### 内存占用高

- **构成**：fp32 包常驻 ~2.6 GB（RoBERTa 1.2 GB 是大头）+ 峰值 RSS 更高。
  加载期的 HuBERT / SV / prompt_encoder 会话算完参考特征即释放
  （常驻 −1.2 GB，默认行为）。
- **处理（按收益排）**：动态 INT8 混合精度包（常驻 1.25 GB，听测通过，
  见 `docs/performance.md` §动态 INT8）→ min 音色包（不带 RoBERTa，
  ~0.8–0.9 GB，降级行为见 deployment.md §4.1）→ `--threads` 调小
  （影响的是 CPU，不是内存）。

### 冷启动慢

- **构成**：SHA-256 校验占一半（2 GB 级的包要全扫）。
- **处理**：`--no-verify`（信任模式，−51.5%；代价是不再校验完整性）→
  fp16 分发包（−21% 磁盘，冷启动只 +0.7 s）→ 动态 INT8 包（加载 −26%）
  → .ort 形态（冷加载 −23%，代价包体 +13% 且与转换侧 ORT 版本绑定）。
  量化包没有白给的精度 —— 每个角色都要过精度门控（`tools/gate/`）。

### 合成延迟高 / RTF 差

- **先归一**：跨进程 RTF 不可比（采样路径不同），用 `genie-bench` 的
  ms/token 对比。
- **旋钮**：线程数（默认 8）；自旋默认**关**（12/16 线程 × spin=1 是实测
  机器的已知不稳定区，且对延迟无收益，`docs/performance.md` §M5.3）。
  解码热路径的 KV cache 图内管理（Transpose+Concat ≈ 21%）是上游图结构
  决定的，宿主侧动不了。

---

## 4. 打包 / 分发

### bundle 解包后跑不起来（找不到 libonnxruntime）

- **原因**：解包后移动了 `bin/` 或 `lib/` 的相对位置 —— 安装产物的
  RUNPATH 是 `$ORIGIN/../lib`，bin 与 lib 必须保持平级。
- **处理**：整个解包目录一起搬；自定义布局要自己设 `LD_LIBRARY_PATH`。

### Windows 版跑不起来

- **原因**：/MD 链接需要目标机有 VC 运行时；`onnxruntime.dll` 必须与
  exe 同目录（bundle 已放好，散装部署要自己带）。
- **处理**：装 [VC Redist](https://aka.ms/vs/17/release/vc_redist.x64.exe)，
  或整个 bundle 目录使用。

### .ort 格式包换机器 / 换 ORT 版本后异常

- **原因**：.ort 与转换侧的 ORT 版本、图优化级别、硬件类绑定
  （`docs/deployment.md` §5.6）。跨环境分发用 .onnx 形态，部署机上重转。

### ARM（aarch64）产物怎么构建

- **交叉编译**：`cmake --preset arm64-release`
  + aarch64 版 ORT 包（见 ci.yml 的 `arm64` job）。托管 CI 只做编译验证，
  运行验证要在 aarch64 机器上做。
- **交叉编译的坑**：toolchain 的 `CMAKE_FIND_ROOT_PATH_MODE_LIBRARY/INCLUDE`
  必须用 NEVER —— `ONLY` 模式会把 `find_path/find_library` 的绝对路径
  HINTS 也 re-root 到 sysroot 下，ORT 解析会静默失败、后端被强制关闭
  （首个 arm64 CI job 实测踩中）。

---

## 5. 音色包 / 量化

### 想要更小的包

- 端侧：`make_voice_package.py --preset min-zh`（§4.1，~270/~450 MB）；
  按需 `min-zh-roberta`。再往下是核心权重的物理地板（t2s/vits 以 Conv
  为主，int8 收益趋零、静态档被精度门控拒绝 —— `docs/performance.md`
  §量化）。

### 量化包怎么生成 / 怎么判定"能不能用"

- `tools/deploy/quantize_package.py`（默认动态档、只量化 roberta /
  prompt_encoder）→ **每个角色过精度门控**（`tools/gate/`，分级容差）→
  听测终审。门控没过的角色 manifest 写回 fp32（逐模型覆盖）。
  静态 INT8（Conv 参与）是 opt-in 第二档，vits 是唯一幸存者。
- 跨进程 token 一致率**不能当质量指标**（采样种子随进程变）—— 听觉
  终审只能 WAV A/B（`docs/regression.md` 精度门控一节）。
