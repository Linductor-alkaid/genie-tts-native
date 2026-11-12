# 发布检查单

发版不是"打个 tag"，是把一个**已经验证过的状态**固定下来。

## 发版前

- [ ] `master` 的 CI 全绿（不是"本地过了"，是 GitHub Actions 过了）
- [ ] 本地各构建都过一遍：

  ```bash
  # debug（无 ORT、无 kairo）—— 纯逻辑
  cmake --preset debug -DGENIE_ENABLE_ORT=OFF -DGENIE_ENABLE_KAIRO=OFF
  # release + kairo
  cmake --preset release
  # asan + ubsan
  cmake --preset asan
  # warnings-as-errors
  cmake --preset warnings
  ```

  每个都要 `ctest --output-on-failure` 全绿。
  （TSan 只在 CI 跑 —— 本地同时开 asan + tsan 会把 3 GB 内存的机器 OOM 掉。）
- [ ] `GENIE_UPDATE_GOLDEN` 没被误提交（黄金数据要么没变，要么变得有理由）
- [ ] `CHANGELOG.md` 有本次版本的条目
- [ ] `docs/roadmap.md` 的里程碑状态是准的
- [ ] `docs/upstream-audit.md` §9.4 的偏差表与当前代码一致
- [ ] 版本号对得上：`CMakeLists.txt` 的 `project(... VERSION ...)`、
      `CHANGELOG.md`、tag 名
- [ ] 编码没损坏：全仓扫一遍 U+FFFD（`grep -rn $'\xef\xbf\xbd'`）

## 发版

二进制产物与 Release 由 `.github/workflows/release.yml` 自动完成
（tag 推送触发；三平台 bundle：linux-x64 / linux-aarch64 / windows-x64，
捆绑的 ONNX Runtime 版本写在 workflow 的
`ONNXRUNTIME_RELEASE_VERSION`，当前 1.31.0 —— 与 `docs/performance.md`
的实测口径一致；ci.yml 的 1.22.0 是编译验证口径，两者刻意不同）：

- [ ] 首次或改动后先 **dry run**：master 上手动触发
      `Release` workflow（`workflow_dispatch`，勾选 dry_run）——
      只构建三平台产物，不建 Release；全部绿再进下一步
- [ ] 从 `master` 打 tag（**不要**从分支打）：

  ```bash
  git tag -a v0.5.0 -m "v0.5.0"
  git push origin v0.5.0
  ```

- [ ] `publish` job 自动完成：版本一致性检查（产物 `genie-tts version`
      == tag，历史上两次漏 bump 的自动化防线）→ 从 CHANGELOG 提取
      对应段作 Release 说明 → 挂产物与 sha256
- [ ] Release 页面人工补一段**已知限制**（不要只写好消息；可从下面的
      "已知限制"分类里挑适用项）
- [ ] dry run 是验证新平台/新 ORT 版本的途径：改了
      `release.yml` 或 `make_runtime_bundle.py` 之后必须重跑一次

## 发版后

- [ ] 在干净环境里验证一遍安装包：

  ```bash
  cmake --install build/release --prefix /tmp/genie-install
  # 下游 find_package(genie) 能链接 genie::genie
  ```

- [ ] 确认导出的符号集干净（`nm -D` 只应有 `genie_*`）
- [ ] 下载 Release 的 bundle，解包直跑（无 LD_LIBRARY_PATH）：
      `genie-tts version`、用 §4.1 的 min 音色包合成一句话
- [ ] Windows 产物需要目标机装 VC 运行时（/MD 链接）—— Release 说明里写明

## v0.4.0 的已知限制

诚实写出来，比被人发现好。

### 能力边界

- **jieba 的 posseg HMM（词性标注）未移植** —— 上游调的是
  `jieba_fast.posseg.lcut`，本移植对齐的是 `jieba.lcut`。两者的切分在
  少量低频/OOV 词上不同（"里种着""那本新"等），POS 依赖的变调分支
  只能取默认行为。变调/儿化的实测差距：**韵母序列 42/44 一致**
  （jieba 路径），详见 [`docs/upstream-audit.md`](upstream-audit.md) §9.6。
- **韩文 mecab-ko 形态标注未移植** —— g2pk2 的四条依赖形态标记的音变
  规则不触发（`/J /E /P /B` 标注），其余规则全部照做，见 §18。
- **日文词典（103 MB）不入仓库** —— 由音色包角色 `japanese_dictionary`
  在运行时给出；缺词典时**明确报错**，不静默回退到中文前端。
- **MSVC 上默认不编 OpenJTalk** —— 内置的 mecab 是 POSIX 系代码，
  在 MSVC 上能否过没有验证过，所以默认关掉（显式
  `-DGENIE_ENABLE_OPENJTALK=ON` 会被尊重）。Linux 上默认开。
- **英文 WordSegmenter 词典（12 MB）不入仓库** —— 由音色包角色
  `english_word_segment` 给出；缺失时 OOV 词直接交给 GRU。
- **fp16 权重分发只替换主图初始化器** —— 子图 / 节点属性 / 稀疏张量里
  指向 fp16 文件的外部引用**加载即报错**（防止后端按 fp32 偏移去读
  fp16 文件）；两种解释都装不下、非 FLOAT、未对齐、缺 length 同样报错，
  不做歧义回退。
- **参考音频在构造期绑定** —— 不能在运行时更换参考音频（上游支持）。
  代价是加载期的 HuBERT / 说话人验证 / prompt_encoder 三个会话算完即可
  释放（常驻 −1.2 GB）；将来加"换参考"API 要重新加载这三个模型（~0.9 s）。
  见 upstream-audit §23。

### 验证边界

- **本环境无法验证真实模型的音频质量** —— 上游模型权重在 HuggingFace 上，
  开发环境不可达，所以端到端只验证到 I/O 契约层。合成质量的验收
  需要在能下载权重的环境里做。
- **PCM 级黄金语料缺失** —— 文本层的 8 套黄金语料已锁定，音频层待补。
- **混读基线未采集** —— 混读（拉丁字母相邻中文）路径的缺陷已由 #34 修复，
  但 `genie-bench` 的混读基线还没入库（默认样例是纯中文，保持与既有
  基线可比）。
- **性能基线只在单机上测过** —— `docs/performance.md` 的数字来自
  i9-13900KF / ORT 1.31.0；换硬件、换 ORT 版本、换音色包都要重测。

### 已知不影响正确性的取舍

- **主机侧没有采样器** —— 采样在 ONNX 图内（上游同款），
  `SynthesisOptions::seed` / CLI `--seed` 是预留字段，当前不生效。
- **word2ph 按单字统计** —— 与上游一致（上游同样是逐字统计，
  §9.4 有勘误记录）。
- **RoBERTa / HuBERT 的可选模型缺失时降级为全零特征** —— 能合成，但韵律会差。
- **音色包信任模式跳过 SHA-256 校验** —— `EngineConfig::verify_voice_package`
  关闭后冷启动 −51.5%，但**模型完整性不再校验**；默认仍开启校验。
- **音频整段推送（不是分块流式）** —— `synthesize_streaming()` 当前是
  合成完一次性推入通道，首个可播放音频块时间 == 端到端耗时；
  真正的分块输出属 M6 待办。
- **高线程数 × 自旋是实测机器的已知不稳定区** —— 采集期间设备死机过一次
  （v0.2.0 也在同机记录过 ORT 内部段错误/浮点异常）；该组合对延迟无收益，
  默认配置（8 线程 × 自旋关）已规避。
