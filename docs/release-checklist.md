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

- [ ] 从 `master` 打 tag（**不要**从分支打）：

  ```bash
  git tag -a v0.3.0 -m "v0.3.0"
  git push origin v0.3.0
  ```

- [ ] 建 GitHub Release，把 `CHANGELOG.md` 里对应段落贴进去
- [ ] Release 里写清楚**已知限制**（不要只写好消息）

## 发版后

- [ ] 在干净环境里验证一遍安装包：

  ```bash
  cmake --install build/release --prefix /tmp/genie-install
  # 下游 find_package(genie) 能链接 genie::genie
  ```

- [ ] 确认导出的符号集干净（`nm -D` 只应有 `genie_*`）

## v0.3.0 的已知限制

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

### 验证边界

- **本环境无法验证真实模型的音频质量** —— 上游模型权重在 HuggingFace 上，
  开发环境不可达，所以端到端只验证到 I/O 契约层。合成质量的验收
  需要在能下载权重的环境里做。
- **PCM 级黄金语料缺失** —— 文本层的 8 套黄金语料已锁定，音频层待补。
- **性能基线缺失** —— 没有 CPU Benchmark 报告，M5 尚未开始。

### 已知不影响正确性的取舍

- **主机侧没有采样器** —— 采样在 ONNX 图内（上游同款），
  `SynthesisOptions::seed` / CLI `--seed` 是预留字段，当前不生效。
- **word2ph 按单字统计** —— 与上游一致（上游同样是逐字统计，
  §9.4 有勘误记录）。
- **RoBERTa / HuBERT 的可选模型缺失时降级为全零特征** —— 能合成，但韵律会差。
