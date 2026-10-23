# 发布检查单

发版不是"打个 tag"，是把一个**已经验证过的状态**固定下来。

## 发版前

- [ ] `master` 的 CI 全绿（不是"本地过了"，是 GitHub Actions 过了）
- [ ] 本地四种构建都过一遍：

  ```bash
  # debug（无 ORT、无 kairo）—— 纯逻辑
  cmake -S . -B build/debug -G Ninja -DGENIE_ENABLE_ORT=OFF -DGENIE_ENABLE_KAIRO=OFF
  # release + kairo
  cmake -S . -B build/release -G Ninja -DCMAKE_BUILD_TYPE=Release -DGENIE_ENABLE_KAIRO=ON
  # release + ORT
  cmake -S . -B build/ort -G Ninja -DCMAKE_BUILD_TYPE=Release -DGENIE_ONNXRUNTIME_ROOT=...
  # asan + ubsan
  cmake -S . -B build/asan -G Ninja -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"
  ```

  四个都要 `ctest --output-on-failure` 全绿。
- [ ] `GENIE_UPDATE_GOLDEN` 没被误提交（黄金数据要么没变，要么变得有理由）
- [ ] `CHANGELOG.md` 有本次版本的条目
- [ ] `docs/roadmap.md` 的里程碑状态是准的
- [ ] 版本号对得上：`CMakeLists.txt` 的 `project(... VERSION ...)`、
      `CHANGELOG.md`、tag 名

## 发版

- [ ] 从 `master` 打 tag（**不要**从分支打）：

  ```bash
  git tag -a v0.1.0 -m "v0.1.0"
  git push origin v0.1.0
  ```

- [ ] 建 GitHub Release，把 `CHANGELOG.md` 里对应段落贴进去
- [ ] Release 里写清楚**已知限制**（不要只写好消息）

## 发版后

- [ ] 在干净环境里验证一遍安装包：

  ```bash
  cmake --install build/ort --prefix /tmp/genie-install
  # 下游 find_package(genie) 能链接 genie::genie
  ```

- [ ] 确认导出的符号集干净（`nm -D` 只应有 `genie_*`）

## v0.1.0 的已知限制

诚实写出来，比被人发现好：

- **英文 / 日文 / 韩文 G2P 未实现** —— 只做了中文。接口留成了可插拔的
  `TextFrontend`，见 [`docs/upstream-audit.md`](upstream-audit.md) §9.4。
- **多音字取最常用读音** —— 上游用 g2pM 做上下文判定，本项目用词频兜底。
- **变调 / 儿化 / 分词 / 繁简 / 日期时间量词未实现**。
- **RoBERTa / HuBERT 的可选模型缺失时降级为全零特征** —— 能合成，但韵律会差。
- **本环境无法验证真实模型的音频质量** —— 上游模型权重在 HuggingFace 上，
  开发环境不可达，所以端到端只验证到 I/O 契约层。
