# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### Added

- **精度门控基建**（M0.2 容差框架 / M5 量化前置）：`genie-gate capture` 用包装
  会话工厂把真实流水线里每个模型的输入输出逐张量落盘（引擎零改动）；
  `tools/gate/precision_gate.py` 把变体模型放在捕获输入上重放并按
  `tools/gate/tolerances.json` 的分级容差裁决（token 逐位一致按噪声地板
  相对、连续张量 rel-L2 + 余弦、音频 advisory）。`genie-gate e2e` 输出
  基线/变体语义 token 一致率与听测 WAV。见 `docs/regression.md` 精度门控一节。

## [v0.4.0] - 2026-11-10

本版本是**性能优化专版**：M0.3 建立了首份性能基线，M5 在其上完成三轮优化 ——
解码热路径的宿主拷贝消除、ORT 会话参数调优、加载期的内存与并行度改造。
同机实测（i9-13900KF / ORT 1.31.0 / v2ProPlus 音色包）：

| 指标 | 优化前 | 优化后 |
|---|---|---|
| 冷启动（信任模式，fp32 包） | 3766 ms | **1738–1846 ms**（建会话并行，同轮对比） |
| 冷启动（信任模式，fp16 包） | 4387 ms | **2039 ms** |
| 常驻 RSS（fp32 包） | 3351 MB | **2075 MB** |
| RTF（短 / 中 / 长） | 0.718 / 0.628 / 0.778 | **0.468 / 0.476 / 0.564** |
| stage_decoder ms/token | 18.3 / 18.6 / 26.1 | **12.5 / 14.3 / 18.0** |
| 进程平均核占用 | 17.1 / 11.6 / 9.1 | **4.8 / 4.6 / 4.2** |

另外，**默认（校验开启）口径**的冷启动 7536 ms 里有一半是音色包 SHA-256
校验 —— 信任模式（`--no-verify`）去掉它后是 3589 ms（−51.5%）。

数字只在同硬件 + 同 ORT 版本上可比；跨轮次的绝对值有几百毫秒的漂移
（本机就出现过同配置 3766 / 3952 ms 两轮读数），测量环境由 bench 的 JSON
自证，原始记录在 `docs/benchmarks/`。

### Added

- **全链路分阶段计时**（M0.3 性能基线的采集端）：

  - `SynthesisResult` 新增 `timing` 字段：一次合成的各阶段耗时
    （language_detect / frontend / text_bert / t2s 及其三个子阶段
    t2s_encoder、t2s_first_stage、t2s_stage_decoder×N / vocoder），
    含端到端总时长。
  - `Engine::load_timing()` 返回构造期的加载耗时
    （音色包校验、每个 ONNX 会话、参考特征、全局嵌入）—— 冷启动基线。
  - `T2SDecoder::decode()` / `decode_loop()` 增加可选的
    `TimingReport*` 出参（默认 nullptr，不影响既有调用）。
  - CLI `synth` 命令新增 `--timing` 旗标：把加载与合成两张阶段表打到
    stderr。

  计时**常开**、没有开关：每阶段只做两次 `steady_clock` 采样
  （纳秒级），相对秒级的合成开销可以忽略。报告随结果返回而不是
  打日志 —— 呈现方式由调用方决定（CLI 打表格、基准工具算统计、
  SDK 使用者直接编程消费）。

- **`genie-bench` 基准工具 + 首份性能基线报告**（M0.3 收口，roadmap M7
  的"bench 待补"兑现）：

  - `tools/bench/`：warmup 消抖 + N 次重复 + 每阶段 mean/min/median/max
    统计，输出 RTF（墙钟/音频时长）、峰值 RSS、进程平均核占用，
    `--json` 落盘机器可读结果。默认短 / 中 / 长三档纯中文样例，
    `--text` 可覆盖。独立于 CLI 开关（`GENIE_BUILD_BENCH`）。
  - `docs/performance.md`：本机实测基线（i9-13900KF / ORT 1.31.0 /
    v2ProPlus 音色包）。关键结论：t2s 自回归循环占合成耗时 74%–86%
    （约 21–23 ms/token）；冷启动 7.6 s 里音色包 SHA-256 校验占一半；
    当前配置下 RTF 0.61–0.75（实时成立）。
  - 同时记录了一个基准过程中暴露的缺陷：混读（拉丁字母相邻中文）
    长文本触发 bert_chunks 与音素序列对不上的错误 —— 已由同版本的 #34
    修复，混读基线待补。

- **`genie-bench` 测量基建硬化**（M5 优化前后对比的前提）：

  - JSON 输出自包含采集环境：ORT 版本（新增公开 API
    `genie::onnxruntime_version_string()`）、OS、CPU 型号 —— 基线数字
    只在同硬件 + 同 ORT 版本上可比，测量记录必须能自证环境。
  - 输出解码步数 `iterations` 与 `stage_decoder ms/token`（终端 + JSON）：
    采样在 ONNX 图内、不可复现，每次运行的 token 数都不同，
    跨版本对比必须按 ms/token 归一（`TimingReport` 的 count 此前
    已采集但 bench 未消费）。
  - JSON 增加每 run 原始样本 `samples`（墙钟 / RTF / 步数 / 核占用 /
    逐阶段毫秒）、`total_wall_ms`、RTF median、顶层 `errors` 计数；
    有文本失败时终端打 warning（退出码行为不变，由脚本查 `errors`）。
  - 基线 JSON 入库：`docs/benchmarks/baseline-2026-11-06.json`
    （此前基线数据在 /tmp，未受版本控制）。
  - 修正 `docs/performance.md` 与本工具注释里仍声称"混读长文本会失败"
    的过时表述 —— 已由 #34 修复；混读基线待补。

- **模型会话并行加载（冷启动 3.77 → ~1.8 s）**：kairo 升至 53ee306
  （默认线程池从逐核绑定改为 OS 自由调度 —— 解除"ORT 线程继承单核
  亲和性"的阻塞；显式 `cpu_affinity` 行为不变）。调度层新增
  `detail::parallel_for`（fork-join；调用线程自带一份活、异常重抛最小
  下标、无 kairo 时串行）。引擎的 fp16 上转（按文件）与建会话（按模型）
  各以 8 路并行（concurrency = intra_op_threads）；计时新增
  `load/sessions`。实测会话段 ~2.9×（3.35 s → ~1.16 s），合成延迟不变；
  决策与绑核问题的来龙去脉见 docs/architecture.md §2.14。

- **`SessionConfig::parallel_prepack`**（opt-in，默认关）：建会话时把权重
  预打包摊到本会话的 intra-op 线程池（ORT `session.prepack.enable_parallel`
  —— 1.31 起才有，更早的版本忽略它）。实测冷启动 -8%（fp32 包三轮均值
  3.77 → 3.46 s），但常驻内存 +0.5 GB，因此默认关。`genie-bench
  --parallel-prepack <0|1>` 复现，JSON 记录该项。（本条先于"模型会话并行
  加载"落地 —— 当时 kairo 默认池逐核绑定，跨会话并行建会话会被 ORT 的
  intra-op 线程继承单核亲和性而否决；kairo 升至 53ee306 解除绑定后，
  建会话并行已落地，见上一条与 docs/architecture.md §2.14。）

- **fp16 权重分发支持**（M5，借鉴上游 Genie-TTS 的分发格式）：

  - 音色包 manifest 新增 `"weight_format": "fp16"`（缺省 fp32）：外部
    权重用 HF 上游的 fp16 打包布局（图声明 FLOAT、offset/length 为 fp32
    字节单位，文件在 offset/2 处存 fp16、大小恰为一半）。引擎加载期流式
    扫描各模型的外部张量、按"图声明区间 vs 文件大小"判定存储格式，
    fp16 打包文件**整文件上转**为 FP32（结果恰是 fp32 bin 的布局，不改写
    图；多个模型共用的文件只转一次），经 ONNX Runtime
    `AddExternalInitializersFromFilesInMemory` 以内存缓冲替换后建会话。
    运行时仍是 FP32。
  - 实测（信任模式）：包体积 2.4 → 1.9 GB（-21%）；冷启动 4.49 s，比
    fp32（3.72 s）只多 ~0.7 s（上转 0.39 s，扫描 8 个模型 6 ms）。
  - 新增公开 API：`scan_external_weights` / `classify_external_weights` /
    `read_fp16_weights_as_fp32`（onnx_graph.hpp）；
    `create_session(path, config, external_weights)` + `ExternalWeightBuffer`
    （session.hpp）。
  - 约束：只替换主图初始化器 —— 子图 / 节点属性 / 稀疏张量里指向 fp16
    文件的外部引用加载即报错（防止后端按 fp32 偏移去读 fp16 文件）；
    两种解释都装不下、非 FLOAT、未对齐、缺 length 同样报 ModelLoadError，
    不做歧义回退。
  - `tools/deploy/make_voice_package.py --weights fp16`：保留 HF 的
    `*_fp16.bin`（目标名不变 —— external location 写死在图里），manifest
    自动登记 weight_format。自包含模型（RoBERTa）不受影响。
  - 测试：`tools/make_synthetic_model.py --external-fp16` 生成回归资产；
    集成测试覆盖"不注入则加载失败 / 注入后立即释放缓冲再推理、逐元素
    校验 / 引擎层四个角色共用一个文件只上转一次"；上转对全部 65536 个
    half 值逐位精确（穷举单测）。
  - 设计演进：首版（#41/#42）在 wire-format 层把外部权重内联进 protobuf
    再从内存建会话（`inline_external_fp16_weights` /
    `create_session_from_memory`，未发布即移除）—— 自包含模型也被整份
    重写、权重被搬运五六遍，冷启动 9.47 s。决策与取舍见
    docs/architecture.md §2.13，数据见 docs/performance.md 与
    docs/benchmarks/2026-11-10-load/。

### Changed

- **加载期会话用完即释（常驻 -1.2 GB）**：HuBERT / 说话人验证 /
  prompt_encoder 只在加载期算参考特征与全局嵌入，结果缓存后引擎在构造
  结束前释放这三个会话，并把空闲页还给系统（glibc `malloc_trim`）。
  fp32 包加载后常驻 3351 → 2075 MB，合成延迟不变。上游支持运行时换
  参考音频所以必须常驻；本引擎参考音频在构造期绑定（差异与影响见
  docs/upstream-audit.md §23）。t2s 双解码器权重共享调研过：ORT 的
  `AddInitializer` / `PrepackedWeightsContainer` 实测省不出来（图优化
  重排的副本才是重复存储的来源），结论记录在 docs/performance.md §内存。
- **`genie-bench` 的 JSON 记录冷启动**：新增 `verify_voice_package`（是否
  跳过包校验 —— 冷启动里最大的一块可开关开销）与 `load`（引擎构造的
  分阶段耗时）。此前加载表只打在终端上，无法事后比对。另新增
  `rss_after_load_kb` / `rss_end_kb`（当前 RSS，加载后与退出前）——
  峰值之外，常驻量才是长期占用。

- **ORT 会话默认值调优**（M5.3，本机扫描数据驱动）：

  - `SessionConfig::allow_spinning` 默认改**关**
    （`session.intra_op.allow_spinning=0`）。自旋是为"下一个算子马上
    就来"准备的忙等；引擎一次合成只有一个会话在干活，闲置会话的自旋
    线程纯属内耗。实测：延迟持平或更优（短句 12.5 vs 14.4 ms/token），
    进程核占用从 9–19 降到 4–5。
  - 线程数默认保持 8：12/16 线程在长句上劣化 11–12%（工作摊到 P/E
    两类核上），收益为负。
  - `genie-bench` 新增 `--spinning <0|1>`（默认 0，与引擎默认一致），
    JSON 记录 `spinning`；扫描原始数据入库
    `docs/benchmarks/2026-11-06-sweep/`，分析与默认值依据见
    `docs/performance.md` §参数扫描、`docs/architecture.md` §2.12。
  - ⚠️ 高线程数 × 自旋是实测机器的已知不稳定区（采集期间设备死机
    一次；v0.2.0 也在同机记录过 ORT 内部段错误/浮点异常），
    该组合对延迟无收益，默认配置已规避。

- **解码热路径的宿主拷贝消除**（M5.2，语义不变）—— M0.3 基线显示
  `t2s/stage_decoder` 占合成耗时 62%–83% 且每 token 耗时随长度上升，
  审计发现每步解码把 KV cache（48 个张量，长句末段约 34 MB）完整拷贝
  约五遍：

  - `T2SDecoder` 解码循环改 move 语义：`read_output` / `build_stage_feed`
    对 y_emb 与 KV cache 走 move（state 每轮被新输出整体替换，中间态
    无人再读）；tokens 仍拷贝（循环结束后的 EOS 归零与切片要用）。
  - `to_ort_value` 零拷贝包装 Tensor 存储：`CreateTensor` 直接引用
    输入缓冲，废除"ORT 分配器分配 + memcpy"（契约：feed 生命周期覆盖
    Run、ORT CPU EP 不改写输入；风险与回退见 architecture.md §2.11）。
  - 新增 `Tensor::from_raw(type, shape, const void*)`（单次 memcpy），
    `from_ort_value` 改走它 —— 原先"拷进临时 vector 再 memcpy"两遍。
  - 会话级复用 `Ort::MemoryInfo`，删除每 run / 每张量新建的分配器。
  - 决策记录：architecture.md §2.11；审计记录：upstream-audit.md §21。

### Fixed

- **超长中文文本无法一次合成**：RoBERTa 的位置嵌入只有 512 行，纯中文
  长文本此前整段编码 —— 531 个 token 实测撞出 "512 by 531" 广播错误，
  合成直接失败（上游同样整段编码、限制一致）。文本前端现在在中文段
  超过单块预算（500 字，软切下限 380）时按标点切 BERT 分块，引擎复用
  混读路径的逐块编码再拼接：超长文本一次合成完成，块尾落在句读上；
  预算内的行为逐字节不变。有意偏差记录见 docs/upstream-audit.md §24。

- **CI 的集成测试一直是空跑**：合成测试模型（`tests/models/synthetic/`）
  被 .gitignore 排除、CI 又从不生成，`genie_integration_tests` 的 7 个用例
  （按名字绑定、引擎构造、加载计时、fp16 上转加载路径）与
  `OnnxGraph.ReadsSyntheticModelSignature` 在 CI 里全部 GTEST_SKIP ——
  ctest 把跳过算通过，job 绿而空。现在 ORT job 先用
  `tools/make_synthetic_model.py` 生成资产（onnx / numpy 钉版本），并单独
  断言集成用例没有任何 SKIPPED；生成器补上输出目录的自动创建（干净
  checkout 里该目录不存在）。

- **CI runner 镜像升级后单元测试编译失败**：`tests/unit/test_english.cpp`
  使用 `std::find` 但依赖其它头文件间接带入 `<algorithm>`，ubuntu-latest
  镜像升级后的 GCC 不再传递包含 —— 显式补上。纯移植性修复，无行为变化。

- **中英混读合成失败**：混读文本里的英文块产出音素与 word2ph 却没有进
  BERT 分块（2026-10-29 接入英文 G2P 时引入）—— 分块不再连续覆盖
  word2ph，RoBERTa 装配时报
  `bert chunks cover N phones, sequence has M`，且英文块音素越多越早
  触发（长混读句必然命中）。上游语义（审计 §6.3）是英文块**全零行 +
  逐块拼接覆盖全序列**；现在英文块以 `BertChunk::zero_features` 占位进
  分块，装配时直接填零行不跑模型。回归测试改用**挂了英文前端**的混读
  前端 —— 旧测试的裸前端把英文块整体丢弃，分块不变量恒真，拦不住这个
  缺陷（`genie-bench` 首跑即暴露）。


## [v0.3.1] - 2026-10-31

### Fixed

- **部署文档里"银行分行"的读音写错了**：原文写"两个行分别读 háng/xíng" ——
  实际**两个都读 háng**（yín háng fēn háng）。

  错在哪：那句话描述的是 **g2pM 模型层**的判定（模型确实把"分行"判成 xíng），
  而**最终输出**经过上游同款的词级读音修正表（4.5 万词）之后已经是两个 háng。
  文档把中间层当成了最终结果。

  | 层 | 银行分行下午三点关门 | 行长在银行上班 |
  |---|---|---|
  | g2pM 原始输出 | yín háng fēn **xíng** | xíng zhǎng zài yín **xíng** |
  | + 词级修正表（上游同款） | yín háng fēn **háng** | xíng zhǎng zài yín **háng** |

  连带修正了同源的几处表述：审计 §9.4 的偏差表、§9.7 的"忠实保留上游怪癖"段
  （那里还顺带把"银行分行下午三点关门"写成模型读 háng —— 实际模型给"分行"
  读 xíng），以及单测 `G2pm.DisambiguatesByContext` 的注释（那是**模型层**的
  判定，修正表会把它改回来 —— 见 `G2pmFrontend.UsesContextReadings`）。

  上游是同样的"模型 → 修正表"两层结构，所以最终读音与上游一致。

## [v0.3.0] - 2026-10-30

### Fixed

- **变调 oracle 的边界比对是恒真的**（本次同步文档时发现）：
  `tools/verify_sandhi.py` 的 `flatten_units()` 把词面**拼接**后再比，
  而拼接结果恒等于原文 —— 于是"分词边界一致"这个指标永远为真，
  一次都没抓到过真实差异。审计 §9.6 里"变调域边界 44/44 一致"这个数
  就是这么来的。现在按**词切分**逐词比。修好后重测，两条路径的真实数字是：

  | 路径 | 边界一致 | 韵母序列一致 |
  |---|---|---|
  | jieba（生产路径） | 40/44 = 90.9% | 42/44 = 95.5% |
  | 降级路径（无 jieba） | 17/44 = 38.6% | 30/44 = 68.2% |

  换句话说：**带 jieba 的音色包**比不带 jieba 的更接近上游 ——
  14 处韵母差异里 12 处由 jieba 直接消掉。

- **文档与代码脱节（jieba 已在 #24 合入，文档还写着"本移植不引入 jieba"）**：
  §9.6 / §9.7 / §9.8 与 `src/text/tone_sandhi.hpp`、
  `src/text/chinese_frontend.cpp`、`src/engine/g2pm.{cpp,hpp}`、
  `src/text/frontend_factory.cpp` 的注释都按"没有 jieba"写的，
  与实际行为相反。现在全部改成"有 jieba 走生产路径、没有走降级路径"，
  并把 §9.6 的数字换成实测值。

- **三处真实的编码损坏**（U+FFFD）：`include/genie/t2s.hpp`、
  `third_party/openjtalk/CMakeLists.txt`、
  `tools/g2p/generate_pinyin_table.py` 各有一个字被替换成替换字符。

- **`project(... VERSION 0.1.0)` 没跟着发版走** —— v0.2.0 发布时也忘了改，
  于是 `genie-tts version` 一直报 0.1.0。现在 bump 到 0.3.0。

- **v0.2.0 一直没在 CHANGELOG 里单列一节** —— 它的内容一直堆在
  `[Unreleased]` 下面。现在按内容拆回 `[v0.2.0] - 2026-10-28]`。

- **打包脚本漏拷 `t2s_shared_fp32.bin`**：`make_voice_package.py` 的
  `TTS_WEIGHTS` 少了这一项 —— 而它是 `t2s_first_stage_decoder` 与
  `t2s_stage_decoder` **两张图共用的**那份 external data ✓，
  图里的 `location` 写死了这个文件名 ✓。漏掉的后果不是打包时少一个文件 ✓，
  而是**推理时**才报 "no such file: t2s_shared_fp32.bin" ✓ ——
  那时候人已经在调模型参数了 ✓，很难联想到是打包脚本漏了 ✓。
  现在它被**列为必需** ✓：缺了当场报错 ✓，并说清楚是哪个角色的 external data ✓。
  顺带把另外几份外部权重也改成必需 ✓（原来缺了是**静默跳过** ✓）✓，
  V2ProPlus 的 prompt_encoder 权重跟着 prompt_encoder 图走 ✓。

- **非中文路径无条件跑 RoBERTa**：RoBERTa 是**中文**模型 ✓，
  上游的 `_get_phones_and_bert_pure_lang` 里 english / korean / japanese
  三个分支给的都是**全零** ✓。本项目以前不管什么语言都跑 ✓ ——
  音色包里带了 RoBERTa 又合成英文/韩文/日文时 ✓，
  RoBERTa 的 token 行数与音素数对不上 ✓ → `expand_by_word2ph` 直接抛异常 ✓，
  合成整个失败 ✓。现在只有中文（含中英混读）路径跑 RoBERTa ✓，
  其余语言走全零 ✓（与上游一致 ✓）。

- **缺词典时静默回退到中文前端**：合成时按语言选前端 ✓，
  以前是"选不到就退回中文前端" ✓ —— 日文没有 OpenJTalk 词典时 ✓，
  日语文本被中文前端的 `pattern_filter` 过滤掉 ✓，
  合成出来是**静音** ✓，而真正的原因（缺 103 MB 的词典 ✓）没人看得见 ✓。
  现在选不到前端就**明确报错** ✓，并说清楚缺哪个角色、怎么接 ✓。
  同一条链路上的另一处：**参考文本**以前固定用中文前端转音素 ✓，
  日文音色包的参考文本会被转成拼音 ✓ —— 现在按**音色包声明的语言**选 ✓
  （与上游 `ReferenceAudio.set_text(prompt_text, language)` 一致 ✓）。
  **没编 OpenJTalk 的构建**（MSVC 上默认如此 ✓）同样不回退 ✓ ——
  那时候日文前端根本不存在 ✓，回退就等于把日文过滤成静音 ✓。

- **测试替身里的除零**：`tests/support/fake_session.hpp` 的 HuBERT 分支
  用 `sum / wave.size()` 算填充值 ✓，波形为空时是浮点除零 ✓ →
  特征里全是 inf/NaN ✓，后面的断言会以"数值不对"的形式失败 ✓，
  根因却完全看不出来 ✓。现在空波形直接填 0 ✓。

- **日文前端"只声明不装配"**：#28 把"缺词典静默回退"改成了明确报错 ✓，
  但 engine 的日文前端成员一直只声明未赋值 ✗ —— 音色包即使带了
  `japanese_dictionary`，`frontend_for(Japanese)` 仍返回 nullptr ✓，
  日文从"合成静音"变成"报错但仍合成不了" ✓。现在加载期按
  `ModelRole::JapaneseDictionary` 显式装配日文前端 ✓；
  `make_voice_package.py` 同步把词典目录登记进 manifest（目录条目不带
  sha256 ✓ —— engine 靠 manifest role 找词典路径 ✓，不登记的话包里有
  词典也等于没有 ✓）。新增回归测试 `Engine.JapaneseWithDictionarySynthesizes`
  （需 `GENIE_TEST_JAPANESE_DICT` 指向真实词典 ✓，缺失时 skip ✓）。

- **`english_word_segment` 打包成死重**：`make_voice_package.py` 把 ngram
  分词词典目录拷进包里 ✗，但没登记进 manifest ✗ —— engine 是靠
  `has(EnglishWordSegment)` 这个 role 找它的 ✓，不登记则 OOV 分词器
  永远不可用 ✓，英文遇到词典外词时**静默降级到 GRU 兜底** ✗，跳过
  "ngram 分词再查词典"路径 ✓（§17）。现在目录条目同样登记进
  manifest（不带 sha256 ✓）。

- **移除两个源文件末尾的 NUL 填充**：`src/text/language_split.cpp` 与
  `tools/g2p/dump_language_split.cpp` 的代码内容完整，但末尾被工作区文件
  系统填充 NUL 到 4096 块边界（约 2 KB），导致 git 将其识别为二进制文件、
  无法在 diff 中审阅。编译器与 clang-format 忽略 NUL（此前 CI 全绿的
  原因），#21 已合并入库，本条将其剥离为纯文本。

### Added

- **jieba 生产路径的黄金语料**：`tests/golden/sandhi_jieba.json`
  （44 段）锁住"音色包带 jieba 角色"时的变调/儿化行为 ——
  之前只有降级路径的黄金语料，生产路径没人守。
  配套单测 `SandhiGolden.JiebaCorpusMatchesLockedOutput`。
  `tools/g2p/dump_sandhi.cpp` 新增 `GENIE_JIEBA_MODEL` 环境变量切换分词源。

- **日语 G2P 实装**：移植上游 `G2P/Japanese/JapaneseG2P.py` 的整条链路 ——
  文本规范化（% / ％ → パーセント、折叠连续标点、Unicode 小写）、
  按非日语字符切段、OpenJTalk 全上下文标签 → 音素 + 韵律符号
  （`#` / `[` / `]`）、全角标点转半角。
  **OpenJTalk 的运行时以源码形式内置**（`third_party/openjtalk/`，
  BSD 3-Clause ✓，取自 `r9y9/open_jtalk` 的 `1.11` 分支 ✓ —— 与上游
  用的 pyopenjtalk 同源 ✓）。
  词典约 103 MB ✓ —— **不入仓库** ✓（与英文的 wordsegment 词典同理 ✓），
  由 `tools/deploy/fetch_openjtalk_dictionary.py` 从 pypi 的 openjtalk
  wheel 取出（附 SHA-256 校验 ✓），运行时从音色包新增的角色
  `japanese_dictionary` 读取 ✓。
  CLI 新增 `--japanese-dict <目录>` ✓。
  oracle 用**上游的 Python 实现**（pyopenjtalk ✓）当基准，
  124 句语料逐音素一致 ✓。
  顺带修掉 CLI 的一个 bug ✓：`--english-dir` 的分支排在了"未知选项"
  之后 ✓，于是永远走不到 ✓（直接报 unknown option ✓）。
  见 docs/upstream-audit.md §19。

- **韩语 G2P 实装**：移植上游 `G2P/Korean/KoreanG2P.py` + g2pk2 的整条链路 ——
  拉丁字母转韩文、数字转韩文（汉字数词 / 固有数词）、惯用语表、
  组合字母拆解、12 条特殊音变规则、402 条"받침 + 起首辅音"规则表、
  4 组连音规则、重新组字、兼容字母输出。规则与数据全部内置
  （`src/text/korean/`，由 `tools/g2p/generate_korean_tables.py` 从 pip 上的
  `g2pk2` / `jamo` 包导出 ✓，Apache-2.0）。引擎按检测到的语言选前端：
  韩文文本走韩文前端，不需要任何音色包资源。
  oracle 用**上游的 Python 实现**当基准（mecab-ko 换成 stub ✓），
  53 句语料逐音素一致。
  **偏差**：g2pk2 的 mecab-ko 形态标注那一步没有移植（mecab 是 C++ 库 +
  100 MB 词典 ✓），依赖标注的四条音变规则因此不触发 ——
  记录在 docs/upstream-audit.md §18.3。

- **英文 G2P 实装**：移植上游 `G2P/English/` 的整条英文链路 ——
  规范化（20 多条规则，手写扫描器，不引 `std::regex`）、nltk TweetTokenizer
  分词、CMU 发音词典查表、averaged perceptron 词性标注、同形异音词按 POS
  选读音、ngram 分词（OOV 兜底的 GRU 前向）。
  数据经 `tools/deploy/convert_cmu_dict.py` / `convert_pos_tagger.py` /
  `convert_english_g2p.py` 转为容器格式随仓库发布
  （`third_party/english/`，共 12 MB，CMU BSD / NLTK Apache-2.0 /
  g2p Apache-2.0）。音色包新增四个可选角色 `english_cmu_dict`（硬依赖）、
  `english_pos_tagger`、`english_g2p_model`、`english_word_segment`。
  引擎按语言选前端：纯英文走英文前端，混读路径的英文块也交给英文前端
  （此前是整块丢弃）。oracle 用**上游的 Python 实现**当基准，
  94 句语料的规范化 / text_normalize / 音素序列三项逐条一致。
  见 docs/upstream-audit.md §17。

- **jieba 分词实装**：移植 jieba 0.42.1 的分词算法（词典 DAG 前向扫描
  + 对数概率逆序 DP + OOV 的 HMM 细分，jieba_fast 与其结果一致），
  词典与 HMM 概率经 `tools/deploy/convert_jieba.py` 打包为
  `third_party/jieba/jieba.bin`（8.3 MB，MIT，随仓库发布）。oracle 与
  jieba Python 103 句语料逐词一致。音色包新增可选角色 `jieba`
  （`make_voice_package.py --jieba`）：分词边界成为变调合并与词级读音
  修正的词边界（与上游流水线一致）；缺失时退回词表贪心匹配 +
  整段最长匹配修正（#23 的降级路径）。POS 依赖的变调分支留待
  下一切片（posseg HMM）。见 docs/upstream-audit.md §9.8。

- **多音字 g2pM 实装**：上游用 g2pM（kakaobrain 单层双向 LSTM，1.6 MB）
  做整句上下文的多音字判定；本移植逐行复刻其 numpy 前向（C++，无 ORT
  依赖），权重经 `tools/deploy/convert_g2pm.py` 转成单一容器随仓库发布
  （`third_party/g2pm/g2pm.bin`，Apache-2.0）。音色包新增可选角色
  `g2pm`（`make_voice_package.py --g2pm`），缺失时回退单字最常用读音。
  oracle 与上游 `G2pM()` 47 句语料逐字一致。附带上游的词级读音修正表
  （polyphonic.pickle，4.5 万词，随容器发布）：g2pM 的已知误判
  （"长大"读 cháng、"银行分行"读 fēn xíng）由词表兜底修正，
  与上游 ChineseG2P 的流水线位置一致。
  同步勘误审计：上游 `word2ph` 实为逐字统计，jieba 影响的是 POS 变调
  分支与词级读音修正，非 word2ph 粒度（docs/upstream-audit.md §9.7）。

- **中英混读分块（`hybrid-chinese-english`）**：移植上游 `split_language`
  （`re.split(r'[a-zA-Z]+')` 切块，与上游正则逻辑 24/24 语料逐块一致），
  每个中文块各自走完整 TextNormalizer + G2P，RoBERTa 按块编码再拼接
  （`PhonemeSequence::bert_chunks` 携带分块与 word2ph 对齐）。
  此前混读文本走整段简化规则，中文块里的日期 / 电话读不到正确读法
  （例：`"Mira打13800138000了"` 现在正确读"幺三八…"）。
  块边界阻断变调，与上游逐块 G2P 的行为一致。
  英文块暂且丢弃（英文 G2P 未实现，记录在案 docs/upstream-audit.md §9.4）。
  `genie-tts g2p` 对混读文本改为按块显示规范化结果（整段显示与实际
  喂给模型的分块不一致）。

- **部署教程与部署脚本**：`docs/deployment.md` 手把手覆盖「下载上游权重
  （HF `High-Logic/Genie`，含 hf-mirror 镜像用法）→ fp16 外部权重转 fp32 →
  组装音色包 → 校验与首次合成」全流程，命令均可直接复制执行。
  `tools/deploy/convert_fp16_weights.py`：整文件 fp16→fp32 转换，可选
  ONNX 图外部区间校验（连续性 + 总长精确匹配，不一致拒绝转换）；
  `tools/deploy/make_voice_package.py`：按上游目录结构组装音色包
  （改名 + SHA-256 + manifest.json，capabilities 按输入自动置位）。
  README 增加部署入口、模型来源说明与 tools 目录树。

- **上游行为入档：句尾挤压**：vits 输出时长严格等于语义 token 数 × 40 ms，
  EOS 采样时机（采样在 ONNX 图内，见审计 §5）的运行间方差会让末音节被
  挤压，听感是"说了一半音频就结束"。实测确认与上游同分布、非移植缺陷；
  缓解方案（自动重采 / 尾部补留白）留待 M6 单独立项。
  见 docs/upstream-audit.md §10.5。

- **中文变调与儿化（ToneSandhi / Erhua）**：对齐上游 `ToneSandhi.py` /
  `Erhua.py` 的"不/一"变调、三声连读、轻声词表与儿化处理 —— "你好"
  读"ní hǎo"、"不怕"读"bú pà"、"一天"读"yì tiān"、"三个"的"个"
  轻声、"媳妇儿"儿化。上游依赖 jieba 分词与词性标注；本移植内嵌
  上游四张词表（约 510 词）做贪心最长匹配，POS 依赖分支取记录在案的
  默认行为。oracle 比对（上游 Python 当基准）：**接上 jieba 之后**
  变调域边界 90.9% 一致、逐字韵母 95.5% 一致（数字在 2026-10-30
  修好边界比对逻辑后重测过一次，见本版本 Fixed 第一条）。
  见 docs/upstream-audit.md §9.6（设计对齐 issue #16）。

### Changed

- **审计 §9.6 重写**：补上两条路径的实测数字、剩余 6 处差异的逐条成因
  （4 处边界差异来自上游用 `jieba_fast.posseg.lcut` 而本移植对齐
  `jieba.lcut` —— posseg 走的是带 POS 约束的 HMM；2 处韵母差异是
  `_split_word` 的 `cut_for_search` 切点与"慢慢"的 POS 依赖分支）。
- **路线图按代码逐项核对**：M1 / M2 / M3 转 ✅，M0 / M4 / M6 / M7 转 🚧；
  补记"采样在图内、无主机侧采样器"、"word2ph 上游也是逐字统计"等。
- **发布检查单重写**：从 v0.1.0 口径更新到 v0.3.0，
  并把已知限制分成"能力边界 / 验证边界 / 不影响正确性的取舍"三类。
- **README 的里程碑表**：M2 从 🚧 转 ✅（四种语言的 G2P 全部落地），
  并补上已发布版本列表。

## [v0.2.0] - 2026-10-28

### Added

- **完整中文文本规范化**（对齐上游 `TextNormalizer`）：日期 / 时间 / 时间范围、
  手机号 / 固话 / 400 号、分数 / 百分比 / 区间 / 版本号 / 幂、量词 / 单位 / 温度、
  繁简转换、数学运算、希腊字母、带圈数字。

  少了这一步，合成出来的中文**读法就是错的**：
  `2024年` 读成"二千零二十四年"（上游读"二零二四年"）、
  `13812345678` 读成"一百三十八亿一千二百三十四万五千六百七十八"
  （上游读"幺三八幺二三四五六七八"）。

  验证方式：把上游的 Python 包当 **oracle**，63 条语料逐条比对，全量一致
  （`tools/verify_normalizer.py`），语料固化在 `tests/golden/normalizer.json`。

- **CLI `normalize` 子命令**：只看规范化这一步的结果，方便排查读法问题。

- **参考音频内容特征（chinese-hubert-base）**：音色包新增可选
  `chinese_hubert` 角色（ONNX 图 + 同目录 fp32 外部权重，部署侧由
  上游 fp16 发布文件转换），`ssl_content` 从全零降级升级为真参考
  音频 HuBERT 特征（[1, 768, T']，图内已含转置）。审计与排查
  记录见 docs/upstream-audit.md §14.4。

- **中文韵律特征（RoBERTa）**：音色包新增可选 `roberta` 角色 +
  同目录 `vocab.txt` 词表，中文合成的 `text_bert` / `ref_bert`
  从全零降级升级为真 RoBERTa 特征 —— tokenizer 编码规范化文本、
  推理、按 word2ph 展开（判定顺序照抄上游 `_expand_roberta_output`）。
  分词器自带 C++ 实现（BertNormalizer → BertPreTokenizer → WordPiece），
  与 HuggingFace tokenizers 0.23.3 逐 ID 黄金对齐
  （`tests/golden/roberta_tokenizer.json`）。没有 roberta 角色时
  保持全零降级，与上游一致。审计记录见 docs/upstream-audit.md §14。

### Fixed

- **折叠连续标点按字节做，把多字节汉字吃掉半个**：
  "四"（E5 9B 9B）后两个字节相同，按字节折叠会变成 E5 9B —— 输出乱码。
  现在按**码点**折叠（`src/text/punctuation.hpp`）。

- **字符过滤按字节做**：把"月"（E6 9C 88）的 0x88 当成全角括号的一部分删掉，
  结果是"月"变成半个字。现在按码点过滤。

- **中文音素转换系统性跑偏（重大，读字不准的根源）**：音节 → 音素
  一步此前用自推规则替代上游的 opencpop 词表查询，导致
  翘舌韵母全部错成普通 i（十/是/字/次…约 1/6 常用音节）、
  `üe` 韵母丢字（"月"整个没有韵母）、"二"丢 EE 声位、
  "语"的 ü 错成 u。已内嵌上游 opencpop-strict.txt 词表（429 条，
  `tools/g2p/generate_opencpop_table.py` 生成）并逐条对齐
  `_pinyin_to_opencpop_phones`；12,010 字与 Python 参考实现
  （pypinyin + 同一词表）全量比对 0 差异。
  转换摘要与黄金契约快照同步更新。

- **参考音频缺少上游的静音尾巴（重要）**：上游 `load_audio()` 在重采样后
  给参考波形拼接 0.3 秒静音（`SILENCE_TO_APPEND_S`），`ref_audio` 与
  HuBERT 特征都带这个尾巴；静音帧向自回归解码器标记"参考内容结束"。
  本实现此前漏掉拼接，接通真 HuBERT 特征后 T2S 在第 1 轮即采样到 EOS，
  输出退化成参考音频的复读。已按上游补齐（32 kHz 下 9600 采样），
  黄金契约快照同步更新（ssl 50→65 帧、ref_audio 32000→41600 采样）。

- **`Engine(root, config)` 两参构造未装配会话**：它在函数体里调用 `Engine(...)`
  构造了一个临时对象又立刻析构，当前对象的 `impl_` 保持未初始化 ——
  一参构造（CLI `synth` 走的路径）把请求委托给它，首次合成必然段错误。
  改为初始化列表里的委托构造，并新增走 ONNX Runtime 真实加载路径的
  集成测试锁定（构造后能读到 manifest 与采样率）。

- **`ssl_content` 零张量长度与上游不一致**：上游帧数来自 16 kHz 波形
  （HuBERT 步长 320 → 50 帧/秒），此前按 32 kHz 帧数除 320，长度多了一倍。
  长度参与 t2s_encoder 的注意力归一化，会稀释韵律注意力 ——
  这也是部分合成结果可懂度差的嫌疑来源之一。
  详见 docs/upstream-audit.md §11。

- **CLI 线程超额订阅**：Engine 会创建 4-6 个 ORT 会话，每会话默认
  开满全部核心的线程池，多核机器上合计上百个自旋线程，高负载时段
  在 ORT 内部出现段错误/浮点异常。CLI 现在默认每会话 8 线程，
  新增 `--threads <n>` 可覆盖。

- **`--seed` 的诚实化**：ONNX Runtime 没有公开图内采样器的种子接口，
  seed 一直以来就无法让结果可复现。CLI 帮助与 docs/usage.md 如实标注
  该限制，并新增 `--threads` 的文档；README 相应说法同步修正。

## [v0.1.0] - 2026-10-23

首个版本。完整 C++ 推理链路 + C ABI + CLI，不依赖 Python / PyTorch。

### Added

- **V2ProPlus 全局说话人嵌入**：`speaker_verification`（16 kHz 波形 → `sv_emb`）
  接 `prompt_encoder`（32 kHz 波形 + `sv_emb` → `ge` / `ge_advanced`），
  与上游 `ReferenceAudio.update_global_emb` 同序。
  说话人验证模型缺失时**在加载期报错** —— V2ProPlus 的硬依赖。
- **参考音频条件**（`genie::load_reference`）：从音色包的 `reference/` 装配
  `ref_seq` / `ref_bert` / `ssl_content` / `ref_audio`。
  RoBERTa 与 HuBERT 缺失时退化为**全零特征**（与上游一致）。
- **C ABI 共享库**（`libgenie_tts.so` / `genie_tts.dll`）：纯 C 头文件，
  Go / Rust / Python ctypes 直接 dlopen 即可，不需要链接 C++ 运行时。
  新增 `genie_engine_synthesize_wav()` 与 `genie_phonemize()`。
  错误码与 C++ 侧 `genie::ErrorCode` 是同一套枚举值，编译期静态断言钉死。
- **合成引擎**（`genie::Engine`）：把音色包、模型会话、文本前端与调度层
  串成一次完整合成。`synthesize()` 同步，`synthesize_streaming()` 走 kairo
  执行器并通过 `AudioChunkChannel` 交付音频块。
- **T2S 自回归解码**（`genie::T2SDecoder`）：逐行对齐上游 `t2s_cpu()`，
  包括 `range(0,500)` 的下标语义、`y[0,-1]=0` 的置零、`y[:,-idx:]` 在
  `idx==0` 时全取这几个**容易写错**的细节。详见 docs/upstream-audit.md §10。
- **会话工厂扩展点**（`genie::SessionFactory`）：可替换推理后端，
  也让测试能注入假会话，在没有权重、没有 ORT 的环境里跑完整流水线。
- **音素符号表**（`genie::symbols`）：重新实现上游 `SymbolsV2.py` 的生成规则，
  732 个符号与上游**逐字一致**（顺序 + 全表摘要双重锁定）。
  符号 ID 错一位不会报错、只会合成出乱码，所以这一层必须硬锁。
- **拼音 → 音素**（`genie::pinyin`）：复刻 pypinyin 的 `to_initials()` /
  `to_finals_tone3()` 规则，已对 **41,923 个汉字**与 pypinyin 全量比对一致。
- **中文文本前端**（`genie::text::TextFrontend`）：汉字 → 音素符号 → 符号 ID。
  多音字取最常用读音；变调 / 儿化 / 分词未实现，差异记录在 docs/upstream-audit.md。
- **文本规范化**（`genie::text::normalize`）：全角→半角、阿拉伯数字→中文读法、
  标点映射与折叠、目标语言字符过滤。
- **内置 UTF-8 工具**（`genie::utf8`）：逐字符解码、码点编码、全半角转换。
- **CLI `g2p` 子命令**：不需要模型就能看到模型会收到的 text_seq。
- **音色包（voice package）**：`genie::VoicePackage` 解析并校验 manifest.json，
  支持按角色定位模型、**SHA-256 完整性校验**、必需角色的加载期检查。
  示例见 `examples/voice-package/`。
- **自带 JSON 解析器**（`genie::json`）：RFC 8259 最小子集，带嵌套深度上限。
- **自带 SHA-256**（`genie::sha256`）：用于模型完整性校验。
- **项目骨架**：CMake 体系（`CMakePresets.json` + `cmake/` 模块）、C++20 编译基线、
  告警策略、sanitizer 开关。
- **核心类型**：`genie::Error` / `genie::ErrorCode`、`genie::Tensor`（float32 / int64 / bool）、
  `genie::TensorSpec`、`genie::ModelSession` 抽象。
- **ONNX Runtime 后端**：按**名字**绑定模型输入输出（上游 Python 依赖位置对应，
  实际张量名不一致，详见 `docs/upstream-audit.md`）。
- **音频**：自带 RIFF/WAV 编解码（PCM 8/16/24/32 + IEEE float）、窗函数法带限 sinc 重采样。
- **调度与并发**：基于自有库 [kairo](https://github.com/Linductor-alkaid/kairo) 的
  `genie::Scheduler`（可取消任务）与 `genie::AudioChunkChannel`（有界流式音频通道）。
- **测试**：unit（140 例）/ component（模型 I/O 契约）/ regression（黄金数据）/
  integration（T2S 端到端链路）/ C ABI（纯 C 编译器编译）。
- **工程化**：CI（core / kairo / ort / asan+ubsan / tsan / windows / format）、
  `AGENTS.md` 管理规范、`CODEOWNERS`、PR 模板。

### Fixed

- **`text_seq` / `ref_bert` / `text_bert` 的秩错了**（秩 3 → 秩 2）。
  上游的真实 ONNX 图声明的是：
  - `text_seq: [1, text_length]`
  - `ref_bert: [ref_length, 1024]`
  - `text_bert: [text_length, 1024]`
  之前喂的是 `[1, 1, L]` / `[1, L, 1024]` —— 真模型会直接拒绝输入。
  注意 `pred_semantic` **确实是**秩 3（`[1, 1, L]`），别跟 `text_seq` 搞混。
- **引擎喂给模型的参考特征是空张量**：形状错的输入模型不会报错，只会输出垃圾。
  现在所有特征张量都显式构造，`text_bert` 与 `ref_bert` 也分开了
  （之前是同一个张量复制过去）。
- **调度器退出路径的 use-after-free**：kairo 的 `Executor` 是函数级静态，
  在 `Scheduler::initialize()` 里才被首次引用，因此比 `Scheduler` 单例
  **更晚构造、更早析构**。进程退出时 `Executor` 已经没了，而
  `Scheduler` 的析构函数仍会调 `request_task_cancel()` —— 症状不是崩溃，
  而是**挂住**（kairo 的 shutdown 最多等在途任务 300 秒）。
  析构现在只做 genie 侧的收尾（置取消标志 + 等任务收尾），
  不再触碰 kairo。
- **流式合成的生命周期**：后台任务捕获了引擎指针，而引擎可能在任务还在跑的
  时候被析构；音频通道也是裸指针，调用方先放手就会悬垂。现在引擎析构时
  会取消并**等待**任务收尾，通道改由任务与调用方共同持有（shared_ptr）。

### 已知限制

- **英文 / 日文 / 韩文 G2P 未实现** —— 只做了中文。接口留成了可插拔的
  `TextFrontend`，见 `docs/upstream-audit.md` §9.4。
- **多音字取最常用读音** —— 上游用 g2pM 做上下文判定，本项目用词频兜底。
- **变调 / 儿化 / 分词 / 繁简 / 日期时间量词未实现**。
- **RoBERTa / HuBERT 缺失时降级为全零特征** —— 能合成，但韵律会差。

### 已知差异（与上游 Genie-TTS）

- 上游把 T2S 采样（TopK / 温度 / 重复惩罚）实现在 ONNX 图内，本移植**不提供**主机侧
  `top_k` / `top_p` / `temperature` 参数 —— 这些是图内常量。唯一可控的随机性是
  ONNX Runtime 的全局随机种子。
- 重采样使用自带的窗函数法带限 sinc 实现，非上游的 `soxr 'hq'`。
  量化差异见 `docs/architecture.md`。
[Unreleased]: https://github.com/Linductor-alkaid/genie-tts-native/compare/v0.4.0...HEAD
[v0.4.0]: https://github.com/Linductor-alkaid/genie-tts-native/compare/v0.3.1...v0.4.0
[v0.3.1]: https://github.com/Linductor-alkaid/genie-tts-native/compare/v0.3.0...v0.3.1
[v0.3.0]: https://github.com/Linductor-alkaid/genie-tts-native/compare/v0.2.0...v0.3.0
[v0.2.0]: https://github.com/Linductor-alkaid/genie-tts-native/compare/v0.1.0...v0.2.0
[v0.1.0]: https://github.com/Linductor-alkaid/genie-tts-native/releases/tag/v0.1.0
