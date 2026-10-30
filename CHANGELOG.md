# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### Fixed

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

### Added

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
  默认行为。oracle 比对（上游 Python 当基准）：变调域边界 44/44 一致，
  逐字韵母 68.2% 一致，差异全部源于已记录的 jieba 词边界 / POS 缺失。
  见 docs/upstream-audit.md §9.6（设计对齐 issue #16）。
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

- **移除两个源文件末尾的 NUL 填充**：`src/text/language_split.cpp` 与
  `tools/g2p/dump_language_split.cpp` 的代码内容完整，但末尾被工作区文件
  系统填充 NUL 到 4096 块边界（约 2 KB），导致 git 将其识别为二进制文件、
  无法在 diff 中审阅。编译器与 clang-format 忽略 NUL（此前 CI 全绿的
  原因），#21 已合并入库，本条将其剥离为纯文本。

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

## [0.1.0] - 2026-10-23

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
[Unreleased]: https://github.com/Linductor-alkaid/genie-tts-native/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Linductor-alkaid/genie-tts-native/releases/tag/v0.1.0
