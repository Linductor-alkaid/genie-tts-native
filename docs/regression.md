# 回归层（Golden Dataset）

这个目录下的测试锁的是**喂给模型的东西**。

## 为什么需要它

上游的模型契约（输入名、dtype、形状、**秩**）一旦错位，模型**不会报错**——
它会安静地输出垃圾。这类 bug 单测抓不到，因为单测只验证控制流。

举个真实的例子（v0.1.0 开发期间真的踩了）：

| 张量 | 正确形状 | 曾经的错误 |
|---|---|---|
| `text_seq` | `[1, L]` | `[1, 1, L]` |
| `ref_bert` | `[L_ref, 1024]` | `[1, L_ref, 1024]` |
| `text_bert` | `[L_text, 1024]` | `[1, L_text, 1024]` |

三个 bug 都藏在"名字对得上、类型对得上、维度数看着差不多"里。
是回归层把它们挖出来的 —— 假会话开始校验形状之后，单测立刻报错。

## 锁的是什么

`tests/regression/golden/model_io_trace.json` 记录了**每次推理时每个模型的
全部输入**：名字、dtype、形状。

覆盖的用例：

| 用例 | 文本 | 覆盖的点 |
|---|---|---|
| `chinese_basic` | `你好，世界。` | 常规中文 + 标点 |
| `chinese_single` | `啊` | 单字、零声母 |
| `chinese_digits` | `一二三` | 连续单字 |
| `mixed_punctuation` | `你好！世界？` | 感叹号 / 问号 |

## 怎么用

正常跑：

```bash
ctest -R regression --output-on-failure
```

改了模型契约、文本前端或引擎结构之后，**先确认改动是有意的**，
再重新生成黄金数据并 **review diff**：

```bash
GENIE_UPDATE_GOLDEN=1 ctest -R regression
git diff tests/regression/golden/model_io_trace.json
```

黄金数据变了但没人说得清为什么 → 这就是回归层在提醒你：
这次改动动了模型的输入契约，需要有人签字。

## 假会话也守契约

`tests/support/fake_session.hpp` 的 `validate_feed()` 会照着真模型的 spec
校验每个输入的：

- 名字必须在模型的输入列表里
- dtype 必须一致
- **秩**必须一致
- 静态维度必须一致（`-1` 表示动态维，不校验）

所以假会话不是"随便返回点什么"的替身 —— 它是**契约的守卫**。
新增一个模型输入时，记得同时更新它的 spec。

## 上游真图在哪

`tests/component/test_model_contract.cpp` 直接解析上游的 ONNX 图
（`onnx.load(..., load_external_data=False)`），逐维核对输入输出。
那些图是 MIT 许可的，随仓库内置在 `tests/models/`。

真图是**权威**来源：本文档里的表都是从图里读出来的，不是抄来的。

## 精度门控（M0.2 容差框架 / M5 量化）

上面锁的是**输入的形状**；精度门控锁的是**数值本身**——它是量化、图优化
变更、后端升级这类"会改变每一位数字"的改动的裁决机制。实现在
`tools/gate/`（`genie-gate` C++ 捕获工具 + `precision_gate.py` 比对脚本）。

### 两步流程

1. **捕获**（C++，零引擎改动）：`genie-gate capture` 用包装会话工厂跑一次
   真实合成，把每个模型真实收到的输入与产生的输出逐张量落盘
   （裸二进制 + `capture.json` 元数据）。
2. **比对**（Python）：`precision_gate.py gate` 把变体模型放在捕获输入上
   重放，与捕获的 fp32 输出逐张量算指标，按 `tolerances.json` 裁决。

### 分级容差（AGENTS.md §5.3 的落地）

| 输出类型 | 裁决 | 说明 |
|---|---|---|
| token 类（`prompts` / `y`） | 逐位一致，**地板相对** | 采样在 ONNX 图内，fp32 自己重放也会翻近平局的 token（实测地板 ≈0.99）；变体用 `--floor-report` 跟 fp32 自身比，不得更差 |
| 连续张量（特征 / KV cache） | rel-L2 + 余弦双门槛 | fp32 跨构建重放的地板在 1e-6 量级，绝对容差天然覆盖 |
| 音频（vits `audio`） | advisory | 图内 z 噪声使波形运行间本就不同；听觉终审在 e2e 听测 |

### e2e 门控

`genie-gate e2e --baseline <pkg> --variant <pkg>`：同 run 序号配对的语义
token 前缀一致率 + 双份 WAV 听测。确定性事实（2026-11-11 实测）：
同进程里两个同配置引擎**逐 run 序号完全一致**（fp32 自比 WAV 逐字节相同），
跨进程会漂移——所以变体 e2e 永远跟基线自比（噪声地板）成对解读，
见 `precision_gate.py e2e`。

### 与 Python 上游黄金语料的边界

本门控的基线是 **C++ fp32 自基线**，回答"变体相对 fp32 掉了多少精度"。
M0.2 定义的"与 Python 上游逐级比对到 PCM"仍需要能下载上游权重的环境
（见 roadmap M0.2），那是另一件事——门控机制两者通用。
