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
