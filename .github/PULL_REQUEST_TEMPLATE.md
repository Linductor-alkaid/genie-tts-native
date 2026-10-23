## 提交前请确认

- [ ] 本次改动属于**一个**里程碑步骤（见 `docs/roadmap.md`）
- [ ] 所有测试通过：`ctest --preset debug --output-on-failure`
- [ ] 新增/修改的公开 API 同步更新了 `include/genie/` 下的头注释
- [ ] 行为变更同步更新了 `CHANGELOG.md` 的 Unreleased 段
- [ ] 没有把模型权重、音频样本、构建产物提交进来
- [ ] 提交信息符合下方的格式约定

## 提交信息格式

```
<type>(<scope>): <中文摘要>

<正文：为什么这么改，而不是做了什么>
```

- `type`：`feat` / `fix` / `refactor` / `perf` / `test` / `docs` / `build` / `ci` / `chore`
- `scope`：模块名，如 `runtime` / `text` / `audio` / `gpt` / `acoustic` / `scheduler` / `cli` / `abi`

示例：

```
feat(scheduler): 基于 kairo 接入可取消的流式音频通道

上一版用 std::thread + 裸 mutex 实现并发，多会话场景下线程数不可控、
取消语义要靠轮询共享标志。改用 kairo 的 Executor + MpscChannel 之后：
线程池由 kairo 统一管理，通道有界且无锁，取消走 StopToken。
```

## 关联

- 关联 Issue：使用 `#123`（如适用）
- 破坏性变更：正文中标注 `BREAKING CHANGE:` 并说明迁移方式
