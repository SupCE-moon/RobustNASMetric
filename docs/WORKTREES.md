# 这个项目为什么有好几个文件夹

## 一句话

正常的项目只有一个代码文件夹。本项目因为过去两个月试过几条不同的技术路线，
每条都单独复制了一份出去改，所以现在磁盘上有好几个文件夹，装着**不同日期**的代码。

## 现在该用哪个

**只用这一个**：

```
D:\Project\RobustNAS\DynamicBN\DynamicBN20251203\robust_metric
```

它 2026-11-19 起检出 `codex/server-mainline-merge-20261112`，是最新代码，
`pytest tests -q` 应该是 746 passed / 2 skipped。

## 其余的在哪、是什么

`.worktrees\` 下面 5 个（都在 `robust_metric` 里面）：

| 文件夹 | 日期 | 是什么 | 能删吗 |
|---|---|---|---|
| `mainline-consolidation` | 11-19 | 我这两周干活的地方，已 detach 到同一提交 | 可以（内容都在主线里） |
| `server-mainline` | 09-18 | 服务器归档时的版本 | 可以 |
| `three-space-benchmark` | 09-22 | 三空间实验 | 可以 |
| `activation-survival` | 10-31 | 激活存活指标实验 | 可以（含 16.9 MB 官方数据，删前确认） |
| `repository-reconciliation` | 11-05 | 对账设计文档 | 可以 |

`C:\Users\22465\.codex\worktrees\` 下面还有 3 个（`adv-excess-metric`、
`ba02`、`strict-metrics-e37`），是更早的 codex 工作区，也可以删。

## 为什么当初要复制

因为一次只想改一条路线，又不想弄坏正在用的版本。git 的 worktree 机制就是为了
这个——同一个仓库可以同时摊开多份，互不干扰。

代价就是现在这样：容易搞混"我看的是哪一份"。

## 清理建议

删之前确认三件事：① 里面没有未提交的改动（`git status` 空）；② 内容已并入主线
（`git log mainline-consolidation..HEAD` 为空）；③ 没有只存在于那里的数据文件。

建议等服务器那边完全接手实验之后再删，那时这些备份就没有价值了。
