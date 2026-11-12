# 仓库对账清单 2026-11-12

本目录记录"服务器 / 本地 worktree / 多个分支 / GitHub"四方对账的第一步：**本机一侧**的
完整盘点和安全导入。服务器一侧尚未执行（等 SSH 凭据），因此这不是最终对账结论。

## 1. 数字总览

| 项目 | 数量 | 说明 |
| --- | --- | --- |
| 参与盘点的 worktree | 9 | 见 `local-worktrees.json` |
| 脏 worktree | 4 | 主工作树、adv-excess、three-space、本整理分支 |
| 主工作树未提交改动 | 180 | 6 个已跟踪修改 + 174 个未跟踪文件 |
| 判定为可导入 | 124 | 2.19 MiB，已复制并逐文件校验 SHA-256 |
| 判定为暂缓（留在服务器/本地） | 49 | 原始逐样本行、超大合并表、二进制表格 |
| 判定为拒绝 | 1 | Office 锁文件 |
| 只以补丁形式保留 | 6 | 已跟踪但未提交的修改 |

## 2. 来源与追溯

- 来源工作树：`D:\Project\RobustNAS\DynamicBN\DynamicBN20251203\robust_metric`
- 来源 HEAD：`784fc74333`（分支 `main`，领先 `origin/main` 7 个提交，领先 `new-origin/main` 0 个）
- 逐文件字段：`path`、`status`、`action`、`reason`、`size`、`sha256`
- 完整判定表：`import-plan-main.json`（每个文件一条，含被拒绝和暂缓的原因）

导入规则写在 `tools/import_worktree.py`，判定只依赖"路径 + 大小"，不做语义猜测，因此可复现：

```bash
python tools/import_worktree.py --source <源工作树> --dest . \
    --plan results/manifests/reconciliation-20261112/import-plan-main.json
```

加上 `--apply` 才会复制；复制后重新计算目标文件哈希，与源哈希不一致立即中止。

## 3. 导入了什么

- `docs/`：22 份实验报告与设计文档，包括 TRNAS 修复、E36/E37 严格跨空间最终报告、
  独立验证边界、RB40 留出权重恢复、e25–e35 系列验证记录。
- `scripts/`：20 个分析与训练脚本。
- `src/`：`nb201_space.py`（`e36_batched.py` 被暂缓，见下文）。
- `tests/`：3 个测试文件。
- `tools/`：研究进展文档生成与校验工具。
- `outputs/`：124 项中的小型结果表与报告（严格总表 CSV、各实验汇总 CSV、`*_report.md`）。

## 4. 刻意没有导入的内容

| 内容 | 原因 |
| --- | --- |
| `outputs/**/per_source/*.csv` | 逐模型原始行，体积随实验线性增长，属于服务器产物，靠 manifest 记录 |
| 两个 13.94 MiB 的 `metrics_values_*.csv` | 合并原始值表，超过 10 MiB 导入舒适阈值，可由服务器原始行重建 |
| `*.xlsx`、`preview.png`、`*.inspect.ndjson` | 二进制或工具转储，CSV 版本已在库中 |
| `pytest_temp_full/**` | pytest 临时目录，不是实验结果（7 个文件） |
| `node_modules/**` | 生成表格时安装的 JS 依赖（含 19 MiB DLL） |
| `~$*.xlsx` | Office 锁文件 |

## 5. 只以补丁形式保留的 6 个文件

主工作树有 6 个**已跟踪但从未提交**的修改，其中 `src/robust_metric_lab/metrics.py`
新增 571 行（`perturbation_capacity_iqr/multiscale/topk`、`jacobian_spectral_*` 等）。
这些内容在服务器上可能有更新版本，因此**没有**直接落到本分支，而是保存为：

- `patches/main-uncommitted-784fc74-20261112.patch`（83,983 字节）
- SHA-256：`d3329f9ca31cf5a1a0a83b39b3e8a22c14db6f6cda707e66c8a4d51680f3a0d8`

被暂缓的 `src/robust_metric_lab/e36_batched.py` 与 `tests/test_e36_batched.py` 依赖该补丁里的
`_e36_cross_layer_stats` / `_e36_within_layer_stats`；单独导入会让测试变红，因此成对保留在
wip 快照分支上。另有 5 个 `scripts/analyze_e2x_*.py` 引用补丁中的新指标名，它们作为"结果如何产生"
的证据保留，但在补丁合并前不可直接运行。

## 6. 已知局限

- 只覆盖本机；服务器 `/workspace/robust_metric_lab_nas` 尚未盘点，因此"谁是主线"仍待确认。
- 判定是机械规则，不做内容语义判断；同名不同内容的文件需要人工复核。
- `local-worktrees.json` 中 `adv-excess-metric` 的 4,506 个未跟踪文件绝大多数是 `node_modules`，
  已按策略标记，未导入。

## 7. 复现本次盘点

```bash
python tools/inventory_worktrees.py --repo <仓库根> --output <输出.json>
python tools/inventory_tree.py --root <目录> --output <输出.json>
python tools/import_worktree.py --source <源> --dest . --plan <计划.json>
python -m pytest tests -q   # 57 passed
```
