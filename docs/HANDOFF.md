# 交接说明（HANDOFF）

写给"下一个接手的人"（可能是几周后的我，也可能是你）。目标：**不读完整历史也能安全开工。**

最后更新：2026-11-12，分支 `codex/mainline-consolidation-20261112`。

---

## 1. 三十秒了解现状

- 项目研究"特征空间稳定性指标能否预测对抗鲁棒性"，目前 84 项严格榜里 TRNAS Φ 第 1，自研指标第 6。
- 代码分散在 **9 个 worktree + GitHub + 服务器**，四者版本不一致，主线尚未确定。
- **服务器现在连不上**（17010/17018/17013 均 connection refused），这是当前第一阻塞。
- 本机主工作树有 571 行未提交的指标代码，已保存为补丁，未合并、待评审。

细节看 [`docs/PROJECT_STATUS.md`](PROJECT_STATUS.md)（状态、结论分级、未完成问题）
与 [`docs/DECISIONS.md`](DECISIONS.md)（每条规则背后的理由）。

## 2. 开工前必读的三个坑

**坑 1：在 worktree 里跑测试必须设 `PYTHONPATH`。**

共享 `.venv` 的 editable 安装指向**主工作树**，直接 `pytest` 会静默地测另一份代码：

```powershell
$env:PYTHONPATH = "<当前worktree>\src"
& "<repo>\.venv\Scripts\python.exe" -m pytest tests -q
```

**坑 2：`git status` 默认折叠未跟踪目录。** 盘点必须加 `--untracked-files=all`，否则 4500 个文件
只会显示成 1 行，哈希根本拿不到。

**坑 3：中文文件名。** PowerShell 控制台会显示乱码，但路径本身没问题。看真实路径用
`git -c core.quotePath=false status`，或让 Python 以 UTF-8 输出。

## 3. 目录约定

| 位置 | 放什么 |
| --- | --- |
| `src/robust_metric_lab/` | 指标与攻击的实现（`metrics.py` 是核心） |
| `scripts/` | 一次性分析与训练脚本，名字对应实验编号 e25–e36 |
| `docs/experiments/` | 每个实验的结论文档，**结论以这里为准**，不以聊天记录为准 |
| `docs/theory/` | 指标定义与数学推导 |
| `outputs/` | 小型结果表（可入库）；逐样本原始行不入库 |
| `results/manifests/` | 盘点清单、补丁、哈希证据 |
| 服务器 | 大权重、数据集、checkpoint、逐样本原始输出 |

## 4. Git 规矩

- 不直接改 `main`；每件事一个分支（`codex/<主题>-<日期>`）。
- 不用 `git push --force`。
- 提交信息写"为什么"，不只写"改了什么"。
- 权重 / 数据 / `.venv` / 缓存 / 密码 / token 不入库；准入规则见 `tools/repository_policy.py`。
- 远端：`new-origin` = https://github.com/SupCE-moon/RobustNASMetric （主仓库）；
  `origin` = cduai/robust_metric（历史仓库，不要推）。

## 5. 盘点工具怎么用

```powershell
# 所有 worktree 的 HEAD、脏文件数与逐文件哈希
python tools/inventory_worktrees.py --repo <仓库根> --output <out.json>

# 任意目录的树清单（含 SHA-256）
python tools/inventory_tree.py --root <目录> --output <out.json>

# 把别的 worktree 的未提交成果安全搬过来（先 dry-run 出计划，再 --apply）
python tools/import_worktree.py --source <源> --dest . --plan <plan.json> [--apply]

# 服务器只读普查（密码走环境变量，不入库）
$env:ROBUST_METRIC_SSH_PASSWORD = "..."
python tools/server_inventory.py --host <ip> --port <port> --user <user> `
    --docker wangjiahao --output results/manifests/server.json
```

## 6. 现在该做什么（恢复访问后）

1. 跑 `server_inventory.py`，确认服务器代码与本地 9 个 worktree 的哈希差异。
2. 用哈希定主线：谁包含谁、谁更新。**不要用时间戳判断**（worktree 的 mtime 会因为 checkout 而变）。
3. 主线确定后，把本分支的 124 个导入文件与主线做一次并集，冲突处逐一核对。
4. 服务器大结果写 manifest：绝对路径、生成脚本、日期、大小、SHA-256。
5. 回填 `PROJECT_STATUS.md` 的"已验证事实"，并把本文第 1 节的阻塞项删掉。

## 7. 不要做的事

- 不要为了"指标好看"删掉失败或无效的指标（`E36_E37_严格实验问题与排除说明.md` 记录了排除规则）。
- 不要把相关性说成因果。
- 不要拿训练完成的模型与随机初始化模型直接比（攻击强度不可比）。
- 不要引用 2026-10-09 那版名次（已被 10-24 版取代）。
- 不要在 `main` 上"顺手"提交。

## 8. 2026-11-12 第二轮：主线已判定并合并（接手者必读）

上一版第 6 节把"定主线"当成未来任务，这件事已经做完，结论也改过一次，务必按新结论走：

- **代码主线 = `893a994`（`codex/strict-metrics-e37` 一线，2026-10-09）**，判据是
  `metrics.py` 的**函数集严格超集**（56 个函数 vs `dc6fb9c` 的 51 个），不是字节数、不是提交时间。
- **`dc6fb9c` 只是"服务器基线提交"**：服务器目录据记录从它导出，但它不是最新代码。
- 三条线都已并入 `codex/server-mainline-merge-20261112`：
  `dc6fb9c`（→ `7400d49`）、`893a994`（→ `92c5d10`）、`dfc67f2` 的 68 个文件 + `2bcb95c` 的 9 个未跟踪文件（→ `13c1d49`）。
- 九棵树对照现在只剩"已解释的残差"：数据集、xlsx 预览、超限 CSV、egg-info、`.tmp`。
  明细 `results/manifests/reconciliation-20261112/tree-comparison.{json,md}`，可一键重算（见第 5 节工具）。
- `pytest tests -q` = **477 passed / 3 skipped**。

两个新踩到的坑（按顺序看，能省你半天）：

1. **worktree 里套 worktree**：`ba02`、`strict-metrics-e37` 下面还有一层 `robust_metric/`。
   按 worktree 根盘点会得到 391 / 441 个假"独有文件"；要按内层目录盘点。
2. **换行符**：`server-mainline` 是 CRLF 检出的，56 个"内容不同"里 46 个只是换行符。
   比对前先归一化，否则差异被夸大 5 倍。

还剩的活（顺序不变）：服务器 SSH 一通就跑 `server_inventory.py`，把服务器停在哪个提交、有哪些大结果落成 manifest；
然后才谈"三空间独立留出集"和 `feature_activation_sign_survival_sum` 的终态。

---

## 9. 交付前自检：clone 一份再跑测试

在本机跑绿**不等于** GitHub 上是完整的。本机 `.venv` 是 editable 安装、指向主工作树，
9 个 worktree 共享它，很容易测到另一份代码（P2 坑）。交付前做这一件事：

```powershell
git clone --depth 1 --branch codex/server-mainline-merge-20261112 https://github.com/SupCE-moon/RobustNASMetric.git $env:TEMP\rm_clone
$env:PYTHONPATH = "$env:TEMP\rm_clone\src"
Push-Location $env:TEMP\rm_clone; pytest tests -q; Pop-Location
```

2026-11-12 实测：727 个文件 / 21.75 MiB，最大单文件 1.76 MiB，`477 passed, 3 skipped in 103.15s`。
文件数比 `git ls-files`（738）少 11 个，是被 `.gitignore` 排除的生成物，不是缺失。

配套的本地大产物清单在 `results/manifests/local/`（422 文件 / 1,107.2 MiB 全量哈希），
它告诉你"报告里的数字对应哪个具体文件"，但文件本体不入库。
