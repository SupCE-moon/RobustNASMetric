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

# 不信任报告里的名次时：从原始行重算严格排名，与官方表逐值比
python tools/verify_strict_ranking.py --rows <原始行.csv> [--repair <repair_rows.csv>] `
    --ranking <strict_ranking.csv> --output results/audits/recompute.json --report results/audits/recompute.md
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

## 10. 还可用的一条独立验证：指标冗余度

`scripts/analyze_metric_redundancy.py` 不训练模型、不读服务器，只用已入库的 13.94 MiB 原始行
（哈希记录了但文件不入库）就能回答"84 项里有几个独立信号"。当有人质疑"84 项是不是同一件事数了 84 遍"时，
先跑这个脚本：`results/audits/metric-redundancy-20261112.md` 里给了 ρ ≥ 0.40/0.50/0.60 三档的簇结构。
注意报告里带 `⚠n/8` 的指标只覆盖部分设置，它们的相关系数不能和其他指标直接比。

## 11. 改了文档里的数字之后

如果你（或未来的我）在文档里改动了一个**榜单上的数**（分数、名次、两个指标之差），
不要只改文字。跑一遍：

```powershell
python tools/verify_doc_claims.py --table <86 项总表> --json <临时 json> --report <临时 md>
```

退出码 0 = 33 条声称仍然对得上；非 0 会直接打印 `MISMATCH <metric>.<field>: doc=... table=...`。
如果确实要改一个数（例如发布了新版榜单），先改 `tools/verify_doc_claims.py` 里的 `expected`，
再改文档——顺序反了就会永远红灯。

## 12. 指标索引与"声明 vs 事实"的警告

`docs/theory/METRIC_INDEX.md` 是自动生成的，**不要手改**——改完下次重建就没了。
要改指标本身：改 `src/robust_metric_lab/metric_registry.py` 或 `configs/metric_eligibility.json`，
然后重建索引。

如果你在索引里看到"⚠ 需要人看的不一致"一节，说明某个指标被资格表声明为 `implementation_error`，
却又在发布表上有名次。当前有 3 条（TRNAS 三项），是 10-24 修复前后的口径差，已在 D-016 记录。
**确认之前不要擅自把资格表改成 `cross_space`**，理由见 D-016。

## 13. 交接前最后一分钟：文档路径体检

`docs/` 里到处是「详见 `results/audits/<报告名>.md`」。这类承诺最容易随文件改名而失效，
而失效时没有任何报错。交付前请跑：

```powershell
python tools/check_doc_links.py `
  --json results/audits/doc-links-20261112.json `
  --report results/audits/doc-links-20261112.md
```

判据只有一条：输出里的 `missing_required` 必须是 **0**。其余三桶（`planned` / `external` /
`optional`）非零是正常的，报告里每一条都写了归入该桶的理由。

新增一条豁免（`ALLOWLIST`）时**必须写 reason**，格式见
`tools/check_doc_links.py` 顶部的字典：`(文档, 路径) -> 为什么它合理缺席`。
没有 reason 的豁免半年后就没人知道它是不是漏了。

改完文档不要忘了顺序：先跑体检 → 再跑 `tools/verify_doc_claims.py`（数字）→ 最后提交。

## 14. RB40 名次的置信区间（2026-11-12）

`tools/bootstrap_rb40.py` 按**模型**重采样，给 RB40 的相关系数与名次配 95% 区间。

    python tools/bootstrap_rb40.py \
      --rows outputs/strict_all_metrics_table_20261024_trnas_repaired/sources/metric_values_original_strict.csv \
      --repair outputs/strict_all_metrics_table_20261024_trnas_repaired/sources/repair_rows.csv \
      --table outputs/strict_adv_excess_release_20261024T110422Z/RobustMetric_严格全部指标总表_86项_三位小数.csv \
      --json results/audits/bootstrap-rb40-20261112.json \
      --report results/audits/bootstrap-rb40-20261112.md

**必须传 `--repair`**：三个 TRNAS 指标在 Kang2021Stable 上因下溢失败，修复值在
`repair_rows.csv`；漏掉它，这三个指标会各缺一个模型而被整列丢弃——恰好丢掉要看的指标。
（`--repair` 省略时会自动在同目录找 `repair_rows.csv`，但显式传更安全。）

判据：区间覆盖 0 就是**方向不知道**，不许写成"失效"；写结论时带上区间，不要只写点估计。
配合读：`results/audits/bootstrap-rb40-20261112.md`、D-022。
## 15. Φ 对漂移族的增量检验（2026-11-12）

`scripts/audit_phi_incremental.py` 回答的是"Φ 排第一，是不是 35 个漂移指标的另一种写法"。
判据写在 `docs/experiments/增量检验方案_Φ对特征漂移族_20261112.md`，**先于结果定死**（D-019）。

    python scripts/audit_phi_incremental.py \
      --rows outputs/strict_all_metrics_table_20261024_trnas_repaired/sources/metric_values_original_strict.csv \
      --redundancy results/audits/metric-redundancy-20261112.json \
      --json results/audits/phi-incremental-20261112.json \
      --report results/audits/phi-incremental-20261112.md \
      --resamples 2000

三个必须记住的点：

1. **架构身份是 `grid:id`**。DARTS 与 NAS-Bench-201 都把架构编号成 0..N，用裸 id 会把
   `darts:0` 与 `nb201:0` 当成同一个网络（独立单元会从 204 掉到 203）。
   测试 `test_build_frame_qualifies_arch_ids_by_grid` 守着。
2. **方向按注册表翻转**。漂移族是 `lower`；不翻转时 `ρ(B,y)` 是负的，读起来像"漂移越大越鲁棒"。
3. **Δρ 必须和 `ρ(B,y)` 一起读**。控制变量与 target 无关时，偏相关退化成 `ρ(Φ,y)`，
   这时高 Δρ 说明的是"基准失效"，不是"Φ 增量大"。真正有说服力的是 B 也强的设置
   （`nb201_e0`：ρ(B,y)=0.831，Δρ 仍 0.522）。

结论：Δρ = +0.695 [+0.583, +0.769]、p = 0.0005、嵌套 R² 增量 +0.476。
**可以说**"Φ 不是漂移族的重复"，**不可以说**"Φ 更好是因为它测一致性"（机制未验证）。
配合读：`results/audits/phi-incremental-20261112.md`、D-023。

## 16. 注册指标 ↔ 实现入口审计

`tools/audit_metric_implementation.py` 回答"每个注册指标到底在哪算出来"。

```
python tools/audit_metric_implementation.py --json out.json --report out.md            # 静态，秒级，只需 pandas
python tools/audit_metric_implementation.py --dynamic --json out.json --report out.md # 额外前向一次小 ResNet
```

- 输出三档：`one-pass bundle`（70）/ `other module`（15）/ `registry only`（1）。
- **退出码 1 是有效结果**：当前就有 1 个 `registry only`（`feature_adv_excess_centered_unit_l2`），
  这意味着注册表和实现之间有一个真实缺口，不要把它当成"工具坏了"。
- 新增指标时：先写进 `metric_registry.py`，再确认它出现在某一档里，然后跑这个工具。
- 看见 `perturbation_capacity_importance_delta` 是 NaN 不要慌，那是 bundle 的占位，
  真值在 `evaluate_importance_student.py`。

## 17. 指标不变量审计

```
python tools/audit_metric_invariants.py --json out.json --report out.md
```

- 秒级，不需要数据集、不需要 GPU；内置一个固定种子的两层卷积网络。
- 退出码 1 只在"零扰动不为 0"或"出现 NaN"时返回——这两种一定是错的。
- `decreasing` 只报告、不算失败：符号约定问题需要人判断（见 `docs/DECISIONS.md` D-025）。
- 改完任何指标实现后跑一次。新增指标如果名字以 `_delta` 结尾，会自动进入距离类检查；
  如果它本来就不该在零扰动时为 0（例如输入自身的属性），加进 `NOT_A_DISTANCE`。

## 18. 有符号 vs 绝对 capacity 审计（2026-11-12）

`tools/audit_signed_capacity.py` 回答「`perturbation_capacity_signed_delta` 与
`perturbation_capacity_delta` 是不是同一个指标」。答案是不是：设置内秩相关中位数 +0.017，
4/8 设置与 target 的相关符号相反。

**用法**：

```powershell
python tools/audit_signed_capacity.py `
  --rows   outputs/strict_all_metrics_table_20261024_trnas_repaired/sources/metric_values_original_strict.csv `
  --repair outputs/strict_all_metrics_table_20261024_trnas_repaired/sources/repair_rows.csv `
  --json   results/audits/signed-capacity-20261112.json `
  --report results/audits/signed-capacity-20261112.md
```

**三个必须知道的点**：

1. **`--repair` 不能省**。三个 TRNAS 指标在 Kang2021Stable 上因 float32 下溢是 `failed`，
   修复值单独存在 `repair_rows.csv`；`load_panel` 先 repair 再 dropna，否则要看的指标整列消失。
2. **退出码 1 表示「不是同一指标」**，这是本工具的正常输出之一，不是崩溃。
   CI 若把它当失败处理，需要显式接受这个语义。
3. **`|signed| <= abs` 是配对自检**，560 行必须零例外。不为 0 说明两列不是同一批样本，
   上面的相关全不可信。

**不要做**：不要因为它「单调下降」就去改注册表的 `direction`。那是改历史（D-021、D-025），
而且本轮已证明下降是「方向净额」的正常表现，不是符号写错。
## 19. 交付前跑一条命令（2026-11-12）

```powershell
python tools/run_audit_suite.py
```

它按依赖顺序跑完 11 个审计，写 `results/audits/audit-suite.{json,md}`。
判据：**`failed` 必须为 0**；`skipped` 不为 0 时，结论只能引用上次在完整数据上的运行。

- 两个步骤的退出码 1 是正常的（`implementation` 报已知缺口、`signed_capacity` 报「两者不同」），
  suite 已把它们写进各自的 `expect_exit`；报告里仍打印退出码，方便你核对。
- 大输入不在仓库里时，加 `--rows-root <主工作树>\outputs`，或设 `ROBUST_METRIC_OUTPUTS`。
- 只想跑几条：`--only doc_claims,doc_links`。

**不要**把这 11 条命令再手抄进别的文档。要改参数就改 `tools/run_audit_suite.py`，
否则两份参数迟早不一致。


## 20. RQ2 的零成本检验：RobBench 在哪里、怎么重跑（2026-11-12）

**数据位置**（不在仓库里，按策略不入库，必须用绝对路径）：

```
<主工作树或 activation-survival worktree>/data/official_nasrobbench201_source/
  NAS-RobBench-201_all.json   15.6 MiB，cifar10 下有 6466 个架构的训练后结果
  meta.json                    2.1 MiB，只有 id → nb201-string / isomorph
```

**重跑一条**：

```
python scripts/audit_early_metrics_vs_robbench.py `
  --rows <主工作树>\outputs\strict_all_metrics_table_20261024_trnas_repaired\sources\metric_values_original_strict.csv `
  --robbench <上面那个 json> `
  --oracle-table <主工作树>\outputs\e28_capacity_family\nb201_e0_v3.csv `
  --json results/audits/early-vs-robbench-20261112.json `
  --report results/audits/early-vs-robbench-20261112.md
```

`--oracle-table` 是可选的，给了就会做一次 **target 同源核对**（把本仓库的 `oracle_target`
与 RobBench 的 `autoattack` 按 arch_id 对齐，报匹配数、秩相关、最大绝对差），并在报告里
二选一措辞：是同一份数据（局限是「只有一套 target」），还是不是（局限是「target 选择本身
是不确定性来源」）。**结论不同，措辞必须跟着变**——不能不做核对就沿用旧措辞。

走 suite 更省事：`python tools/run_audit_suite.py --only early_vs_robbench`。
`resolve_robbench()` 会依次找 `--robbench`、`ROBUST_METRIC_ROBBENCH` 环境变量、
当前 worktree 与 `git worktree list` 里每个 worktree 的
`data/official_nasrobbench201_source/NAS-RobBench-201_all.json`。找不到时这一步标 `skipped`，
**不会**拿别的文件顶替。

**这个检验的判据**（改之前先读 D-028）：

- 只看**偏相关**（控制训练后 clean 准确率），不看原始相关。原始相关在这份数据里几乎必然很高，
  因为 clean 与 AutoAttack 的秩相关是 +0.982。
- 判定阈值写死在 `DIRECT_RHO=0.30` / `VIA_CLEAN_RHO=0.10`，改阈值等于改结论，必须走决策记录。
- 结论措辞只能是「相关/排序」，不能写「因果」或「训练早期就能预测」（跨流程，不是跨 target：
  目标与本仓库同源，见 `--oracle-table` 的核对输出）。

**下一步不是训练**：RobBench 有 6466 个架构，目前只有 64 个有 e0 指标。先把 e0 指标扩到
数百个架构（每个约 4 秒），比训练 16 个架构便宜得多，且能把 ±0.25 的区间压窄。
