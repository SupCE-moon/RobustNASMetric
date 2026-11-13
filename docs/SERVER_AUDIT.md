# 服务器只读普查（2026-11-13）

本轮把服务器 `/workspace/robust_metric_lab_nas`（容器 `wangjiahao`）与 GitHub 主线
`codex/server-mainline-merge-20261112` 做了第一次**逐字节对账**，并解决了此前一个
悬了很久的问题：「本地原始行和服务器发布包到底是不是同一份东西」。

结论先说：

1. **源码同源**。服务器 `src/robust_metric_lab/` 26 个模块里 25 个与本分支逐字节相同，
   唯一不同的 `metric_registry.py` 是「本分支多一条 adv-excess 注册」——因为那个补丁
   在服务器上打失败了（`.rej` 还在）。
2. **数据同源**。本地 47,600 行原始记录与服务端发布包的 85 个指标逐行同值
   （`metric_value` 最大绝对差 `0.000e+00`）。
3. **第 84 项指标补回来了**。取回服务器上 `feature_adv_excess_centered_unit_l2` 的
   560 行后，84 个入榜指标可以**全部**离线复算，`max_abs_error = 0.000e+00`、
   名次顺序完全一致。此前"只能复算 83 项"的限制解除。
4. 服务器上确实存着大量不该进 Git 的东西（48 GB，其中 9.7 GB 是 `student_metrics/`），
   已在下面列出清单，供 manifest 使用。

## 怎么连的

| 项 | 值 |
| --- | --- |
| 入口 | `guzhengkai@49.232.181.138:17010` |
| 主机密钥 | `ssh-ed25519`，指纹 `SHA256:6170fe70d1769091ea5c279f48f6ad5f`（与 `known_hosts` 匹配，未用 AutoAddPolicy） |
| 容器 | `wangjiahao`（`nvcr.io/nvidia/pytorch:24.10-py3`，Up 8 weeks） |
| 宿主机 | `516server`，`Linux 6.11.0-1025-oem`，uptime 91 天 |
| 方式 | 全程只读：`ls` / `sha256sum` / `docker exec ... python3` 读文件；下载只取两个 CSV |

密码由用户在本轮提供，只用于这一次连接，**没有写进任何文件、脚本或 Git 历史**。
临时脚本用 `.tmp_` 前缀（已被 `.gitignore` 忽略）。

一个操作细节：短时间内连续建 SSH 连接会被服务端的 `MaxStartups` 限流，表现为
`SSHException: Error reading SSH protocol banner`。这不是网络故障，隔几秒重试即可。

## 项目规模

```
/workspace/robust_metric_lab_nas   48 GB
├── outputs/                        9.7 GB   （1683 个文件）
│   └── student_metrics/            9.7 GB
├── data/                           361 MB
├── scripts/                        1.4 MB   （96 个 .py + 21 个 .sh）
├── docs/                           1.1 MB   （50 个 .md）
├── src/                            608 KB   （26 个 .py）
├── tests/                          768 KB   （42 个 .py）
├── configs/                        48 KB
└── shared_benchmarks/              ~7 GB    （多个 1.1–1.66 GB 的 RobustBench 权重）
```

`shared_benchmarks/` 里的权重包括 `Rebuffi2021Fixing_106_16_cutmix_ddpm.pt`、
`Amini2024MeanSparse_S-WRN-94-16.pt`、`Bartoldson2024Adversarial_WRN-94-16.pt`、
`Kang2021Stable.pt` 等。按 D-001/D-002 的策略，这些**只登记不入库**。

## 源码对账

`src/robust_metric_lab/*.py` 逐个算 sha256，与本地分支同名文件比对：

| 结果 | 数量 | 说明 |
| --- | --- | --- |
| 完全相同 | 25 / 26 | 含 `metrics.py`（76,626 B / 56 个函数，sha256 `0db17614…`）、`student_runner.py`、`e37_metrics.py`、`activation_sign_survival.py` |
| 不同 | 1 / 26 | `metric_registry.py`：服务器 `1aab7d25…`，本分支 `182cabb7…` |

差异原因不是"服务器更新"，而是**补丁在服务器上没打成功**：

- 服务器上存在 `metric_registry.py.orig` 与 `metric_registry.py.rej`；
- `.rej` 的内容正是 `feature_adv_excess_centered_unit_l2` 的 `MetricSpec` 4 行，
  以及 `configs/metric_eligibility.json` 对应的 1 行；
- 本分支是先打上补丁、再加 `--extra` 复算逻辑的版本。

所以：**服务器源码 = 本分支源码 − adv-excess 注册**，其余逐字节相同。这条证据把
"主线 = `893a994` 那一线、`dc6fb9c` 只是归档时点"的判断从推断变成了实测。

## 发布包

位置（注意：文档里以前写的是 `outputs/strict_adv_excess_release_.../`，少了
`student_metrics/wangjiahao/` 两级）：

```
/workspace/robust_metric_lab_nas/outputs/student_metrics/wangjiahao/
  strict_adv_excess_release_20261024T110422Z/
```

| 文件 | 大小 | 备注 |
| --- | --- | --- |
| `metric_values_merged.csv` | 14,840,212 B | sha256 `e2590610cf9e33cadad3a78278bcbed04ba20069fc6e4307eed4325416cf0f74` |
| `metric_eligibility.json` | 31,863 B | 86 条资格声明 |
| `provenance.json` | 714 B | 哈希链，见下 |
| `release_verification.json` | 121 B | `PASS` / 48,160 / 84 + 2 |
| `ranking/strict_ranking.csv` | — | 84 行官方排名 |
| `ranking/excluded_metrics.csv` | — | 2 行 |
| `ranking/confound_diagnostics.csv` | — | — |
| `ranking/analysis/{correlations,all_setting_table}.csv` | — | — |
| `ranking/analysis/analysis_report.md` | — | — |
| `analysis/adv_excess_comparison.json` | — | — |
| `STATUS.env` | — | `state=BUILDING` 后 `state=PASS` |

`metric_values_merged.csv` 的构成（用 `csv` 模块精确统计，不是 `awk`——文件里有带逗号的
字段，`awk -F,` 会切错）：

- 48,160 行 = 86 个指标 × 8 个设置 × 560 个模型/架构
- `status`：`ok` 47,600 / `failed` 560
- 失败的 560 行正好是被排除的两个指标：`racl_fixed_network_log_upper_proxy` 520 行、
  `racl_arch_lipschitz_log_upper` 40 行
- `feature_adv_excess_centered_unit_l2`：560 行，**全部 `ok`**

`provenance.json` 记录的哈希链：

| 字段 | 值 |
| --- | --- |
| `old_release` | `.../strict_trnas_direct_ode_release_20261024T000000Z` |
| `old_raw_sha256` | `deabce82…` |
| `new_run` | `.../adv_excess_full_20261024T101746Z` |
| `new_run_manifest_sha256` | `18459ccd…` |
| `new_raw_sha256` | `947f53c7a035211515af1552df3c2fa2468cb09fb6537112c75bc8fb5d196833` |
| `merged_raw_sha256` | `e2590610…` |
| `verification_sha256` | `1f666158…` |

## 数据对账

本地 `outputs/strict_all_metrics_table_20261024_trnas_repaired/sources/metric_values_original_strict.csv`
是 47,600 行 / 85 个指标 / 17 列，sha256 `b5137e2b3fe89ce5a99a99ea94bafa48d1abb81c45837551e5b1a54fe42ac92d`，
列名与服务器 `metric_values_merged.csv` **完全相同**。

按 `(setting, model_or_arch_id, metric_name)` 做内连接：

| 检查 | 结果 |
| --- | --- |
| 本地 47,600 个键能否在服务端 48,160 行里找到 | **47,600 / 47,600** |
| `metric_value` 最大绝对差 | **0.000e+00** |
| `target_value` 是否全同 | 是 |
| 服务端多出的 560 行 | 全部是 `feature_adv_excess_centered_unit_l2` |

也就是说：**本地那份 = 服务端 merged 去掉 adv-excess 之后的结果，逐值相同**。以前
`results/audits/strict-ranking-recompute-20261112.json` 里"83 项可复算、第 84 项原始行
只在服务器"的说法，缺的就是这 560 行。

那 560 行已取回并存进仓库：

```
outputs/strict_all_metrics_table_20261024_trnas_repaired/sources/metric_value_rows_adv_excess.csv
  172,751 B   sha256 947f53c7a035211515af1552df3c2fa2468cb09fb6537112c75bc8fb5d196833
```

它的 sha256 **与 `provenance.json` 的 `new_raw_sha256` 完全相同**——说明取自
`adv_excess_full_20261024T101746Z` 那次运行，链路闭合。

## 84 项复算

```powershell
python tools/verify_strict_ranking.py `
  --rows     <主工作树>\outputs\strict_all_metrics_table_20261024_trnas_repaired\sources\metric_values_original_strict.csv `
  --extra    outputs\strict_all_metrics_table_20261024_trnas_repaired\sources\metric_value_rows_adv_excess.csv `
  --repair   outputs\strict_all_metrics_table_20261024_trnas_repaired\sources\repair_rows.csv `
  --ranking  outputs\strict_adv_excess_release_20261024T110422Z\strict_ranking.csv `
  --output   results\audits\strict-ranking-recompute-20261112.json `
  --report   results\audits\strict-ranking-recompute-20261112.md
```

输出：

```
recomputed=84 compared=84 max_abs_error=0.000e+00 exact=True order_identical=True
```

八个抽查名次的差值（发布表三位小数 vs 复算）：

| 指标 | 复算 | 发布 | 差 |
| --- | --- | --- | --- |
| `trnas_consistency_phi` | 0.7822 | 0.782 | 0.0000 |
| `trnas_r_score` | 0.7378 | 0.738 | 0.0000 |
| `adversarial_loss` | 0.6143 | 0.614 | 0.0000 |
| `trnas_activation_psi` | 0.5981 | 0.598 | 0.0000 |
| `natural_loss` | 0.5913 | 0.591 | 0.0000 |
| `feature_measure_multiview_max_delta` | 0.5241 | 0.524 | 0.0000 |
| `feature_measure_multiview_rms_delta` | 0.4950 | 0.495 | 0.0000 |
| `fsp_delta` | 0.4794 | 0.479 | 0.0000 |

两个被排除的指标仍被拒，理由与官方一致：`racl_arch_lipschitz_log_upper` 在
`rb40_mature` 上 0/40 行 ok；`racl_fixed_network_log_upper_proxy` 在七个设置上 0 行 ok。

## 服务器上的大型产物（manifest 用）

| 路径（相对项目根） | 规模 | 处理 |
| --- | --- | --- |
| `outputs/student_metrics/` | 9.7 GB | 只登记 |
| `data/` | 361 MB | 只登记 |
| `shared_benchmarks/*.pt` | ~7 GB | 只登记，权重不入库 |
| `outputs/.../e36_full_metric_design_v1/retry_bundles.20260929T105701Z/` | 含 1.4 GB 权重副本 | 只登记；是重试残留 |

## 服务器上的残留物（不该被误当成源码）

| 路径 | 说明 |
| --- | --- |
| `src/robust_metric_lab/metric_registry.py.orig` / `.rej` | 补丁失败现场，正好是 D-031/本文件第 2 节的证据 |
| `.venv.failed_task8_20260918T023403Z/` | 一次失败的虚拟环境 |
| `.strict-e37-deploy-20260930.tar.gz`（73 KB） | 部署包 |
| `backups/` | 备份 |
| `outputs/student_metrics/wangjiahao/` 下的多个 `*_2026xxxxTxxxxxxZ/` | 历次运行目录，`STATUS.env` 记录状态 |

## 这份报告能说明什么、不能说明什么

**能说明**：

- 本分支的源码就是服务器在跑的那份（差一条在服务器上没打上的注册补丁）。
- 本仓库里那 47,600 行原始记录与服务端发布包同源同值，可以放心用来复算。
- 84 个入榜指标的排名可以从原始行完整重建，误差为 0。

**不能说明**：

- 不能说明发布表里的**每一个数值**都由本仓库这一版代码产出——第 23 轮的
  `published-e0` 审计发现发布表 e0 列里存在 `clamp_max(1e3)` 挡不住的量级
  （最高 5.87e10），说明生成 e0 的那一版代码不在本仓库历史里。
- 不能说明服务器上其他运行目录（`student_metrics/` 下还有多个）与这份发布包一致；
  本轮只核对了 `strict_adv_excess_release_20261024T110422Z` 一个。
- 不能说明 `feature_adv_excess_centered_unit_l2` 的计算过程正确，只说明它的原始行
  与 provenance 记录的哈希一致、并且能让复算达到零误差。

## 后续

- [x] 源码逐文件哈希对账
- [x] 发布包哈希链与行数核对
- [x] 本地原始行与发布包逐行比对
- [x] 取回 adv-excess 560 行并入库
- [x] 84 项离线复算零误差
- [ ] 服务器上其余运行目录的清单（`student_metrics/` 下逐个 `STATUS.env` + 大小 + 哈希）
- [ ] `shared_benchmarks/` 权重的逐个哈希（1.1–1.66 GB，可采样）
- [ ] 发布表 e0 列的生成代码溯源（第 23 轮遗留）
