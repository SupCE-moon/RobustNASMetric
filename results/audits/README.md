# 离线复核（audits）

这个目录放"不依赖服务器、只用仓库里的东西重新算一遍"的验证报告。
目的不是产出新结论，而是给已有结论**加一道独立检查**。

| 文件 | 检查什么 | 结论 |
| --- | --- | --- |
| `strict-ranking-recompute-20261112.json` / `.md` | 从 47,600 行原始记录重算八设置严格排名，与官方 `strict_ranking.csv` 逐值比较 | 83 项最大绝对误差 `0.000e+00`，名次顺序一致；2 项 RACL 因 `ok` 行数为 0 被排除；第 84 项（`feature_adv_excess_centered_unit_l2`）原始行只在服务器，离线不可复算 |

## 怎么重跑

```powershell
python tools/verify_strict_ranking.py `
  --rows   <主工作树>\outputs\strict_all_metrics_table_20261024_trnas_repaired\sources\metric_values_original_strict.csv `
  --repair outputs\strict_all_metrics_table_20261024_trnas_repaired\sources\repair_rows.csv `
  --ranking outputs\strict_adv_excess_release_20261024T110422Z\strict_ranking.csv `
  --output results\audits\strict-ranking-recompute-20261112.json `
  --report results\audits\strict-ranking-recompute-20261112.md
```

退出码 0 表示逐值全部落在 `1e-9` 容差内；非 0 表示有不一致，报告里会指出是哪个
`指标::设置` 差了多少。

## 附带核对：指标方向（2026-11-12）

计算 Spearman 之前要对 `direction == "lower"` 的指标取负号。方向一旦标反，整个名次会翻转，
所以这条不能只写在文档里。核对方式：

- 取 `metric_values_original_strict.csv` 的 `direction` 列（逐行声明）；
- 取 `src/robust_metric_lab/metric_registry.py` 的 `METRIC_SPECS`（代码里的声明）；
- 逐个比。

**结果**：85 个指标全部出现在注册表中，方向**零冲突（85/85）**；注册表比这批原始行多 1 个
`feature_adv_excess_centered_unit_l2`——正是第 84 名那条后加的指标。

## 为什么原始行不在仓库里

`metric_values_original_strict.csv` 是 13.94 MiB 的逐模型原始表，按 `.gitignore` 与
`tools/repository_policy.py` 的规则不入 Git（数据类大文件）。它在本机主工作树的
`outputs/` 下，哈希已记在 `results/manifests/local/outputs-inventory.json`。
仓库里保留的是**它的产物**（总表、排名表、发布验证 JSON）和这份复算报告。

## 指标冗余度（2026-11-12）

回答"84 项榜到底有几个独立信号"。脚本：

```powershell
python scripts/analyze_metric_redundancy.py ``
  --rows <主工作树>\outputs\strict_all_metrics_table_20261024_trnas_repaired\sources\metric_values_original_strict.csv ``
  --json results\audits\metric-redundancy-20261112.json ``
  --report results\audits\metric-redundancy-20261112.md
```

**结果**：85 个指标、8 个设置；ρ ≥ 0.5 → 21 簇，最大簇 35 个（`feature_measure_*` 漂移变体），
`perturbation_capacity_*` 9 个一簇，协方差/谱距离 9 个一簇，TRNAS Φ/R/Ψ 3 个自成一簇，
输入敏感性 6 个一簇（`local_logit_lipschitz`、`jacobian_spectral_norm`、`logit_kl`、
`input_loss_delta`、`input_grad_l2`、`racl_input_gradient_lower`），
`jacobian_effective_rank` 与 `jacobian_spectral_decay` 与主流**负相关**（-0.190 / -0.149）。

**读法**：
- 平均相关越高（最高 `feature_measure_robust_hybrid_rel_delta` +0.447），说明该指标越没有新信息。
- 覆盖不足的指标在表里带 `⚠n/8`：`racl_fixed_network_log_upper_proxy` 只有 1/8 个设置、
  `racl_arch_lipschitz_log_upper` 只有 7/8。**它们是"缺设置"，不是"数值低"。**
- 这是描述性分析：只说明谁和谁重复，不说明谁正确；相关在"模型排序"上算，只对排序用途有效。

## 文档声称 vs 发布表（2026-11-12）

回答"文档里写的数字，和 CSV 对得上吗"。脚本：

```powershell
python tools/verify_doc_claims.py ``
  --table outputs\strict_adv_excess_release_20261024T110422Z\RobustMetric_严格全部指标总表_86项_三位小数.csv ``
  --json results\audits\doc-claims-20261112.json ``
  --report results\audits\doc-claims-20261112.md
```

**结果**：33 条声称全部通过，最大偏差 0.0002（三位/四位小数舍入）；四条结构不变量全过。

**怎么加新条目**：在 `tools/verify_doc_claims.py` 的 `claims()` 里加一条
`field(...)` 或 `diff(...)`，写清 `doc`（文档路径）、`where`（小节）、`metric`、`field`、
`expected`。**不要把不在榜单上的数字加进来**（阶段稳定性、耗时、作废版本的值），
它们的出处不是这张表，放进来只会制造假警报。

**边界**：只证明"文档与表一致"，不证明表正确。表正确性见同目录的
`strict-ranking-recompute-20261112.md`。

## 指标总索引（2026-11-12）

86 个指标的定义、三空间资格、名次、方向、覆盖，全部 join 到 `docs/theory/METRIC_INDEX.md`。
重建命令：

```powershell
python tools/build_metric_index.py ``
  --table outputs\strict_adv_excess_release_20261024T110422Z\RobustMetric_严格全部指标总表_86项_三位小数.csv ``
  --rows  <主工作树>\outputs\strict_all_metrics_table_20261024_trnas_repaired\sources\metric_values_original_strict.csv ``
  --markdown docs\theory\METRIC_INDEX.md ``
  --json results\audits\metric-index-20261112.json
```

**`--rows` 可省略**：省略时"覆盖"列显示 `—` 并注明"未提供原始行，留空（不是 0）"。
原始行 13.94 MiB 按策略不入库，所以从 clone 出来的仓库重建索引时，覆盖列会是空的——这是**正确的**，
不要为了填满它而把 13.94 MiB 塞进 Git。

**退出码**：三份数据源的指标名集合不一致时返回 1，并打印 `PROBLEM ...`。
这说明注册表和资格表脱节了（例如新加了指标但忘了写资格），必须修好再看索引。

**已知警告**：TRNAS 三项在资格表里仍是 `implementation_error`，却被发布表排了名次（见 D-016）。
这是**待确认**的口径问题，不是 bug，索引每次都会列出来。

## 文档路径体检（2026-11-12）

`tools/check_doc_links.py` 扫描 88 份 Markdown，抽出 389 条仓库内路径引用，
确认它们指向的文件确实存在。

```powershell
python tools/check_doc_links.py `
  --json results/audits/doc-links-20261112.json `
  --report results/audits/doc-links-20261112.md
```

结果：`missing_required = 0`；`planned = 25`（计划文档里的历史文件名）、
`external = 1`（教师交付包内脚本）、`explained = 3`（文档已写明是服务器/外部文件）、
`optional = 12`（`outputs/` 产物，按策略不入库）。

**修掉的 5 处真实错误**：`README.md` 里指向 `metric_baselines.md` 的路径少了 `theory/` 一层、
`docs/theory/current_metric_table.md` 里指向 `current_metric_table_readable_zh.md` 的路径同样少了 `theory/`、
`docs/experiments/实验管理.md` 里 3 处 `e25_k2_k8_k16_three_source_validation_report.md`
（实际 `e25_k8_three_source_validation_report.md`）与 2 处 `e35_final_summary.md`（实际 `e35.md`）。

**边界**：只证明路径存在，不证明文件内容与文档描述一致。
