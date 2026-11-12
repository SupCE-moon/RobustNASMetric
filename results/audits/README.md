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

## 设置权重与分网格名次（`setting-weights-20261112`）

    python tools/analyze_setting_weights.py \
      --table outputs/strict_adv_excess_release_20261024T110422Z/RobustMetric_严格全部指标总表_86项_三位小数.csv \
      --json results/audits/setting-weights-20261112.json \
      --report results/audits/setting-weights-20261112.md

回答"八设置简单平均实际在平均什么"。把八列拆成 RB40 / NB201 / DARTS 三个网格分别排名，
算各网格与最终榜的 Kendall τ，并列出 7 个重点指标在四套名次里的位置。

结论：与只 NB201 的 τ = +0.926，与只 RB40 的 τ = −0.127。
**引用名次时必须说明是哪一套名次**，详见 D-021。

注意：这只是重读已发布表，**不改协议、不改任何数字**。
RB40 只有 40 个模型，该列是点估计、没有置信区间，不能当定论引用。
## RB40 名次的置信区间（`bootstrap-rb40-20261112`）

    python tools/bootstrap_rb40.py \
      --rows outputs/strict_all_metrics_table_20261024_trnas_repaired/sources/metric_values_original_strict.csv \
      --repair outputs/strict_all_metrics_table_20261024_trnas_repaired/sources/repair_rows.csv \
      --table outputs/strict_adv_excess_release_20261024T110422Z/RobustMetric_严格全部指标总表_86项_三位小数.csv \
      --json results/audits/bootstrap-rb40-20261112.json \
      --report results/audits/bootstrap-rb40-20261112.md

按**模型**（40 个 RB40 网络）重采样 2000 次，给每个指标的 RB40 相关系数与名次配 95% 区间。

两条容易踩的坑：
1. 必须传 `--repair`。三个 TRNAS 指标在 Kang2021Stable 上因 float32 下溢失败过，
   修复值在 `repair_rows.csv` 里；不合并这份补丁，这三个指标会各缺一个模型而被整列丢弃——
   恰好丢掉本审计要看的指标。测试 `test_repair_rows_restore_a_failed_metric` 守住这一点。
2. 复现核对用 5.5e-4 而不是 1e-6。发布表只存三位小数，重算与表的最大差不可能小于半步。

结论：Φ 与 `adversarial_loss` 稳定为正；`trnas_r_score` / Ψ 的区间跨 0，
所以"在已训练模型上失效"这句话没有依据，只能说"40 个模型分辨不出来"。详见 D-022。
## Φ 对漂移族的增量检验（`phi-incremental-20261112`）

    python scripts/audit_phi_incremental.py \
      --rows outputs/strict_all_metrics_table_20261024_trnas_repaired/sources/metric_values_original_strict.csv \
      --redundancy results/audits/metric-redundancy-20261112.json \
      --json results/audits/phi-incremental-20261112.json \
      --report results/audits/phi-incremental-20261112.md \
      --resamples 2000

基准 B 是冗余度审计里那 35 个漂移指标的组内百分位秩均值（不在这里重新聚类：
基准可动，就能被调到"Φ 看起来有增量"）。独立单元是 204 个架构，不是 560 行。

结果：Δρ = +0.695 [+0.583, +0.769]、置换 p = 0.0005（2000 次，0 次超过）、
嵌套模型 R² 0.163→0.640、同漂移分位内秩相关 +0.637。稳健性对照：PCA 0.685、oracle 单指标 0.676。

读的时候注意：Δρ 高有一半来自"B 在 RB40/DARTS 上本来就不预测 target"，
控制无效变量等于没控制。看 `nb201_e0`（ρ(B,y)=0.831、Δρ 0.522）才是关键证据。
详见 D-023。

## 注册表 ↔ 实现入口

`tools/audit_metric_implementation.py` 把 86 个注册名解析到具体模块：70 个在一次性 bundle、
15 个在专门模块、1 个（`feature_adv_excess_centered_unit_l2`）只有声明。
它的静态模式只读源码（AST），不需要 torch，所以可以在读 CSV 的机器上跑。

```
python tools/audit_metric_implementation.py --json results/audits/metric-implementation.json --report results/audits/metric-implementation.md
```

退出码 1 表示存在 "只有声明、没有实现" 的名字——这是需要人看的结果，不是工具故障。
`--dynamic` 会额外前向一次小 ResNet，用来确认 bundle 的字段真的被填上（NaN 除外）。

## 指标基本不变量

`tools/audit_metric_invariants.py` 检查 75 个字段的三条最低性质：零扰动时为 0（距离类）、
处处有限、随扰动单调上升。它用一个固定种子的两层卷积网络，秒级完成。

```
python tools/audit_metric_invariants.py --json results/audits/metric-invariants.json --report results/audits/metric-invariants.md
```

当前结果：69 通过、1 个有记录的非单调（盒计数维度）、1 个已知占位 NaN、
1 个随扰动下降（`perturbation_capacity_signed_delta`，见 `docs/PROJECT_STATUS.md` §6 第 0 条）、
3 个记录字段。`decreasing` 不是失败，是"需要人判断符号约定"。

## 有符号 vs 绝对 capacity（2026-11-12）

`tools/audit_signed_capacity.py` → `signed-capacity-20261112.{json,md}`。
输入是 47,600 行合并表（13.9 MiB，按 D-002/D-012 不入库）+ `repair_rows.csv`。

重跑命令见 `docs/HANDOFF.md` §18。三个容易踩的点：`--repair` 不能省、
退出码 1 表示「结论是两者不同」（正常）、`|signed|<=abs` 配对自检必须为 0 例外。

结论：两者不是同一指标的符号版本（秩相关中位数 +0.017，4/8 设置符号相反）。
## 交付前自检汇总（2026-11-12）

`tools/run_audit_suite.py` → `audit-suite.{json,md}`。一条命令跑完 11 个审计，
`failed` 必须为 0；`skipped` 表示这台机器缺大输入，不等于通过。

本次：11 通过 / 0 失败 / 0 跳过（`implementation`、`signed_capacity` 的退出码 1 是结论）。

## 早期指标 vs 训练后鲁棒性（2026-11-12）

`scripts/audit_early_metrics_vs_robbench.py` → `early-vs-robbench-20261112.{json,md}`。

用 NAS-RobBench-201 现成的训练后结果（6466 个架构，其中 64 个有本仓库的 e0 指标）
检验 RQ2。关键不在「相关系数多大」，而在**控制 clean 准确率之后还剩多少**：
这份数据里 clean 与 AutoAttack 的秩相关是 +0.982。

结果：84 个指标里 `direct` 28 / `via_clean` 17 / `suppressed` 9 / `weak` 30 / `uncovered` 1。
Φ 的原始相关 +0.849、偏相关 +0.170（区间跨 0）；站得住的是输入敏感性族与 `feature_measure_swd`。

重跑：`python tools/run_audit_suite.py --only early_vs_robbench`（需要 `--robbench` 或环境变量
`ROBUST_METRIC_ROBBENCH`，文件 15.6 MiB 不入库）。判据与局限见 D-028。

**target 同源核对**：加 `--oracle-table <outputs/e28_capacity_family/nb201_e0_v3.csv>` 会把本仓库
的 `oracle_target` 与 RobBench 的 `autoattack` 对齐核对（64/64、秩相关 +1.000、最大绝对差 0），
结论是「同一份数据」→ 局限改写为「只有一套 target，无法外部验证」，而不是「两套 target 可能矛盾」。
suite 在 `outputs/e28_capacity_family/nb201_e0_v3.csv` 存在时会自动传这个参数。

## e0 覆盖扩到 400 个架构（2026-11-12）

`scripts/expand_nb201_e0_coverage.py` 把 e27 配方用到 400 个 NB201 架构（必含已发布 64 个），
`scripts/wide_e0_to_rows.py` 转成 audit 吃的 long 格式。产物：

- `outputs/e28_capacity_family/nb201_e0_n400_20261112.csv`（400 行，不入库，按策略只在本机）
- `results/audits/early-vs-robbench-n400-20261112.md`（400 架构）
- `results/audits/early-vs-robbench-same64-20261112.md`（同一配方重跑 64 个，作为对照）

**对照怎么读**：`发布表 64 → 当前代码 64` 只变数据来源（+0.575 → +0.357），
`当前代码 64 → 当前代码 400` 只变样本量（+0.357 → +0.357，区间收紧）。
所以 RQ2 的幅度应表述为「约 +0.35」，而不是发布表口径的 +0.5 量级（D-029）。

**不要做的事**：这份 400 表只有 70 个 zero-cost 指标，**不含 TRNAS Φ/R/Ψ**（需要训练轨迹），
不要用它去重估第 3 节第 17 条里 Φ 的 +0.170；也不要把它与发布表的 e0 数字放进同一张比较表
（`feature_delta` 差约 20 倍，见 `PROJECT_STATUS` §6 第 10 条）。
## RQ2 的第二个混杂：参数量（2026-11-12）

`scripts/audit_rq2_confounders.py` 在控制 clean 之外再控制参数量（由 `Nb201TinyNetwork(ops)` 实算），
报告三方偏相关与 2000 次按架构 bootstrap 区间。产物 `results/audits/rq2-confounders-20261112.md`。

要点：ρ(target,params)=+0.710、ρ(clean,params)=+0.692，但控制后信号**不降反升**
（最强 `feature_delta` +0.371 → +0.422）——参数量在这批架构里是抑制变量，不是解释变量。
判定分布 survives 12 / shrinks 7 / explained_by_capacity 5 / weak_before_capacity 45。

**别把 `weak_before_capacity` 读成 `explained_by_capacity`**：前者的 `rho_clean` 本来就 < 0.30。
## 发布表 e0 的逐列核对（2026-11-12）

`tools/audit_published_e0_sanity.py` 按代码里的 `clamp_max(1e3)` 上界，把发布表 e0 的每个指标判成
`clean` / `spiked` / `divergent` / `uncomparable`。产物 `results/audits/published-e0-20261112.md`。

结果：70 列里 `clean` **0 个**、`spiked` 14 个、`divergent` 55 个、`uncomparable` 1 个。
**在这张表被重算之前，不要引用它上面得到的任何幅度**（第 3 节第 17 条的更正、D-031）。
## RQ2 的扰动敏感性，以及一次偏相关口径更正（2026-11-12）

脚本：`scripts/audit_e0_perturbation_sensitivity.py`，报告：
`results/audits/e0-perturbation-20261112.{json,md}`。

三个配置跑同一批 400 架构：`base` = FGSM 8/255 单步、`weak` = FGSM 4/255、`pgd` = PGD-3 8/255。
结论：`base` 上最强的 `feature_measure_robust_delta`（+0.400）在两个变体里变成 −0.27，
前五名重合 0/5，指标级排序相关约 0.03–0.04；但 `input_grad_l2` 三个配置都在 +0.37 以上。
所以"哪个早期指标最好"是配置的性质，不是架构的性质。

**同期的口径更正**：`audit_rq2_confounders.partial_rho` 原来只秩化 x/y、控制列用原值，
与 `audit_early_metrics_vs_robbench.partial_spearman` 不一致（+0.422 vs +0.145）。
已统一为全秩化，`rq2-confounders-20261112.{json,md}` 与 `PROJECT_STATUS.md` §3 第 19 条按新口径重写。
以后要引用 RQ2 的数字，用 +0.400（`feature_measure_robust_delta`），不要再用 `feature_delta` 的 +0.42。

**重跑**：见 `docs/HANDOFF.md` §25；`--reuse` 可只重算分析。该脚本需要 GPU，不进 `run_audit_suite.py`。


## e0 指标对权重种子的稳定性（2026-11-12）

`e0-weight-seed-20261112.{json,md}`，由 `scripts/audit_e0_weight_seed.py` 生成（100 架构 × 3 种子，约 11 分钟 GPU）。

回答的问题是"e0 指标是架构的属性，还是那次初始化的运气"。两个数字必须分开看：

- `counts.stable_value`（本次 **53/69**）：同一架构跨种子的指标值是否同序。
- `counts.stable_signal`（本次 **55/69**）：每个种子单独算的偏相关是否同号同量级。

**输出空间整族（`adversarial_loss`、`natural_loss`、`logit_kl`、`input_loss_delta`）值稳定只有
0.03–0.17、种子间符号翻转**，不能用于早期预测。输入敏感性与分布距离类稳。

生成前请读 `docs/HANDOFF.md` §26：不要加 `.eval()`（会让结果与发布表不可比），
`--output-dir` 必须显式给，只改分析时用 `--reuse`。