# 交付前自检汇总

- 通过 **12** / 失败 **0** / 跳过 **0**
- 发布表：`D:\Project\RobustNAS\DynamicBN\DynamicBN20251203\robust_metric\.worktrees\mainline-consolidation\outputs\strict_adv_excess_release_20261024T110422Z\RobustMetric_严格全部指标总表_86项_三位小数.csv`
- 原始行：`D:\Project\RobustNAS\DynamicBN\DynamicBN20251203\robust_metric\outputs\strict_all_metrics_table_20261024_trnas_repaired\sources\metric_values_original_strict.csv`

| 步骤 | 状态 | 退出码 | 秒 | 这一步在查什么 | 最后一行输出 |
| --- | --- | --- | --- | --- | --- |
| `doc_claims` | ok | 0 | 0.7 | 文档里引用的每个数字都能在发布表里对上 | claims=33 failed=0 invariant_failures=0 |
| `doc_links` | ok | 0 | 0.3 | 文档里引用的每个路径都存在 | docs=99 checked=488 missing_required=0 planned=25 external=1 explained=3 optional=16 |
| `metric_index` | ok | 0 | 0.7 | 86 个指标的注册、资格、名次、覆盖能拼成一张表 | metrics=86 problems=0 |
| `setting_weights` | ok | 0 | 1.6 | 八设置平均里每个设置各占多少分量 | metrics=84 tau(all,nb201)=0.9258 tau(all,rb40)=-0.127 weights={'rb40': 0.125, 'nb201': 0.625, 'darts': 0.25} |
| `implementation` | ok | 1 | 0.6 | 每个注册指标都有代码真的把它算出来 | registered=86 bundle=70 other=15 registry_only=1 missing=0 |
| `invariants` | ok | 0 | 5.1 | 零扰动为 0、处处有限、随扰动单调 | decreasing       perturbation_capacity_signed_delta |
| `strict_ranking` | ok | 0 | 2.9 | 83 个名次能从原始行离线复算出来，误差为零 | rejected: {"racl_arch_lipschitz_log_upper": "missing or degenerate settings: rb40_mature (0/40 ok rows)", "racl_fixed_ne |
| `redundancy` | ok | 0 | 1.3 | 85 个指标实际上聚成几个独立信号 | rho>=0.40: 15 clusters |
| `bootstrap_rb40` | ok | 0 | 4.0 | RB40 上每个指标的相关性有没有 95% 区间 |   adversarial_loss: rho=+0.743 [+0.487,+0.903] rank=2 (1-14) positive (whole interval above zero) |
| `phi_incremental` | ok | 0 | 264.5 | Φ 相对 35 个漂移指标是否还有增量信息 | verdict: has incremental value |
| `signed_capacity` | ok | 1 | 1.6 | 有符号与绝对 capacity 漂移是不是同一个指标 | verdict='not the same metric' median_rho=0.017284798534798536 flips=4/8 pairing_violations=0 |
| `early_vs_robbench` | ok | 0 | 126.7 | 早期指标预测的是鲁棒性，还是只预测了 clean 准确率 | n=64 rho(target,clean)=+0.982 best=feature_spectral_participation_ratio_static partial=+0.575 counts={'direct': 28, 'sup |

## 怎么读

- `skipped` 不是通过。它表示这台机器上没有那份大输入（通常因为原始行按策略不入库），
  结论只能引用上次在完整数据上的运行结果，不能当成刚跑过。
- `signed_capacity` 的退出码 1 是**结论**（两者不是同一指标），不是崩溃，
  因此它的期望退出码是 0 或 1；其他步骤只接受 0。
- `early_vs_robbench` 需要 NAS-RobBench-201 的 `NAS-RobBench-201_all.json`（16 MB，按策略不入库）。找不到时这一步会被跳过，而不是用别的文件顶替。
- 这个汇总不重算任何数字，它只负责把十二个审计的**参数接线**固定下来并跑一遍。
