# RB40 名次的置信区间（bootstrap，2026-11-12）

输入：`D:\Project\RobustNAS\DynamicBN\DynamicBN20251203\robust_metric\outputs\strict_all_metrics_table_20261024_trnas_repaired\sources\metric_values_original_strict.csv`（sha256 `b5137e2b3fe89ce5…`）
，`D:\Project\RobustNAS\DynamicBN\DynamicBN20251203\robust_metric\.worktrees\mainline-consolidation\outputs\strict_adv_excess_release_20261024T110422Z\RobustMetric_严格全部指标总表_86项_三位小数.csv`。

重采样单位是**模型**而不是指标行：40 个已训练防御模型，有放回抽 2000 次，种子 20261112，取 2.5% / 97.5% 分位。

## 最要紧的一句

RB40 单独排名与已发布八设置榜的一致度 τ = **-0.143**，95% 区间 [-0.243, +0.018]。区间跨过 0，说明**不能断言两套排序反向**，只能说它们不一致。

## 重点指标

| 指标 | RB40 点估计 | 95% 区间 | 结论 | RB40 名次（区间） | 已发布名次 |
| --- | ---: | --- | --- | --- | ---: |
| `trnas_consistency_phi` | +0.576 | [+0.310, +0.761] | positive (whole interval above zero) | 12 (2–34) | 1 |
| `trnas_r_score` | -0.184 | [-0.459, +0.136] | inconclusive (interval covers zero) | 73 (37–81) | 2 |
| `trnas_activation_psi` | -0.194 | [-0.471, +0.128] | inconclusive (interval covers zero) | 76 (38–82) | 4 |
| `adversarial_loss` | +0.743 | [+0.487, +0.903] | positive (whole interval above zero) | 2 (1–14) | 3 |
| `feature_measure_multiview_max_delta` | +0.302 | [-0.030, +0.589] | inconclusive (interval covers zero) | 35 (17–43) | 6 |
| `feature_measure_multiview_rms_delta` | +0.226 | [-0.119, +0.521] | inconclusive (interval covers zero) | 38 (24–46) | 7 |

## 怎么读

- **区间覆盖 0 = 不知道方向**。这时不能说"该指标在已训练模型上失效"，也不能说"它有效"，
  只能说 40 个模型的证据不足以区分。要更窄的区间就得加模型，而不是加大重采样次数。
- **名次区间比点名次重要**。若区间是 60–80，说"第 73"会给出并不存在的精度。
- 这套区间只回答"在 RB40 这 40 个模型上排得稳不稳"，不回答"换成另外 40 个模型会怎样"。
  后者需要新的模型集合，不是重采样能给的。
- **区间宽不等于指标没用**。它反映的是"40 个模型能提供多少信息"：
  一个真实相关 0.5 的指标，在 40 个模型上算出来，95% 区间大致就有 ±0.3 宽。

## 为什么按模型重采样

每个模型对 84 个指标各贡献一行，这些行不是独立证据（同一张网络的不同测量）。若按行重采样，等于把 1 个模型的 84 行当成 84 个独立样本，区间会被算得过窄。

## 复现核对

把算出的每个指标 RB40 相关系数与发布表 `rb40_mature` 列逐项相减：最大绝对误差 **4.991e-04**，不一致指标 0 个。
误差为 0 说明这份审计与冻结表算的是同一批数，不是另找了一份数据。

## 榜上有、但本审计算不了的指标

这些指标的原始行不在本机（只在服务器），所以没有区间，也不补数：

- `feature_adv_excess_centered_unit_l2`

## 这条检查能证明什么，不能证明什么

**能**：给出"RB40 上的名次"这套说法的误差范围，从而判断哪些名次差别值得写进论文。

**不能**：证明某个指标机制上更好；证明换一批模型后排序不变；替代已发布的八设置榜（那份协议没变，也不会变）。
