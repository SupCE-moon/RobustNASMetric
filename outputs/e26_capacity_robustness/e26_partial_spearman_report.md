# E26：Partial Spearman — 控制 clean accuracy 后的容量-鲁棒性关联

日期：2026-07-31 | 数据：E25 K=8 三源共享对抗聚合

## 目的

检验 `perturbation_capacity_delta` 是否只是 clean accuracy 的代理。若控制 clean accuracy 后 Spearman 大幅下降，则容量未提供独立信息。

## 方法

- 零阶 Spearman：`capacity_delta` vs `rb_robust_acc`
- Partial Spearman：控制 `clean_acc`（rank residuals → Pearson）
- Semi-partial：`capacity_delta` vs `rb_robust_acc` 残差（移除 clean_acc 线性贡献后）
- 10000 次 bootstrap 95% CI

## 结果

| 类型 | adj-Spearman | 95% CI |
|---|--:|--:|
| 零阶 capacity vs robust | 0.7444 | [0.4297, 0.9164] |
| Partial capacity vs robust \| clean | **0.7692** | [0.4269, 0.9177] |
| Semi-partial (residual) | **0.6316** | [0.2487, 0.8635] |

Clean accuracy 单独 vs robust：Spearman = 0.3132

## 结论

控制 clean accuracy 后容量-鲁棒性关联不降反升（+0.0248）。容量在半偏相关下仍保留 0.6316 且 CI 不跨零。**容量不是 clean accuracy 的代理**，它捕获了 clean accuracy 不能解释的鲁棒性差异。
