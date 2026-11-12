# 早期信号在控制「参数量」之后还剩多少（RQ2 第二个混杂）

输入：`D:\Project\RobustNAS\DynamicBN\DynamicBN20251203\robust_metric\outputs\e28_capacity_family\nb201_e0_n400_20261112.csv`（sha256 `b618ecf32fede27655386ba7fc81c87c9f76e45bc5429d221dfa25e779a9014c`）
架构数：400；参数量范围 91226–2699770
相关：ρ(target,clean) = +0.983、ρ(target,params) = +0.710、ρ(clean,params) = +0.692

## 结论

- **12 个指标在同时控制 clean 与参数量后仍有 |偏相关| ≥ 0.30**：它们没有因为「架构更大」而消失。
- 7 个指标在加入参数量后明显缩小（> 0.10）：原来的一部分信号其实是「更大的架构更容易同时漂移得多、也训得更好」。
- 5 个指标原本 ≥ 0.30、加控制后掉到 0.30 以下：它们的独立信息基本就是规模。
- 45 个指标在只控 clean 时就已经 < 0.30，谈不上被参数量解释——它们本来就没有超出 clean 的信号。
- 最强的一个是 `feature_delta`：只控 clean 时 +0.371，再加参数量后 +0.422（**上升**）。参数量在这里更像一个抑制变量——它与 clean 同向、与漂移类指标不同向，控掉它反而让剩余变异与 target 的关系更清楚。这只是解释，不是已验证的机制。

判定：`rho_clean` 已经 < 0.30 → weak_before_capacity（本来就不强，不算被解释）；否则 `rho_clean_params` < 0.30 → explained_by_capacity；比 `rho_clean` 低 0.10 以上 → shrinks；符号反了 → flips；其余 → survives。

## 前 15 个指标

| 指标 | ρ(target) | ρ(仅控 clean) | ρ(控 clean+参数) | 95% 区间 | 判定 |
| --- | --- | --- | --- | --- | --- |
| `feature_delta` | +0.760 | +0.371 | **+0.422** | [+0.335, +0.490] | survives |
| `feature_measure_flow_rel_allpairs_delta` | +0.738 | +0.357 | **+0.417** | [+0.333, +0.478] | survives |
| `feature_measure_spatial_gram_rel_delta` | +0.731 | +0.334 | **+0.396** | [+0.309, +0.470] | survives |
| `feature_measure_spatial_gram_delta` | +0.732 | +0.332 | **+0.394** | [+0.307, +0.468] | survives |
| `feature_measure_relu_active_spatial_delta` | +0.730 | +0.330 | **+0.390** | [+0.302, +0.464] | survives |
| `feature_measure_relu_active_spatial_rel_delta` | +0.729 | +0.330 | **+0.390** | [+0.302, +0.464] | survives |
| `feature_measure_flow_rel_delta` | +0.675 | +0.320 | **+0.389** | [+0.311, +0.458] | survives |
| `feature_measure_flow_rel_multiscale_delta` | +0.675 | +0.321 | **+0.388** | [+0.309, +0.456] | survives |
| `feature_measure_relu_active_hybrid_rel_delta` | +0.707 | +0.309 | **+0.383** | [+0.299, +0.453] | survives |
| `feature_measure_spectrum_js_delta` | +0.777 | +0.468 | **+0.376** | [+0.282, +0.451] | survives |
| `feature_measure_relu_active_hybrid_delta` | +0.702 | +0.296 | **+0.371** | [+0.288, +0.441] | weak_before_capacity |
| `feature_measure_relu_active_delta` | +0.694 | +0.288 | **+0.365** | [+0.283, +0.435] | weak_before_capacity |
| `jacobian_spectral_decay` | -0.776 | -0.499 | **-0.362** | [-0.442, -0.267] | shrinks |
| `perturbation_capacity_relu_active_multiscale_delta` | +0.772 | +0.468 | **+0.361** | [+0.270, +0.436] | shrinks |
| `feature_measure_spectrum_hellinger_delta` | +0.768 | +0.465 | **+0.361** | [+0.267, +0.437] | shrinks |

## 怎么读

- 参数量的计算来自 `Nb201TinyNetwork(ops)` 的实际参数个数，不是估计值；C=16、N=5 下取值很有限，
  所以参数量实际上是一个「op 类型组合」的分组变量，控制它等于在同一档规模的架构内部比较。
- 参数量与 clean、target 本身相关（见上方三个数）。如果后两者都随规模上升，那么
  「控参数量」会同时拿掉一部分真实信号与一部分混杂，读到接近 0 不等于指标无用，
  只说明在这批架构里它没有超出规模的独立信息。
- 这里仍然只是相关，不是因果；也没有控制训练难度或 RobBench 自身的协议差异。

## 复现

```powershell
python scripts/audit_rq2_confounders.py --wide D:\Project\RobustNAS\DynamicBN\DynamicBN20251203\robust_metric\outputs\e28_capacity_family\nb201_e0_n400_20261112.csv --json <out.json> --report <out.md>
```
