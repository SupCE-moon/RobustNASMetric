# 指标基本不变量审计

- 小网络：两层卷积（`tools/audit_metric_invariants.py::TinyNet`），种子 0，6 张 8×8 图
- 扰动网格：0.00/255, 0.25/255, 0.50/255, 1.00/255, 2.00/255, 4.00/255, 8.00/255
- 判据：`eps=0` 时距离类应为 0（容差 1e-06）；随 `eps` 单调上升要求 Spearman ≥ 0.7
- 共检查 **75** 个字段

## 结论

| 判定 | 数量 |
| --- | --- |
| 通过 | 69 |
| 已知非单调（有记录） | 1 |
| 已知占位 NaN | 1 |
| 随扰动**下降**（方向与其他 delta 相反） | 1 |
| 记录字段（不是指标） | 3 |

## 需要人看的

| 指标 | 判定 | Spearman | 起点值 | 说明 |
| --- | --- | --- | --- | --- |
| `perturbation_capacity_importance_delta` | 已知占位 NaN | — | nan | NaN placeholder; real value comes from the two-pass path |
| `perturbation_capacity_signed_delta` | 随扰动**下降**（方向与其他 delta 相反） | -1.000 | 0.000e+00 | decreases with epsilon: sign opposite to the other deltas |

## 怎么读

- `通过` 只说明这个指标在**这一个随机小网络**上守住了三条基本性质，不说明它测的东西有意义，也不说明它在真实 CIFAR 模型上同样表现。
- `随扰动下降` 不等于 bug：可能它本来就有符号（例如相对变化量），但注册表把它声明成 `lower` 时，它与其他 55 个 delta 的方向约定就不一致了，跨指标比较排名时要单独说明。
- `不随扰动有序变化` 常见于量化/离散量（盒计数维度）或饱和量，这类指标在小扰动区间的排序能力要单独验证。
- 这个审计**不检查**指标与鲁棒准确率的相关性——那是榜单在做的事。
