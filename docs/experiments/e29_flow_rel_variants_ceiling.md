# E29: flow_rel 变体推高零成本天花板 + RB20 饱和诊断

日期：2026-08-15

## 材料档案

- 类型：指标精化 + 失效诊断
- 状态：已完成（负结果：零成本天花板 ~0.63 被坐实；RB20 失效诊断为「饱和」而非宽度混杂）
- 问题：能否通过 flow_rel 的变体（全层对、归一化相关）推高零成本去混杂 partial 超过 0.628？flow_rel 在成熟模型上失效的原因是什么？
- 新指标：`feature_measure_flow_rel_allpairs_delta`、`feature_measure_flow_corr_rel_delta`
- 数据集：现有 CIFAR-10，无下载
- 搜索空间：RobustBench-20（成熟）、NAS-Rob-Bench-201 e0（零成本）

## 动机

E28 已确认：零成本去混杂后唯一存活的信号是「跨层/全局交互」（flow_rel、feature_delta）。E29 试图在这条主线上继续往上推，并诊断 flow_rel 在成熟模型上失效（RB20 ρ=0.205）的原因。

## 新指标

| 字段 | 定义 |
|---|---|
| `feature_measure_flow_rel_allpairs_delta` | 所有层对（i<j，不只相邻）的 FSP 相对漂移平均 |
| `feature_measure_flow_corr_rel_delta` | channel 归一化（纯方向相关）后的跨层相关相对漂移 |

## 主要结果

### NB201 e0 零成本（去混杂 partial 是核心判据）

| 指标 | raw | partial（去 op 混杂） |
|---|---:|---:|
| `feature_measure_flow_rel_allpairs_delta` | **+0.843** | +0.628 |
| `feature_measure_flow_rel_multiscale_delta` | +0.797 | +0.627 |
| `feature_measure_flow_rel_delta`（原） | +0.796 | +0.626 |
| `feature_delta` | +0.845 | +0.615 |
| `feature_measure_flow_corr_rel_delta` | +0.645 | +0.480 |

### RB20 饱和诊断

按鲁棒性排序看 flow_rel 分布：

| 现象 | 值 |
|---|---|
| Standard（非鲁棒，rob=0.000） | flow_rel = **0.718**（离群） |
| 19 个鲁棒模型 | flow_rel = **0.10–0.24**（挤在一起，无趋势） |

flow_rel 能干净区分「自然训练 vs 对抗训练」（0.718 vs ~0.15），但无法区分「更鲁棒 vs 次鲁棒」（0.10–0.24 全噪声）。

## 关键发现

1. **全层对只提高 raw（+0.796→+0.843），partial 基本持平（+0.626→+0.628）**。多出的 raw 信号几乎全是 op 组成（被去混杂剥掉），说明「长程接线」已被相邻层对 FSP 覆盖。

2. **归一化相关（纯方向）掉到 +0.480**，说明 raw FSP 的**幅值结构**也携带部分交互信号。修正 E28 的说法：准确说是「符号 + 幅值结构」都携带交互信号，只有绝对离散度（capacity）不携带。

3. **flow_rel 在 RB20 失效的原因是「饱和」而非宽度混杂**。它测的是跨层接线（架构固定属性），能区分「是否对抗训练」，但无法区分训练强度——因为 19 个鲁棒模型的接线是固定的，其鲁棒性差异来自训练质量（capacity 测的就是这个）。

## 结论

- 零成本去混杂天花板被三个独立指标（flow_rel / multiscale / allpairs）共同钉在 **~0.63**，raw 可更高（0.843）但增量全是 op 组成。
- flow_rel 的 RB20 失效是「接线 vs 训练」的 regime 分界，非宽度/攻击强度问题，不可通过归一化修复。

## 输出文件

```text
outputs/e28_capacity_family/nb201_e0_v2.csv      （含 allpairs/corr_rel 的 NB201 结果）
src/robust_metric_lab/metrics.py                  （+allpairs/corr_rel 两个指标）
```
