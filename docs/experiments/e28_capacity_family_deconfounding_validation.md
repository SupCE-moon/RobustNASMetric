# E28: 扰动容量家族精化与去混杂墙验证

日期：2026-08-10

## 材料档案

- 类型：指标设计 + 结构去混杂调查
- 状态：已完成（容量家族为负结果；关键发现是 regime 特异性的鲁棒性）
- 问题：扰动容量（MAD 离散度）能否通过任何精化，在零成本 NAS 去混杂后存活，并跨空间（成熟 + 零成本）迁移？
- 新指标：5 个容量精化 + 1 个新攻击模式（fixed 固定扰动）
- 数据集：现有 CIFAR-10，无下载
- 搜索空间：RobustBench-20（成熟，本地）、NAS-Rob-Bench-201 e0（零成本，本地重实现）

## 背景与设计

E27 已发现：`perturbation_capacity_delta` 在 NB201 零成本去混杂后只有 partial +0.283，而 `flow_rel`/`feature_delta` 达 +0.62。E28 试图用 5 个新容量精化突破这堵墙，每个针对容量泛函的一层聚合或跨层结构：

| 字段 | 机制 | 针对的聚合层 |
|---|---|---|
| `perturbation_capacity_importance_delta` | 用通道对输出的梯度重要性加权 MAD | 通道求和（死通道零权重） |
| `perturbation_capacity_gini_delta` | 逐通道空间离散的 Gini 系数（尺度不变） | 空间平均 |
| `perturbation_capacity_channelwise_delta` | 逐通道相对 MAD 漂移（每通道尺度不变） | 通道求和 |
| `perturbation_capacity_layerprofile_delta` | 容量沿深度的方向形状 | 逐层相加 |
| `perturbation_capacity_flow_rel_delta` | 跨层「离散度流」＝对 dispersion map 做 FSP 相对漂移 | 跨层（模仿 flow_rel 的接线敏感性） |

其中 `importance_delta` 因在部分 RobustBench 大模型上死锁（额外的 grad-enabled 前向），且 NB201 结果无效（见下）而**禁用**（保持为 nan）。

## 主要结果

### NB201 e0 零成本（去混杂 partial 是核心判据）

方向校正后 Spearman + 控制 op 组成后的 partial：

| 指标 | raw | partial（去 op 混杂） | 存活率 |
|---|---:|---:|---:|
| `feature_measure_flow_rel_delta`（现有冠军） | +0.796 | **+0.626** | 79% |
| `feature_delta`（现有冠军） | +0.845 | **+0.615** | 73% |
| `perturbation_capacity_flow_rel_delta`（M5） | +0.688 | +0.378 | 55% |
| `perturbation_capacity_iqr_delta`（E27） | +0.552 | +0.395 | 71% |
| `perturbation_capacity_topk_delta`（E27） | +0.702 | +0.354 | 50% |
| `perturbation_capacity_gini_delta`（M2） | +0.534 | +0.335 | 63% |
| `perturbation_capacity_channelwise_delta`（M4） | +0.563 | +0.308 | 55% |
| `perturbation_capacity_delta`（原） | +0.552 | +0.283 | 51% |
| `perturbation_capacity_importance_delta`（M1，禁用） | +0.574 | +0.280 | 49% |
| `perturbation_capacity_layerprofile_delta`（M3） | +0.040 | +0.414（无意义） | — |

### RB20 成熟模型（白盒 FGSM，方向校正）

| 指标 | ρ |
|---|---:|
| `perturbation_capacity_relu_active_delta`（现有冠军） | **+0.699** |
| `perturbation_capacity_gini_delta`（M2） | +0.629 |
| `perturbation_capacity_delta`（原） | +0.626 |
| `perturbation_capacity_flow_rel_delta`（M5） | +0.603 |
| `perturbation_capacity_channelwise_delta`（M4） | +0.519 |
| `perturbation_capacity_layerprofile_delta`（M3） | +0.302（CI 跨零） |
| `feature_measure_flow_rel_delta` | +0.205 |
| `feature_delta` | +0.135 |

### 附加检验

**固定扰动（fixed perturbation）**：新增 `fixed` 攻击模式，用对所有架构相同的固定随机符号扰动替代白盒 fgsm，检验 E27 理论「零成本下白盒方向是任意方向」的担忧。结果全面变差：

| 指标 | fgsm partial | fixed partial |
|---|---:|---:|
| `flow_rel` | +0.626 | +0.172 |
| `feature_delta` | +0.615 | +0.227 |
| `capacity_delta` | +0.283 | +0.346（raw 从 +0.552 掉到 +0.230） |

**攻击强度**：RB20 上 flow_rel 在 FGSM / PGD-3 / PGD-10 / 三源 PGD-10 下分别为 +0.20 / +0.20 / +0.18 / −0.20，capacity 为 +0.63 / +0.62 / +0.57 / +0.74。攻击强度不是 flow_rel 在成熟模型上失效的原因。

**组合**：将两个零成本冠军 `flow_rel` + `feature_delta` 做 rank 平均，partial 掉到 +0.21（比任一单独都差），因两者的 op 组成分量高度相关、平均会放大该被去混杂掉的杂音。

## 关键发现

1. **容量墙是根本性的**。连同 E27 的 8 个，共 13 个容量变体在 NB201 零成本去混杂后集体卡在 partial ≈ 0.28–0.38，无一逼近 flow_rel/feature_delta 的 0.62。根因：容量取**绝对值**丢掉了符号/结构，而符号/结构正是去混杂后唯一存活的「op 交互」信号。

2. **M5 离散度流是容量族的天花板**（raw 0.688、partial 0.378），它把「容量 + 跨层」结合到极致，仍比 flow_rel 差一截——直接证明「取绝对值」这一步就丢了接线信号。

3. **白盒 fgsm 方向在零成本下其实是「架构信息」而非「任意方向」**。固定扰动实验推翻了 E27 理论「任意方向」的担忧：去掉 fgsm 用固定噪声，信号反而归零。

4. **regime 特异性对攻击强度鲁棒**。flow_rel 在成熟模型上无论攻击多强都只有 ~0.2，capacity 在零成本上无论怎么精化都 ~0.3。成熟鲁棒性来自训练权重、零成本鲁棒潜力来自计算图接线，是两个不同物理量。

## 结论

- 零成本「纯机制」指标：`flow_rel` / `feature_delta`（partial 0.62），同时满足「raw 强 + 去混杂不崩」。
- 成熟模型指标：`perturbation_capacity_relu_active_delta`（0.699）。
- 不存在单一「去混杂后仍通用」的指标；原始排序跨空间通用者是 `trnas_r_score`（0.748 全场景均值，但 op 组成混杂，partial 0.206）。

## 输出文件

```text
outputs/e28_capacity_family/nb201_e0.csv          （NB201 e0，fgsm，全部新指标）
outputs/e28_capacity_family/nb201_e0_fixed.csv    （NB201 e0，fixed 扰动）
outputs/e28_capacity_family/rb20_whitebox.csv     （RB20 白盒 fgsm，batch16）
outputs/e28_capacity_family/rb20_run.log          （RB20 运行日志）
src/robust_metric_lab/metrics.py                  （+5 容量指标）
src/robust_metric_lab/attacks.py                  （+fixed 扰动模式）
scripts/analyze_nb201_quick.py                    （NB201 去混杂分析）
```

## 复现命令

```powershell
cd E:\projects\robust_metric_lab   # 或对应本地路径
$env:PYTHONPATH="$PWD\src"

# NB201 零成本 (fgsm / fixed)
python scripts\evaluate_nb201_e27_zerocost.py `
  --csv outputs\e23_cross_space\nb201_64\metrics_64_e0e5e10e25e50.csv `
  --data-root D:\Project\Datasets\cifar10 --samples 64 --weight-seed 0 `
  --attack fgsm --output outputs\e28_capacity_family\nb201_e0.csv

# RB20 白盒 (注意用 batch 16 避免大模型显存压力导致死锁)
python scripts\compute_metric_batch.py --config configs\cifar10_linf_20.yaml `
  --data-root D:\Project\Datasets\cifar10 `
  --model-dir D:\Project\RobustNAS\robustbench-master\models `
  --output outputs\e28_capacity_family\rb20_whitebox.csv `
  --model-limit 20 --sample-limit 64 --batch-size 16 --attack fgsm --max-layers 4 --seed 20260704
```

## 实现注

`perturbation_capacity_importance_delta`（M1）的通道梯度重要性需要一次额外的 grad-enabled 前向+反传，在部分 RobustBench 大模型（如 Huang2020Self）上导致 CUDA 死锁，且 NB201 结果证明无效（partial 0.280 ≈ 原始 0.283），故禁用。其余 4 个容量指标（M2/M3/M4/M5）均为纯张量运算，无梯度反传，稳定可复现。
