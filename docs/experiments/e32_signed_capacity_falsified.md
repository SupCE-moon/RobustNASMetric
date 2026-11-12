# E32: signed capacity 证伪 + capacity/gram 通用性-强度权衡

日期：2026-08-29

## 材料档案

- 类型：机制证伪 + 跨设置通用性诊断
- 状态：已完成（关键发现：signed capacity 全空间死亡；capacity 族「成熟强但非通用」；gram 族在 NB201+RB20 内通用、DARTS 上失效；DARTS 空间可测但 gram/capacity 失效，trnas 有效）
- 问题：E31 提出「capacity 取绝对值丢符号 → 才不通用」的机制假说。E32 直接检验该假说的对照（signed capacity），并厘清 capacity 族与 gram 族之间的「成熟强度 ↔ 通用性」权衡。
- 新指标：无（本轮为证伪 + 诊断，不引入新指标）
- 数据集：现有 CIFAR-10，无下载；纯分析 + DARTS 零成本重跑（20 架构）
- 搜索空间：NB201（64，e0/e5/e10/e25/e50）、RB20（成熟异构 20）、DARTS（20，e0）

## 动机

E31 的结论是「gram 族通用、capacity 不通用（e5 翻转）」，并给出机制解释：「capacity 取绝对值丢符号，因此只在成熟模型有效」。这一机制可被直接证伪：代码里早已实现的 `perturbation_capacity_signed_delta = (adv−clean)/|clean|` 保留了符号，若该机制成立，signed 版本应处处为正。E32 用纯分析（现有 CSV）检验这一点，同时把 capacity 族的每个变体都放到「通用性」这一标尺上。

## 核心结果

### 1. signed capacity 全空间死亡（机制证伪）

方向校正 Spearman（lower is better 已翻转）：

| 指标 | NB201 e0 (n=64) | RB20 (n=20) | DARTS e0 (n=20) |
|---|---:|---:|---:|
| `perturbation_capacity_signed_delta` | **−0.081** | **−0.160** | **−0.282** |
| `perturbation_capacity_delta`（unsigned） | +0.552 | +0.575 | −0.304 |
| `fsp_delta`（gram 参照） | +0.599 | +0.450 | +0.099 |

signed capacity 在三个空间全部接近 0 或方向相反，**比 unsigned 更差**。这说明容量漂移的信号在**幅度**里，不在符号里；E31 的「取绝对值丢符号」机制解释**不成立**。

### 2. capacity 族：成熟强，但非通用

NB201 各训练阶段方向校正 Spearman（e0/e5/e10/e25/e50）：

| 指标 | e0 | e5 | e10 | e25 | e50 | min | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| `perturbation_capacity_delta` | +0.703 | **−0.397** | +0.187 | +0.237 | +0.208 | −0.397 | +0.188 |
| `perturbation_capacity_relu_active_delta` | +0.398 | **−0.458** | −0.278 | −0.409 | −0.475 | **−0.475** | **−0.245** |
| `perturbation_capacity_above_mean_delta` | +0.419 | **−0.458** | −0.137 | −0.117 | −0.169 | −0.458 | −0.092 |

**关键**：`perturbation_capacity_relu_active_delta` 是 RB20 上最强的 Ours 指标（见下），但它在 e5 之后全程为负（mean −0.245）——是最不通用的 capacity 变体。ReLU 门控让它在成熟模型上更强，却让通用性更差。

### 3. gram 族：NB201 + RB20 内通用（DARTS 上失效）

| 指标 | e0 | e5 | e10 | e25 | e50 | min | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fsp_delta` | +0.773 | +0.731 | +0.690 | +0.648 | +0.669 | **+0.648** | +0.702 |
| `feature_measure_struct01_delta` | +0.765 | +0.724 | +0.684 | +0.632 | +0.656 | +0.632 | +0.692 |
| `feature_measure_moment_delta` | +0.788 | +0.730 | +0.707 | +0.673 | +0.648 | +0.648 | +0.709 |

### 4. 通用性-强度权衡（RB20 全指标排名）

RB20 方向校正 Spearman 前 20（vs adv_acc, n=20）：

| 排名 | 指标 | adj |
|---:|---|---:|
| 1 | `perturbation_capacity_relu_active_delta` | +0.710 |
| 2 | `adversarial_loss`（成熟 loss 读出） | +0.657 |
| 3 | `perturbation_capacity_relu_active_multiscale_delta` | +0.653 |
| 4 | `input_loss_delta` | +0.588 |
| 5 | `perturbation_capacity_multiscale_delta` | +0.575 |
| 5 | `perturbation_capacity_delta` | +0.575 |
| 7 | `perturbation_capacity_topk_delta` | +0.567 |
| … | … | … |
| 16 | `feature_measure_struct01_delta` | +0.453 |
| 17 | `fsp_delta` | +0.450 |
| 25 | `feature_measure_moment_delta` | +0.316 |

RB20 前 7 名里 capacity 族占 6 席（relu_active 以 +0.71 甚至超过 adversarial_loss），gram 族只到 +0.45。这正是「成熟强度」与「通用性」的镜像。

### 5. DARTS 空间可测，但 gram/capacity 失效（方法学修正）

| 空间 | oracle 字段 | std | range |
|---|---|---|---:|
| DARTS (20) | final_pgd_accuracy | 0.0077 | [0.453, 0.484] |
| NB201 (64) | oracle_target | 0.0555 | [0.182, 0.429] |
| RB20 (20) | adv_acc | 0.0860 | [0.313, 0.719] |

DARTS 的 oracle 方差虽小（std 0.0077，是 NB201 的 1/7、RB20 的 1/11），但排序真实可辨——trnas 族在 DARTS 上相关 0.50～0.78（E23 full-test：`trnas_r_score` 0.750、`trnas_consistency_phi` 0.776）。**此前「DARTS 不可测」是误判**（早期版本据 std 小错误地排除 DARTS）：gram/capacity 在 DARTS 上接近 0 是这些指标本身不跨空间，而非该空间无法检验。gram 族的跨空间通用性应限定为 NB201 + RB20 两空间，DARTS 是 gram 的失效空间。

## 关键发现

1. **signed capacity 被证伪**：在 NB201 e0、RB20、DARTS 三处全为负/近零，且劣于 unsigned。E31 的「取绝对值丢符号」机制解释错误——容量漂移的信息在幅度，符号是噪声。

2. **capacity 族的「成熟强」与「通用性」严格互斥**：RB20 最强（relu_active +0.71）恰是跨阶段最差（mean −0.245）。ReLU 门控放大了成熟模型上的区分度，也放大了训练中期的方向翻转。

3. **gram 族在 NB201 + RB20 内「处处为正」**：NB201 全阶段 min ≥ 0.63、mean ≥ 0.69，RB20 0.32–0.45。但 DARTS 上失效（e50/e100 接近 0 或负）。代价是成熟模型上只有中等强度（+0.45）。

4. **DARTS 可测，gram/capacity 在其中失效**：DARTS 的 oracle 方差虽小（std 0.0077）但排序真实可辨，trnas 族相关 0.50～0.78。「DARTS 不可测」是早期误判，已更正。

## 结论

- 「扰动容量」（capacity）方向（capacity 族）是**成熟模型专用**：在 RB20 上最强（relu_active +0.71），但跨训练阶段方向翻转（e5 后为负），不满足「处处不差」的通用性要求。
- **gram 族是「跨协议尺度不变 + 去混杂非平凡」的通用候选**（`fsp_delta`、`feature_measure_moment_delta`、`feature_measure_struct01_delta`），但 DARTS 上失效（跨空间不完整）。E32 同时**否定了**用 signed capacity 突破 RB20 天花板的路线。
- 通用性与成熟强度存在真实的权衡：gram 族用「稳定但中等」换取「NB201+RB20 处处为正」。若论文需要一个「尺度不变 + 去混杂非平凡」指标，gram 族是唯一候选；若需「跨空间零成本排序」，trnas 族更强；若需「成熟模型最强」，`perturbation_capacity_relu_active_delta` 是候选，但必须如实报告其跨阶段翻转。

## 输出文件

```text
scripts/analyze_e32_generality.py      （signed/unsigned/gram 三端点 + 去混杂分析）
outputs/e32_darts_e0_zerocost.csv      （DARTS 20 架构零成本全指标）
```
