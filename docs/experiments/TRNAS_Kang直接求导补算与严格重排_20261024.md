# TRNAS Kang 数值修复与严格重排（2026-10-24）

## 结论与版本边界

新版本 `strict_trnas_direct_ode_release_20261024T000000Z` 在 `wangjiahao` 容器内通过独立发布校验：47,600 条原始行、47,600 个唯一键、83 个满足完整 560 行条件的跨空间指标、2 个仍排除的 RACL 空间特异指标。旧版 `strict_final_release_20261008T000000Z` 保持不变，不得将两个版本的名次混写。

| TRNAS 指标 | 新严格名次 | All-setting mean | RB40 原始相关 | RB40 控制模型家族与参数量后相关 |
|---|---:|---:|---:|---:|
| `trnas_consistency_phi` | 1 | 0.782223 | 0.575985 | -0.013215 |
| `trnas_r_score` | 2 | 0.737770 | -0.183570 | 0.091217 |
| `trnas_activation_psi` | 4 | 0.598075 | -0.193915 | 0.065179 |

名次是按原严格协议的八个 setting 简单平均得出，不代表指标已在每个空间或控制结构后都最优。尤其 `trnas_consistency_phi` 的 RB40 原始相关在结构控制后接近零。零成本均值由 NB201 e0 与 DARTS e0 平均，新总表还保留 5 个 NB201 epoch 的独立列和 `nb201_mean`。

## 失败根因与补算方法

旧版每个 TRNAS 指标只有 `rb40_mature / Kang2021Stable` 一个失败键，错误为 `AssertionError: underflow in dt 0.0`。完整堆栈定位到 TRNAS 对模型执行 `1e-6` 交叉熵权重上升步之后，扰动模型再次前向时，SODEF 的自适应 ODE 求解器下溢。原 `odeint_adjoint` 求导路线产生的最大参数变化约为 `1.06e19`；隔离对照中，`torchdiffeq.odeint` 直接自动求导路线的最大参数变化约为 `1.13e-6`。该修复只改变 Kang 的 ODE 梯度计算路线，不改变模型前向求解器、样本、FGSM 扰动、TRNAS 公式或冻结的权重步长。

两次 CPU 隔离计算均得到 Kang 的 `Psi=0`、`Phi=686.1439208984375`、`R-Score=0`，且干净与对抗输入在两种路线下的前向 logits 最大绝对差均为 `0`。正式补算脚本只生成这三个失败键；安全合并器拒绝替换任何已有 `status=ok` 的键，并记录新旧文件哈希。严格排名程序再次逐一核验 83 个入榜指标各有 560 条唯一、有限、`status=ok` 记录。新版本中原有 80 个指标的八个 setting 分数和 All-setting mean 与旧版逐项完全相同（最大差 `0`）；只是加入 TRNAS 后名次顺延。

## 证据位置

- 服务器新发布：`/workspace/robust_metric_lab_nas/outputs/student_metrics/wangjiahao/strict_trnas_direct_ode_release_20261024T000000Z/`
- 数值修复原始记录与诊断：该目录下 `repair_rows.csv`、`diagnostic.json`。
- 只替换失败键的审计证据：`repair_manifest.json`。
- 正式名次、排除项、结构去混杂：`ranking/strict_ranking.csv`、`ranking/excluded_metrics.csv`、`ranking/confound_diagnostics.csv`。
- 独立完整性检查：`release_verification.json`，其 `status=PASS`。
- 可复现脚本：`scripts/repair_kang_trnas_direct_ode.py`；旧数值失败证据仍保存在旧发布中。

## 已知限制与本地测试

Kang 使用与其余模型不同的数值求导路线，虽然前向完全一致且数学指标定义不变，报告中应显式披露。没有因此重定义 RACL 两项或把空间特异量强行入榜。TRNAS 高名次仍需结合结构去混杂、留出验证与运行成本解读，不能只引用总体排名。

本地定向测试 `tests/test_kang_trnas_repair.py` 与 `tests/test_trnas_metrics.py` 共 5 项通过；全项目测试 39 项通过、2 项失败。两项失败均在既有的 `tests/test_e36_batched.py`，原因是本地未提交的 `src/robust_metric_lab/e36_batched.py` 导入缺失的 `_e36_cross_layer_stats`；本次未改动这些 E36 文件，也不将其误报为 TRNAS 修复成功的测试。
