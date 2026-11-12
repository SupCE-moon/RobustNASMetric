# 文档里的路径引用：存在性检查

- 扫描文档：103 份
- 检查过的仓库内路径：506 条
- 应当修复的缺失（`docs/ src/ scripts/ tests/ tools/ configs/ results/`）：**0** 条

## 1. 应当修复的缺失

这些文档承诺「详见某文件」，但文件不在仓库里。这是唯一一组算错误的。

无。

## 2. 计划/设计文档里的历史文件名

`docs/superpowers/plans/` 与 `specs/` 记录的是**当时打算创建**的文件，实际实现常改名（例如 `run_activation_holdout.py` 最终叫 `train_activation_holdout_nb201.py`）。这些是历史，不是断链，但列出来便于对照。

| 文档 | 引用的路径 |
| --- | --- |
| `docs/superpowers/plans/2026-09-17-three-space-benchmark-implementation.md` | `docs/experiments/e25_plus_three_space_validation.md` |
| `docs/superpowers/plans/2026-09-17-three-space-benchmark-implementation.md` | `docs/experiments/student_benchmark_environment.md` |
| `docs/superpowers/plans/2026-09-17-three-space-benchmark-implementation.md` | `docs/experiments/student_benchmark_smoke_report.md` |
| `docs/superpowers/plans/2026-09-17-three-space-benchmark-implementation.md` | `docs/experiments/student_three_space_final_report.md` |
| `docs/superpowers/plans/2026-09-17-three-space-benchmark-implementation.md` | `docs/superpowers/specs/2026-09-17-new-robust-metric-design.md` |
| `docs/superpowers/plans/2026-09-17-three-space-benchmark-implementation.md` | `scripts/verify_student_benchmark_bundle.py` |
| `docs/superpowers/plans/2026-10-24-adversarial-excess-feature-drift-implementation.md` | `docs/experiments/对抗超额特征漂移_严格跨空间结果_20261024.md` |
| `docs/superpowers/plans/2026-10-25-darts-e37-layer-diagnostic-implementation.md` | `outputs/student_metrics/wangjiahao/e37_full_metric_design_v1/run_manifest.json` |
| `docs/superpowers/plans/2026-10-27-activation-sign-survival-independent-validation.md` | `scripts/analyze_activation_holdout.py` |
| `docs/superpowers/plans/2026-10-27-activation-sign-survival-independent-validation.md` | `scripts/run_activation_holdout.py` |
| `docs/superpowers/plans/2026-10-27-activation-sign-survival-independent-validation.md` | `scripts/verify_activation_holdout.py` |
| `docs/superpowers/plans/2026-10-27-activation-sign-survival-independent-validation.md` | `tests/test_activation_holdout_analysis.py` |
| `docs/superpowers/plans/2026-10-27-activation-sign-survival-independent-validation.md` | `tests/test_activation_holdout_gate.py` |
| `docs/superpowers/plans/2026-10-27-activation-sign-survival-main-grid.md` | `docs/experiments/activation_sign_survival_source_audit_20261027.md` |
| `docs/superpowers/plans/2026-10-31-darts-independent-holdout-weights-and-targets.md` | `scripts/analyze_activation_holdout.py` |
| `docs/superpowers/plans/2026-10-31-darts-independent-holdout-weights-and-targets.md` | `scripts/seal_activation_holdout_darts_selection.py` |
| `docs/superpowers/plans/2026-10-31-darts-independent-holdout-weights-and-targets.md` | `scripts/train_activation_holdout_darts.py` |
| `docs/superpowers/plans/2026-10-31-darts-independent-holdout-weights-and-targets.md` | `scripts/verify_activation_holdout_package.py` |
| `docs/superpowers/plans/2026-10-31-darts-independent-holdout-weights-and-targets.md` | `tests/test_seal_activation_holdout_darts_selection.py` |
| `docs/superpowers/plans/2026-10-31-darts-independent-holdout-weights-and-targets.md` | `tests/test_train_activation_holdout_darts.py` |
| `docs/superpowers/plans/2026-10-31-darts-independent-holdout-weights-and-targets.md` | `tests/test_verify_holdout_targets.py` |
| `docs/superpowers/plans/2026-11-05-repository-reconciliation-implementation.md` | `results/manifests/reconciliation-2026-11-05/README.md` |
| `docs/superpowers/plans/2026-11-05-repository-reconciliation-implementation.md` | `results/manifests/reconciliation-2026-11-05/imported-files.json` |
| `docs/superpowers/plans/2026-11-05-repository-reconciliation-implementation.md` | `results/manifests/reconciliation-2026-11-05/local-worktrees.json` |
| `docs/superpowers/plans/2026-11-05-repository-reconciliation-implementation.md` | `results/manifests/reconciliation-2026-11-05/server-inventory.json` |

## 3. 外部交付包内的路径

| 文档 | 引用的路径 |
| --- | --- |
| `docs/experiments/教师交接README服务器核对_20261031.md` | `docs/student_robust_metric_benchmark/README.md` |

## 4. 已在文档里说明为「服务器/容器上的文件」的引用

文档自己写明该文件不在本仓库，因此不算断链；记录原因以免日后被误删理由。

| 文档 | 引用的路径 | 原因 |
| --- | --- | --- |
| `docs/experiments/activation_holdout_provenance_audit_20261027.md` | `scripts/train_activation_holdout_nb201_v2.py` | 文档写明这是「服务器新文件」，与本地 train_activation_holdout_nb201.py 字节相同 |
| `docs/experiments/student_benchmark_issue_log.md` | `docs/experiments/source_snapshot/verifier_after_recovery.log` | 文档写明日志保存在容器项目内，不是本仓库文件 |
| `docs/experiments/教师交接README服务器核对_20261031.md` | `scripts/verify_student_benchmark_bundle.py` | 教师交付包内自带的验收脚本，随交付包提供，不属于本项目仓库 |

## 5. `outputs/` 下的缺失（信息性）

这些是「在生成结果的机器上存在」的路径，按策略不入库。列在这里是为了让
真正的拼写错误仍能被看见，而不是为了让它们全部变绿。

| 文档 | 引用的路径 |
| --- | --- |
| `docs/DECISIONS.md` | `outputs/e28_capacity_family/nb201_e0_n400_20261112.csv` |
| `docs/PROJECT_STATUS.md` | `outputs/e28_capacity_family/nb201_e0_n400_20261112.csv` |
| `docs/experiments/DARTS_E37_逐层诊断结果与问题记录_20261025.md` | `outputs/student_metrics/wangjiahao/darts_e37_layer_diagnostic_20261025T120240Z/frozen_manifest.json` |
| `docs/experiments/activation_holdout_provenance_audit_20261027.md` | `outputs/student_metrics/wangjiahao/activation_holdout_nb201_weights_20261031/training_plan.json` |
| `docs/experiments/activation_holdout_provenance_audit_20261027.md` | `outputs/student_metrics/wangjiahao/activation_holdout_nb201_weights_20261031_v2/training_plan.json` |
| `docs/experiments/activation_holdout_provenance_audit_20261027.md` | `outputs/student_metrics/wangjiahao/activation_holdout_provenance_20261031/darts_selection_unsealed.json` |
| `docs/experiments/activation_holdout_provenance_audit_20261027.md` | `outputs/student_metrics/wangjiahao/activation_holdout_provenance_20261031/nb201_selection_unsealed.json` |
| `docs/experiments/activation_holdout_provenance_audit_20261027.md` | `outputs/student_metrics/wangjiahao/activation_holdout_provenance_20261031/prior_exposure_inventory.json` |
| `docs/experiments/e25_k8_three_source_validation_report.md` | `outputs/e25_k2_k8_k16_three_source_validation/robustbench20_k8_bootstrap_ci.csv` |
| `docs/experiments/student_benchmark_issue_log.md` | `outputs/student_metrics/wangjiahao/existing_metrics_full_v3_cpu_repair/logs/kang_trnas_diagnosis.log` |
| `docs/experiments/增量检验方案_Φ对特征漂移族_20261112.md` | `outputs/strict_all_metrics_table_20261024_trnas_repaired/sources/metric_values_original_strict.csv` |
| `docs/experiments/实验管理.md` | `outputs/command_validation/e03_join_check.csv` |
| `docs/experiments/实验管理.md` | `outputs/command_validation/e05_multisource_aggregate_check.csv` |
| `docs/experiments/实验管理.md` | `outputs/command_validation/e08_fixture_search_check.csv` |
| `docs/experiments/实验管理.md` | `outputs/e21_trnas_mmd/metrics_uniform512_step1e6.csv` |
| `results/audits/README.md` | `outputs/e28_capacity_family/nb201_e0_n400_20261112.csv` |

## 这条检查能证明什么

- 能证明：文档里指向仓库内文件的每一处引用，指到的文件**确实在**。读者顺着「详见 xxx」走下去不会撞空。
- 不能证明：文件内容与文档描述一致——那要靠 `verify_doc_claims.py`（数字）和人工阅读（论述）。