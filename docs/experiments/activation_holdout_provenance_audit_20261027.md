# 激活状态保持指标：三空间独立验证材料审计

审计时间：2026-10-31（北京时间）。范围仅限 `wangjiahao:/workspace/robust_metric_lab_nas` 和指定本地 RobustMetric 项目；本次仅查路径与源码，不读取本候选的任何留出分数。

## 当前材料状态

- RB40：服务器 `outputs/student_metrics/wangjiahao/holdout_checkpoint_recovery_20261027/` 保有替换权重、恢复报告和 `recovery_verification.json`。先前 11/11 加载检查只证明材料可用，不是 Φ/Ours 留出成绩。
- NB201：历史命令引用的 `outputs/e19_zero_cost_uniform512/checkpoints/` 在允许项目中不存在；现存 `outputs/nasrobbench201_server/e19_zero_cost/` 只有已分析过的 CSV/报告，没有该批权重。历史 512 架构已参与研究，不能直接称为新的独立包。
- DARTS：现存 `outputs/e23_cross_space/darts20/` 有 20 架构指标、目标和报告 CSV，但目录下未发现对应检查点；历史 20 架构也已参与研究。旧目标不能配给新训练权重。
- `shared_benchmarks/robust_metric_v1/` 当前只有 RB40、NB201 主网格 64 和 DARTS 主网格 100 的目录；没有已封存的三空间独立包。

据此，目前只能标记为 **`external_validation_incomplete`**。不能因主榜可能领先而声称超过 TRNAS Φ 已获独立验证，不能用 RB40 11/11 代替完整 371 键。

## 审计边界与下一步

已新增纯结构重叠审计代码 `scripts/holdout_overlap_audit.py`：用空间与 ID、DARTS 每节点两条边的顺序无关签名、权重哈希三道规则排除旧材料；缺目标来源或旧结构清单不全者标为 unresolved。它不读取成绩。5 项专门测试及本地完整 163 项测试通过。

下一步先取得或重建**未用于以往研究选择**的 NB201 64 架构五阶段权重及匹配的官方鲁棒性目标、DARTS 20 架构 e0/e100 权重及同协议完整 10,000 图 PGD20 目标。逐项核对结构、训练和目标来源，再在查看本候选留出成绩前封存 371 键。材料短缺时保持未完成状态，不缩小样本。

## 清单程序进度（2026-10-31）

已实现 `scripts/build_activation_holdout_manifest.py` 的纯结构校验与独占写入封存函数：要求 11+320+40 键、阶段共用架构、唯一键、有限目标、权重与目标哈希绑定及来源哈希。新加 10 项针对性测试；本地完整测试 173 项通过。**尚未创建或封存真实留出清单**：NB201/DARTS 未见满足独立性与目标来源要求的完整材料，目标验证器也尚未完成。封存函数的 `targets_verified` 字段只是后续目标验证报告的入口，不能单独证明目标有效；调用前必须由目标验证器核实其报告、实物和协议。
