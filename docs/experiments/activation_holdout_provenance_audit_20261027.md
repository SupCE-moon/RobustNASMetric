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

后续核对主网格定义发现：DARTS e0 和 e100 都预测同一 e100 最终鲁棒性目标。因此 e0 留出条目的目标权重哈希应绑定同一架构的 **e100 权重**，而非其 e0 权重。已用先失败再通过的测试修正清单规则，并要求 NB201/DARTS 同架构各阶段目标数值与来源一致；针对性 12 项、完整 175 项测试通过。该修正发生在任何真实留出清单封存或分数计算前。

已新增 DARTS 目标**元数据**核验：冻结完整 CIFAR-10 测试集 10,000 个顺序 ID，PGD-20 的 `epsilon=8/255`、步长 `2/255`、单次随机起点、batch 128、seed 400000 和现有 Normalize 参数。错误目标权重、9999 张图、攻击参数、批大小或种子均会被拒绝。它目前**不能证明**数据文件存在、文件哈希正确或 PGD 实际执行；真实材料仍缺，后续须加入逐文件/逐目标证据核验，不能单靠声明字段封存。

进一步新增 `verify_target_file`：对项目内 DARTS 目标 JSON 逐字节核对 SHA-256，要求恰好 10,000 个有序测试图 ID、10,000 个布尔正确性结果，按逐图结果重算准确率，并对最终权重和完整攻击协议做交叉核对。20 项针对性测试通过。当前旧评估脚本只输出汇总准确率，**不会产生这种逐图证据**；须为新的独立目标运行追加式逐图日志，并另行核对实际攻击代码和权重。服务器项目内尚未发现新留出 NB201 官方目标 JSON，因此没有把历史主网格目标假定为新留出源。

## 官方 NB201 来源复核（2026-10-31）

本项目历史 E23 命令指定 `data/nasrobbench201/NAS-RobBench-201_all.json` 和 `data/nasrobbench201/meta.json`，读取 `cifar10` 下的 `autoattack` 数值；`choose_architectures` 仅从可追溯 canonical ID 中等概率抽样。首次核对时，这两份 JSON 在允许的服务器项目内不存在。NAS-RobBench-201 作者的 [官方代码与下载入口](https://github.com/TT2408/nasrobbench201#download-our-benchmark-dataset) 提供数据文件。新包选择前须核验官方文件的字节哈希、数据字段和全部历史 ID/结构排除集。

### 官方源文件补齐（2026-10-31，覆盖上一段缺失状态）

通过作者公开 Drive 文件夹只读取得原始文件，并直接传至唯一允许的 `wangjiahao:/workspace/robust_metric_lab_nas/data/official_nasrobbench201_source_20261031/`。本地指定项目内亦保留同字节副本 `data/official_nasrobbench201_source/`。本地与服务器逐文件 SHA-256 一致：

| 文件 | 字节 | SHA-256 |
|---|---:|---|
| `NAS-RobBench-201_all.json` | 15,622,110 | `b670b9ce5a7e6cd29735c07806efa493ba0f7023cc46940b856e214a70b3a308` |
| `meta.json` | 2,112,697 | `54b6d2c2780cfb63de98e9d67e72db9cb0c02be426191d429dbe02c898d4577e` |

服务器端解析确认 `cifar10` 有 6,466 条架构记录，6,466 条均存在 `autoattack` 字段；`meta.json` 有 15,625 个 ID。作者代码另克隆到项目内 `data/official_nasrobbench201_repo_f933a51/`，固定提交 `f933a51be7026985a6b5fbf9ab82956acb14e3d4`。这些文件**只是目标来源**；尚未完成全历史排除、选取独立 64 架构、训练五阶段权重或留出成绩计算，不能据此声称独立验证完成。

## 历史架构排除集初查（2026-10-31）

只读取项目内 CSV 的 ID/结构列，不提取或查看新候选留出分数。对 `outputs` 中路径含 `nasrobbench201`/`nb201` 且有 `arch_id` 或 `canonical_id` 的文件，以及主网格 NB201 清单，找到 52 个来源文件；以官方 `meta.json` 的 `isomorph` 映射合并，得到 **1,301 个曾参与结果生成的 canonical NB201 ID**，没有发现无法映射到官方元数据的 ID。该数是当前 CSV 扫描下界，尚未完成所有 JSON/原始尝试的排除审计，不能直接从余下架构抽 64 个并称独立。

DARTS 在主网格和历史结果中发现 8 份带签名的 CSV，合计 **120 个不同原始结构签名**（主网格 100、E23/E24 20）；另有 10 份只有 `arch_id`、没有签名的历史表，包括 E27/E32 与 E23/E24 汇总。E27 的评估脚本默认输入 E23 的 20 架构表，但 E32 的逐结构出处还须用原始命令或检查点映射进一步核实。编号 `0..19` 不能独自证明跨实验是同一结构；在出处不清时把它们保留为 unresolved，不封存 DARTS 新留出包。

对 E27/E32 追加来源核对：`scripts/evaluate_darts_e27_zerocost.py`（SHA-256 `594a623f04f206dedda706c68ab9f75eb5facc04a0db1176878a5ad08e9c2bd4`）默认直接读取 E23 DARTS20 的含签名 CSV，并逐行解析其签名构造 e0 模型。E32 与 E27 v3 的 20 个编号全部相同，57 个共同旧指标列中有 **53 列在 20/20 行逐值完全相等**；E32 文件 SHA-256 为 `a6824c4c5ccceeeda3b8f47f32779b8a5c0185214dcce8a492e49b8f1552e707`，E27 v3 为 `9284712578205e8d205a2155087ab1194e3fc5f6b2df4baf2352232fdc90a52f`。这强烈支持 E32 表承接 E27/E23 的 20 架构旧行，但 E32 新增列的原始运行命令/权重映射未见，故仍保留 unresolved，而不把相同编号当成充分证明。

## 可重跑的排除清单（2026-10-31）

新增 `scripts/holdout_exposure_inventory.py`，从项目内历史 CSV 与 JSON 只提取身份字段；官方 `meta.json` 负责 NB201 同构归并，DARTS 签名按每节点两边顺序无关规范化；对每份身份来源记录字节 SHA-256。3 项针对性测试先失败再通过，本地完整测试 **198 项通过**。本地/服务器脚本 SHA-256 均为 `d62eefc82d63e955afb13b3e1fd571ed7e7418a4f656bb2665e0962e8bcad9e5`，其依赖的结构审计脚本两端哈希均为 `23ce9a67631ef7faeecb69c1fff824ed63d69ffdd437dc4c870b83ab8cf48da2`。

服务器新建、独占写入 `outputs/student_metrics/wangjiahao/activation_holdout_provenance_20261031/prior_exposure_inventory.json`，288,440 字节，SHA-256 `5d920fa1546fda39462f4ba71ad52e330ae5a00bef8647fd48005a7b5392cdc94`。内容含 **1,301 个 NB201 canonical ID、120 个 DARTS 规范化签名、859 份来源文件哈希**。扫描全部 `outputs` JSON 后，NB201 只见已在 CSV 排除集中的 64 个主网格 ID，未增加新 ID；一份无法解析的 E37 代码哈希 JSON 不含任何架构字段。清单还保留 **453 份只有 DARTS 编号的来源**（多数为主网格逐键诊断与原始尝试），状态明确标为 `unresolved_darts_id_only`。这不是已通过的独立性证明；须把每类编号追溯到冻结签名后才可封存 DARTS 留出。
