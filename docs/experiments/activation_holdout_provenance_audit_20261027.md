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

## NB201 留出架构预选冻结（2026-10-31）

新增 `scripts/select_activation_holdout_nb201.py`，只接收官方 `cifar10` 架构 **键**、官方元数据和已用 canonical ID；以数字序排序后由 `random.Random(20261027).sample(...,64)` 等概率抽样，**不接收或读取目标值**。3 项针对性测试、完整 201 项本地测试通过。脚本在本地/服务器 SHA-256 均为 `50b164f7e3ebcd1a09f7f0b82417ef1f63546faa122e81499fc0abda46d846c1`。

官方 6,466 个 CIFAR-10 架构键扣除 1,301 个已用 canonical ID 后剩 5,165 个可候选。官方仓库中的 `meta-node-4.pth`（SHA-256 `275f76034ee3a4df01c39176101c036642520380d771c61a287b8f75daad02e0`）含 15,625 个结构，与官方 `meta.json` 对全部 6,466 个 canonical 架构的结构字符串逐一一致；选中第一个架构 `2936` 已在 CPU 上成功构造 615,386 参数模型（只验证加载，不计算 metric）。

服务器独占写入 `outputs/student_metrics/wangjiahao/activation_holdout_provenance_20261031/nb201_selection_unsealed.json`，10,099 字节，SHA-256 `07eeafcc8dbc245ffab169173678b9bf53f7382b41a73c4690783a694ef012a1`。文件固定了选择种子、算法、64 个 ID/结构、e0/e5/e10/e25/e50 的预期 320 键，以及官方数据、元数据、模型结构、排除清单和选择代码的来源哈希。独立读回确认 64 个 ID 与结构唯一、与排除集无交集、文件不含目标值字段。状态仍为 `unsealed_weights_and_targets_missing`：这只是架构选择，不是 320 份权重、正式目标验证或任何留出结论。

## NB201 留出权重训练计划（2026-10-31）

只读检查主网格现有 `arch_5998_epoch_50_seed_0.pt` 的 `metadata.run_config`，核对其训练配置为 CIFAR-10 全量 50,000 张、C=16/N=5、50 epoch、PGD-7（ε=8/255、步长 2/255、随机起点）、batch 256、SGD momentum 0.9 / weight decay 5e-4、OneCycle 最大 lr 0.1、seed 0；该检查点 SHA-256 为 `e19cc651ea1fd709f899ea10e77b4b5bd47c9583fa20aaf22e86afa23c437f69`。新留出训练沿用这些参数，不运行指标计算，也不读取官方目标值。

新增 `scripts/train_activation_holdout_nb201.py`。它锁定选择清单、旧训练函数源码 `452c9d84e011d589f7dc1c4b23b482aec4f54e18a20620904dfbcdbabadb22be`、服务器攻击函数源码 `65b348cedccfb4fcea1aafc322c2f9984ec6c23d01b0bef232ebe760c8aeee13`、官方 `meta-node-4.pth`、官方仓库提交及项目内 CIFAR-10 训练文件哈希。每个架构使用新建的独立尝试目录；仅在五份检查点哈希收齐后写完成收据，失败尝试原样保留。重启时只跳过完成收据和五份实物均匹配的架构；损坏收据会报错，不会静默重训或覆盖。4 项针对性测试先失败后通过；本地完整 **205 项通过**。

第一版服务器脚本 `scripts/train_activation_holdout_nb201.py` SHA-256 `9e9cf3cda7b45550131b805d01158fdd0a8207a158c35dd91bd42b18bf97ad60`，曾仅执行 `--plan-only`，在 `outputs/student_metrics/wangjiahao/activation_holdout_nb201_weights_20261031/training_plan.json` 留下 SHA-256 `ff6e8af15fbe50e19c34d40aa5cfd2289a8b3f2521b6db83ad9fc979f46d43b0`。发现计划未锁定执行脚本自身后，在运行任何训练前将该版标记为**仅供审计、不得执行训练**，不覆盖其文件。

修订版本地 `scripts/train_activation_holdout_nb201.py` 与服务器新文件 `scripts/train_activation_holdout_nb201_v2.py` 字节 SHA-256 均为 `99bd26c2b8fd4849641ec407e0fb892151adc6fc6205667ec2251e5140537081`，调用时必须提供匹配的 `--expected-runner-sha`。服务器仅执行 `--plan-only`，独占生成 `outputs/student_metrics/wangjiahao/activation_holdout_nb201_weights_20261031_v2/training_plan.json`，SHA-256 `26fb65655fc60c98272ca2df751aba2d1c0a676dc37b7ce49176f67ab1430bd2b`。该版才是后续训练依据；读回确认 64 架构、320 预期权重键、6 个 CIFAR-10 训练文件哈希、无目标字段。**未启动 GPU 训练，未产出留出权重或成绩。** 先等待当前 560 键主实验流程使用 GPU，之后再按冻结计划训练；若主实验结果未达到预定主榜门槛，仍保留此未查看分数的留出包供下一轮候选使用。

## DARTS 留出结构预选，尚未封存（2026-10-31）

补查 `wangjiahao` 项目内 E32 记录 `docs/experiments/e32_signed_capacity_falsified.md`（SHA-256 `e961c7bb1bc9133c6fc8dff1bc98b316cc6f3e3e4757c1f40a253f1d2b00158e`）：文档称 E32 复用既有材料并对 20 个 DARTS 架构作零成本重跑，列出 `outputs/e32_darts_e0_zerocost.csv`；其分析脚本 SHA-256 为 `823a8027b1f71e3b3c35878c15210731e672f5a042c4c48d0f5ce69534778cfc`。这些材料仍未给出新列所用的逐结构签名、检查点映射或原始运行命令，因此 **E32 未解状态不撤销**。历史身份清单中 453 份 DARTS 只有编号的来源仍按未解保留。

新增 `scripts/select_activation_holdout_darts.py`，只读 120 个已知历史规范化结构签名，以 `seed=20261027` 从项目原有 DARTS 离散空间的固定随机流中按顺序取前 20 个不重叠结构，分配新的编号 `1000..1019`；不接收任何目标或指标值。选择脚本本地/服务器 SHA-256 均为 `2778206ea8b8d2200e35f37713d23cc8e8f31e8ac273bf3bd36573461b466aae`，空间生成源码两端 SHA-256 均为 `3904d99614bae0d8218603ddf966a07e572a54d32d5a0d33b9518aa2ee0bb572`。选择代码的 2 项针对性测试先失败后通过。

服务器独占写入 `outputs/student_metrics/wangjiahao/activation_holdout_provenance_20261031/darts_selection_unsealed.json`（SHA-256 `f3e272c91dbe0ed2746f7d40ebc8cbc7f497519bf9d5286d1c48bdde81231e86`），固定种子、候选流、20 个 ID/结构、40 个预期阶段键、相关来源哈希和 `unresolved_id_only_source_count=453`。读回核对 20 个规范化结构唯一且与已知 120 个不重叠，无目标字段。**状态为 `unsealed_prior_id_only_sources`，不是独立性证明**；在 E32 等缺失出处补齐前，不对这 20 个结构作正式留出比较或宣布跨空间外部验证完成。

用户随后说明 E32 当时似乎由 AI 执行，当前不知道原始运行命令、日志或逐结构映射。该回忆不能代替可核验的来源证据；E32 继续记为 `unresolved`，不通过删除该历史表、仅凭编号相同或对照指标数值相近来解除排除风险。DARTS 留出架构仍只是一份未封存的预选清单。

## DARTS 留出目标逐图证据准备（2026-10-31）

只读核对主网格 `darts100` e100 检查点元数据和既有 `scripts/run_darts_metric_trajectory.py` / `scripts/evaluate_darts_checkpoints.py`：主网格为 C=16、8 cells、全量 CIFAR-10、TRADES PGD-10 β=6、100 epoch、SGD momentum 0.9 / weight decay 2e-4、cosine、batch 128、seed 0。旧完整测试目标在 e100 权重上用 PGD-20、ε=8/255、步长 2/255、随机起点、seed 400000、10,000 图；上述两份服务器源码 SHA-256 分别为 `033c160a8bfe4021e825b7e85a40b21a4ee186c932876118b27221c6227498b2` 和 `08409844ca21c65d244e14b1477457433353af19967db99d06983f6885537b9c`。旧评估器只给准确率汇总，没有逐图证据。

在本地隔离工作区新增 `scripts/evaluate_activation_holdout_darts_targets.py`（SHA-256 `9c2f1387b805338892419bdcfa4c73718be2b2da1fd289eec9b19e6dcb228ced`），把原 PGD 批次循环的逐图布尔结果保存为独占写入的 10,000 图 JSON，并生成与 `scripts/verify_holdout_targets.py` 匹配的目标条目。测试确认逐图平均值与原 `pgd_accuracy` 在相同随机种子下完全一致、10,000 图文件可由现有验证器复算且不能覆盖、缺图与非布尔值拒绝。3 项针对性测试和本地完整 **210 项通过**。当前代码**仅在本地准备，未在服务器部署或运行**；未解 DARTS 出处、20 份 e100 权重及正式目标材料仍未补齐。

服务器同步尝试在 2026-10-31 03:28 UTC 遇到自动审批审查的使用额度限制：`verify_holdout_targets.py` 的传输命令已提交但尚未独立核哈希，`evaluate_activation_holdout_darts_targets.py` 的 base64 输入已发送而结束标记操作被拒，故**不能声称两份服务器文件已完成同步**。审查消息注明这是额度问题、不是安全性否决，提示北京时间 14:49 后重试。当前 SSH 会话仍在等待该结束标记；额度恢复后先核对终端和文件状态，绝不覆盖既有文件或另开渠道绕过审查。

## 留出封存入口加固（2026-10-31）

本地 `scripts/build_activation_holdout_manifest.py` 新增针对完整 371 键的两项结构约束：目标准确率须在 `[0,1]`，NB201 不得以不同 ID 复用同一结构，DARTS 用节点内双边顺序无关的规范化签名检查重复。封存函数现在必须读取一份实际存在、SHA-256 与清单一致、`status=PASS`、计数为 11+320+40、对象列表哈希与本清单精确一致的目标验证报告；单独设置 `targets_verified=true` 不再足以封存。新增边顺序别名、缺报告、报告 SHA/对象不匹配等测试，本地完整 **215 项通过**。

该入口加固只防止标志位或错配报告造成的误封存，**尚不证明报告本身逐文件重算了 RB40/NB201/DARTS 目标**。正式封存仍须先完成独立目标验证器的实物、来源及协议核对；目前没有真实 371 键留出包，也没有调用封存函数生成正式文件。新代码仍仅在本地隔离工作区，待自动审批额度恢复后再按哈希同步到 `wangjiahao`。

## NB201 官方目标逐项核对（2026-10-31）

本地 `scripts/verify_holdout_targets.py` 加入官方 NB201 源文件加载和逐条核对，当前 SHA-256 `f84eaa5a9f811ee3139f8da1b62d228daba568c07682e06ac1f3ec5020e4343d`。加载前要求官方结果 JSON 与 `meta.json` 分别匹配既定 SHA-256 `b670b9ce5a7e6cd29735c07806efa493ba0f7023cc46940b856e214a70b3a308`、`54b6d2c2780cfb63de98e9d67e72db9cb0c02be426191d429dbe02c898d4577e`；逐条要求 canonical ID、官方结构字符串、来源标识/哈希和 `autoattack` 数值与清单一致。合成样例测试覆盖错值、错结构、非规范 ID、错源哈希；实际官方文件只读格式检查得到 CIFAR-10 6,466 条、元数据 15,625 ID，没有输出新选 64 架构的目标值。针对性 22 项、本地完整 **217 项通过**。

此函数尚未形成 320 键的真实目标验证报告，也没有补齐 RB11 来源核对或 DARTS 逐图文件。服务器早先传输的旧版 `verify_holdout_targets.py` 状态仍待独立核哈希；本次本地新版不得视为服务器已同步或正式封存证明。

继续在本地新增 `verify_nb201_target_rows`，要求五个阶段各 64 条、总计精确 320 个唯一 `(setting, canonical_id)` 键、同一 64 ID 集合、e0 结构唯一，并逐键复核官方 AutoAttack 目标。只输出 PASS、计数、键集哈希和官方源哈希，不输出各架构目标值。测试先失败后通过，覆盖 319 键、重复键、错官方目标与非字符串 ID；本地完整 **218 项通过**。当前本地验证器 SHA-256 更新为 `bda61022f4e90d65086cf703d8a3f925e595e98abb214aabc31aba23005d9367`，上一段哈希仅记录当时版本；服务器未同步此更新，未对真实 64 架构运行批量验证。

## RB11 冻结目标与恢复映射核对准备（2026-10-31）

指定本地项目的 `docs/experiments/RB40留出权重恢复验证_20261027.md` 与 `outputs/student_metrics/wangjiahao/holdout_checkpoint_recovery_20261027/recovery_verification.json` 只读复核：原冻结 RB11 manifest 预期 SHA-256 `b529fc7b722f1314a4cac9c6bf9ee383043d489e374d87e2119d8cce88fce372`，原 11 权重哈希清单预期 SHA-256 `9ddf7faca73ccdfa7be0a8eb7a73f5ac337aaab80406bf1e8b6de2b8069fd7e4`；仅 `Rebuffi2021Fixing_70_16_cutmix_ddpm` 改用替换权重 SHA-256 `0769d395cbf77dc49caeab7e4c48fec5299e694937bd0b598293410a0e39a9bb`。恢复 JSON 的本地 SHA-256 为 `263c0eea26bdf8c324ecb851f5b1f3440f872c30a0963789b1a57512414a2f90`，按该哈希只读解析得到加载检查 `PASS, 11/11`。用户确认替换文件来自 RobustBench 登记链接，但没有独立发布的官方文件哈希，故不把这一来源声明写成字节级官方认证。

本地验证器新增冻结 RB11 清单的哈希/11 模型/目标范围检查，以及 11 条目标与该清单逐项一致、恢复记录与单文件替换哈希一致的纯函数。合成测试覆盖 10/11、改目标、改替换哈希与恢复 JSON 错哈希；本地完整 **219 项通过**。当前 `scripts/verify_holdout_targets.py` SHA-256 为 `f4c9b017450e09d3b1c470646d0278b8d7012c620f6f4a8b64cb6374d1c6fa9f`。**原 RB11 目标 manifest 和权重哈希清单不在本地隔离工作区，真实 11 条目标尚未运行本验证器核对**；该逻辑也尚未在服务器同步。恢复记录只证明权重可加载，不证明任何 Ours/Φ 留出结果。

## DARTS 目标批量核对准备（2026-10-31）

本地验证器新增 `verify_darts_target_rows`：要求 DARTS e0/e100 各 20 条、合计 40 个唯一键，两个阶段使用同一 20 个架构编号和同一最终权重目标；逐个读取 e100 的 10,000 图 PGD-20 正误文件、验证文件哈希并复算准确率。逐图 JSON 现在拒绝重复字段和非有限 JSON 常量，避免解析器覆盖前一字段。合成测试覆盖错阶段数量、两阶段目标错配、重复字段等；本地完整 **221 项通过**。当前 `scripts/verify_holdout_targets.py` SHA-256 为 `4539ac8b459e4239d046644f023c97cb4f1bdf8e178b7796c72fe07571bf65ee`。这些是验证代码及合成测试，**真实 20 个 DARTS 目标文件尚不存在，40 键未验证，留出集未封存**。服务器同步仍受此前审批额度限制，不能声称服务器已有此版本。

进一步新增逐检查点文件核验：每个键只能出现一次，文件路径必须留在指定项目内，按 1 MiB 块读取并复算 SHA-256；同一路径不能对应冲突的哈希。合成测试覆盖正确文件、错哈希、越界路径和重复键，本地完整 **222 项通过**。当前本地验证器 SHA-256 为 `fd87bbfb4ae151b898313af8afa0d88db9d99c232adf8c4392c5563a995561ac`。该函数仍是留出整体验证的组件，**尚未读取真实 371 份材料，也不能据此宣布外部验证完成**。

## 三空间 371 键目标验收入口（2026-10-31）

本地新增 `verify_holdout_package_targets`，先检查 371 键结构，再按项目内实际字节复算所有检查点哈希，逐项核对 RB11 冻结目标与恢复映射、NB201 官方 64×5 目标、DARTS 20×2 的逐图目标证据。RB11 另要求旧 11 项哈希清单与恢复报告的既定 SHA-256 一致，并将未替换的十份权重哈希加旧损坏件哈希与冻结清单对齐。输出报告绑定全部对象的规范化哈希，仅包含分组计数和证据哈希，不输出目标数值。端到端合成材料覆盖 371 个键、20 份 10,000 图文件和实际权重字节；篡改权重后验收失败。本地完整 **225 项通过**，当前验证器 SHA-256 为 `438dafc9f8eb4c94610bf20a41d79e090feef61bcb79f4461a4c6fc5b1f7a982`。

这只是本地代码与合成数据核验。冻结 RB11 哈希清单的实际格式及真实 371 项材料仍待在 `wangjiahao` 项目核对，服务器新版尚未同步。E32 结构出处未解，DARTS 预选仍未封存；即使合成测试通过，也**没有真实 371 键验证报告、独立 Ours/Φ 分数或胜出结论**。

后续代码审查发现 RB11 旧权重清单只比较哈希多重集合，若两个模型交换权重仍可能通过。已加入逐模型检查点**文件名与冻结哈希清单同名条目**的绑定，并以交换两份哈希的反例验证拒绝；371 键端到端合成材料相应改为 11 个独立命名的 RB 旧文件加一份替代文件。本地完整 **225 项通过**，当前验证器 SHA-256 `d0734cef9487c444709d608abe3c1013a0d22e3e9ee4192385901b326506cac7`。仍需读取服务器原始清单，确认其真实格式与模型的 `artifact_files` 对应关系；当前代码的合成测试**不能替代该实物核对**。

## E32 DARTS 历史身份的本地数值指纹（2026-10-31）

在指定本地项目内只读比对三份旧输出：`outputs/e32_darts_e0_zerocost.csv`（SHA-256 `a6824c4c5ccceeeda3b8f47f32779b8a5c0185214dcce8a492e49b8f1552e707`）、`outputs/e27_darts_e0_zerocost_v3.csv`（`9284712578205e8d205a2155087ab1194e3fc5f6b2df4baf2352232fdc90a52f`）和 `outputs/e23_cross_space/darts20/metrics_20_e0e50e100.csv`（`f58f70a48ea939204b5977973102497896ba4001e6188afb1dcfc163fdf05cc7`）。E32/E27v3 均为 ID 0..19 的 20 行；共同的 57 个数值字段中，**53 个字段在全部 20 个同 ID 行的 CSV 文本值完全一致**，用这 53 列作跨行指纹时每一 E32 行均唯一命中 E27v3 的同 ID 行。另四个 Jacobian/梯度字段有差异。E32 与 E23 e0 的 20 个同 ID `final_pgd_accuracy` 也逐项相同，E23 e0 有 20 个唯一结构签名。E27 原评估脚本默认从 E23 的 20 架构 CSV 读 `signature`；这些事实**强烈提示** E32 复用了 E27/E23 那批架构或其中大部分原始评估数值。

该比对是数值指纹推断，**不是 E32 原始命令、权重路径或逐 ID genotype 映射的直接证据**；不能据此把 E32 从 `unresolved` 改成已核验，也不能宣称新 DARTS 20 预选与全部历史结构不重叠。下一步仍需在 `wangjiahao` 原始材料中找 E32/E27v3 的运行记录或更直接的来源映射；若找不到，保留未解并阻止独立留出封存。
