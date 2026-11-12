# 项目状态（PROJECT_STATUS）

最后更新：2026-11-12
更新人：接管整理（分支 `codex/mainline-consolidation-20261112`）
状态标记：**服务器一侧尚未盘点**（SSH 端口不可达，见第 7 节）。以下"已验证事实"仅限本机与
已入库文档；凡涉及服务器现状的，一律标注为待核实。

---

## 1. 研究目标（不变）

网络内部特征在干净样本与扰动样本之间的变化，能否预测一个图像分类模型的真实对抗鲁棒性？

两个子问题：

1. **已训练防御模型**：特征空间稳定性指标能否预测/排序其当前真实鲁棒准确率？
2. **未训练/训练早期架构**：训练前或早期的指标，能否预测该架构按固定协议训练完成后的鲁棒性潜力？

子问题 2 有额外约束（`m00001`）：不能拿随机初始化模型自己的白盒梯度造攻击，要用固定扰动、
共享对抗样本或成熟模型生成的扰动；预测目标是"按同一协议训练完成后"的鲁棒准确率。

## 2. 当前口径：哪一份结果算数

**权威榜单**：服务器 `outputs/student_metrics/wangjiahao/strict_adv_excess_release_20261024T110422Z/`，
`release_verification.json` = PASS，48,160 条唯一原始行，84 项入榜，2 项 RACL 因证据不足排除。

| 指标 | all_setting_mean | 名次 |
| --- | --- | --- |
| TRNAS Φ | 0.7822 | 1 |
| R-Score | 0.7378 | 2 |
| Ψ | 0.5981 | 4 |
| E37 multiview max | 0.5241 | 6 |
| RMS | 0.4950 | 7 |

**已作废、不得再引用名次**：`docs/experiments/E36_E37_严格跨空间实验最终报告.md`（2026-10-09，
`strict_final_release_20261008T000000Z`，80 项，当时 max/RMS 排第 3/4）。该版本已被 2026-10-24
的 TRNAS 修复版取代。**同一篇文档里若出现两套名次，一律以 10-24 版为准。**

**2026-11-12 复核（直接从 CSV 读出，不依赖报告叙述）**：`RobustMetric_严格全部指标总表_86项_三位小数.csv`
共 86 行 = 84 条 `ranked` + 2 条 `excluded`（两条 RACL 因"空间不可比"排除，理由与文档一致）。
前 8 名与上表逐位吻合：Φ 0.782 / R-Score 0.738 / adversarial_loss 0.614 / Ψ 0.598 /
natural_loss 0.591 / multiview max 0.524 / RMS 0.495 / fsp_delta 0.479。
每条指标的 `raw_rows` 均为 560，`release_id` 均为 `strict_adv_excess_release_20261024T110422Z`。
另注：本仓库 `outputs/strict_all_metrics_table_20261024_trnas_repaired/` 那份是**83 项**的前身快照
（`raw_rows` 47,600），数值与总表一致但少一条指标——不要把它当成最终发布版。

## 3. 已完成的关键实验（已验证事实）

1. **TRNAS 数值修复**（`docs/experiments/TRNAS_Kang直接求导补算与严格重排_20261024.md`）
   旧版 `rb40_mature/Kang2021Stable` 求导失败，报错 `AssertionError: underflow in dt 0.0`。
   改为 `torchdiffeq.odeint` 直接自动求导后，最大参数变化从 1.06e19 降到 1.13e-6
   （CPU 隔离跑出 Ψ=0、Φ=686.1439208984375、R-Score=0）。只替换失败键，其余 80 项原分数差为 0。

2. **与 TRNAS 的结构匹配与阶段稳定性**（`docs/experiments/TRNAS与自研指标结构匹配及阶段稳定性_20261024.md`）
   八设置均值差：max−Φ = −0.258 [−0.353, −0.159]、max−R = −0.214、max−Ψ = −0.074、RMS−Φ = −0.287。
   阶段秩相关：Φ 在 NB201 e0/e50 为 0.914、DARTS e0/e100 为 0.995；
   自研 multiview max 仅 0.568 / −0.197，RMS 0.493 / −0.243。
   RB40 控制家族与参数量后偏相关：Φ −0.013、max 0.186、RMS 0.231（原始 0.576/0.302/0.226）。

3. **自研指标定位**（`docs/experiments/` 与 activation-survival 分支报告）
   NB201 五阶段 max 0.686/0.735/0.750/0.677/0.657，RMS 0.663/0.716/0.761/0.653/0.630；
   同期 Φ 为 0.849/0.825/0.797/0.753/0.771，逐阶段领先。DARTS e100 上 max 0.006、RMS −0.036，
   而 Φ 0.847。单模型评估耗时 3.08–5.43 秒。

4. **DARTS E37 逐层诊断**（`docs/experiments/DARTS_E37_逐层诊断最终验收补充记录_20261026.md`）
   v4 最终验收 PASS（200 键、100 跨阶段对），保存分数最大绝对误差 9.96e-17。
   **仅诊断，不入总榜**，不能据此宣称超过 Φ。

5. **教师交接核对**（`docs/experiments/教师交接README服务器核对_20261031.md`）
   huangyue 容器只读核对：README SHA-256 `46b0da7e…`；huangyue 的 RB40 缺 15/45 artifact；
   两容器 `TRANSFER_STATUS.json` 均为 IN_PROGRESS；三张主网格 manifest 哈希
   RB40 `5b519799…`、NB201 `a158c609…`、DARTS `7761b4a4…`。
   **结论：教师数据包尚未齐备，不能宣称已交付。**

## 4. 可信结论（分三级）

**已验证事实**

- 在当前的 84 项严格榜上，TRNAS Φ 排第 1（0.7822），自研 multiview max 排第 6（0.5241）。
- 自研指标在 NB201 五阶段内部排序能力可用（0.63–0.76），但**跨空间**（DARTS）几乎失效（≈0）。
- RB40 上自研指标与鲁棒性的相关性，在控制家族与参数量后大幅下降（max 0.576→0.186）。

**合理推断**

- 自研指标的排序力主要来自"架构族内的相对差异"，而非跨族的普适规律；
  这解释了为什么 NB201 内有效、DARTS 上失效。
- Φ 的优势部分来自其结构与模型容量/家族无关，但这只是推断，尚未做消融验证。

**尚待验证的假设**

- 特征空间稳定性是否是**因果**因素，而不只是与鲁棒性共变的伴随现象——目前所有结果都是相关性。
- 训练早期指标能否预测"训练完成后"的鲁棒性——需要同一协议训练到底的对照，目前尚无完整证据链。

## 5. 四方差异盘点结果（2026-11-12，本机可完成的部分）

**结论：`codex/server-mainline`（`dc6fb9c`）就是服务器主线的本地代表。** 依据是内容哈希，不是提交图，
也不是时间戳。

决定性证据：主工作树里那 6 个"从未提交"的文件，与 `dc6fb9c` 里的版本**逐字节相同**。

| 文件 | 归一化后大小 |
| --- | --- |
| `src/robust_metric_lab/metrics.py` | 67,447 |
| `src/robust_metric_lab/attacks.py` | 2,628 |
| `src/robust_metric_lab/metric_registry.py` | 5,867 |
| `scripts/compute_metric_batch.py` | 17,579 |
| `docs/experiments/实验管理.md` | 109,498 |
| `docs/theory/理论管理.md` | 22,357 |

含义：**那 571 行指标代码从来没有丢**，它已经提交在 `server-mainline` 上，只是没进 `main`。

九棵树的横向对比（目标 = 本分支，见 `results/manifests/reconciliation-20261112/tree-comparison.md`）：

| 工作树 | 独有文件 | 独有体积 | 内容不同 |
| --- | --- | --- | --- |
| main | 64 | 31.73 MiB | 7 |
| adv-excess | 27 | 0.85 MiB | 4 |
| ba02 | 391 | 14.65 MiB | 0 |
| strict-metrics-e37 | 441 | 15.01 MiB | 0 |
| activation-survival | 78 | 17.47 MiB | 3 |
| repository-reconciliation | 2 | 0.01 MiB | 2 |
| **server-mainline** | **37** | **0.62 MiB** | **56** |
| three-space-benchmark | 51 | 0.68 MiB | 10 |

**一个必须记住的陷阱**：`server-mainline` 那 56 个"内容不同"里，**46 个只是换行符不同**
（该工作树按 CRLF 检出）。真正的差异只有 10 个文件。凡是不做换行符归一化就下结论的对比，
都会把结果夸大 5 倍。

## 6. 未完成的问题（按优先级）

1. **服务器现状未知**：`/workspace/robust_metric_lab_nas` 的代码、结果、manifest 未盘点（阻塞，见第 7 节）。
2. **主线未确认**：本地 `main`（784fc74）、`codex/server-mainline`（dc6fb9c）、
   `codex/three-space-benchmark`（2bcb95c）、服务器三者谁是最新，需要按内容哈希判定。
3. **三空间独立留出集仍缺**：`三空间独立验证材料审计_20261025.md` 指出 371 键独立验证包未冻结。
4. **激活状态保持候选结果未知**：`feature_activation_sign_survival_sum` 的 560 键全网格
   在服务器上跑完了没有，无从确认（等待 SSH）。
5. **571 行未提交指标代码未评审**（见第 6 节）。
6. **教师数据包未齐**：huangyue 缺 15/45 RB40 artifact。

## 7. 本地未提交资产（已确认为 server-mainline 的子集，不再有丢失风险）

主工作树 `784fc74` 有 6 个已跟踪文件被改动但从未提交，其中
`src/robust_metric_lab/metrics.py` 新增 571 行（`perturbation_capacity_iqr/multiscale/topk`、
`jacobian_spectral_norm/effective_rank/spectral_decay` 等）。

已保存为补丁：`results/manifests/reconciliation-20261112/patches/main-uncommitted-784fc74-20261112.patch`
（83,983 字节，SHA-256 `d3329f9ca31cf5a1a0a83b39b3e8a22c14db6f6cda707e66c8a4d51680f3a0d8`）。
依赖该补丁的两个文件（`e36_batched.py`、`test_e36_batched.py`）以原始字节保存在
`results/manifests/reconciliation-20261112/held/`。

## 8. 当前阻塞：服务器 SSH 不可达

2026-11-12 16:2x 实测（`socket.connect_ex`）：

| 端口 | 结果 | 含义 |
| --- | --- | --- |
| 17010 | error 10061 | 主机在线，但该端口**没有服务监听**（连接被拒绝） |
| 17018 | error 10061 | 同上 |
| 17013 | error 10061 | 同上 |
| 22 | open | 是另一台机器（host key `SHA256:Wd29pYQlkJNZzh4e4jst7fYt5H1o68eopbfdHflonTQ`，与 known_hosts 中三条记录都不匹配），`guzhengkai` 认证失败 |

`known_hosts` 里记录的三个目标主机密钥分别是 17010 `SHA256:uxigsIW4dwxlfweYzCEhF5iJkF5qmx0vZ41/VIk5Xsk`、
17018 `SHA256:IL7ZtDjAKWmmK/ihyFsgJ4IqMV0Q6GEPoibfUwZVwqQ`、17013 `SHA256:TARUCHl9kMyYacOF34dNCoazUPmDyRpjVs8O5lmIcRA`。

判断：容器（`wangjiahao`）或其端口映射已下线/重启换了端口。需要恢复端口映射或取得新端口后重试。

## 9. 下一步

1. 服务器只读普查：`tools/server_inventory.py`（只 stat + sha256，不写服务器）。
2. 内容比对：服务器源码 vs 9 个 worktree，按 SHA-256 而不是时间戳判定谁新。
3. 确定主线分支，合并/重基于服务器主线。
4. 服务器大结果建 manifest（绝对路径、生成脚本、日期、大小、SHA-256）。
5. 补齐 `docs/HANDOFF.md`，并回填本文件的"已验证事实"。
