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

**结论（第二轮修正）：代码主线是 `strict-metrics-e37` 一线（`893a994`，2026-10-09）。**
第一轮只看"主工作树那 6 个从未提交的文件与 `dc6fb9c` 逐字节相同"，就判定 `dc6fb9c` 是主线；
第二轮通读全部快照后发现 `893a994` 的 `metrics.py` 是 `dc6fb9c` 的**严格超集**，结论必须改。

判据用函数集，不用字节数（字节数会骗人，函数集不会）：

| 提交 | 提交日期 | `metrics.py` 字节 | 顶层函数数 | 与其它分支的关系 |
| --- | --- | --- | --- | --- |
| `784fc74`（main） | 2026-09-30 | 42,297 | 31 | 最旧 |
| `dc6fb9c`（codex/server-mainline） | 2026-09-18 | 67,447 | 51 | main 的超集 |
| `2bcb95c`（three-space） | 2026-09-22 | — | 52 | 比 server-mainline 多 1 |
| `893a994`（strict-metrics-e37） | 2026-10-09 | 76,626 | **56** | **`dc6fb9c` 的严格超集** |
| `dfc67f2`（activation-survival） | 2026-10-31 | 43,562 | 33 | 只比 main 多 2，不含 e36/e37 |

`893a994` 相对 `dc6fb9c` 多出的 5 个函数：`_e36_cross_layer_stats`、`_e36_spectrum_probabilities`、
`_e36_within_layer_stats`、`_safe_cosine_distance`、`_symmetric_eigvalsh_stable`。
注意 `dfc67f2` 提交最晚（10-31），代码底子却是最老的一支——**提交时间新不等于代码新**。

`dc6fb9c` 的定位因此收窄为"服务器基线提交"（服务器目录据记录从它导出），而不是"最新代码"。

**处理方式：两条线都并进来，冲突按函数集判胜负。**
1. 合并 `dc6fb9c` → `7400d49`。当时判定它是主线，这个判定后被修正；合并本身保留，因为它是服务器基线的凭据。
2. 合并 `893a994` → `92c5d10`。4 处冲突：`metrics.py`、`metric_registry.py`、
   `configs/metric_eligibility.json`、`tests/test_student_runner.py`，全部取 `893a994` 侧（超集），
   再把 adv-excess 声明和它的 eligibility 行补回。补之前有 3 个测试失败，原因是
   `metric_eligibility.py` 拒绝"已注册却没有资格行"的指标。
3. 导入 `dfc67f2` 的 68 个文件 + `2bcb95c` 的 9 个未跟踪文件 → `13c1d49`。

测试：合并前 225 passed / 2 skipped；合并 `893a994` 后 284 passed；
导入完全部 leftover 后 **477 passed / 3 skipped**（3 个 skip：2 个符号链接不可用 + 1 个缺 matplotlib）。

九棵树最终对照（目标 = 本分支，明细见 `results/manifests/reconciliation-20261112/tree-comparison.md`）：

| 工作树 | 独有文件 | 独有体积 | 内容不同 | 独有内容是否已解释 |
| --- | --- | --- | --- | --- |
| main | 29 | 31.17 MiB | 35 | xlsx / preview / 两张大 CSV / `.tmp` / egg-info，均按策略 defer |
| adv-excess | 3 | 0.67 MiB | 16 | 教师版 xlsx + 预览图（由已入库 CSV 经 `.mjs` 生成） |
| ba02 | 0 | 0 | 12 | 无独有文件；差异只是它停在旧提交 |
| strict-metrics-e37 | 0 | 0 | 10 | 已全部并入 |
| activation-survival | 2 | 16.91 MiB | 13 | 只有 `data/official_nasrobbench201_source/`（数据集，不入 Git） |
| repository-reconciliation | 0 | 0 | 12 | 已全部并入 |
| server-mainline | 0 | 0 | 54 | 已全部并入 |
| three-space-benchmark | 5 | 0.01 MiB | 33 | 只有 egg-info（新策略已拒绝） |

**两个必须记住的陷阱**：

1. `server-mainline` 那 56 个"内容不同"里，**46 个只是换行符不同**（该工作树按 CRLF 检出）。
   不做 CRLF→LF 归一化就下结论，会把差异夸大 5 倍。
2. `ba02` 与 `strict-metrics-e37` 两个 worktree 里**套着第二层 `robust_metric/` 目录**。
   按 worktree 根盘点会得到 391 / 441 个"独有文件"的假象；按 `robust_metric/` 子目录盘点，
   `ba02` 独有 0 个、`strict-metrics-e37` 39 个（已全部并入）。

## 6. 未完成的问题（按优先级）

1. **服务器现状未知**：`/workspace/robust_metric_lab_nas` 的代码、结果、manifest 未盘点（阻塞，见第 8 节）。
2. **服务器那一侧仍未确认**：本地已判定代码主线是 `893a994` 一线并能全部合并，
   但服务器目录 `/workspace/robust_metric_lab_nas` 究竟停在哪一版，仍要等 SSH 才能核实（见第 8 节）。
3. **三空间独立留出集仍缺**：`三空间独立验证材料审计_20261025.md` 指出 371 键独立验证包未冻结。
4. **激活状态保持候选结果未知**：`feature_activation_sign_survival_sum` 的 560 键全网格
   在服务器上跑完了没有，无从确认（等待 SSH）。
5. **571 行未提交指标代码未评审**（内容与 `dc6fb9c` 逐字节相同，见第 7 节）。
6. **教师数据包未齐**：huangyue 缺 15/45 RB40 artifact。

## 7. 本地未提交资产（已确认为 `dc6fb9c` 与 `893a994` 的子集，不再有丢失风险）

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

**已完成的部分**（2026-11-12，第二轮 + 第三轮）：

- 主线判定（`893a994`）、三条线合并、9 棵树对照归零、测试 477 passed / 3 skipped。
- 服务器大盘点清单已写好（`results/manifests/server/PLAN.md` + `targets.json`），SSH 一通就能执行。
- **本地大产物已建 manifest**：`results/manifests/local/`，422 个文件、1,107.2 MiB，全部 SHA-256（无采样），
  记录绝对根路径 + 相对路径，可逐个核对身份（含 1 GB holdout checkpoint、18 个 `adv_cache`、两份 13.9 MiB 合并表）。
- **GitHub 分支已做可复现性验证**：从 `SupCE-moon/RobustNASMetric` 全新 `--depth 1` clone 评审分支，
  727 个文件 / 21.75 MiB，最大文件 1.76 MiB（无二进制），`PYTHONPATH=<clone>/src pytest tests -q`
  → 477 passed / 3 skipped（103 秒）。说明这条分支不依赖本机任何未提交文件。

**仍然按顺序剩下的事**：

1. **服务器只读普查**（阻塞在 SSH）：`tools/server_inventory.py`，只 stat + sha256，不写服务器。
2. **用第 1 步的结果核对主线判定**：服务器若出现本地没有的源码文件，说明服务器更新，需要再并一轮。
3. **服务器大结果落 manifest**：按 `results/manifests/server/PLAN.md` 第 4 节的四条命令执行。
4. **三空间独立留出集**：371 键验证包未冻结，这是当前最影响"结论可信度"的缺口（见第 6 节第 3 条）。
5. **`feature_activation_sign_survival_sum` 终态**：560 键全网格跑完没有、发布验证过没有，只能上服务器查。
6. **进入第二阶段研究**：主线整理完成后，才谈新指标与新实验（用户明确要求先整理、后设计）。
