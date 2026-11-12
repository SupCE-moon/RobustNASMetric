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
另注：本仓库 `outputs/strict_all_metrics_table_20261024_trnas_repaired/` 是**前身快照**，
文件名写着 85 项：85 行 = 83 条有 `strict_rank` + 2 条空（排除），`raw_rows` 47,600 = 85 × 560。
**报数规则**：说"多少项"必须连 `release_id` 一起说。本仓库有三个入榜数，按时间：
**80**（`outputs/strict_all_metrics_table_20261024/`，85 行；无名次的 5 项实测为
`racl_arch_lipschitz_log_upper`、`racl_fixed_network_log_upper_proxy`、`trnas_activation_psi`、
`trnas_consistency_phi`、`trnas_r_score`，后三者正是 Kang 求导失败拖垮的）→
**83**（`..._trnas_repaired/`，85 行，修复后 3 项恢复名次）→ **84**（最终榜，86 行 = 84 + 2 条排除）。
三者用的是同一批原始记录，差别只在"哪几项算进了榜"。

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

6. **严格排名离线复算（2026-11-12 新增，服务器不可达时做的独立验证）**
   不从报告抄名次，而是用 `tools/verify_strict_ranking.py` 从 47,600 行原始记录重算一遍：
   逐 metric × 逐 setting 算 Spearman、按 `direction` 定向、八设置平均、再排序。
   **结果：83 项逐值最大绝对误差 0.000e+00，名次顺序完全一致**；2 项 RACL 的 `ok` 行数为 0（全是 failed），
   因此被排除——与官方"证据不足"的排除理由精确对应。
   边界：官方 84 项里那 1 项 `feature_adv_excess_centered_unit_l2` 的原始行只在服务器，
   本地这批行冻结在它加入之前，**离线无法复算**，只能等 SSH。
   详见 `results/audits/strict-ranking-recompute-20261112.md`。

7. **指标冗余度：85 项里到底有几个独立信号（2026-11-12 新增）**
   榜单上有 84 项，但"84 项都独立"和"84 项是 5 个信号的变体"是两件完全不同的事，
   审稿人一定会问。`scripts/analyze_metric_redundancy.py` 把每个指标做方向归一化后，
   在每个设置内算指标两两的 Spearman 相关，八个设置取平均，再对 `1 - rho` 做平均连接聚类。
   **结果：ρ ≥ 0.5 时分成 21 簇，最大的一簇有 35 个指标**（全是 `feature_measure_*` 特征漂移变体），
   第二簇 9 个（`perturbation_capacity_*`），第三簇 9 个（协方差/谱距离）。
   值得注意：**TRNAS 的 Φ / R / Ψ 三个指标自成一簇**，与那 35 个主流特征漂移指标不同族；
   输入敏感性的 6 个指标（`local_logit_lipschitz`、`jacobian_spectral_norm`、`logit_kl`、
   `input_loss_delta`、`input_grad_l2`、`racl_input_gradient_lower`）自成一簇；
   `jacobian_effective_rank` 与 `jacobian_spectral_decay` 与主流**负相关**（-0.19 / -0.15），是独立甚至反向的信号。
   边界：这是描述性分析，只说明哪些指标携带重复信息，**不能**说明哪个指标"正确"；
   两个覆盖不足的 RACL 指标（1/8 与 7/8 个设置）已单独标注，它们的相关系数不能与 560 个模型的指标直接比较。
   详见 `results/audits/metric-redundancy-20261112.md`。

8. **文档声称 vs 发布表：33 条关键数字逐条对账（2026-11-12 新增）**
   同一个结论在报告、状态文档、交接笔记里被反复复述，每次复述都是一次抄错的机会。
   `tools/verify_doc_claims.py` 手工维护 33 条"文档说了什么"的清单，逐条回到 86 行发布表核对：
   冠军分数与名次、四对差值（max−Φ −0.258、max−R −0.214、max−Ψ −0.074、RMS−Φ −0.287）、
   NB201 五阶段的三组相关值、DARTS 塌陷值（max 0.006、RMS −0.036、Φ 0.847）。
   **结果：33/33 通过，最大偏差 0.0002（纯属三位小数与四位小数的舍入）**；
   结构检查也全过（86 行 = 84 入榜 + 2 排除、`raw_rows` 恒为 560、名次连续 1..84、`release_id` 唯一）。
   边界：这只证明"文档与表一致"，**不证明表本身正确**——后者由第 6 条的独立复算保证。
   详见 `results/audits/doc-claims-20261112.md`。

9. **指标总索引（2026-11-12 新增）：86 个指标一张表**
   指标定义在 `src/robust_metric_lab/metric_registry.py`、三空间资格在 `configs/metric_eligibility.json`、
   分数在发布表、原始行在 `metric_values_original_strict.csv`。回答"某个指标到底入没入榜、
   方向是什么、属于谁、三空间通不通用"要翻四个文件。
   `tools/build_metric_index.py` 把四者 join 成 `docs/theory/METRIC_INDEX.md`（自动生成，勿手改）：
   名次 / 家族（**按名字前缀猜的**）/ 方向 / 归属 / 八设置均值 / 覆盖（x/8）/ 资格。
   三份数据源的指标名集合必须完全一致，否则工具以非零退出码报 `PROBLEM`；本次 86/86 一致。

   **索引顺带查出一处口径矛盾（尚未处理）**：`configs/metric_eligibility.json` 把
   `trnas_consistency_phi` / `trnas_r_score` / `trnas_activation_psi` 标成
   `implementation_error`，理由写着 "Kang2021Stable float32 step underflow is unresolved under the
   frozen 1e-6 definition"；但 2026-10-24 的 `torchdiffeq.odeint` 直接求导修复已把该 underflow 解掉
   （最大参数变化 1.13e-6），三个指标在发布表上分别排第 1 / 2 / 4。
   即：**配置记录的是修复前的状态，发布表记录的是修复后的结果，两者都没有错，但没人写下哪条是当前口径。**
   索引每次生成都会在"⚠ 需要人看的不一致"一节列出这三条，等确认后一次性改配置。
   详见 `docs/theory/METRIC_INDEX.md` 与 D-016。

10. **文档路径引用体检（2026-11-12 新增）：389 条路径、0 条断链**
    "详见 `results/audits/xxx.md`" 是项目对可追溯性的承诺，文件不在就等于承诺落空。
    `tools/check_doc_links.py` 扫 88 份 Markdown 里所有反引号路径与 Markdown 链接（389 条），
    按四类归位：应当修复的缺失 / 计划文档里的历史文件名 / 外部交付包路径 /
    `outputs/` 下的产物路径。体检抓到并修掉 5 处真错：

    | 文档 | 原写法 | 实际文件 |
    | --- | --- | --- |
    | `README.md` | `docs/metric_baselines.md` | `docs/theory/metric_baselines.md` |
    | `docs/theory/current_metric_table.md` | `docs/current_metric_table_readable_zh.md` | `docs/theory/current_metric_table_readable_zh.md` |
    | `docs/experiments/实验管理.md`（3 处） | `e25_k2_k8_k16_three_source_validation_report.md` | `e25_k8_three_source_validation_report.md` |
    | `docs/experiments/实验管理.md`（2 处） | `e35_final_summary.md` | `e35.md` |

    另有 30 条进"计划文档"桶（`docs/superpowers/plans/` 写的是当时打算创建、后来改了名的脚本，
    例如 `run_activation_holdout.py` 最终叫 `train_activation_holdout_nb201.py`），
    3 条进"已说明为服务器/外部文件"桶（附理由），12 条 `outputs/` 缺失属正常。
    详见 `results/audits/doc-links-20261112.md` 与 D-018。

## 4. 可信结论（分三级）

**已验证事实**

- 在当前的 84 项严格榜上，TRNAS Φ 排第 1（0.7822），自研 multiview max 排第 6（0.5241）。
- 榜单的 85 个指标并不互相独立：ρ ≥ 0.5 时聚成 21 簇，最大一簇 35 个（`feature_measure_*` 漂移变体），
  TRNAS 的 Φ/R/Ψ 自成一簇，`jacobian_effective_rank` 与主流负相关。
- 自研指标在 NB201 五阶段内部排序能力可用（0.63–0.76），但**跨空间**（DARTS）几乎失效（≈0）。
- RB40 上自研指标与鲁棒性的相关性，在控制家族与参数量后大幅下降（max 0.576→0.186）。

**合理推断**

- 自研指标的排序力主要来自"架构族内的相对差异"，而非跨族的普适规律；
  这解释了为什么 NB201 内有效、DARTS 上失效。
- Φ 的优势部分来自其结构与模型容量/家族无关，但这只是推断，尚未做消融验证。
- Φ 之所以能排第 1，一个**待检验**的解释是：它测的是"表示的一致性/平滑度"，而 35 个
  `feature_measure_*` 测的是"扰动后特征漂移的幅度"。两者不同族，所以 Φ 提供的排序信息是互补的；
  但这需要"只保留 35 个漂移类指标、再用 Φ 做增量检验"才能验证，目前只是从聚类结构推出的合理猜测。

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
7. **TRNAS 三项的资格口径待确认**（2026-11-12 新增，不影响任何已发布数字）：
   `configs/metric_eligibility.json` 仍把 Φ/R/Ψ 标为 `implementation_error`，但它们在发布表上
   有第 1/2/4 名。配置记录修复前、发布表记录修复后，需要有人确认后统一（见 D-016）。
   在此之前索引会一直显示这三条警告。

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
