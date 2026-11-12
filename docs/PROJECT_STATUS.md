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
    「详见某份审计报告」是项目对可追溯性的承诺，文件不在就等于承诺落空。
    `tools/check_doc_links.py` 扫 88 份 Markdown 里所有反引号路径与 Markdown 链接（389 条），
    按四类归位：应当修复的缺失 / 计划文档里的历史文件名 / 外部交付包路径 /
    `outputs/` 下的产物路径。体检抓到并修掉 5 处真错：

    | 文档 | 原写法 | 实际文件 |
    | --- | --- | --- |
    | `README.md` | 少了 `theory/` 一层的相对路径 | `docs/theory/metric_baselines.md` |
    | `docs/theory/current_metric_table.md` | 同目录文件却写成 `docs/` 下 | `docs/theory/current_metric_table_readable_zh.md` |
    | `docs/experiments/实验管理.md`（3 处） | 写成了 `e25_k2_k8_k16_three_source_validation_report.md` | `e25_k8_three_source_validation_report.md` |
    | `docs/experiments/实验管理.md`（2 处） | 写成了 `e35_final_summary.md` | `e35.md` |

    另有 30 条进"计划文档"桶（`docs/superpowers/plans/` 写的是当时打算创建、后来改了名的脚本，
    例如 `run_activation_holdout.py` 最终叫 `train_activation_holdout_nb201.py`），
    3 条进"已说明为服务器/外部文件"桶（附理由），12 条 `outputs/` 缺失属正常。
    详见 `results/audits/doc-links-20261112.md` 与 D-018。

11. **八设置平均实际上是"NB201 平均"（2026-11-12 新增，改变对榜单的读法）**
    `tools/analyze_setting_weights.py` 把冻结表的八列拆成三个网格分别排名，结果：

    | 网格 | 设置数 | 在平均里的权重 | 与最终榜的 Kendall τ |
    | --- | ---: | ---: | ---: |
    | RobustBench（已训练防御模型） | 1 | 12.5% | **−0.127** |
    | NAS-Bench-201（5 个训练阶段） | 5 | 62.5% | **+0.926** |
    | DARTS（2 个训练阶段） | 2 | 25.0% | +0.339 |

    即：**84 项名次有九成以上由 NB201 的训练阶段决定，而与"已训练防御模型"的排序几乎无关、略反向。**

    具体到指标（全 8 设置名次 → 只 RB40 名次）：

    | 指标 | 全 8 设置 | 只 RB40 | 只 NB201 |
    | --- | ---: | ---: | ---: |
    | `trnas_consistency_phi` | 1 | 12 | 2 |
    | `trnas_r_score` | 2 | **73** | 1 |
    | `trnas_activation_psi` | 4 | **76** | 3 |
    | `adversarial_loss` | 3 | **2** | 20 |
    | `feature_measure_multiview_max_delta` | 6 | 35 | 7 |

    在 40 个真实已训练防御模型上，最强的是 `feature_adv_excess_centered_unit_l2`（0.766）与
    `adversarial_loss`（0.743）。

    **这些点估计必须配置信区间读**（`results/audits/bootstrap-rb40-20261112.md`，
    按模型重采样 2000 次，种子 20261112）：

    | 指标 | RB40 点估计 | 95% 区间 | 判定 |
    | --- | ---: | --- | --- |
    | `trnas_consistency_phi` | +0.576 | [+0.310, +0.761] | 稳定正相关 |
    | `adversarial_loss` | +0.743 | [+0.487, +0.903] | 稳定正相关 |
    | `trnas_r_score` | −0.184 | [−0.459, +0.136] | **方向不确定** |
    | `trnas_activation_psi` | −0.194 | [−0.471, +0.128] | **方向不确定** |
    | `feature_measure_multiview_max_delta` | +0.302 | [−0.030, +0.589] | **方向不确定** |
    | `feature_measure_multiview_rms_delta` | +0.226 | [−0.119, +0.521] | **方向不确定** |

    两套排序的一致度 τ = −0.143，95% 区间 [−0.243, +0.018]：**区间跨 0，不能说"反向"，
    只能说"不一致"**。此前"R-Score 会把模型排反"的说法是过度解读，已撤回。

    所以对 RQ1 的正确表述是：**在已训练防御模型上，能确认有效的只有 Φ 与对抗损失两项；
    R-Score/Ψ 的方向在 40 个模型上分辨不出来。** 要分辨它们需要更多模型，
    不是更多重采样次数。

    **复现核对**：本审计算出的 84 个 RB40 相关系数与发布表 `rb40_mature` 列的最大绝对误差
    0.000499（即三位小数舍入的一半），零个不一致指标。

    **这不说明榜单算错了**——冻结协议就是八设置简单平均，权重高是协议的一部分。
    它说明的是：**引用"第 N 名"时必须说明是哪一套名次**，不能把全 8 设置的名次
    直接读成"在已训练防御模型上的排序力"。这一条直接影响 RQ1 的表述方式。
    详见 `results/audits/setting-weights-20261112.md` 与 D-021。

    **局限**：区间只说明"在这 40 个模型上排得稳不稳"，不说明"换 40 个模型会怎样"；
    后者需要新的模型集合。另外 `feature_adv_excess_centered_unit_l2` 的原始行只在服务器，
    本机无法为它配区间（它不在上表里）。

12. **Φ 对特征漂移族有实质增量（2026-11-12 新增，按预注册判据执行）**
    第 7 条发现 85 个指标塌成 21 簇、最大簇 35 个全是特征漂移变体，因此必须回答：
    "Φ 排第一，是不是只是漂移族的另一种写法？"

    判据在 `docs/experiments/增量检验方案_Φ对特征漂移族_20261112.md` 里**先于结果写死**
    （D-019），本轮只是执行。基准 B = 35 个漂移指标的组内百分位秩均值，
    独立单元是 **204 个架构**（不是 560 行）。

    | 检验 | 结果 | 读法 |
    | --- | --- | --- |
    | T1 偏秩相关 Δρ | **+0.695**，95% 区间 [+0.583, +0.769] | 控制漂移后 Φ 仍解释大量排序 |
    | 置换检验 | p = 0.0005（2000 次，0 次超过观测） | 极不可能是偶然 |
    | T2 嵌套模型 R² | 0.163 → 0.640，增量 **+0.476** | 加 Φ 后解释力翻近四倍 |
    | T3 同漂移分位内的秩相关 | **+0.637**（8 个设置都可用） | 不是靠区分"整层漂移大小" |

    两个稳健性对照都排除了"是基准构造方式在作怪"：
    PCA 第一主成分替代平均秩 → Δρ = +0.685；每设置挑族内与 target 最相关的**单个**指标
    （乐观上限，真实预测里做不到）→ Δρ = +0.676。

    逐设置明细（Δρ 必须与 ρ(B,y) 一起读）：

    | 设置 | n | ρ(Φ,y) | ρ(B,y) | ρ(Φ,B) | Δρ |
    | --- | ---: | ---: | ---: | ---: | ---: |
    | darts_e0 | 100 | +0.840 | +0.463 | +0.620 | +0.795 |
    | darts_e100 | 100 | +0.847 | **−0.028** | −0.110 | +0.850 |
    | nb201_e0 | 64 | +0.849 | **+0.831** | +0.823 | **+0.522** |
    | nb201_e5 | 64 | +0.825 | +0.581 | +0.530 | +0.749 |
    | nb201_e10 | 64 | +0.797 | +0.658 | +0.570 | +0.682 |
    | nb201_e25 | 64 | +0.753 | +0.467 | +0.385 | +0.702 |
    | nb201_e50 | 64 | +0.771 | +0.545 | +0.500 | +0.686 |
    | rb40_mature | 40 | +0.576 | **−0.125** | −0.085 | +0.572 |

    **关键解读（避免过度声称）**：Δρ 高有两个来源。一是在 RB40 与 DARTS 上，
    漂移族平均对鲁棒性几乎没有预测力（−0.125 / −0.028），控制一个无效变量等于没控制，
    偏相关自然退化成 ρ(Φ,y)；二是在漂移族真正强的地方，Φ 仍有增量。
    **最有说服力的是 `nb201_e0`**：那里 B 与 target 的相关高达 0.831、Φ 与 B 也高度共线（0.823），
    在这种"最难分"的条件下 Δρ 依然有 0.522。

    **可以说**：Φ 不是漂移族的另一种写法，它带来了这些指标没有的信息。
    **不可以说**：Φ 更好是因为它测了"一致性"而不是"漂移幅度"——那是机制解释，
    本检验不涉及。**也不可以说**因果。

    **局限**：只有 85 个指标（第 86 个 adv-excess 的原始行只在服务器）；三个网格的 target
    语义不同，混平均是对齐冻结协议而非自然选择；B 的成员来自一次 ρ≥0.5 的冻结聚类，
    换阈值会变。详见 `results/audits/phi-incremental-20261112.md` 与 D-023。

13. **注册表 ↔ 实现入口对照审计**（2026-11-12，`tools/audit_metric_implementation.py`）：
    86 个注册指标里，**70 个**由一次性 bundle `compute_feature_metrics` 直接返回，
    **15 个**由专门模块计算（`e37_metrics.py` 的多视图/谱/流相关 9 个、
    `student_runner.py` 的 RACL 3 个与 TRNAS 3 个），**0 个**在仓库里完全没有实现痕迹。
    唯一例外：`feature_adv_excess_centered_unit_l2` 只在 `metric_registry.py` 出现——
    它的漂移公式在 `adv_excess_metric.py`（`_unit_centered`）里，但**没有任何代码把这个名字写成输出键**；
    这与"它的原始行只存在于服务器、离线无法复算"是同一件事的两面。
    另一个记录：`perturbation_capacity_importance_delta` 在 bundle 里被刻意写成 `float("nan")`，
    真正计算在 `importance_batched.py` / `evaluate_importance_student.py`（两遍法），
    所以"NaN"在这里表示"不在这个入口算"，不是"算失败"——与 D-014 的"缺失不是零"是同一条原则。
    静态解析（不 import torch）保证在只有 pandas 的机器上也能跑；`--dynamic` 才真的前向一次小 ResNet。

### 14. 指标基本不变量审计（2026-11-12）

**问题**：§3 第 13 条只证明"有代码产出这个数"，没证明"这个数像个距离"。
一个指标可以在注册表里、有实现、有测试，却在零扰动时不为 0，或者随扰动增大反而变小。

**做法**：`tools/audit_metric_invariants.py` 用固定种子的两层卷积小网络，把扰动从
0 逐级加到 8/255，对 75 个字段检查三条最低要求：
`eps=0` 时距离类必须为 0；所有值有限；随 `eps` 单调上升（Spearman ≥ 0.7）。

| 判定 | 数量 | 含义 |
|---|---|---|
| 通过 | **69** | 三条都守住 |
| 已知非单调（有记录） | 1 | `feature_box_dimension_delta`，盒计数维度是量化几何量，不足一像素时整数量跳变 |
| 已知占位 NaN | 1 | `perturbation_capacity_importance_delta`，真值在两步法入口 |
| 随扰动**下降** | **1** | `perturbation_capacity_signed_delta` |
| 记录字段（不是指标） | 3 | `num_layers` / `num_candidate_layers` / `sampled_layer_indices` |
| 零扰动不为 0 / 出现 NaN | **0** | — |

**唯一需要人看的信号**：`perturbation_capacity_signed_delta` 随扰动增大而单调**变负**
（Spearman = −1.000），而它和其他 55 个 delta 一样被声明为 `lower`。
定义是 `(adv_capacity − clean_capacity) / |clean_capacity|`（`metrics.py:1285`），
是个有符号的相对变化，不是绝对差。

**这一步不能说明什么**：小网络是随机初始化的，"通过"只说明它在这个网络上守住了距离的基本性质，
不代表在真实 CIFAR 模型上同样成立，也不代表该指标与鲁棒准确率相关。

**原样保留**：`pytest tests/test_audit_metric_invariants.py` 里有一条断言绑定当前事实
（`signed_capacity_delta` 判为 `decreasing`）。若将来改了它的定义或方向，必须显式改那条测试，
而不是让它悄悄变绿。

### 15. 有符号与绝对 capacity 漂移是不是同一个指标（2026-11-12）

**问题**：§3 第 14 条发现 `perturbation_capacity_signed_delta` 随扰动增大而单调下降，
与它注册的 `lower`（越小越好）方向相反。最自然的猜测是"它和 `perturbation_capacity_delta`
只差一个符号"——如果是，两者在榜单上的名次应该能互相印证。

**代码事实**：`metrics.py:1265` 与 `:1286` 是同一批量的两种写法，
`capacity_deltas` 用 `(adv − clean).abs() / (|clean| + 1e-8)`，`capacity_signed_deltas` 用
`(adv − clean) / (|clean| + 1e-8)`。差别就是一个 `abs()`：前者先逐样本取绝对值再平均，
后者带符号平均。因此 `|signed| ≤ abs` 恒成立。

**做法**：`tools/audit_signed_capacity.py` 读 560 行发布原始行（8 设置 × 模型/架构，含 TRNAS 修复行），
逐设置算四组秩相关：signed 与 target、abs 与 target、signed 与 abs、比值与 target。

| 设置 | n | ρ(signed, target) | ρ(abs, target) | ρ(signed, abs) | 负值占比 |
|---|---|---|---|---|---|
| `rb40_mature` | 40 | +0.041 | −0.616 | −0.326 | 0.90 |
| `nb201_e0` | 64 | −0.264 | +0.216 | +0.010 | 0.06 |
| `nb201_e5` | 64 | +0.197 | +0.308 | +0.441 | 0.05 |
| `nb201_e10` | 64 | −0.243 | −0.184 | +0.446 | 0.23 |
| `nb201_e25` | 64 | −0.613 | +0.100 | +0.024 | 0.73 |
| `nb201_e50` | 64 | −0.839 | +0.543 | −0.566 | 0.95 |
| `darts_e0` | 100 | −0.107 | −0.119 | +0.505 | 0.01 |
| `darts_e100` | 100 | −0.404 | +0.412 | −0.758 | 1.00 |

**已验证事实**：两者在设置内的秩相关**中位数只有 +0.017**，4/8 个设置里它们与 target 的相关
**符号相反**；配对自检 `|signed| ≤ abs` 在 560 行上 **0 例外**（说明两列确实来自同一批样本，
差异不是 join 出错）。

**解释（合理推断，机制未验证）**：signed 是带符号平均，batch 内"capacity 上升"的样本会抵消
"capacity 下降"的样本，抵消程度逐模型变化极大——`darts_e100` 全部同号（signed ≈ −abs），
`darts_e0` 几乎全正（signed ≈ +abs），`nb201_e0` 模型级 94% 为正但 batch 内混合（ρ 仅 +0.01）。
所以 signed 度量的是"**方向净额**"，abs 度量的是"**幅度总额**"。

**可以说**：这两个指标不是同一指标的符号版本，在榜单上不能互相印证，各自的名次要单独解释。

**不可以说**：不能说哪一个"更好"。本轮没有控制模型容量，也没有证据说明 signed 的方向约定是
设计选择还是实现副产物；`analyze_e32_generality.py` 里已有对两者端点一致性的早期观察，
本轮只是把它变成可复跑、带配对自检的量化。

### 16. 交付前自检的统一入口（2026-11-12）

**问题**：仓库里已经有 11 个审计脚本，每个参数不同（`--rows`、`--repair`、`--table`、
`--ranking`、`--redundancy`）。交接文档里列 11 条命令，实际结果是有人跑六条、
有人把 CSV 传错位置，而错的那条通常不会报错——它只是算出了另一个数。

**做法**：`tools/run_audit_suite.py` 把参数接线固定在代码里，按依赖顺序跑完全部 11 步，
输出 `results/audits/audit-suite.{json,md}` 一张汇总表。依赖顺序是硬约束：
冗余度矩阵喂给 Φ 增量检验，所以 `redundancy` 必须排在 `phi_incremental` 之前。

**本次运行结果**（`--rows-root` 自动定位到主工作树）：**11 通过 / 0 失败 / 0 跳过**，
总耗时约 3 分钟。其中两条的退出码是 1 而不是 0，都是**结论**不是崩溃：

| 步骤 | 退出码 | 最后一行 |
|---|---|---|
| `implementation` | 1 | `registered=86 bundle=70 other=15 registry_only=1`（已知的 adv-excess 缺口，D-024） |
| `signed_capacity` | 1 | `verdict='not the same metric' median_rho=0.017`（D-026） |

**两个设计取舍**：

1. **大输入缺失时标 `skipped`，不标通过**。原始行 13.9 MiB 按策略不入库，
   在一台干净 clone 上跑，7 个依赖它的步骤会被跳过。报告里明确写「skipped 不是通过，
   不能当成刚跑过」——这是 D-014「缺失不是零」的同一条原则。
2. **`--rows-root` 三级回退**：显式参数 → `ROBUST_METRIC_OUTPUTS` 环境变量 →
   `git worktree list` 里的主工作树。第三级是为了「在一台只有这份仓库的机器上也能跑」，
   但它会让「跳过」变成「实际跑了主工作树的数据」，所以报告开头会打印实际用的 `rows_root`。

**顺带修掉的自污染**：`check_doc_links.py` 只跳过 `doc-links-` 前缀的报告，
而 suite 里那份报告叫 `doc-links.md`（无日期），于是上一轮报告里列的 25 条缺失路径
被当成新引用又报了一遍。现在改为 `GENERATED_REPORTS = ("doc-links", "doc-claims", "audit-suite")`
按前缀跳过，并加了两条测试：生成报告不扫、人写报告（`metric-redundancy-*.md`）仍然扫。

### 17. 早期指标预测的是鲁棒性，还是只预测了 clean 准确率（2026-11-12）

**背景**：RQ2 一直缺「训练完成后的真实鲁棒准确率」。本轮在
`.worktrees/activation-survival/data/official_nasrobbench201_source/` 找到官方
NAS-RobBench-201 发布（15.6 MiB，按策略不入库），其中 cifar10 有 **6466 个架构**的训练后
`val_acc`、`val_pgd_3.0_acc`、`val_pgd_8.0_acc`、`autoattack` 等字段。e0 的 64 个架构
**64/64 全部能对上**，因此 RQ2 的第一步不需要训练。

**必须先看的陷阱**：在这份数据里，训练后 clean 准确率与 AutoAttack 准确率的秩相关是
**+0.982**。任何能预测「这个架构好不好训」的早期指标，都会自动看起来像鲁棒性预测器。
所以每个指标都报三列：`ρ(target)`、`ρ(clean)`、偏相关（控制 clean 后的秩相关）。

**运行**：`scripts/audit_early_metrics_vs_robbench.py`（已接入 `run_audit_suite` 第 12 步），
`n=64`，bootstrap 2000 次（按架构重采样），置换 2000 次，种子 20261112。

**结果（已验证事实）**：84 个指标判定分布 **direct 28 / via_clean 17 / suppressed 9 /
weak 30 / uncovered 1**。

| 指标 | ρ(target) | ρ(clean) | 偏相关 [95%] | 判定 |
|---|---|---|---|---|
| `trnas_consistency_phi` | +0.849 | +0.847 | **+0.170** [−0.171, +0.479] | weak |
| `trnas_r_score` | +0.848 | +0.843 | +0.197 [−0.081, +0.425] | weak |
| `trnas_activation_psi` | +0.580 | +0.614 | −0.150 [−0.391, +0.181] | suppressed |
| `fsp_delta` | +0.745 | +0.732 | +0.199 [−0.097, +0.450] | weak |
| `feature_delta` | +0.690 | +0.672 | +0.215 [−0.056, +0.426] | weak |
| `feature_measure_multiview_max_delta` | +0.686 | +0.693 | +0.040 [−0.215, +0.308] | via_clean |
| `feature_measure_swd` | +0.738 | +0.679 | **+0.512** [+0.282, +0.636] | direct |
| `adversarial_loss` | +0.600 | +0.541 | **+0.432** [+0.217, +0.573] | direct |
| `feature_spectral_participation_ratio_static` | +0.202 | +0.096 | **+0.575** [+0.288, +0.696] | direct |

偏相关前 10：`feature_spectral_participation_ratio_static` +0.575、`jacobian_effective_rank`
+0.565（原始相关 **−0.487**，控制后翻号）、`input_loss_delta` +0.536、`jacobian_spectral_norm`
+0.527、`jacobian_spectral_decay` +0.523、`input_grad_l2` +0.522、`racl_input_gradient_lower`
+0.522、`local_logit_lipschitz` +0.519、`feature_measure_swd` +0.512、`feature_spectral_erank_static` +0.508。

**怎么读**：RQ2 的「表面强信号」（Φ 的原始相关 +0.849）大部分由 clean 准确率传递，
控制后只剩 +0.170 且区间跨 0。真正在控制 clean 后仍然站住的，是**输入敏感性族**
（input_grad_l2、local_logit_lipschitz、input_loss_delta、jacobian_spectral_norm）
与 `feature_measure_swd`，量级 +0.5。换句话说：**「早期指标能预测训练后鲁棒性」有了第一份
正面证据，但这批证据不属于 Φ/R-Score。**

**target 从哪来（同源核对，2026-11-12 补做）**：仓库里 `outputs/e28_capacity_family/nb201_e0_v3.csv`
有一列 `oracle_target`。把它与 RobBench 的 `autoattack`（threeseed）按 arch_id 对齐：64 个架构
**64/64 匹配，秩相关 +1.000，最大绝对差 0.00e+00** → 两者是**同一份数据**。所以本报告不存在
「两套 target 互相矛盾」的问题；真正的局限是**只有一套 target**，没有第二套独立结果可以
交叉验证。这也意味着一开始把「跨协议」写成首要局限是把问题说反了——目标同源恰恰说明
target 的选择没有引入额外不确定性，代价是无法做外部验证。重跑时加 `--oracle-table` 会
自动做这项核对并写进报告与 JSON（suite 里按 `e28_capacity_family/nb201_e0_v3.csv` 是否存在
决定是否传入）。

**这份结果不能说明什么**（报告里逐条列出）：目标来自官方固定训练协议、早期指标是本仓库对
**随机初始化**网络做的一次前向（同一份 target，但没有第二套协议可交叉验证）、单次初始化、
n=64、非因果、RobBench 的 `autoattack` 不是最强评估配置。
控制 clean 只排除了一条混杂路径，参数量与可训练性仍可能同时影响两者。

**产出**：`results/audits/early-vs-robbench-20261112.{json,md}`、`tests/test_audit_early_metrics_vs_robbench.py`（16 个测试）。

### 18. e0 覆盖扩到 400 个架构，并发现「发布表的 e0 与当前代码不一致」（2026-11-12）

**做了什么**：把 e27 的 zero-cost 配方用到 400 个 NAS-Bench-201 架构上（`scripts/expand_nb201_e0_coverage.py`，
必含已发布的 64 个；`--data-root` 指向 `data/cifar10`；约 2.3 秒/架构，总计约 16 分钟）。
产物 `outputs/e28_capacity_family/nb201_e0_n400_20261112.csv`：400 行 / 400 唯一 / 78 列，
target 0.149–0.4385（均值 0.3642），唯一全 NaN 列是已知的 `perturbation_capacity_importance_delta`。

**先验证配方可复现**：用 e27 配方重算 64 个已发布架构，与 `nb201_e0_v3.csv` 按 arch_id 对齐，
**67 个共有列里 64 列逐位相同（max|diff| = 0）**；3 个微差是 `margin_input_grad_ratio` 3.9e-03、
`jacobian_effective_rank` 4.8e-06、`jacobian_spectral_norm` 1.5e-11（梯度与特征分解的数值不确定性）。

**关键限制·必须先说**：e0 只能算 `MetricResult` 的 70 个指标。榜单是 85 个，缺的 15 个在
`e37_metrics.py`（9 个）与 `student_runner.py`（racl 3 + trnas 3）。更要紧的是 **TRNAS Φ/R-Score/Ψ
需要训练轨迹（e0→e50 的权重演化），单次随机初始化前向根本算不出来**。所以这次扩覆盖
**不能**用来重估第 17 条里 Φ 的 +0.849 / +0.170，只能重估 70 个 zero-cost 指标。

**结果（同一配方，只变样本量）**：

| 数据 | n | ρ(target,clean) | 最高偏相关 | direct / weak / suppressed / via_clean |
| --- | --- | --- | --- | --- |
| 当前代码，64 架构 | 64 | +0.982 | `feature_measure_cov_delta` **+0.357** [−0.014, +0.566] | 8 / 55 / 2 / 4 |
| 当前代码，400 架构 | 400 | +0.983 | `perturbation_capacity_topk_delta` **+0.357** [+0.260, +0.437] | 16 / 45 / 2 / 6 |
| 发布表，64 架构 | 64 | +0.982 | `feature_spectral_participation_ratio_static` **+0.575** [+0.288, +0.696] | 28 / 30 / 9 / 17 |

**怎么读**：(1) 从 64 扩到 400，最高偏相关**没有变大**（都是 +0.357），只是**区间从跨 0 收紧到不跨 0**
——扩大覆盖买到的是确定性，不是更强的信号。(2) 400 架构下 top-5 的偏相关都在 +0.32~+0.36，
是一**簇**指标共同携带一点超出 clean 的信息，没有单个指标突出。(3) 发布表那一行与当前代码
同一批架构给出 +0.575，明显高于 +0.357，说明**数据源差异会显著改变结论的幅度**（不是样本量）。

**顺带发现的真问题**：发布表的 `nb201_e0` 指标值与当前代码重算的值不一致——
`feature_delta` 发布范围 0.0021–0.2702、当前代码 0.0004–0.0110（约 20 倍），但两者秩相关 0.836、
与 target 的相关 −0.806 vs −0.845（方向一致）。原因未查清（可能是 `metrics.py` 演进而发布表未重算，
或 e23 用了不同的归一化）。**没有改发布表**，记入第 6 节第 10 条。

**产出**：`results/audits/early-vs-robbench-n400-20261112.{json,md}`、
`results/audits/early-vs-robbench-same64-20261112.{json,md}`、
`scripts/expand_nb201_e0_coverage.py`、`scripts/wide_e0_to_rows.py`、
`tests/test_expand_nb201_e0_coverage.py`（14 个）、`tests/test_wide_e0_to_rows.py`（11 个）。

## 4. 可信结论（分三级）

**已验证事实**

- 在当前的 84 项严格榜上，TRNAS Φ 排第 1（0.7822），自研 multiview max 排第 6（0.5241）。
- 榜单的 85 个指标并不互相独立：ρ ≥ 0.5 时聚成 21 簇，最大一簇 35 个（`feature_measure_*` 漂移变体），
  TRNAS 的 Φ/R/Ψ 自成一簇，`jacobian_effective_rank` 与主流负相关。
- **84 项名次约 93% 由 NB201 的 5 个训练阶段决定**（τ = +0.926），与只 RB40 的排名几乎无关
  （τ = −0.127）；在 40 个已训练防御模型上，`trnas_r_score` 是 **−0.184**（第 73/84），
  `trnas_activation_psi` 第 76/84，而 `adversarial_loss` 第 2、`feature_adv_excess_centered_unit_l2` 第 1。
- 自研指标在 NB201 五阶段内部排序能力可用（0.63–0.76），但**跨空间**（DARTS）几乎失效（≈0）。
- RB40 上自研指标与鲁棒性的相关性，在控制家族与参数量后大幅下降（max 0.576→0.186）。
- **早期指标的原始相关主要来自 clean 准确率**：在 64 个 NB201 架构上，
  `ρ(训练后 clean, 训练后 AutoAttack) = +0.982`；控制 clean 后，Φ 的偏相关只剩 +0.170
  （区间 [−0.171, +0.479]），R-Score +0.197，Ψ 为 −0.150（符号翻转）。
  84 个 e0 指标里 17 个属于「与 clean 高度共线、控制后消失」，28 个仍保持 ≥0.3 的偏相关
  （最强 `feature_spectral_participation_ratio_static` +0.575）。

**合理推断**

- 自研指标的排序力主要来自"架构族内的相对差异"，而非跨族的普适规律；
  这解释了为什么 NB201 内有效、DARTS 上失效。
- Φ 的优势部分来自其结构与模型容量/家族无关，但这只是推断，尚未做消融验证。
- Φ 的排序信息**不是** 35 个漂移指标的重复：增量检验（2026-11-12 执行）给出
  Δρ = +0.695 [+0.583, +0.769]、p = 0.0005、嵌套模型 R² 增量 +0.476，
  且在漂移族最强的 `nb201_e0`（ρ(B,y)=0.831）上仍有 +0.522。因此
  "Φ 只是漂移族的另一种写法"这个替代解释已被排除（D-023）。
  剩下的是机制问题：**为什么**它能补上漂移族缺的那部分信息，仍是推断（见上一条）。

**尚待验证的假设**

- 特征空间稳定性是否是**因果**因素，而不只是与鲁棒性共变的伴随现象——目前所有结果都是相关性。
- 训练早期指标能否预测"训练完成后"的鲁棒性：**已有正面证据，量级约 +0.35，且混杂已识别**。
  在 400 个 NB201 架构上（当前代码口径），控制训练后 clean 准确率后，最好的一簇漂移/容量类指标
  仍有 +0.32~+0.36 的偏相关，95% 区间 [+0.260, +0.437] 不含 0；同一配方在 64 个架构上也是 +0.357，
  只是区间更宽。但在发布表的历史 e0 数据上同一批架构给出 +0.575——**两个口径不能混用**（第 3 节第 18 条）。
  要把这变成更强的结论，需要的是**不同种类的早期信号**（不是更多同一个指标）与第二套训练协议。

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

0. ~~`perturbation_capacity_signed_delta` 的方向约定待定~~（2026-11-12 部分回答，见第 3 节第 15 条）。
   §3 第 14 条测出它随扰动增大而单调变负，与其他 55 个 delta 相反；
   同时它在发布表里 `direction=lower`，在 NB201 晚期设置的原始相关是 −0.839
   （即「越负、target 越高」）。第 15 条的对照证明它和 `perturbation_capacity_delta`
   **不是同一个指标的符号版本**（秩相关中位数 +0.017，4/8 设置符号相反），
   所以「单调下降」不是 bug，而是「方向净额」与「幅度总额」的差别。
   **仍开放**：两种解释仍未被区分——一是符号约定本来如此，二是**分母效应**
   （鲁棒模型 clean capacity 可能更小，同样的绝对下降被 `|clean_capacity|` 放大）。
   区分办法：控制 `perturbation_capacity`（clean 侧）做偏秩相关，看相关性是否显著缩小。
   **在此之前，跨指标比较时要单独说明这一个指标的方向，不要把它和另外 55 个 delta 一起解释。**


1. **服务器现状未知**：`/workspace/robust_metric_lab_nas` 的代码、结果、manifest 未盘点（阻塞，见第 8 节）。
2. **服务器那一侧仍未确认**：本地已判定代码主线是 `893a994` 一线并能全部合并，
   但服务器目录 `/workspace/robust_metric_lab_nas` 究竟停在哪一版，仍要等 SSH 才能核实（见第 8 节）。
3. **三空间独立留出集仍缺**：`三空间独立验证材料审计_20261025.md` 指出 371 键独立验证包未冻结。
4. **激活状态保持候选结果未知**：`feature_activation_sign_survival_sum` 的 560 键全网格
   在服务器上跑完了没有，无从确认（等待 SSH）。
5. **571 行未提交指标代码未评审**（内容与 `dc6fb9c` 逐字节相同，见第 7 节）。
   2026-11-12 的对照审计（第 3 节第 13 条）已确认这些名字都有实现入口，
   但"哪个函数用哪个入口"的逐行数学评审仍未做。
6. **教师数据包未齐**：huangyue 缺 15/45 RB40 artifact。
7. **TRNAS 三项的资格口径待确认**（2026-11-12 新增，不影响任何已发布数字）：
   `configs/metric_eligibility.json` 仍把 Φ/R/Ψ 标为 `implementation_error`，但它们在发布表上
   有第 1/2/4 名。配置记录修复前、发布表记录修复后，需要有人确认后统一（见 D-016）。
   在此之前索引会一直显示这三条警告。
8. ~~Φ 的增量检验只有方案、还没跑~~（2026-11-12 已完成，见第 3 节第 12 条）：
   `scripts/audit_phi_incremental.py` 已按方案实现并执行，结论是 Φ 确实带来增量信息
   （Δρ = +0.695）、不是漂移族的重复。剩下的开放问题是机制，不是这个检验本身。
9. **RQ2 的混杂控制只做了 clean 准确率这一项**（2026-11-12 新增，见第 3 节第 17 条）：
   RobBench 的 clean 与 AutoAttack 秩相关 +0.982，控制 clean 后 Φ 降到 +0.170。
   但参数量、训练难度、RobBench 自身的训练协议都还没控制。
   ~~e0 指标也只有 64 个架构覆盖~~ → 已扩到 400（第 3 节第 18 条），结论是偏相关没有变大、
   只是区间收紧了。要继续必须换**不同的**早期信号，而不是更多的同一个指标。
10. **发布表的 `nb201_e0` 指标值与当前代码重算不一致**（2026-11-12 新增，见第 3 节第 18 条）：
    `feature_delta` 相差约 20 倍（0.0021–0.2702 vs 0.0004–0.0110），秩相关 0.836，与 target 的相关
    方向一致（−0.806 vs −0.845）。在查清之前，**不要把发布表的 e0 数字与当前代码新算的数字放进同一张
    比较表**，也不要用后者去"纠正"前者。查清办法：找到 e23 那一批的实际生成脚本与 `metrics.py` 的
    历史版本，逐指标比对实现；在此之前只记录，不改任何已发布文件（D-029）。

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

**第二阶段计划已写好**：`docs/ROADMAP.md`（2026-11-12 第一版）。它把两个研究问题拆开看：
RQ1（已训练模型的排序）基本已回答，只剩"Φ 增量检验"与"target 混网格异质性"两处收尾；
RQ2（早期指标预测训练后鲁棒性）**只有指标稳定性证据、没有预测证据**，是本项目最缺的一环。
路线图给出 16 架构 × 3 seed 的试点设计、算力估算（单卡 6–10 天）、必须做的三项对照
（控制参数量的偏相关、随机基线、固定扰动 vs 自生成扰动两种口径）以及 M1–M5 里程碑。
