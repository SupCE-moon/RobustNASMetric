# 学生三空间实验问题记录

本文记录 RB40、NAS-Bench-201（NB201）和 DARTS 三空间统一评测期间实际遇到的问题。记录只依据可检查的文件、日志和测试，不把推测或未成功的结果写成事实。

## 实验范围

- 执行环境：`wangjiahao` 容器
- 容器内项目：`/workspace/robust_metric_lab_nas`
- 基准材料：`/workspace/robust_metric_lab_nas/shared_benchmarks/robust_metric_v1`
- 八个 setting：RB40 mature；NB201 e0/e5/e10/e25/e50；DARTS e0/e100
- 固定协议：seed 20260808、sample offset 0、每个模型 64 个样本、扰动半径 8/255
- 完整网格规模：68 个注册指标，共 38,080 行唯一结果

## 问题与处理

### 1. 原始交接状态与实际材料不一致

- 现象：原始 `TRANSFER_STATUS.json` 仍为 `IN_PROGRESS`，而部分说明认为材料可以直接使用；RB40 的 45 个 artifact 中只有 19 个最初可直接确认完整。
- 风险：只看状态文件会阻止实验；忽略文件完整性又可能使用缺失或错误权重。
- 处理：不修改老师的原始状态文件，逐项按 manifest 检查存在性、大小和 SHA-256；补齐缺失材料后重新运行验收脚本。
- 验证：验收器返回 RB40 40 行/45 个 artifact、NB201 64 个架构/320 个 checkpoint、DARTS 100 个架构/200 个 checkpoint，最终显示 `PASS`。日志保存在容器项目的 `docs/experiments/source_snapshot/verifier_after_recovery.log`。

### 2. 宿主机共享目录无读取权限

- 现象：从宿主机共享材料复制到 `wangjiahao` 容器时出现 `permission denied`，`docker cp` 流程中止。
- 根因：登录账号对老师提供的宿主机路径没有足够读取权限，而不是容器内目标目录损坏。
- 处理：保留失败现场；只把已获授权且能够读取的材料复制到 `/workspace/robust_metric_lab_nas`，没有修改其他容器或项目。
- 结果影响：首次复制没有产生可用实验结果，后续通过逐项恢复和校验解决。

### 3. Python 依赖不完整

- 现象：环境中最初没有 `robustbench`；安装后导入仍因缺少 `timm` 失败。
- 根因：包安装与可导入不是同一层验证，RobustBench 的传递依赖未完整满足。
- 处理：在项目独立虚拟环境中补齐依赖，并以逐模块导入检查验证 `torch`、`torchvision`、`robustbench`、`yaml`、`pandas`、`scipy`、`tqdm` 和项目包。
- 结果影响：依赖修复前的探测结果作废，没有写入指标总表。

### 4. NB201 checkpoint 必须由老师的官方实现构建

- 现象：仅凭通用/替代模型结构不能保证 NB201 权重键、BN 状态和最终标签语义一致。
- 风险：权重虽然可能“加载”，但网络结构或参数映射错误会使指标不可比较。
- 处理：将老师提供的 NB201 官方库放入容器项目，并由该实现构建网络后加载 checkpoint；增加回归测试约束。
- 验证：相关实现和测试已纳入代码，提交 `dc6fb9c`。

### 5. RB40 大模型谱分解不稳定

- 现象：Hendrycks 模型在 CUDA float32 上对最后一个 640 通道特征层执行 `eigvalsh` 时不收敛，导致 `feature_delta` 等谱类指标失败。
- 根因：输入为有限值，但大矩阵在 GPU float32 对称特征值分解上存在数值不稳定；同一矩阵在 GPU/CPU float64 可以成功。
- 处理：为对称特征值分解增加稳定降级路径：优先原计算，失败后提升到 float64，必要时转 CPU 计算；不替换指标公式、不伪造值。
- 验证：真实 Hendrycks 权重上的 `feature_delta` 返回有限值；新增谱稳定性回归测试；提交 `4224dba`。

### 6. 首轮完整网格包含可修复失败

- 现象：`existing_metrics_full_v1` 写满 38,080 行，但有 2,358 行状态为失败，因此“行数完整”不等于“实验成功”。
- 根因：其中一部分来自缺失/错误的 RB40 权重及上述谱分解问题；另有一部分属于指标定义上的不适用或明确禁用。
- 处理：保留 v1 原始输出不覆盖，建立 `existing_metrics_full_v2` 作为修复版；只删除并重算 1,238 个可修复的 RB40 失败键，保留可审计的不可用状态。
- 当前状态：v2 已保留 36,842 行，尚缺 1,238 行待续跑；在补齐前不能声称实验完成。

### 7. v2 首次续跑的 run_id 不一致

- 现象：从 v1 派生 v2 时，复制的行仍带有 `run_id=existing_metrics_full_v1`，续跑器拒绝把不同 run_id 混入一个输出文件。
- 根因：派生文件时只复制了 CSV，未同步规范化行内 run_id。
- 处理：在原子重写过程中将保留行统一改为 `existing_metrics_full_v2`，并在继续计算前检查所有行的 run_id。
- 验证：v2 当前 36,842 行的 run_id 均为 v2，唯一键没有重复。

### 8. RB40 空 epoch 导致断点续跑误判

- 现象：CSV 中 RB40 的 `checkpoint_epoch=None` 被规范化为空字符串 `""`；续跑器构造待运行键时仍使用 Python `None`。于是已经存在的行被误判为缺失，随后写入阶段又被重复键保护拒绝。
- 根因：读取侧和请求侧对可空 epoch 的规范化规则不一致。
- 处理：新增统一的请求键构造函数，将 `None` 与持久化格式一样规范化为 `""`，并同时用于预加载判断和逐指标跳过判断。
- 验证：新增回归测试 `test_resume_normalizes_rb40_none_epoch_before_pending_check`；本地完整测试为 153 passed、2 skipped。修复同步服务器并通过服务器测试后，再恢复 1,238 行计算。

### 9. v2 网格完整但 11 个大模型发生显存不足

- 现象：v2 达到 38,080 个唯一键，计算脚本退出 0，但独立复核发现 1,784 行失败，比按设计允许的 1,120 多 664 行；额外失败均位于 RB40 的 11 个大模型。
- 根因：单卡 24 GiB RTX 4090 在保持 64 样本协议的特征、几何和 TRNAS 计算中发生 CUDA OOM。原网格验收只打印失败分布，没有把“非预期失败为零”作为硬条件。
- 处理：保留 v2 原始结果，从中派生 `existing_metrics_full_v3_cpu_repair`，只重算 664 个失败键，保持相同样本、权重与指标公式，改用 CPU 并增加严格失败检查。
- 结果：661 个键恢复成功；另 3 个键发生下述 ODE 数值异常。v3 严格验收以退出码 1 结束，未误报完成。

### 10. Kang2021Stable 的 TRNAS 权重扰动引发 ODE 下溢

- 现象：剩余 3 个失败是 `Kang2021Stable` 的 `trnas_activation_psi`、`trnas_consistency_phi`、`trnas_r_score`，均为 `AssertionError: underflow in dt 0.0`。
- 定位：普通前向、FGSM 和 TRNAS 激活项正常；错误在权重更新后对对抗输入做特征一致性前向时，由 `torchdiffeq` 自适应时间步检查抛出。权重步长 1e-6、1e-7、1e-8 都失败，零步长对照成功。
- 处理原则：零步长消除了指标定义中的权重扰动，不可作为正式数值。保留 3 行失败状态，在八 setting All-setting mean 中使 TRNAS 三项保持 incomplete；只允许将这 3 行作为经诊断的明确例外列入最终验收报告，不可填零或插值。
- 诊断日志：容器项目 `outputs/student_metrics/wangjiahao/existing_metrics_full_v3_cpu_repair/logs/kang_trnas_diagnosis.log`。

### 11. 服务器入口暂时不可达

- 现象：2026-09-22 继续处理时，`49.232.181.138:17010` 多次 TCP 连接检查失败；无法上传最终验收脚本或读取最终分析文件。
- 处理：已准备严格验收脚本并通过本地 Bash 语法检查；保持服务器现有 v3 CSV 和日志不变，不改用其他容器，也不猜测最终排名。
- 后续结果：2026-09-22 入口恢复；核对 v3 CSV 后执行严格验收，允许且只允许 1,120 个定义边界失败和 3 个经诊断的 ODE 例外。脚本退出 0，生成八 setting 分析与 All-setting mean 总表。

## 最终数据与结论（2026-09-22）

- 最终版本：`existing_metrics_full_v3_cpu_repair`；v1、v2 保留不覆盖。
- `grid_validation_with_declared_exceptions.log`：38,080 行、38,080 个唯一键；36,957 行有效、1,123 行失败；其中 1,120 行为既定定义边界，3 行为 `Kang2021Stable` 的 TRNAS ODE 下溢；其他失败为 0。
- `STATUS.env`：`COMPLETE_WITH_DECLARED_EXCLUSIONS`。该状态刻意不称为无例外的完全成功。
- 分析结果：62/68 个指标在八 setting 均有效，6 个 incomplete 不参与总排名。前五为 `adversarial_loss` 0.614290、`natural_loss` 0.591311、`fsp_delta` 0.479411、`feature_measure_flow_corr_rel_delta` 0.477195、`feature_measure_moment_delta` 0.472490。
- E29 指标 `feature_measure_flow_corr_rel_delta` 本次总表排第 4，但 RB40 为 -0.154、DARTS e100 为 0.005；排名是同一批数据上的探索性观察，不能替代独立样本验证。
- 服务器完整测试：155 passed。

## 有意保留的不可用指标

下列状态属于定义边界，不应为了提高名次而填造数值：

- `racl_arch_lipschitz_log_upper`：不适用于 RB40，共 40 行。
- `racl_fixed_network_log_upper_proxy`：不适用于 NB201/DARTS，共 520 行。
- `perturbation_capacity_importance_delta`：实现中明确禁用，共 560 行。

这些指标不会进入要求八个 setting 均有效的 All-setting mean 排名；报告将区分“算法不适用”和“运行故障”。

## 完成判据

只有同时满足以下条件才将实验标记为完成：

1. `metric_values.csv` 恰好包含 38,080 个唯一协议键；
2. 除上述有意不可用项外，不存在未解释失败；
3. 八 setting 分析表与 All-setting mean 排名可由原始 CSV 重新生成；
4. 完整测试、材料验收和分析命令均有日志证据；
5. 不覆盖 v1，不修改其他容器或项目。
