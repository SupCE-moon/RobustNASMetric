# 激活状态保持指标：正式实验运行记录

## 固定范围与出处

- 预注册候选：`feature_activation_sign_survival_sum`，方向 higher。
- 主网格：RB40 40；NB201 64 架构 × e0/e5/e10/e25/e50；DARTS 100 架构 × e0/e100，共 560 个键。
- 对照：`strict_adv_excess_release_20261024T110422Z` 中的 `trnas_consistency_phi`；旧榜不得替代。
- 代码来源：本地隔离工作区 `codex/activation-survival`，服务器仅部署到 `wangjiahao:/workspace/robust_metric_lab_nas`。历史 v3/E36/E37/strict 目录未作为输出目录。

## 2026-10-30 冻结输入

服务器新目录：`outputs/student_metrics/wangjiahao/activation_sign_survival_20261030T153734Z/`。

冻结命令：

```bash
python scripts/activation_survival_manifest.py \
  --bundle-root shared_benchmarks/robust_metric_v1 \
  --data-root data/cifar10 \
  --strict-csv outputs/student_metrics/wangjiahao/strict_adv_excess_release_20261024T110422Z/metric_values_merged.csv \
  --run-dir outputs/student_metrics/wangjiahao/activation_sign_survival_20261030T153734Z
```

15:37 UTC 启动。16:01 UTC 同一进程仍在读取权重，PID 143011，累计读取约 14.3 GB；`freeze.log` 暂无错误输出，`frozen_manifest.json` 尚未生成。进程 I/O 优先级已设为 idle，以免占用其他人的磁盘资源。此状态只证明任务尚在运行，**不证明输入完整**。不得因等待过久而重新启动同一冻结任务，也不得减少 560 个键。

同一时刻 GPU 使用率 100%，空闲显存约 9961 MiB；没有启动候选的 GPU 计算，没有结束或调整他人进程。待冻结成功后，按项目显存门槛进行不参与排名的运行检查，再启动正式实验。

## 严格发布接口核对

16:05 UTC 只读检查服务器严格发布脚本：旧 `build_adv_excess_strict_release.py` 把上一候选的名称、方向及 48,160 行终态写死，不能直接拿来发布本候选。通用 `build_strict_metric_ranking.py` 与 `verify_strict_release.py` 使用 `metric_eligibility` 和 `METRIC_BY_NAME` 校验方向、八设置、唯一性及完整性。因此本候选算完后必须使用单独、经测试的新发布入口，在其进程内显式登记 `feature_activation_sign_survival_sum`（higher、Ours、cross_space），调用相同的通用严格排名与验证函数；不能更改当前正在冻结的服务器 `src/`，也不能放宽旧 RACL 排除。新入口及其哈希须写入新发布的出处记录。此项尚未实施，不能将候选 CSV 直接宣称为严格总榜。

## 待补证据

- 冻结任务终态、560 键与全部输入哈希，失败时保留 `freeze.log` 原文。
- 三空间运行检查的样本数、层调用对齐、耗时、峰值显存和原始失败证据。
- 560/560 唯一、有限、`status=ok` 核验；严格新发布；与 Φ 的八设置差值和架构配对区间。
- 若主榜胜出，再按独立验证计划冻结三空间留出材料；未过独立门槛不得宣布超过 Φ。
