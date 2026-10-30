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

## 2026-10-30/31 输入冻结与显存等待

冻结任务成功生成 `frozen_manifest.json`，摘要 `6584139ffb2562a80ec58eaa9e893a7e4e63b5940962bc38a1e03806d17400f9`。独立读取核对：payload 摘要相符、560/560 键唯一、560 个不同权重路径、八设置分别为 40/64×5/100×2；服务器注册表哈希 `1aab7d25894de7cc77c24effda8f9e669bfc8a61c373839a04ba7c316e6a63db` 与隔离副本相同。`freeze.log` 原始输出保留在新目录。以上只证明输入清单冻结成功，不是任何指标结果。

审批恢复后，新增非排名运行检查脚本，仅新建 `scripts/smoke_activation_survival.py` 与 `scripts/wait_and_run_activation_survival_smoke.sh`，本地/容器 SHA-256 逐份一致，分别为 `7efe020bdb409a0745aac0e1ce203ac1c017ef0c8817af10bb7d557a53a13698`、`145eff171b9f34fac629b99e94362dc8e9bb37b0a6adb99147eed1b199b9110d`。容器 `bash -n` 及 Python `--help` 通过，本地完整测试 154/154 通过。

20:36 UTC 在 `wangjiahao` 中启动独立 `docker exec -d` 显存监控，PID 143655；状态保存在新目录 `smoke_launcher/STATUS.env`。它沿用项目原门槛：其他计算进程总显存占用不超过 1000 MiB，连续三次、相隔 60 秒；满足后才对 RB40、NB201 e0、DARTS e0 各一个冻结清单中的最小权重模型运行固定 64 样本检查。启动时占用 8412 MiB，状态为 `WAITING_GPU`，**未开始 GPU 指标计算**。失败写独立尝试 JSON 和日志，不能当作正式排名。

20:40 UTC 另启动完整主网格的**等待器**（PID 143784），脚本 SHA-256 `5c3b9795bd377df34d1ddefb8c3be38d6aed578ec82415e487d53e9b225121d7` 与本地相同，容器 `bash -n` 通过，初始 `state=WAITING_SMOKE`。它仅在三空间非排名 smoke 全部 PASS 且再次满足相同显存门槛后才按固定八设置运行 560 键；每个设置有独立追加式尝试目录与日志，任一失败即停，原始失败键保留，后续仅能用同源同清单补缺失键。目前等待器**没有运行正式指标**。新严格发布仍须 560/560 核验和独立合并，等待器完成不等于入榜。

20:47 UTC 再启动验证等待器（PID 144010，`state=WAITING_FULL_GRID`）。新增 `collect_activation_survival_attempts.py` 与 `wait_and_verify_activation_survival.sh` 仅放在之前不存在的服务器脚本路径；SHA-256 与本地一致，分别为 `fb5c52ef533a1ae6bb100ba580bbab05020b45937451bf9f76dd2733c3695f14`、`51a887bbb9fa9df9a9b6b07aca600ae0e9846f8d8ad0c89fa2963ab5e3be5029`，容器语法及 CLI 检查通过。本地收集器做了包含 560 成功 + 1 旧失败的测试，完整测试 157/157 通过。验证等待器仅在八设置全部运行完成后把原尝试逐字节复制到新 `attempts_flat`，保留失败证据，再以 I/O idle 优先级重哈希全部输入并调用独立验证器。任何不完整或不一致均停在失败状态，不生成严格总榜。

20:53 UTC 最终严格发布等待器已在容器运行（PID 144296，`state=WAITING_VERIFIED_560`）。`build_activation_survival_strict_release.py`、`analyze_activation_survival.py` 和 `wait_and_release_activation_survival.sh` 是新文件，服务器/本地 SHA-256 分别一致为 `2836ffca49fdbac1ff4802a457f58485bca9a89f3b295aea54c51683ba79c399`、`5aefaeb2f9dfbf497fe90a99cfd5e649aef426d7c1e00475496e802d17eb5e3c`、`1b724236e8e4964cc30f6fd93c6ff731bf852b5d4174b433fa6eb1405118ffc7`。容器 CLI/语法检查通过，本地完整测试 158/158 通过。此等待器只在候选 560/560 核验 PASS 后调用同一通用严格排名/验证函数，于新目录 `activation_sign_survival_strict_release_from_20261030T153734Z` 创建总榜，并生成完整八设置与 Φ 的架构配对对照。主榜是否胜出仍由固定门槛决定；即便胜出，三空间独立验证未完成也不能宣称最终超过 Φ。
