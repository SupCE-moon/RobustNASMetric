# 服务器大型结果 manifest：先定清单，再等 SSH

这份文档解决的是"服务器一通，手忙脚乱"的问题。它现在就能写完，因为需要的路径、生成脚本、
日期和预期大小在 `docs/experiments/*.md` 里都有记录。等 SSH 恢复，只剩执行。

## 1. 为什么要单独做 manifest

服务器上的东西分两类，处理方式完全相反：

| 类型 | 例子 | 去哪 | 为什么 |
| --- | --- | --- | --- |
| 源码、测试、配置、文档、小表 | `src/`、`tests/`、`configs/`、`docs/`、86 项总表 | GitHub | 靠 diff 才能协作和复查 |
| 数据集、权重、checkpoint、逐样本原始行 | CIFAR-10、45 个 RobustBench 权重、`per_source/*.csv` | 留在服务器 | 几十 GB，Git 存不下也用不上 |

第二类不能进 Git，但**必须可追溯**：一个数字能不能被信任，取决于能不能回答"它来自哪个文件、
哪个脚本、哪一天、有没有被改过"。manifest 就是这份答案。没有它，三个月后没人能证明
`0.7822` 不是手抄的。

## 2. 清单（机器可读版本：`targets.json`）

路径全部从实验报告里摘抄，不是凭记忆补的；`expect` 一栏写的是报告**声称**的事实，
这样现场普查一旦对不上，会立刻暴露，而不是被"看起来差不多"糊过去。

| 优先级 | 路径 | 内容 | 报告声称 |
| --- | --- | --- | --- |
| 1 | `/workspace/robust_metric_lab_nas` | 项目根（非 Git 工作区） | 对应提交 `dc6fb9c`，需用哈希核实 |
| 1 | `outputs/student_metrics/wangjiahao/strict_adv_excess_release_20261024T110422Z` | 权威 86 项发布 | 48,160 原始行、84 入榜、2 项 RACL 排除、verification PASS |
| 1 | `shared_benchmarks/robust_metric_v1` | 三张主网格 manifest | RB40 `5b519799…`、NB201 `a158c609…`、DARTS `7761b4a4…` |
| 2 | `outputs/.../darts_e37_layer_diagnostic_20261025T120240Z` | DARTS 逐层诊断 | v4 PASS，200 键 / 100 跨阶段对，canonical `63b8ab30…` |
| 2 | `data/official_nasrobbench201_source_20261031` | NAS-Bench-201 表 | 15.6 MB + 2.1 MB，本地有同字节副本 |
| 2 | `docs/experiments/source_snapshot` | 报告引用的源码快照 | 体积小，核实后应导入 Git |
| 3 | `outputs/.../strict_final_release_20261008T000000Z` | 80 项旧发布 | 已被 10-24 版取代，只作沿革证据 |
| 3 | `outputs/.../strict_trnas_direct_ode_release_20261024T000000Z` | TRNAS 修复中间产物 | 最大参数变化 1.13e-6 |
| 3 | `outputs/.../existing_metrics_full_v3_cpu_repair` | 560 行主网格 CPU 修复 | 86 项总表的上游 |

## 3. 哈希策略（为什么不是"全部算 SHA-256"）

用户要求"记录 SHA-256"，但服务器上有几十 GB 权重和 checkpoint。全量哈希要跑几小时，
而且会拖住第一次普查——而第一次普查最要紧的是**先知道服务器停在哪一版**。

折中规则（已写进 `targets.json` 的 `hash_policy`）：

- ≤ 2 GiB 的文件：完整 SHA-256。
- > 2 GiB 的文件：记录大小、mtime，外加**首尾各 8 MiB 的 SHA-256**，并在 manifest 里标 `sampled`。

理由：截断、替换、半途中断的拷贝，首尾采样都能发现；而"文件中间某一位被翻转"对 checkpoint
这种一次性产物几乎不会发生，为此花几小时不值得。标了 `sampled` 就不会有人在论文里把它当完整哈希引用。

## 4. 执行顺序（SSH 恢复后照着做）

```powershell
# 0) 只读普查：密码只走环境变量，不写文件、不进命令行历史
$env:ROBUST_METRIC_SSH_PASSWORD = "<运行时输入>"

# 1) 项目根普查（这是判定"服务器停在哪个提交"的关键一步）
python tools/server_inventory.py --host 49.232.181.138 --port 17010 --user guzhengkai `
    --docker wangjiahao --root /workspace/robust_metric_lab_nas `
    --hash-limit-mb 2048 --output results/manifests/server/wangjiahao-root.json

# 2) 与本地树比对，看服务器代码落在哪一支
python tools/compare_inventories.py `
    --target results/manifests/reconciliation-20261112/trees/mainline-consolidation.json `
    --other results/manifests/server/wangjiahao-root.json `
    --output results/manifests/server/server-vs-local.json

# 3) 大结果目录单独普查（结果目录里 per_source 很碎，先只看 > 1 MiB 的）
python tools/server_inventory.py ... --root .../strict_adv_excess_release_20261024T110422Z `
    --output results/manifests/server/release-20261024.json
```

第 2 步的期望结果：`only_in_other` 里出现服务器独有的源码文件 → 说明服务器比本地主线还新，
需要再并一轮；`differs` 很少 → 说明服务器就是 `dc6fb9c` 那一版，主线判定成立。

## 5. 卡在哪

SSH 不可达，实测证据（2026-11-12）：

- `49.232.181.138:17010 / 17018 / 17013` → `connect_ex` 返回 10061（主机在线，端口无监听）。
- `22` 端口开放，但主机密钥 `SHA256:Wd29pYQlkJNZzh4e4jst7fYt5H1o68eopbfdHflonTQ`
  与 `known_hosts` 里三条记录都不匹配，`guzhengkai` 认证失败。

结论：容器或端口映射下线，密码本身大概率是对的。**不要**用 `StrictHostKeyChecking=no` 绕过——
主机密钥对不上时，正确做法是停下来问，而不是关掉校验。
