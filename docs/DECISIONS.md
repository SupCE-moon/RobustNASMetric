# 决策记录（DECISIONS）

记录"为什么这样做"，而不是"做了什么"。每条决策包含：背景、选择、理由、后果。
只追加，不改写；若推翻旧决策，新增一条并注明取代关系。

---

## D-001 Git 准入策略：按路径与大小判定，10 MiB 警告 / 50 MiB 拒绝

**背景**：项目里有 45 个 RobustBench 权重文件、CIFAR-10 数据、checkpoint、逐样本原始行，
以及生成表格时产生的 `node_modules`。历史上没有统一规则，靠人工记得"别提交大文件"。

**选择**：`tools/repository_policy.py` 用一条确定性规则判定每个文件是否可入库：
禁用后缀（权重/数组/压缩包/PDF/密钥）、禁用目录（`.venv`、`checkpoints`、`secrets`）、
文件名含 `secret`/`credential`/`password` 等直接拒绝；≥10 MiB 警告，>50 MiB 拒绝。

**理由**：规则必须能被测试和 CI 复用，而不是写在文档里靠自觉。用"路径 + 大小"而不是内容，
是因为判定时不需要读文件，几万个文件也能秒级完成。

**后果**：两个 13.94 MiB 的合并原始值表被判定为"警告"，导入工具据此暂缓（见 D-003）。

---

## D-002 结果表用"显式清单 + 强制添加"，而不是改宽 `.gitignore`

**背景**：`.gitignore` 只忽略了 `outputs/` 下的少数子目录，大部分 CSV 本来就可入库；
但有些目录（`pytest_temp_full/`、`node_modules/`）必须排除。

**选择**：保持 `.gitignore` 的保守默认（新增 `node_modules/`、`~$*`），
导入时以 `import-plan-main.json` 这份**逐文件清单**为准，清单里 `action=import` 的文件才复制。

**理由**：`.gitignore` 只能按模式排除，无法表达"这个 CSV 要、那个 CSV 不要"。
显式清单同时满足对账规范里的"显式清单"要求，并且每个文件都带 SHA-256，事后可核。

**后果**：将来新增结果文件不会被自动跟踪，需要重新生成清单——这是刻意的：
入库应当是有意识的动作。

---

## D-003 未提交的指标代码只存补丁，不落到整理分支

**背景**：主工作树有 6 个已跟踪文件被改动但从未提交，`metrics.py` 新增 571 行
（`perturbation_capacity_iqr/multiscale/topk`、`jacobian_spectral_*` 等）。
服务器上可能已有更新版本。

**选择**：保存为 `patches/main-uncommitted-784fc74-20261112.patch`（83,983 字节，
SHA-256 `d3329f9c…`），并在 wip 目录保留两个依赖它的文件的原始字节；**不**应用补丁。

**理由**：整理分支的职责是"把现状讲清楚"，不是"合并未经评审的新指标"。若服务器版本更新，
直接应用补丁会制造一个假的合并冲突；先留证据，等服务器盘点结果出来再决定取舍。

**后果**：5 个 `scripts/analyze_e2x_*.py` 在补丁落地前无法直接运行，已在清单中注明。

---

## D-004 在 worktree 里跑测试必须显式设置 `PYTHONPATH`

**背景**：共享 `.venv` 以 editable 方式安装了 `robust_metric_lab`，指向主工作树。
在 `.worktrees/mainline-consolidation` 里直接 `pytest` 时，导入的其实是主工作树的源码——
第一次运行就出现了"测试读的是另一个目录的 metrics.py"的诡异报错。

**选择**：所有 worktree 内的测试命令都前置
`$env:PYTHONPATH="<worktree>\src"`，并在提交信息里记录这一点。

**理由**：否则测试结果与被测代码不是同一份，"57 passed"没有意义。重装 editable 包会污染
主工作树的环境，不可取。

**后果**：这是本项目最容易踩的坑之一，后续任何人（包括我）在 worktree 里验证都必须照做。

---

## D-005 服务器盘点用只读 SSH 普查，密码不落地

**背景**：服务器 `/workspace/robust_metric_lab_nas` 不是 Git 工作区，且本机没有
`sshpass`/`plink`/私钥，只有 OpenSSH 客户端（密码交互式读取 tty，脚本无法喂入）。

**选择**：`tools/server_inventory.py` 用 paramiko 连接，把一段内嵌 Python 程序通过
`python3 -`（或 `docker exec -i <容器> python3 -`）送到服务器执行；程序只做 `os.walk` +
`stat` + `sha256`，输出 JSON 到本地。密码只从环境变量或 `getpass` 读取，
不接受命令行参数、不写文件；主机密钥未知时直接拒绝连接。

**理由**：满足"服务器只读"和"密码不入 Git"两条硬约束；同时把哈希计算放在服务器侧，
避免把 200+ GiB 数据拉回本地。

**后果**：大文件默认只记录大小/时间（`--hash-limit-mb`，默认 64 MiB），
需要完整哈希时用 `--hash-limit-mb -1` 单独跑一次。

---

## D-006 不直接修改 `main`

**背景**：`main` 上有 7 个未推送提交，且工作树脏；`new-origin/main` 与本地 `main` 同指 `784fc74`。

**选择**：全部工作在 `codex/mainline-consolidation-20261112` 分支进行，推送为评审分支，
不碰 `main`，不 force push。

**理由**：主线尚未确定（服务器可能才是最新），此时改默认分支会把两条历史搅在一起。

**后果**：主线确认后，由评审分支合并或重新基于主线整理。

---

## D-007 为一批导入模块应用了 4 行注册声明（有条件的例外）

**背景**：从 `codex/adv-excess-metric` 导入的 `student_runner.py`、`student_benchmark.py` 等
4 个模块，按名字调用 `feature_adv_excess_centered_unit_l2`；这个名字只存在于该 worktree
**未提交**的 `metric_registry.py` 中。不补这 4 行，导入的模块一调用就 `KeyError`，2 个测试变红。

**选择**：把该改动单独保存为 `patches/adv-excess-uncommitted-metric-registry.patch` 并**应用**
（只加一条 `MetricSpec("feature_adv_excess_centered_unit_l2", "lower", "Ours preregistered
adversarial-excess feature drift", "Ours")`）。

**理由**：这与 D-003 的情况不同。D-003 那 571 行是**未评审的指标数学实现**，风险在"算法对不对"；
这里是**声明**（名字、方向、归属），实现代码 `adv_excess_metric.py` 是作为新文件导入的，
评审对象是同一份。不补声明只会留下一堆跑不起来的死代码，比补上更危险。

**后果**：`pytest tests -q` 从 101 passed / 2 failed 变为 **103 passed**。
若将来判定该指标不应入榜，删掉这条声明即可，不影响其他代码。

