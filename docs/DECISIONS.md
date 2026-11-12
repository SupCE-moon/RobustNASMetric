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

---

## D-008 认定 `codex/server-mainline`（dc6fb9c）为服务器主线，并以其为基线合并

**背景**：四方版本不一致，需要回答"哪个分支代表服务器"。可选做法有三种：看提交时间、
看提交图上的领先数、看文件内容哈希。

**选择**：按内容哈希。主工作树里 6 个未提交文件与 `dc6fb9c` **逐字节相同**
（`metrics.py` 67,447 字节等），而 `main`（784fc74）里是旧版（42,297 字节）。

**理由**：提交时间会被 rebase 和 checkout 污染；领先数只说明分叉了多少次，不说明谁新。
内容哈希是唯一能直接证明"这份文件就是那一份"的证据。另有一条旁证：对账 spec 里记录的
`recorded base = dc6fb9cf…` 与 `server-mainline` 的 tip 一致。

**执行**：`git merge dc6fb9c` 到 `codex/server-mainline-merge-20261112`，三处冲突：

| 文件 | 处理 | 理由 |
| --- | --- | --- |
| `.gitignore` | 两边都留 | 忽略规则是并集，删任何一条都会让文件重新变脏 |
| `metric_registry.py` | 取 `--theirs` 再加回 adv-excess 声明 | 服务器版是我们的超集，但缺 adv-excess 那一条 |
| `student_runner.py`（add/add） | 取 `--ours` | 我们的版本把 `None` 的 `checkpoint_epoch` 归一化成 `""`，服务器版保留 `None`，会让续跑键对不上而重复计算 |

**后果**：`pytest tests -q` = **225 passed, 2 skipped**（合并前 103 passed）。
合并后主线代码比 `main` 完整，`main` 不再被当作代码基线。



---

## D-009 修正 D-008：代码主线是 `893a994`（strict-e37），`dc6fb9c` 只是服务器基线提交

**背景**：D-008 用"主工作树 6 个未提交文件与 `dc6fb9c` 逐字节相同"认定 `dc6fb9c` 是主线。
这个推理只证明"那 6 个文件来自 `dc6fb9c`"，不证明"`dc6fb9c` 是最新代码"。
把 9 棵树的文件全部展开后，`strict-metrics-e37`（`893a994`，2026-10-09）的 `metrics.py`
是 76,626 字节 / 56 个顶层函数，而 `dc6fb9c` 是 67,447 / 51 个。

**选择**：判定 `893a994` 为代码主线，并把它合并进整理分支（`92c5d10`）；
`dc6fb9c` 的定位改为"服务器基线提交"。判据用**函数集包含关系**（`ast` 解析后比较函数名集合），
不用字节数，也不用提交时间。

**理由**：
1. 字节大不代表代码新——注释、换行、无关重构都能撑大文件。函数集是严格超集才说明"新代码包含旧代码"。
2. 提交时间会被 rebase / checkout 伪造。实例：`dfc67f2`（activation-survival）提交最晚（10-31），
   但它的 `metrics.py` 只有 43,562 字节 / 33 个函数，底子是 9 月那一版；按时间判会把它当主线。
3. `893a994` 无论按函数集还是按业务内容（e36/e37 指标、strict release 流水线）都是最全的一支。

**执行**：4 处冲突全部取 `893a994` 侧，并补回 adv-excess 声明与其 eligibility 行
（`metric_eligibility.py` 的 `validate` 要求注册表里的每个指标都有资格行，缺一行就 3 个测试变红）。

**后果**：`pytest tests -q` = **284 passed, 2 skipped**。D-008 的合并保留，因为它记录的是
"服务器基线 = 哪一版"这个仍然有效的事实，只是不再等于"最新代码"。

---

## D-010 三个"补丁级"取舍：egg-info 拒绝入库、matplotlib 降为可选依赖、log 文件入库

**背景**：把剩余的 worktree 独有文件并进来时，有三类文件需要当场定规则，否则会一边漏一边脏。

**选择**：
1. `tools/repository_policy.py` 增加规则：路径里出现 `<package>.egg-info/` → `reject`。
   它是 `pip install -e .` 生成的产物，记录的是**这一台机器**的安装路径，进 Git 只会泄露本机路径。
2. `pyproject.toml` 增加可选依赖 `plot = ["matplotlib"]`，
   `tests/test_e37_diagnostic_plot.py` 改为 `pytest.importorskip("matplotlib")`。
3. `outputs/e28_capacity_family/rb20*_run.log`、`outputs/e34_nb201_pgd1_run.log` 入库。

**理由**：
1. egg-info 是唯一让 `three-space` 看起来还有"5 个独有文件"的东西，拒绝后归零。
2. 只有画图脚本需要 matplotlib，指标流水线不需要。把它设成硬依赖会让服务器环境安装变重；
   设成可选又会让测试在没装的机器上红。`importorskip` 是标准做法：跳过是"环境不具备"，
   失败是"代码有问题"，两者不能混。
3. 运行日志是**实验档案**，不是缓存：它记录当时跑的哪条命令、多少轮、有没有报错，
   丢了就无法复现"这个数字是哪次跑出来的"。三个文件合计 17 KB，属于应入库的小结果。

**后果**：`pytest tests -q` = **477 passed, 3 skipped**（3 skip = 2 个符号链接 + 1 个 matplotlib）。
