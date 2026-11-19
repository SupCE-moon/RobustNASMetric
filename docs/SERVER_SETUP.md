# 服务器实验环境（可 git 同步）

本文记录 2026-11-19 在服务器上建立的**可 git 管理的实验克隆**。目标是让
"以后做实验优先在服务器上做"这件事不必靠手工传文件：代码永远从 GitHub 拿，
数据留在原地，实验结果按 manifest 登记。

## 1. 在哪

| 项 | 值 |
|---|---|
| 宿主机 | `guzhengkai@49.232.181.138:17010`（`516server`，Linux 6.11） |
| 容器 | `wangjiahao`（`nvcr.io/nvidia/pytorch:24.10-py3`） |
| 新克隆 | `/workspace/robust_metric_git`（52 MB，其中 `.git` 7.6 MB） |
| 旧目录 | `/workspace/robust_metric_lab_nas`（48 GB，**不动**：数据与历史产物都在这里） |
| 分支 | `codex/server-mainline-merge-20261112` |

旧目录**不是** git 工作区（`CODE_PROVENANCE.txt` 记的是归档时点 `dc6fb9c`），
所以不改造它，而是另建一个干净的克隆。

## 2. 为什么用克隆而不是把旧目录 git init

1. 旧目录 48 GB，`git add .` 即使有 `.gitignore` 也可能误跟踪大文件；
2. 旧目录里 `metric_registry.py` 是"补丁打失败"的状态（`.rej` 还在），
   不适合当成基线提交；
3. 克隆得到的是 GitHub 上的权威版本，与本地、与远端三者一致，出问题可以
   直接 `rm -rf` 重来，不会伤到 48 GB 的数据。

## 3. 网络：直连 GitHub 不通，走 ghproxy

实测（2026-11-19）：

- `curl https://github.com` → 失败；
- `git ls-remote https://github.com/...` → `exit=124`（60 秒超时被杀）；
- 但 `https://ghproxy.net/https://github.com/...` → `exit=0`，正常。

已在容器内配置全局改写，之后写普通 GitHub 地址即可：

```bash
git config --global url."https://ghproxy.net/https://github.com/".insteadOf "https://github.com/"
```

`git fetch --dry-run origin` 现在 `exit=0`。

**注意：镜像通常只支持 fetch，不支持 push。** 所以服务器是**只读远端**：
拉代码可以，推代码不行。在服务器上改的代码要么走 commit + 打包带回，
要么改在本地推。

## 4. 数据：软链接，不复制

`.gitignore` 排除的大件留在旧目录，新克隆里只放指针：

```bash
/workspace/robust_metric_git/data/cifar10                          -> /workspace/robust_metric_lab_nas/data/cifar10
/workspace/robust_metric_git/data/official_nasrobbench201_source   -> /workspace/robust_metric_lab_nas/data/official_nasrobbench201_source
```

三张审计要读的原始表（本地生成的，服务器上原本没有）已上传到
`outputs/sources_strict/`，并在 `.git/info/exclude` 里排除，所以 `git status` 是干净的：

| 文件 | 字节 | 来源 |
|---|---|---|
| `metric_values_original_strict.csv` | 14,620,237 | 本地（47,600 行，等于服务端 merged 的 ok 行） |
| `metric_value_rows_adv_excess.csv` | 172,751 | 仓库内（560 行 adv-excess） |
| `repair_rows.csv` | 1,047 | 仓库内（3 行 TRNAS 修复） |

## 5. 验证证据（2026-11-19 实测）

```
$ python -m pytest tests -q
748 passed in 41.85s

$ python tools/verify_strict_ranking.py --rows ... --repair ... --extra ... --ranking ...
recomputed=84 compared=84 max_abs_error=0.000e+00 exact=True order_identical=True
```

比本机多两个通过项：`tests` 里的符号链接测试在 Linux 上不再 skip。

## 6. 日常怎么用

**在本地改代码**

```bash
git add ... && git commit -m "..." && git push new-origin HEAD
```

**在服务器更新代码**

```bash
cd /workspace/robust_metric_git && git pull --ff-only
```

**在服务器跑实验**（`PYTHONPATH` 指向本克隆的 `src`，不要用旧的全局安装）

```bash
cd /workspace/robust_metric_git
export PYTHONPATH=/workspace/robust_metric_git/src
python scripts/expand_nb201_e0_coverage.py --data-root data/cifar10 ...
```

**跑审计**（十二步一次跑完，见 `docs/HANDOFF.md` §19）

```bash
export PYTHONPATH=/workspace/robust_metric_git/src
python tools/run_audit_suite.py --rows-root outputs
```

## 7. 三个坑

1. **连太快会被限流**：连续 SSH 会报 `SSHException: Error reading SSH protocol banner`
   （`MaxStartups`），重试即可，不要当网络故障。本地的 `.tmp_ssh2.py` 带 4 次退避。
2. **`docker cp` 只能在宿主机跑**：容器内没有 `docker` 命令；上传文件先 SFTP 到
   宿主机 `/tmp`，再 `docker cp` 进容器。
3. **`--data-root` 要指到 `data/cifar10`**，不是 `data`：`torchvision` 期望
   `<root>/cifar-10-batches-py` 这一层。

## 8. 如果克隆丢了怎么恢复

```bash
cd /workspace
git clone --branch codex/server-mainline-merge-20261112 \
    https://ghproxy.net/https://github.com/SupCE-moon/RobustNASMetric.git robust_metric_git
cd robust_metric_git
ln -s /workspace/robust_metric_lab_nas/data/cifar10 data/cifar10
ln -s /workspace/robust_metric_lab_nas/data/official_nasrobbench201_source data/official_nasrobbench201_source
# 再上传三张 sources_strict 的表（见第 4 节）
```

数据没有丢，丢的只是代码副本，重建约一分钟。
