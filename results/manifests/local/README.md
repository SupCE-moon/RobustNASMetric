# 本地大型产物清单（2026-11-12）

这份清单解决一个问题：**项目里有一堆不进 Git 的大文件，它们在哪、多大、是不是原来那个。**
GitHub 只保存代码、文档和小结果；数据集、权重、缓存、合并原始表留在本地或服务器。但"没进 Git"不等于
"不用管"——一旦有人误删或换机器，没有哈希就说不清是不是同一个文件。所以这里给每个文件记下
**绝对路径 + 大小 + SHA-256 + 生成时间**。

- 清单生成时间：2026-11-12T09:11Z（本机时间 17:11）
- 扫描根目录：`D:\Project\RobustNAS\DynamicBN\DynamicBN20251203\robust_metric\outputs`
- 机读清单：`outputs-inventory.json`（422 条，每条含 path / size / mtime_utc / sha256 / classification）
- 重新生成：
  ```
  python tools/inventory_tree.py --root "<项目>\outputs" \
      --output results/manifests/local/outputs-inventory.json --hash-limit-mb 2048
  ```

## 总览

| 项 | 值 |
|---|---|
| 文件数 | 422 |
| 总大小 | 1,107.2 MiB |
| 已计算 SHA-256 | 422 / 422（全量，非采样） |
| 策略判定 allow | 398 |
| 策略判定 warn（>10 MiB） | 2 |
| 策略判定 reject（权重/压缩包） | 22 |

## 按目录

| 目录 | 文件数 | 大小 | 是否入 Git | 说明 |
|---|---|---|---|---|
| `student_metrics/wangjiahao/holdout_checkpoint_recovery_20261027/` | 2 | 1,018.2 MiB | 否 | 从服务器恢复的 holdout checkpoint（WRN-70-16 cutmix-ddpm v2），本机留档 |
| `strict_all_metrics_table_20261024/sources/` | 7 | 14.2 MiB | 否（warn） | 含 13.9 MiB `metric_values_merged.csv`，86 项榜单的合并原值表 |
| `strict_all_metrics_table_20261024_trnas_repaired/sources/` | 11 | 14.1 MiB | 否（warn） | 含 13.9 MiB `metric_values_original_strict.csv`；`delivery_sources.tar.gz` 被拒 |
| `adv_cache/` | 18 | 42.9 MiB | 否 | FGSM/PGD 对抗样本缓存（3 个模型家族 × 4 个 seed × 3 个样本量），可用同一脚本重算 |
| `nasrobbench201_server/fidelity512_e5e10/` | 10 | 3.6 MiB | 部分 | NAS-Bench-201 保真度抽样结果 |
| 其余（各实验小表、报告、图） | 374 | 14.2 MiB | 多数已入 Git | 见 `results/manifests/reconciliation-20261112/` 的导入记录 |

## 22 个被拒绝的文件

| 文件 | 大小 | 原因 |
|---|---|---|
| `student_metrics/.../cifar10_linf_wrn70-16_cutmix_ddpm_v2.pt` | 1,018.2 MiB | 模型权重 |
| `adv_cache/*.pt`（18 个） | 42.9 MiB | 对抗样本缓存（二进制） |
| `strict_..._trnas_repaired/pytest_temp_*/.../arch_1_epoch_10_seed_0.pt`（2 个） | <0.1 MiB | pytest 临时 checkpoint（测试残留） |
| `strict_..._trnas_repaired/sources/delivery_sources.tar.gz` | 0.1 MiB | 压缩包 |

## 已知的口径问题（不要被清单误导）

1. **清单只证明"文件在、哈希是这个"**，不证明内容正确。数值对不对要看对应的 `release_verification.json`。
2. 三份总表的行数关系容易记混，按时间顺序是：`strict_all_metrics_table_20261024/` = 85 行 / **80 入榜**
   （TRNAS 的 Kang 求导还失败，5 项无名次）→ `..._trnas_repaired/` = 85 行 / **83 入榜**（修复后 3 项恢复名次）
   → `strict_adv_excess_release_20261024T110422Z/` = 86 行 / **84 入榜**（多一条 adv-excess 指标）。
   引用排名只能用最后那份 86 行总表。
3. `pytest_temp_*` 是测试残留目录，`outputs/` 里的测试产物不代表实验结果，清理前请确认没有脚本依赖。
4. 本清单**不含**主工作树之外的 8 个 worktree 里的 `outputs/`；那些已按九树对照表逐文件核过
   （见 `results/manifests/reconciliation-20261112/tree-comparison.md`）。
