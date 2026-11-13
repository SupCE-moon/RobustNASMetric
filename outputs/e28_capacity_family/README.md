# e28_capacity_family：RQ2 的原始表在这里

这个目录存的是 **2026-11-12 那一批 RQ2 审计的输入表**。之所以把它们纳入 Git，
是因为审计报告里的每个数字都必须能从这里重算出来；只放脚本和结论、不放输入，
等于让读者相信结论却无法核对。

总大小 2.56 MiB，单文件最大 0.57 MiB，都远低于策略的 10 MiB 警戒线
（见 `tools/repository_policy.py`）。CIFAR-10、RobBench 官方 JSON、模型权重仍然不入库。

## 表清单

| 文件 | 大小 | SHA-256（前 16 位） | 用途 |
| --- | --- | --- | --- |
| `nb201_e0_n400_20261112.csv` | 595,906 | `b618ecf32fede276` | 400 个架构的 e0 指标宽表（78 列） |
| `nb201_e0_verify_20261112.csv` | 96,110 | `d9fc7176732dd272` | 64 架构的配方等价性验证（与 `nb201_e0_v3.csv` 对照） |
| `seedcheck/nb201_e0_seed{0,1,2}_100_64.csv` | 145,819 / 145,542 / 145,203 | `bf0bb3792822fb52` / `36f661cd94d4c7bf` / `e28fbeaefe72c7c6` | 同一批 100 架构在三个权重种子下的 e0 指标 |
| `pert400/nb201_e0_weak_400_64.csv` | 574,930 | `8381adb15175d6ec` | 同一批 400 架构，FGSM ε=4/255 单步 |
| `pert400/nb201_e0_pgd_400_64.csv` | 572,977 | `5615b60bd432bc9e` | 同一批 400 架构，PGD-3 ε=8/255 |

## 每张表怎么生成

**共同前提**：CIFAR-10 解压在 `data/cifar10/cifar-10-batches-py/`。
`--data-root` 必须指向 `data/cifar10`（不是 `data`），否则 torchvision 会报
`Dataset not found or corrupted`——这是路径差一层，不是数据坏了。

```powershell
# 400 架构宽表（每架构约 2.3 秒，全量约 16 分钟）
python scripts\expand_nb201_e0_coverage.py `
  --meta <RobBench>\meta.json --robbench <RobBench>\NAS-RobBench-201_all.json `
  --data-root data\cifar10 --limit 400 --samples 64 --attack fgsm --device cuda `
  --output outputs\e28_capacity_family\nb201_e0_n400_20261112.csv

# 配方等价性验证（64 架构，与 nb201_e0_v3.csv 对照）
python scripts\expand_nb201_e0_coverage.py --same-flags --limit 64 `
  --output outputs\e28_capacity_family\nb201_e0_verify_20261112.csv

# 权重种子（100 架构 × 3 种子，约 11 分钟）
python scripts\audit_e0_weight_seed.py `
  --wide outputs\e28_capacity_family\nb201_e0_n400_20261112.csv `
  --data-root data\cifar10 --limit 100 --seeds 0,1,2 --device cuda

# 扰动口径（400 架构 × 2 变体，约 30 分钟）
python scripts\audit_e0_perturbation_sensitivity.py `
  --wide outputs\e28_capacity_family\nb201_e0_n400_20261112.csv `
  --data-root data\cifar10 --limit 400 --device cuda
```

## 有意不入库的两个派生表

| 文件 | 大小 | 为什么不入 | 怎么再生成 |
| --- | --- | --- | --- |
| `nb201_e0_n400_rows_20261112.csv` | 2.6 MiB | 是宽表的机械转置（28000 行 = 400 × 70），没有新信息 | `python scripts\wide_e0_to_rows.py --wide ... --output ...` |
| `nb201_e0_verify_rows_20261112.csv` | 0.4 MiB | 同上 | 同上 |

## 复现口径上的两个已知差异

1. **`.eval()` 不要加**。发布表的生成脚本 `scripts/evaluate_nb201_e27_zerocost.py`
   从不调用 `.eval()`；加了会让 13 个指标与发布表对不上（见 D-033）。
2. **`clean` 有两个来源**。本目录宽表的 `clean` 是 RobBench 的**训练后**准确率；
   `audit_e0_weight_seed.py` 自己算出的 `clean_acc` 是**随机初始化**网络的准确率，
   两者不可混用（见 D-033 与 HANDOFF §26）。

## 相关报告

- `results/audits/rq2-confounders-20261112.md`
- `results/audits/e0-perturbation-20261112.md`
- `results/audits/e0-weight-seed-20261112.md`
- `results/audits/rq2-stable-subset-20261112.md`
- `results/audits/rq2-selection-value-20261112.md`
