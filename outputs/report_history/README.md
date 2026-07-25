# metric_validation_20_models_report.md 历史版本

v01-v11 由 Codex 本地 session 日志中的原始 `apply_patch` 记录按时间顺序重放得到；v12-v13 分别是 E14、E15 完成后从当前工作文件保存的校验快照。时间均为北京时间（UTC+8）。这些版本不是人工概括或近似恢复。

| 版本 | 北京时间 | 主要状态 |
| --- | --- | --- |
| [v01](metric_validation_20_models_report_v01_history_2026-07-06_233558.md) | 2026-07-06 23:35:58 | 最初创建：white-box FGSM 与 Standard shared-adv 基础结果。 |
| [v02](metric_validation_20_models_report_v02_history_2026-07-06_234421.md) | 2026-07-06 23:44:21 | 补充 Ours / baseline 归属、文献编号和 References。 |
| [v03](metric_validation_20_models_report_v03_history_2026-07-06_234522.md) | 2026-07-06 23:45:22 | 补入 `margin_input_grad_ratio` 对比结果。 |
| [v04](metric_validation_20_models_report_v04_history_2026-07-07_000201.md) | 2026-07-07 00:02:01 | 补入 spatial Gram 系列实验结果与解释。 |
| [v05](metric_validation_20_models_report_v05_history_2026-07-07_002639.md) | 2026-07-07 00:26:39 | 补入 weak / mid / strong 三源 shared-adv 汇总。 |
| [v06](metric_validation_20_models_report_v06_history_2026-07-16_101151.md) | 2026-07-16 10:11:51 | 改为完整 Spearman 排名报告，纳入当时全部已实现指标。 |
| [v07](metric_validation_20_models_report_v07_history_2026-07-16_102428.md) | 2026-07-16 10:24:28 | 纳入 20-model ReLU-active 重算结果并更新完整榜单。 |
| [v08](metric_validation_20_models_report_v08_history_2026-07-16_102452.md) | 2026-07-16 10:24:52 | 修正二阶 feature metric 的结论表述。 |
| [v09](metric_validation_20_models_report_v09_history_2026-07-16_102544.md) | 2026-07-16 10:25:44 | 补充 ReLU-active 原始结果文件、复现信息和参考文献。 |
| [v10](metric_validation_20_models_report_v10_history_2026-07-16_110853.md) | 2026-07-16 11:08:53 | 新增中文快速结论和全部 34 个指标简明排序。 |
| [v11](metric_validation_20_models_report_v11_history_2026-07-16_112754.md) | 2026-07-16 11:27:54 | 新增三个 source 的定义及逐源 Spearman 完整对比；E14 前的最后版本。 |
| [v12](metric_validation_20_models_report_v12_history_2026-07-16_143453.md) | 2026-07-16 14:34:53 | 纳入 ReLU-active perturbation capacity，主榜由 34 个指标更新为 35 个。 |
| [v13（当前快照）](metric_validation_20_models_report_v13_current_2026-07-16_150427.md) | 2026-07-16 15:04:27 | 纳入 above-mean perturbation capacity，主榜由 35 个指标更新为 36 个。 |

当前工作文件仍是 `outputs/metric_validation_20_models_report.md`，本目录中的文件仅用于查看和比较历史，不应直接覆盖当前工作文件。
