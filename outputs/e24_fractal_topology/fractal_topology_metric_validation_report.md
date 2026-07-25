# E24 Fractal/Topology Metric Result

Four `(Ours)` metrics were added to the previous 41-metric comparison without retraining any architecture: symmetric Hausdorff, 95%-quantile Hausdorff, finite-scale box-dimension drift, and multiscale graph fundamental-rank drift.

## Decision Table

| Metric | RobustBench-20 | NB201 e0 | DARTS e0 | Zero-cost mean | Mature mean | Decision |
|---|---:|---:|---:|---:|---:|---|
| Graph fundamental rank | 0.403 | 0.750 | 0.304 | 0.527 | -0.157 | retain as zero-cost topology ablation |
| Hausdorff | 0.328 | 0.704 | 0.197 | 0.451 | -0.243 | reject as general proxy |
| Quantile Hausdorff | 0.400 | 0.715 | 0.131 | 0.423 | -0.244 | reject as general proxy |
| Box dimension | 0.365 | 0.642 | 0.179 | 0.410 | 0.133 | retain as weak stable ablation |

Values are direction-adjusted Spearman, so larger is better. Graph fundamental rank is rank 11/45 by the two-space zero-cost mean. TRNAS R-Score remains substantially stronger at 0.824.

NB201 epoch-0 graph rank has a 95% bootstrap interval of [0.652, 0.812]. Its raw Spearman falls from -0.750 to -0.359 after jointly controlling operation counts and official clean accuracy. DARTS epoch-0 remains -0.304 with an interval crossing zero. The current evidence therefore does not support a cross-search-space robustness claim.

Complete formulas, protocols, commands, all epochs, structural controls, and limitations are in `docs/theory/fractal_topology_feature_metric_design.md` and `docs/experiments/e24_fractal_topology_validation.md`.
