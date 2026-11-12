"""E29 follow-up: is the graph-transition signal real (out-of-sample), and can it
be turned into a clean scalar metric that beats flow_rel's deconfounded 0.628?"""
import numpy as np
import pandas as pd
from scipy import stats

from robust_metric_lab.nasrobbench201 import parse_nb201_arch

OPS = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
TRANSITIONS = [(0, 2), (0, 4), (1, 5), (2, 5)]

df = pd.read_csv("outputs/e23_cross_space/nb201_64/metrics_64_e0e5e10e25e50.csv").drop_duplicates("arch_id")
rows = []
for _, r in df.iterrows():
    ops = parse_nb201_arch(str(r["arch_string"]))
    if ops is None:
        continue
    f = {"oracle": r["oracle_target"]}
    for op in OPS:
        f[f"n_{op}"] = ops.count(op)
    for a, b in TRANSITIONS:
        k = f"t_{ops[a]}_{ops[b]}"
        f[k] = f.get(k, 0) + 1
    rows.append(f)
d = pd.DataFrame(rows).fillna(0)
opc = [f"n_{op}" for op in OPS]
tcols = [c for c in d.columns if c.startswith("t_")]

y = d["oracle"].values
Xb = np.column_stack([np.ones(len(d)), d[opc].values])
Xt = np.column_stack([Xb, d[tcols].values])


def partial_spearman(x, y, controls):
    X = np.column_stack([np.ones(len(y)), controls])
    bx, *_ = np.linalg.lstsq(X, x, rcond=None)
    by, *_ = np.linalg.lstsq(X, y, rcond=None)
    rho, _ = stats.spearmanr(x - X @ bx, y - X @ by)
    return rho


# 5-fold CV: out-of-sample partial of the "transition component"
rng = np.random.RandomState(0)
idx = rng.permutation(len(d))
folds = np.array_split(idx, 5)
test_pred_transition = np.zeros(len(d))
test_y_resid = np.zeros(len(d))
for fold in folds:
    tr = np.array([i for i in range(len(d)) if i not in fold])
    beta_b, *_ = np.linalg.lstsq(Xb[tr], y[tr], rcond=None)
    beta_t, *_ = np.linalg.lstsq(Xt[tr], y[tr], rcond=None)
    for i in fold:
        test_pred_transition[i] = Xt[i] @ beta_t - Xb[i] @ beta_b
        test_y_resid[i] = y[i] - Xb[i] @ beta_b
rho_cv = partial_spearman(test_pred_transition, y, d[opc].values)
print(f"5-fold CV: partial Spearman of learned transition metric = {rho_cv:+.4f}")
print(f"  (vs in-sample ceiling ~0.765, flow_rel=0.628)")

# hand-designed scalars (deconfounded-by-normalization)
d["conv33_chain_density"] = d["t_nor_conv_3x3_nor_conv_3x3"] / (d["n_nor_conv_3x3"] + 1e-6)
d["pool_chain_density"] = d["t_avg_pool_3x3_avg_pool_3x3"] / (d["n_avg_pool_3x3"] + 1e-6)
d["conv33_to_conv_density"] = (d["t_nor_conv_3x3_nor_conv_3x3"] + d["t_nor_conv_3x3_nor_conv_1x1"]) / (d["n_nor_conv_3x3"] + 1e-6)
d["chain_score"] = (d["t_nor_conv_3x3_nor_conv_3x3"] - d["t_avg_pool_3x3_avg_pool_3x3"]) / (d["n_nor_conv_3x3"] + d["n_avg_pool_3x3"] + 1e-6)

print("\nHand-designed scalars (raw / partial vs oracle):")
for c in ["conv33_chain_density", "pool_chain_density", "conv33_to_conv_density", "chain_score"]:
    if d[c].std() == 0:
        continue
    raw, _ = stats.spearmanr(d[c].values, y)
    part = partial_spearman(d[c].values, y, d[opc].values)
    print(f"  {c:26s} raw={raw:+.4f}  partial={part:+.4f}")
