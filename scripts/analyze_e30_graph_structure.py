"""E29: direct computation-graph (op-interaction) metrics for NB201.

Tests whether the op-to-edge arrangement (which op connects to which node),
beyond total op counts, predicts oracle robustness. If the interaction signal
beats flow_rel's deconfounded partial (~0.628), a direct DAG metric is promising.
"""
import numpy as np
import pandas as pd
from scipy import stats

from robust_metric_lab.nasrobbench201 import parse_nb201_arch

OPS = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
# NB201 cell DAG: edge sources [0,0,1,0,1,2]; transitions = which edge feeds which
# node 1 = e0; node 2 = e1+e2; node 3 = e3+e4+e5
# so transitions: e0->e2, e0->e4, e1->e5, e2->e5
TRANSITIONS = [(0, 2), (0, 4), (1, 5), (2, 5)]

df = pd.read_csv("outputs/e23_cross_space/nb201_64/metrics_64_e0e5e10e25e50.csv").drop_duplicates("arch_id")

rows = []
for _, r in df.iterrows():
    ops = parse_nb201_arch(str(r["arch_string"]))
    if ops is None:
        continue
    feat = {"arch_id": r["arch_id"], "oracle": r["oracle_target"]}
    for op in OPS:
        feat[f"n_{op}"] = ops.count(op)
    # transition counts (op_a -> op_b)
    for a, b in TRANSITIONS:
        feat[f"t_{ops[a]}_{ops[b]}"] = feat.get(f"t_{ops[a]}_{ops[b]}", 0) + 1
    # edge-position one-hot: which op at which edge index (0..5)
    for e in range(6):
        feat[f"e{e}_{ops[e]}"] = 1
    rows.append(feat)

d = pd.DataFrame(rows).fillna(0)
opc = [f"n_{op}" for op in OPS]
tcols = [c for c in d.columns if c.startswith("t_")]
ecols = [c for c in d.columns if c.startswith("e")]

y = d["oracle"].values
X_base = d[opc].values
X_base = np.column_stack([np.ones(len(d)), X_base])
beta, *_ = np.linalg.lstsq(X_base, y, rcond=None)
resid = y - X_base @ beta
r2_base = 1 - resid.var() / y.var()
print(f"R^2(oracle | op counts) = {r2_base:.4f}   (residual std = {resid.std():.4f}, oracle std = {y.std():.4f})")

# Does the transition matrix add predictive power beyond op counts?
X_full = np.column_stack([X_base, d[tcols].values])
beta_f, *_ = np.linalg.lstsq(X_full, y, rcond=None)
resid_f = y - X_full @ beta_f
r2_full = 1 - resid_f.var() / y.var()
print(f"R^2(oracle | op counts + transitions) = {r2_full:.4f}   (delta = {r2_full - r2_base:+.4f})")

X_full2 = np.column_stack([X_base, d[ecols].values])
beta_e, *_ = np.linalg.lstsq(X_full2, y, rcond=None)
resid_e = y - X_full2 @ beta_e
r2_e = 1 - resid_e.var() / y.var()
print(f"R^2(oracle | op counts + edge-position) = {r2_e:.4f}   (delta = {r2_e - r2_base:+.4f})")

# partial Spearman of each individual transition count vs oracle (control op counts)
def partial_spearman(x, y, controls):
    X = np.column_stack([np.ones(len(y)), controls])
    bx, *_ = np.linalg.lstsq(X, x, rcond=None)
    by, *_ = np.linalg.lstsq(X, y, rcond=None)
    rho, _ = stats.spearmanr(x - X @ bx, y - X @ by)
    return rho

print("\nTop transition-count features (partial Spearman vs oracle, control op counts):")
res = []
for c in tcols:
    if d[c].std() == 0:
        continue
    rho = partial_spearman(d[c].values, y, d[opc].values)
    res.append((c, rho))
for c, rho in sorted(res, key=lambda x: -abs(x[1]))[:12]:
    print(f"  {c:35s} partial={rho:+.4f}")

# also: raw spearman of each transition feature
print("\nTop transition-count features (raw Spearman vs oracle):")
res2 = []
for c in tcols:
    if d[c].std() == 0:
        continue
    rho, _ = stats.spearmanr(d[c].values, y)
    res2.append((c, rho))
for c, rho in sorted(res2, key=lambda x: -abs(x[1]))[:12]:
    print(f"  {c:35s} raw={rho:+.4f}")
