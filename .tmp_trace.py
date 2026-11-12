import pandas as pd
from scipy import stats

root = r"D:\Project\RobustNAS\DynamicBN\DynamicBN20251203\robust_metric"

pub = pd.read_csv(root + r"\outputs\strict_all_metrics_table_20261024_trnas_repaired\sources\metric_values_original_strict.csv")
pub_e0 = pub[pub["setting"] == "nb201_e0"]
e23 = pd.read_csv(root + r"\outputs\e23_cross_space\nb201_64\metrics_64_e0e5e10e25e50.csv")
e23 = e23[e23["train_epochs"] == 0].drop_duplicates("arch_id")
print("pub metrics", pub_e0["metric_name"].nunique(), "archs", pub_e0["model_or_arch_id"].nunique())
print("e23 cols", len(e23.columns), "rows", len(e23))

shared = sorted(set(pub_e0["metric_name"]) & set(e23.columns))
print("shared metric names", len(shared))
print("sample shared", shared[:8])

rows = []
for m in shared:
    a = pub_e0[pub_e0["metric_name"] == m].copy()
    a["arch_id"] = a["model_or_arch_id"].astype(str)
    a = a[a["status"] == "ok"][["arch_id", "metric_value"]]
    b = e23[["arch_id"]].copy()
    b["arch_id"] = b["arch_id"].astype(str)
    b["v23"] = pd.to_numeric(e23[m], errors="coerce").to_numpy()
    mrg = a.merge(b, on="arch_id").dropna()
    if len(mrg) < 10:
        continue
    diff = (mrg["metric_value"] - mrg["v23"]).abs()
    rho = stats.spearmanr(mrg["metric_value"], mrg["v23"])[0]
    rows.append((m, len(mrg), float(diff.max()), float(diff.mean()), float(rho),
                 float(mrg["metric_value"].min()), float(mrg["metric_value"].max()),
                 float(mrg["v23"].min()), float(mrg["v23"].max())))

frame = pd.DataFrame(rows, columns=["metric", "n", "max_abs", "mean_abs", "rho", "pub_min", "pub_max", "e23_min", "e23_max"])
frame = frame.sort_values("max_abs", ascending=False)
print(frame.to_string(index=False, float_format=lambda v: "{:.3e}".format(v) if abs(v) < 1e-3 else "{:.4f}".format(v)))
print()
print("identical columns (max_abs==0):", int((frame["max_abs"] == 0).sum()), "of", len(frame))
