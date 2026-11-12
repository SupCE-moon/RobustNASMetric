import csv
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


input_path, original_path, output_path = map(Path, sys.argv[1:])
with input_path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

keys = [(r["model_or_arch_id"], r["metric_name"]) for r in rows]
assert len(rows) == 88, len(rows)
assert len(keys) == len(set(keys))
failures = [r for r in rows if r["status"] != "ok"]
assert len(failures) == 8, len(failures)
assert Counter(r["model_or_arch_id"] for r in failures) == {
    "Rebuffi2021Fixing_70_16_cutmix_ddpm": 8
}

with original_path.open(newline="", encoding="utf-8") as handle:
    original_rows = list(csv.DictReader(handle))

# RB40 has 11 unavailable artifacts, so the analyzer marks each row "invalid"
# even though its correlation over the 29 valid models is finite and usable.
original = {}
for row in original_rows:
    if row["setting"] != "rb40_mature":
        continue
    value = float(row["direction_adjusted_spearman"])
    if math.isfinite(value):
        original[row["metric_name"]] = value

by_metric = defaultdict(list)
for row in rows:
    if row["status"] == "ok":
        by_metric[row["metric_name"]].append(row)
assert set(by_metric) <= set(original), sorted(set(by_metric) - set(original))

rng = np.random.default_rng(20260927)
summary = []
for metric, valid in sorted(by_metric.items()):
    assert len(valid) == 10, (metric, len(valid))
    sign = -1.0 if valid[0]["direction"] == "lower" else 1.0
    x = np.asarray([float(r["metric_value"]) for r in valid])
    y = np.asarray([float(r["target_value"]) for r in valid])
    rho = float(spearmanr(sign * x, y).statistic)
    boots = []
    for _ in range(20_000):
        idx = rng.integers(0, len(valid), size=len(valid))
        value = float(spearmanr(sign * x[idx], y[idx]).statistic)
        if np.isfinite(value):
            boots.append(value)
    low, high = np.percentile(boots, [2.5, 97.5])
    summary.append(
        {
            "metric_name": metric,
            "valid_models": 10,
            "failed_models": 1,
            "heldout_direction_adjusted_spearman": rho,
            "ci_low": float(low),
            "ci_high": float(high),
            "bootstrap_repeats": 20_000,
            "original_rb40_direction_adjusted_spearman": original[metric],
            "heldout_minus_original": rho - original[metric],
        }
    )

assert len(summary) == 8, len(summary)
summary.sort(key=lambda r: r["heldout_direction_adjusted_spearman"], reverse=True)
tmp = output_path.with_suffix(output_path.suffix + ".tmp")
if tmp.exists():
    tmp.unlink()
with tmp.open("x", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)
    handle.flush()
    os.fsync(handle.fileno())
os.replace(tmp, output_path)

print("heldout_rows", len(rows), "unique_keys", len(set(keys)))
print("status_counts", dict(Counter(r["status"] for r in rows)))
print("failed_models", dict(Counter(r["model_or_arch_id"] for r in failures)))
print("heldout_ranking", [(r["metric_name"], r["heldout_direction_adjusted_spearman"]) for r in summary])
