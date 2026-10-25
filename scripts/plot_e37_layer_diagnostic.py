"""Render every preregistered E37 layer component, including undefined values."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


ANCHOR_COMPONENTS = (
    "clean_rms", "adv_rms", "clean_centered_l2", "adv_centered_l2",
    "channel", "spatial", "spectrum_fisher", "spectrum_triangular", "spectrum_ks",
)
ROWS = tuple(("anchor", slot, component) for component in ANCHOR_COMPONENTS for slot in range(4)) + tuple(
    ("transition", slot, "flow") for slot in range(3)
)


def plot_correlations(source: Path, output: Path) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    with source.open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    expected = set(ROWS)
    keyed: dict[tuple[str, int, str], dict] = {}
    for record in records:
        key = (record["family"], int(record["slot"]), record["component"])
        if key in keyed:
            raise ValueError(f"duplicate component: {key}")
        keyed[key] = record
        if int(record["pair_count"]) != 100:
            raise ValueError(f"component is not paired 100/100: {key}")
    if len(keyed) != 39 or set(keyed) != expected:
        raise ValueError(f"expected exact 39-component grid; found {len(keyed)}")

    labels = list(ANCHOR_COMPONENTS) + ["flow (adjacent anchors)"]
    columns = ("anchor 0 / flow 0", "anchor 1 / flow 1", "anchor 2 / flow 2", "anchor 3")
    keys = ("stage_spearman", "change_log_parameter_spearman")
    titles = (
        "e0 vs e100 architecture-rank Spearman",
        "Component change vs log(parameter count) Spearman",
    )
    figure, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    for axis, field, title in zip(axes, keys, titles, strict=True):
        matrix = np.full((len(labels), 4), np.nan)
        for row, component in enumerate(ANCHOR_COMPONENTS):
            for slot in range(4):
                value = keyed[("anchor", slot, component)][field]
                if value:
                    matrix[row, slot] = float(value)
        for slot in range(3):
            value = keyed[("transition", slot, "flow")][field]
            if value:
                matrix[-1, slot] = float(value)
        if not np.all(np.isfinite(matrix[~np.isnan(matrix)])):
            raise ValueError(f"nonfinite correlation in {field}")
        color_map = plt.get_cmap("coolwarm").copy()
        color_map.set_bad("#d9d9d9")
        image = axis.imshow(np.ma.masked_invalid(matrix), cmap=color_map, vmin=-1, vmax=1, aspect="auto")
        axis.set_xticks(range(4), labels=columns, rotation=27, ha="right")
        axis.set_yticks(range(len(labels)), labels=labels)
        axis.set_title(title)
        axis.set_xlabel("Predeclared anchor or transition slot")
        axis.set_ylabel("E37 numeric component")
        for row in range(len(labels)):
            for column in range(4):
                value = matrix[row, column]
                label = "n/a" if math.isnan(value) else f"{value:.3f}"
                axis.text(column, row, label, ha="center", va="center", fontsize=8)
        figure.colorbar(image, ax=axis, shrink=0.7, label="Spearman ρ")
    figure.suptitle("DARTS E37 full paired-layer diagnostic (100 architectures; exploratory, non-causal)")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return len(keyed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correlations-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    plot_correlations(arguments.correlations_csv, arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
