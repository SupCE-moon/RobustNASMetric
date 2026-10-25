from __future__ import annotations

import csv
from pathlib import Path
import subprocess
import sys

import pytest


def _write_correlations(path: Path, *, omit: tuple[str, int, str] | None = None) -> None:
    components = (
        "clean_rms", "adv_rms", "clean_centered_l2", "adv_centered_l2",
        "channel", "spatial", "spectrum_fisher", "spectrum_triangular", "spectrum_ks",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "family", "slot", "component", "stage_spearman",
            "change_log_parameter_spearman", "pair_count",
        ))
        writer.writeheader()
        for family, slots, names in (("anchor", range(4), components), ("transition", range(3), ("flow",))):
            for slot in slots:
                for component in names:
                    if (family, slot, component) == omit:
                        continue
                    writer.writerow({
                        "family": family, "slot": slot, "component": component,
                        "stage_spearman": "" if slot == 3 else "0.25",
                        "change_log_parameter_spearman": "" if slot == 3 else "-0.25",
                        "pair_count": 100,
                    })


def test_plot_includes_full_fixed_39_component_grid(tmp_path: Path) -> None:
    from scripts.plot_e37_layer_diagnostic import plot_correlations

    source = tmp_path / "correlations.csv"
    output = tmp_path / "all_components.png"
    _write_correlations(source)
    assert plot_correlations(source, output) == 39
    assert output.is_file() and output.stat().st_size > 1000
    with pytest.raises(ValueError, match="39"):
        _write_correlations(source, omit=("anchor", 2, "spatial"))
        plot_correlations(source, tmp_path / "incomplete.png")


def test_plot_cli_starts_without_external_pythonpath() -> None:
    project = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(project / "scripts/plot_e37_layer_diagnostic.py"), "--help"],
        cwd=project, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
