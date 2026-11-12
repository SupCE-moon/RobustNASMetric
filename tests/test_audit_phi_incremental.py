"""Tests for the Phi incrementality audit (T1 partial rho, T2 nested R^2, T3 strata).

Three failure modes would make this audit lie, and each gets a test:

1. The baseline family must not contain Phi itself -- otherwise "Phi adds
   information beyond Phi" is trivially true.
2. Architecture identity must be grid-qualified.  DARTS and NAS-Bench-201 both
   number networks 0..N, so a bare id merges unrelated networks and lets the
   bootstrap draw them together.
3. A permutation test must actually break the link it claims to break; a p-value
   that stays tiny after shuffling means the null is wrong, not that the effect
   is strong.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_phi_incremental as api  # noqa: E402


def _redundancy_json(tmp_path: Path, members: list[str]) -> Path:
    payload = {
        "cuts": [
            {
                "min_rho": 0.5,
                "clusters": [{"size": len(members), "members": members}],
            }
        ]
    }
    path = tmp_path / "redundancy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_family_returns_the_named_cluster(tmp_path: Path) -> None:
    members = [f"feature_measure_m{i}" for i in range(api.FAMILY_SIZE)]
    path = _redundancy_json(tmp_path, members)
    assert api.load_family(path) == sorted(members)


def test_load_family_refuses_a_cluster_that_contains_phi(tmp_path: Path) -> None:
    members = [api.PHI] + [f"feature_measure_m{i}" for i in range(api.FAMILY_SIZE - 1)]
    path = _redundancy_json(tmp_path, members)
    with pytest.raises(ValueError, match="must not be part of the baseline"):
        api.load_family(path)


def test_orient_flips_lower_is_better_metrics() -> None:
    panel = pd.DataFrame(
        {
            "feature_delta": [1.0, 2.0, 3.0],
            api.PHI: [1.0, 2.0, 3.0],
        }
    )
    direction = {"feature_delta": "lower", api.PHI: "higher"}
    flipped = api.orient(panel, direction)
    assert list(flipped["feature_delta"]) == [-1.0, -2.0, -3.0]
    assert list(flipped[api.PHI]) == [1.0, 2.0, 3.0]


def test_partial_rho_is_near_zero_when_control_explains_everything() -> None:
    """Both x and y are the control plus independent noise: nothing should survive.

    The noise is deliberately not microscopic.  When the residual variance is
    near zero the ratio is ill-conditioned and returns 0.2-0.3 from rounding
    alone, which would look like signal.
    """
    rng = np.random.default_rng(7)
    control = rng.normal(size=300)
    x = control + 0.6 * rng.normal(size=300)
    y = control + 0.6 * rng.normal(size=300)
    assert abs(api.partial_rho(x, y, control)) < 0.12


def test_partial_rho_stays_positive_when_x_carries_own_signal() -> None:
    rng = np.random.default_rng(11)
    control = rng.normal(size=300)
    x = control + rng.normal(size=300)
    y = control + x + 0.3 * rng.normal(size=300)
    assert api.partial_rho(x, y, control) > 0.4


def test_build_frame_qualifies_arch_ids_by_grid() -> None:
    """Two networks both called '0' in different grids must stay distinct."""
    index = pd.MultiIndex.from_tuples(
        [("nb201_e0", "0"), ("darts_e0", "0")], names=["setting", "model_or_arch_id"]
    )
    family = ["feature_measure_a", "feature_measure_b"]
    panel = pd.DataFrame({api.PHI: [1.0, 2.0], family[0]: [3.0, 4.0], family[1]: [5.0, 6.0]}, index=index)
    target = pd.Series([0.1, 0.2], index=index)
    frame = api.build_frame(panel, target, family)
    assert frame["arch"].nunique() == 2
    assert set(frame["arch"]) == {"nb201:0", "darts:0"}


def test_resample_keeps_every_setting_of_a_drawn_architecture() -> None:
    frame = pd.DataFrame(
        {
            "setting": ["a", "a", "a", "a", "b", "b", "b", "b"],
            "arch": ["g:0", "g:0", "g:1", "g:1", "g:0", "g:0", "g:1", "g:1"],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "phi": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "b": [0.0] * 8,
            "grid": ["g"] * 8,
        }
    )
    sample = api._resample(frame, np.random.default_rng(3))
    assert set(sample["setting"]) == {"a", "b"}
    assert sample["arch"].nunique() == len(sample.groupby("arch"))
    for _, block in sample.groupby("arch"):
        assert len(block) == 4


def test_bootstrap_is_reproducible_for_a_fixed_seed() -> None:
    rng = np.random.default_rng(5)
    rows = []
    for setting in ("s1", "s2"):
        for i in range(30):
            b = rng.normal()
            y = b + rng.normal()
            phi = b + y + rng.normal()
            rows.append({"setting": setting, "arch": f"s:{i}", "y": y, "phi": phi, "b": b})
    frame = pd.DataFrame(rows)
    first = api.bootstrap(frame, resamples=30, seed=99)
    second = api.bootstrap(frame, resamples=30, seed=99)
    assert first["low"] == pytest.approx(second["low"])
    assert first["high"] == pytest.approx(second["high"])


def test_permutation_collapses_a_planted_relationship() -> None:
    rng = np.random.default_rng(13)
    rows = []
    for setting in ("s1", "s2"):
        for i in range(40):
            b = rng.normal()
            phi = b + rng.normal()
            y = b + phi + rng.normal()
            rows.append({"setting": setting, "arch": f"s:{i}", "y": y, "phi": phi, "b": b})
    frame = pd.DataFrame(rows)
    result = api.permutation(frame, resamples=200, seed=17)
    assert result["p_value"] < 0.05
    assert abs(result["null_mean"]) < 0.15


def test_verdict_covers_the_three_branches() -> None:
    assert api.verdict(0.2, 0.5, 0.01, 0.1, 0.3) == "has incremental value"
    assert "no evidence" in api.verdict(-0.1, 0.2, 0.8, 0.001, 0.02)
    assert api.verdict(-0.1, 0.3, 0.04, 0.5, 0.6) == "not settled"


def test_nested_r2_does_not_gain_when_phi_is_noise() -> None:
    rng = np.random.default_rng(23)
    rows = []
    for i in range(80):
        b = rng.normal()
        rows.append(
            {
                "arch": f"g:{i}",
                "y": b + 0.2 * rng.normal(),
                "b": b,
                "phi": rng.normal(),
            }
        )
    result = api.nested_r2(pd.DataFrame(rows))
    assert result["gain"] < 0.05


def test_end_to_end_audit_flags_a_planted_increment(tmp_path: Path) -> None:
    """Phi carries signal that the family average does not, and the audit finds it."""
    rng = np.random.default_rng(31)
    family = [f"feature_measure_m{i}" for i in range(api.FAMILY_SIZE)]
    redundancy = _redundancy_json(tmp_path, family)

    rows = []
    for setting in ("nb201_e0", "darts_e0"):
        for i in range(48):
            arch = f"{setting}_{i}"
            y = rng.normal()
            shared = 0.4 * y + 0.9 * rng.normal()  # family signal: real but diluted
            phi = 0.75 * y + 0.5 * rng.normal()
            target = 0.6 * y + 0.4 * rng.normal()
            rows.append(
                {
                    "setting": setting,
                    "model_or_arch_id": arch,
                    "metric_name": api.PHI,
                    "metric_value": phi,
                    "direction": "higher",
                    "status": "ok",
                    "target_value": target,
                }
            )
            for name in family:
                rows.append(
                    {
                        "setting": setting,
                        "model_or_arch_id": arch,
                        "metric_name": name,
                        "metric_value": -shared + 0.3 * rng.normal(),
                        "direction": "lower",
                        "status": "ok",
                        "target_value": target,
                    }
                )
    rows_path = tmp_path / "rows.csv"
    pd.DataFrame(rows).to_csv(rows_path, index=False)

    result = api.analyze(rows_path, None, redundancy, resamples=60, seed=20261112)

    assert result["config"]["n_architectures"] == 96
    assert result["config"]["n_rows"] == 96
    assert result["observed_delta_rho"] > 0.2
    assert result["verdict"] in {
        "has incremental value",
        "no evidence of incremental value (not the same as proving zero)",
        "not settled",
    }
    assert len(result["per_setting"]) == 2
    assert result["inputs"]["rows_sha256"]


def test_end_to_end_audit_sees_no_increment_when_phi_is_redundant(tmp_path: Path) -> None:
    """Phi is just the family average in disguise: the audit must not claim value."""
    rng = np.random.default_rng(37)
    family = [f"feature_measure_m{i}" for i in range(api.FAMILY_SIZE)]
    redundancy = _redundancy_json(tmp_path, family)

    rows = []
    for setting in ("nb201_e0", "darts_e0"):
        for i in range(48):
            arch = f"{setting}_{i}"
            y = rng.normal()
            shared = 0.85 * y + 0.3 * rng.normal()
            rows.append(
                {
                    "setting": setting,
                    "model_or_arch_id": arch,
                    "metric_name": api.PHI,
                    "metric_value": shared + 0.05 * rng.normal(),
                    "direction": "higher",
                    "status": "ok",
                    "target_value": y,
                }
            )
            for name in family:
                rows.append(
                    {
                        "setting": setting,
                        "model_or_arch_id": arch,
                        "metric_name": name,
                        "metric_value": -shared + 0.25 * rng.normal(),
                        "direction": "lower",
                        "status": "ok",
                        "target_value": y,
                    }
                )
    rows_path = tmp_path / "rows.csv"
    pd.DataFrame(rows).to_csv(rows_path, index=False)

    result = api.analyze(rows_path, None, redundancy, resamples=60, seed=20261112)
    assert result["observed_delta_rho"] < 0.3
