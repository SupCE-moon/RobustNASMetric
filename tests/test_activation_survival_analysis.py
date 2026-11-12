"""Preregistered complete-grid comparison against TRNAS Phi."""

from __future__ import annotations

import math

import pytest

from scripts.analyze_activation_survival import (
    COUNTS,
    architecture_bootstrap,
    compare_complete_grid,
    main_gate,
)


def _rows(*, missing: bool = False, duplicate: bool = False) -> tuple[list[dict], list[dict]]:
    candidate: list[dict] = []
    phi: list[dict] = []
    for setting, count in COUNTS.items():
        for index in range(count):
            identifier = f"arch_{index:03d}"
            epoch = None if setting == "rb40_mature" else int(setting.rsplit("e", 1)[1])
            key = {
                "setting": setting, "model_or_arch_id": identifier,
                "checkpoint_epoch": epoch, "target_value": float(index),
                "status": "ok",
            }
            candidate.append({**key, "metric_name": "feature_activation_sign_survival_sum",
                              "metric_value": float(index)})
            phi.append({**key, "metric_name": "trnas_consistency_phi",
                        "metric_value": float(index)})
    if missing:
        phi.pop()
    if duplicate:
        phi.append(dict(phi[-1]))
    return candidate, phi


def test_main_gate_requires_eight_settings_and_positive_interval() -> None:
    deltas = {setting: 0.1 for setting in COUNTS}
    assert main_gate(deltas, ci_low=0.01, valid_rows=560)
    assert not main_gate(deltas, ci_low=0.0, valid_rows=560)
    assert not main_gate(deltas, ci_low=0.01, valid_rows=559)
    assert not main_gate({setting: -0.1 for setting in COUNTS}, ci_low=0.01, valid_rows=560)
    with pytest.raises(ValueError, match="eight settings"):
        main_gate(dict(list(deltas.items())[:-1]), ci_low=0.01, valid_rows=560)


def test_complete_grid_requires_exact_unique_finite_pairing() -> None:
    candidate, phi = _rows()
    result = compare_complete_grid(candidate, phi)
    assert result["valid_rows"] == 560
    assert set(result["setting_deltas"]) == set(COUNTS)
    assert all(math.isclose(value, 0.0) for value in result["setting_deltas"].values())
    assert math.isclose(result["eight_setting_delta"], 0.0)
    for altered in (_rows(missing=True), _rows(duplicate=True)):
        with pytest.raises(ValueError, match="missing|duplicate"):
            compare_complete_grid(*altered)
    candidate[0]["metric_value"] = math.inf
    with pytest.raises(ValueError, match="finite"):
        compare_complete_grid(candidate, phi)


def test_bootstrap_keeps_each_architecture_across_all_stages() -> None:
    candidate, phi = _rows()
    draws = architecture_bootstrap(candidate, phi, draws=3, seed=20261027,
                                   return_sampled_ids=True)
    assert len(draws) == 3
    for draw in draws:
        assert len(draw["setting_deltas"]) == 8
        nb_ids = draw["sampled_ids"]["nb201"]
        darts_ids = draw["sampled_ids"]["darts"]
        assert len(nb_ids) == 64 and len(darts_ids) == 100
        assert all(math.isclose(value, 0.0) for value in draw["setting_deltas"].values())
