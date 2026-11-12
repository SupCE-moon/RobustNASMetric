"""Paired resampling and independent-holdout guard tests."""

import pytest

from scripts.analyze_adv_excess_release import (
    bootstrap_mean_delta,
    bootstrap_stage_stability,
    holdout_material_status,
    interpret_predeclared_gates,
    paired_sample_indices,
    require_release_verified,
    space_balanced_mean,
    stage_pairs,
)


EPOCHS = ("nb201_e0", "nb201_e5", "nb201_e10", "nb201_e25", "nb201_e50")


def test_bootstrap_resamples_same_nb201_architectures_across_five_epochs():
    identifiers = {epoch: ("a", "b", "c") for epoch in EPOCHS}
    draws = paired_sample_indices(identifiers, draws=100, seed=20261024)
    assert len(draws) == 100
    assert all(draw[epoch] == draw["nb201_e0"] for draw in draws for epoch in EPOCHS)


def test_stage_pairing_rejects_missing_or_reordered_architectures():
    e0 = [("a", 0.1), ("b", 0.2), ("c", 0.3)]
    e50 = [("c", 0.4), ("a", 0.5), ("b", 0.6)]
    paired = stage_pairs(e0, e50)
    assert paired == [("a", 0.1, 0.5), ("b", 0.2, 0.6), ("c", 0.3, 0.4)]
    with pytest.raises(ValueError):
        stage_pairs(e0, e50[:-1])
    with pytest.raises(ValueError):
        stage_pairs(e0, [("a", 0.4), ("a", 0.5), ("b", 0.6)])


def test_missing_eleventh_holdout_checkpoint_suppresses_conclusion():
    expected = tuple(f"model-{index}" for index in range(11))
    materials = {name: {"checkpoint_ok": True, "label_ok": True,
                        "sample_hash_ok": True}
                 for name in expected[:-1]}
    result = holdout_material_status(expected, materials)
    assert result["ready"] is False
    assert result["complete"] == 10
    assert result["missing_or_invalid"] == ["model-10"]
    materials["model-10"] = {"checkpoint_ok": True, "label_ok": False,
                             "sample_hash_ok": True}
    assert holdout_material_status(expected, materials)["ready"] is False
    materials["model-10"]["label_ok"] = True
    assert holdout_material_status(expected, materials)["ready"] is True


def comparison_rows():
    rows = []
    for setting in ("rb40_mature", *EPOCHS, "darts_e0", "darts_e100"):
        for index in range(6):
            target = index / 10
            for name, value, direction in (
                ("new", -target, "lower"),
                ("phi", -target, "higher"),
            ):
                rows.append({
                    "setting": setting,
                    "model_or_arch_id": f"{setting.split('_')[0]}-{index}",
                    "metric_name": name, "metric_value": value,
                    "target_value": target, "direction": direction,
                })
    return rows


def test_paired_bootstrap_orients_lower_and_shares_stage_draws():
    result = bootstrap_mean_delta(
        comparison_rows(), "new", "phi", draws=100, seed=20261024
    )
    assert result["point_delta"] == pytest.approx(2.0)
    assert result["ci95_low"] == pytest.approx(2.0)
    assert result["shared_epoch_indices"] is True


def test_partial_setting_comparison_is_not_a_formal_mean():
    rows = [row for row in comparison_rows() if row["setting"] != "darts_e100"]
    with pytest.raises(ValueError):
        bootstrap_mean_delta(rows, "new", "phi", draws=10, seed=20261024)


def test_darts_stability_comparison_uses_same_architectures():
    rows = []
    for setting in ("darts_e0", "darts_e100"):
        for index in range(6):
            rows.extend([
                {"setting": setting, "model_or_arch_id": f"d-{index}",
                 "metric_name": "new", "metric_value": index},
                {"setting": setting, "model_or_arch_id": f"d-{index}",
                 "metric_name": "max",
                 "metric_value": index if setting == "darts_e0" else -index},
            ])
    result = bootstrap_stage_stability(
        rows, "new", "max", draws=100, seed=20261024
    )
    assert result["new_stability"] == pytest.approx(1.0)
    assert result["comparison_stability"] == pytest.approx(-1.0)
    assert result["ci95_delta_low"] > 0


def test_analysis_refuses_unverified_or_incomplete_release():
    verified = {"status": "PASS", "raw_rows": 48160, "unique_raw_keys": 48160,
                "ranked_metrics": 84, "excluded_metrics": 2}
    require_release_verified(verified)
    with pytest.raises(ValueError):
        require_release_verified({**verified, "status": "FAILED"})
    with pytest.raises(ValueError):
        require_release_verified({**verified, "raw_rows": 48159})


def test_three_space_mean_and_preregistered_gates():
    scores = {
        "rb40_mature": 0.3,
        **{epoch: 0.6 for epoch in EPOCHS},
        "darts_e0": 0.9, "darts_e100": 0.9,
    }
    assert space_balanced_mean(scores) == pytest.approx(0.6)
    positive = interpret_predeclared_gates(
        mean_delta_ci_low=0.01, new_stage_ci_low=0.02,
        stability_delta_ci_low=0.03,
    )
    assert positive == {"beats_phi_on_frozen_grid": True,
                        "improves_stage_stability": True}
    boundary = interpret_predeclared_gates(
        mean_delta_ci_low=0.0, new_stage_ci_low=-0.1,
        stability_delta_ci_low=0.03,
    )
    assert boundary == {"beats_phi_on_frozen_grid": False,
                        "improves_stage_stability": False}
