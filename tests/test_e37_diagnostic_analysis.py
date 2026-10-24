from __future__ import annotations

import pytest

from scripts.analyze_e37_layer_diagnostic import analyze_records, join_controls, paired_spearman


def test_constant_component_is_undefined_without_dropping_pairs() -> None:
    assert paired_spearman([2.0] * 100, [3.0] * 100) is None
    with pytest.raises(ValueError, match="100"):
        paired_spearman([1.0] * 99, [2.0] * 99)


def test_control_ids_are_canonical_and_complete() -> None:
    identifiers = [str(i) for i in range(100)]
    controls = [{"model_or_arch_id": i, "log_params": float(i)} for i in range(100)]
    joined = join_controls(identifiers, controls)
    assert len(joined) == 100
    assert joined["17"] == 17.0
    with pytest.raises(ValueError, match="duplicate"):
        join_controls(identifiers, controls + [controls[0]])
    with pytest.raises(ValueError, match="complete"):
        join_controls(identifiers, controls[:-1])


def test_analysis_emits_all_39_components_for_all_100_pairs() -> None:
    sample = {
        "anchors": [{name: 0.1 for name in (
            "clean_rms", "adv_rms", "clean_centered_l2", "adv_centered_l2",
            "channel", "spatial", "spectrum_fisher", "spectrum_triangular", "spectrum_ks",
        )} for _ in range(4)],
        "transitions": [{"flow": 0.2} for _ in range(3)],
    }
    records = [
        {
            "setting": f"darts_e{stage}", "model_or_arch_id": str(identifier),
            "diagnostic": {"sample_components": [sample] * 64},
        }
        for stage in (0, 100) for identifier in range(100)
    ]
    controls = [{"model_or_arch_id": identifier, "log_parameter_count": identifier + 1} for identifier in range(100)]
    rows, correlations = analyze_records(records, controls)
    assert len(rows) == 100 * 39
    assert len(correlations) == 39
    assert all(row["pair_count"] == 100 for row in correlations)
    assert all(row["stage_spearman"] is None for row in correlations)
    assert all(row["structure_interpretation"] == "exploratory_noncausal" for row in correlations)
