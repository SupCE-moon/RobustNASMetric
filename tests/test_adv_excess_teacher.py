"""Display-only three-decimal table; strict ranking stays full precision."""

import pytest

from scripts.prepare_adv_excess_teacher_table import make_teacher_records


SCORES = ("rb40_mature", "nb201_e0", "nb201_e5", "nb201_e10",
          "nb201_e25", "nb201_e50", "darts_e0", "darts_e100",
          "all_setting_mean")


def sample_inputs():
    old = []
    for index in range(83):
        old.append({
            "metric_name": f"old-{index}", "strict_status": "ranked",
            "source_type": "论文/已有方法", "specific_origin": f"source-{index}",
            "literature_experiment_basis": f"evidence-{index}",
            "raw_rows": "560", "ok_finite_rows": "560", "required_rows": "560",
        })
    for name in ("racl_arch_lipschitz_log_upper",
                 "racl_fixed_network_log_upper_proxy"):
        old.append({
            "metric_name": name, "strict_status": "excluded",
            "source_type": "论文/已有方法", "specific_origin": "RACL",
            "literature_experiment_basis": "original failure evidence",
            "raw_rows": "560", "ok_finite_rows": "0", "required_rows": "560",
            "exclusion_category": "space_specific", "exclusion_reason": "not cross-space",
        })
    ranking = []
    for index, name in enumerate(
        ["feature_adv_excess_centered_unit_l2"] +
        [f"old-{item}" for item in range(83)], start=1
    ):
        ranking.append({
            "rank": index, "metric_name": name,
            **{column: 0.123456 for column in SCORES},
        })
    return old, ranking


def test_teacher_table_has_all_86_metrics_and_display_only_rounding():
    old, ranking = sample_inputs()
    records = make_teacher_records(old, ranking, release_id="strict-new")
    assert len(records) == 86
    assert [row["strict_rank"] for row in records[:2]] == [1, 2]
    assert records[0]["all_setting_mean"] == "0.123"
    assert records[0]["nb201_mean"] == "0.123"
    assert records[0]["Zero-cost mean"] == "0.123"
    assert records[0]["source_type"] == "Ours"
    assert records[1]["specific_origin"] == "source-0"
    assert all(row["strict_status"] == "excluded" for row in records[-2:])
    assert ranking[0]["all_setting_mean"] == 0.123456


def test_teacher_table_rejects_incomplete_ranking():
    old, ranking = sample_inputs()
    with pytest.raises(ValueError):
        make_teacher_records(old, ranking[:-1], release_id="strict-new")
