"""Prepare validated, display-only records for the teacher-facing table.

The source ranking retains full precision; this module never recalculates rank.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence


METRIC = "feature_adv_excess_centered_unit_l2"
SCORE_COLUMNS = (
    "rb40_mature", "nb201_e0", "nb201_e5", "nb201_e10", "nb201_e25",
    "nb201_e50", "darts_e0", "darts_e100", "nb201_mean",
    "Zero-cost mean", "all_setting_mean",
)
SETTING_COLUMNS = SCORE_COLUMNS[:8]
COLUMNS = (
    "strict_rank", "metric_name", "strict_status", "historical_45",
    "source_type", "specific_origin", "literature_experiment_basis",
    "raw_rows", "ok_finite_rows", "required_rows", "exclusion_category",
    "exclusion_reason", *SCORE_COLUMNS, "release_id",
)
EXCLUDED = {
    "racl_arch_lipschitz_log_upper",
    "racl_fixed_network_log_upper_proxy",
}


def _display(number: object) -> str:
    value = float(number)
    if not math.isfinite(value):
        raise ValueError("teacher table cannot display a nonfinite score")
    return f"{value:.3f}"


def make_teacher_records(
    historical: Sequence[Mapping[str, object]],
    ranking: Sequence[Mapping[str, object]],
    *, release_id: str,
) -> list[dict[str, object]]:
    """Preserve 85 historical provenances and append one strictly ranked metric."""
    if not release_id or len(historical) != 85 or len(ranking) != 84:
        raise ValueError("teacher table requires 85 old and 84 ranked metrics")
    old_by_name = {str(row["metric_name"]): row for row in historical}
    if len(old_by_name) != 85 or METRIC in old_by_name:
        raise ValueError("historical provenance keys are duplicate or changed")
    old_excluded = {
        name for name, row in old_by_name.items()
        if row["strict_status"] == "excluded"
    }
    if old_excluded != EXCLUDED:
        raise ValueError("RACL exclusion evidence must remain in the table")
    if any(row["strict_status"] != "ranked" for name, row in old_by_name.items()
           if name not in EXCLUDED):
        raise ValueError("historical ranked eligibility changed")
    ranking_by_name = {str(row["metric_name"]): row for row in ranking}
    if (len(ranking_by_name) != 84
            or set(ranking_by_name) != (set(old_by_name) - EXCLUDED) | {METRIC}):
        raise ValueError("ranking names do not match complete historical plus new metric")
    ranks = [int(row["rank"]) for row in ranking]
    if sorted(ranks) != list(range(1, 85)):
        raise ValueError("strict ranks are not consecutive 1..84")

    output = []
    for ranked in sorted(ranking, key=lambda row: int(row["rank"])):
        name = str(ranked["metric_name"])
        if name == METRIC:
            source = {
                "historical_45": "no", "source_type": "Ours",
                "specific_origin": "Ours，2026-10-24 预注册对抗超额特征漂移",
                "literature_experiment_basis": (
                    "项目自研；对抗减随机基线的中心化单位特征漂移；"
                    "预注册设计见 2026-10-24-adversarial-excess-feature-drift-design.md"
                ),
                "raw_rows": "560", "ok_finite_rows": "560",
                "required_rows": "560",
            }
        else:
            source = dict(old_by_name[name])
        numbers = {column: float(ranked[column]) for column in SETTING_COLUMNS}
        overall = float(ranked["all_setting_mean"])
        if not math.isclose(
            overall, sum(numbers.values()) / 8, rel_tol=0, abs_tol=1e-10
        ):
            raise ValueError(f"strict ranking mean mismatch: {name}")
        row = {column: source.get(column, "") for column in COLUMNS}
        row.update({
            "strict_rank": int(ranked["rank"]), "metric_name": name,
            "strict_status": "ranked", "exclusion_category": "",
            "exclusion_reason": "", "release_id": release_id,
        })
        for column, number in numbers.items():
            row[column] = _display(number)
        row["nb201_mean"] = _display(sum(numbers[column] for column in
                                          SETTING_COLUMNS[1:6]) / 5)
        row["Zero-cost mean"] = _display(
            (numbers["nb201_e0"] + numbers["darts_e0"]) / 2
        )
        row["all_setting_mean"] = _display(overall)
        output.append(row)
    for name in sorted(EXCLUDED):
        source = dict(old_by_name[name])
        row = {column: source.get(column, "") for column in COLUMNS}
        row.update({"strict_rank": "", "metric_name": name,
                    "strict_status": "excluded", "release_id": release_id})
        output.append(row)
    return output
