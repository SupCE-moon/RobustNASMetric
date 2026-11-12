"""Tests for the stable-subset join.

The script exists to catch one specific failure: a strong association that is
really a seed artefact. The tests therefore focus on the classification (which
group a metric lands in) and on the verdict logic, not on the statistics, which
are shared with the other audits.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import audit_rq2_stable_subset as audit  # noqa: E402


def _frame(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    target = rng.uniform(0.1, 0.5, n)
    clean = target + rng.normal(0, 0.02, n)
    params = rng.integers(90_000, 2_700_000, n).astype(float)
    return pd.DataFrame(
        {
            "arch_id": [str(i) for i in range(n)],
            "arch_string": ["|nor_conv_3x3~0|"] * n,
            "target": target,
            "clean": clean,
            "params": params,
            "signal_metric": target + rng.normal(0, 0.05, n),
            "noise_metric": rng.normal(0, 1, n),
        }
    )


def _seeds(**overrides) -> dict[str, dict]:
    base = {
        "signal_metric": {"value_rho": 0.95, "signal_verdict": "stable_signal"},
        "noise_metric": {"value_rho": 0.20, "signal_verdict": "signal_flips"},
    }
    base.update(overrides)
    return base


def test_group_of_four_buckets():
    seeds = {
        "both": {"value_rho": 0.95, "signal_verdict": "stable_signal"},
        "value_only": {"value_rho": 0.95, "signal_verdict": "signal_flips"},
        "signal_only": {"value_rho": 0.40, "signal_verdict": "stable_signal"},
        "neither": {"value_rho": 0.40, "signal_verdict": "signal_flips"},
    }
    assert audit.group_of("both", seeds) == "stable_both"
    assert audit.group_of("value_only", seeds) == "stable_value_only"
    assert audit.group_of("signal_only", seeds) == "stable_signal_only"
    assert audit.group_of("neither", seeds) == "unstable_both"


def test_group_of_unmeasured_metric_is_empty():
    assert audit.group_of("missing", {}) == ""


def test_partial_rho_removes_a_shared_driver():
    rng = np.random.default_rng(1)
    driver = rng.normal(0, 1, 200)
    x = driver + rng.normal(0, 0.6, 200)
    y = driver + rng.normal(0, 0.6, 200)
    rho = audit.partial_rho(pd.Series(x), pd.Series(y), [pd.Series(driver)])
    assert abs(rho) < 0.2


def test_partial_rho_keeps_independent_signal():
    rng = np.random.default_rng(2)
    driver = rng.normal(0, 1, 200)
    x = driver + rng.normal(0, 0.3, 200)
    y = x + rng.normal(0, 0.3, 200)
    rho = audit.partial_rho(pd.Series(x), pd.Series(y), [pd.Series(driver)])
    assert rho > 0.4


def test_summarise_reports_the_strongest_per_group():
    rows = [
        {"metric": "a", "partial_rho": 0.40, "value_rho": 0.9, "signal_verdict": "stable_signal", "group": "stable_both"},
        {"metric": "b", "partial_rho": -0.35, "value_rho": 0.9, "signal_verdict": "stable_signal", "group": "stable_both"},
        {"metric": "c", "partial_rho": 0.10, "value_rho": 0.3, "signal_verdict": "signal_flips", "group": "unstable_both"},
    ]
    summary = audit.summarise(rows)
    assert summary["stable_both"]["count"] == 2
    assert summary["stable_both"]["strongest"] == "a"
    assert summary["unstable_both"]["strongest"] == "c"
    assert summary["stable_both"]["above_0_30"] == 2


def test_verdict_flags_an_unstable_metric_near_the_top():
    rows = [
        {"metric": f"m{i}", "partial_rho": 0.5 - i * 0.01, "group": "stable_both"} for i in range(9)
    ]
    rows.append({"metric": "bad", "partial_rho": 0.40, "group": "unstable_both"})
    assert "打折" in audit.verdict(rows, {})


def test_verdict_accepts_a_clean_top_five():
    rows = [
        {"metric": f"m{i}", "partial_rho": 0.5 - i * 0.01, "group": "stable_both"} for i in range(10)
    ]
    assert audit.verdict(rows, {}) == "最强的关联来自值与信号都稳定的指标"


def test_verdict_allows_signal_only_metrics_but_says_so():
    rows = [
        {"metric": f"m{i}", "partial_rho": 0.5 - i * 0.01, "group": "stable_both"} for i in range(3)
    ]
    rows += [
        {"metric": f"s{i}", "partial_rho": 0.4 - i * 0.01, "group": "stable_signal_only"} for i in range(7)
    ]
    message = audit.verdict(rows, {})
    assert "可复现" in message and "信号稳定" in message


def test_scores_skips_metadata_columns_and_ranks_by_absolute_rho():
    frame = _frame()
    rows = audit.scores(frame, _seeds())
    names = [row["metric"] for row in rows]
    assert "arch_id" not in names and "target" not in names
    assert names[0] == "signal_metric"
    assert rows[0]["group"] == "stable_both"


def test_scores_flips_lower_metrics():
    frame = _frame()
    frame["feature_delta"] = -frame["target"]  # lower is better, so it rises as target falls
    rows = audit.scores(frame, {})
    row = next(r for r in rows if r["metric"] == "feature_delta")
    assert row["partial_rho"] == pytest.approx(1.0, abs=0.02)


def test_load_wide_rejects_a_table_without_a_target(tmp_path):
    path = tmp_path / "wide.csv"
    pd.DataFrame({"arch_id": ["0"], "arch_string": ["x"], "clean": [0.5]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        audit.load_wide(path)


def test_load_seed_report_reads_value_rho_and_verdict(tmp_path):
    path = tmp_path / "seed.json"
    payload = {
        "metrics": [
            {"metric": "a", "value": {"mean_rho": 0.91}, "signal_verdict": "stable_signal"},
            {"metric": "b", "value": {"mean_rho": 0.12}, "signal_verdict": "signal_flips"},
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    table = audit.load_seed_report(path)
    assert table["a"]["value_rho"] == pytest.approx(0.91)
    assert table["b"]["signal_verdict"] == "signal_flips"


def test_render_markdown_has_the_sections_a_reader_needs():
    frame = _frame()
    seeds = _seeds()
    rows = audit.scores(frame, seeds)
    report = {
        "schema_version": 1,
        "wide": {"path": "wide.csv", "sha256": "0" * 64, "n": len(frame)},
        "seed_report": {"path": "seed.json", "sha256": "1" * 64},
        "value_threshold": 0.8,
        "controls": ["clean", "params"],
        "verdict": audit.verdict(rows, {}),
        "groups": audit.summarise(rows),
        "metrics": rows,
    }
    text = audit.render_markdown(report)
    assert "## 结论" in text
    assert "## 分组统计" in text
    assert "## 这份报告不能说明什么" in text
    assert "已训练模型" in text
