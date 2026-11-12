"""Tests for ``tools/audit_metric_implementation.py``.

The tool exists to answer "does every registered metric have code behind it?".
Its value is entirely in the cases where the answer is no, so most of these
tests build a tiny fake package and check that the classification really
separates "computed elsewhere", "only declared" and "absent".
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import audit_metric_implementation as ami  # noqa: E402


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def fake_package(tmp_path: Path, *, extra: dict[str, str] | None = None) -> Path:
    src = tmp_path / "pkg"
    write(
        src / "metric_registry.py",
        "class MetricSpec:\n"
        "    def __init__(self, name, direction, attribution, reference):\n"
        "        self.name = name\n"
        "\n"
        "def _specs(names, attribution, reference):\n"
        "    return [MetricSpec(n, 'lower', attribution, reference) for n in names]\n"
        "\n"
        "METRIC_SPECS = [\n"
        "    MetricSpec('in_bundle_delta', 'lower', 'x', 'y'),\n"
        "    *_specs(('grouped_delta', 'elsewhere_delta'), 'x', 'y'),\n"
        "    MetricSpec('lonely_delta', 'lower', 'x', 'y'),\n"
        "]\n",
    )
    write(
        src / "metrics.py",
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass\n"
        "class MetricResult:\n"
        "    in_bundle_delta: float\n",
    )
    write(src / "other.py", "def compute():\n    return 'elsewhere_delta'\n")
    for name, text in (extra or {}).items():
        write(src / name, text)
    return src


def test_registered_names_reads_literal_and_grouped_specs(tmp_path: Path) -> None:
    src = fake_package(tmp_path)
    names = ami.registered_names(src / "metric_registry.py")
    assert names == ["elsewhere_delta", "grouped_delta", "in_bundle_delta", "lonely_delta"]


def test_bundle_fields_reads_the_dataclass(tmp_path: Path) -> None:
    src = fake_package(tmp_path)
    assert ami.bundle_fields(src / "metrics.py") == {"in_bundle_delta"}


def test_mentions_does_not_match_a_longer_name(tmp_path: Path) -> None:
    src = fake_package(tmp_path, extra={"helper.py": "x = 'in_bundle_delta_extra'\n"})
    hits = ami.mentions(sorted(src.glob("*.py")), "in_bundle_delta")
    assert "helper.py" not in hits
    assert "metrics.py" in hits


def test_mentions_matches_a_name_inside_a_call(tmp_path: Path) -> None:
    src = fake_package(tmp_path, extra={"helper.py": "value = compute('grouped_delta')\n"})
    assert set(ami.mentions(sorted(src.glob("*.py")), "grouped_delta")) == {"helper.py", "metric_registry.py"}


def test_bundle_status_wins_over_other_files(tmp_path: Path) -> None:
    row = ami.Row(name="in_bundle_delta", in_bundle=True, files=["metrics.py", "metrics.py"])
    assert row.status == "one-pass bundle"


def test_metric_only_mentioned_by_the_registry_is_reported(tmp_path: Path) -> None:
    row = ami.Row(name="lonely_delta", in_bundle=False, files=["metric_registry.py"])
    assert row.status == "registry only"


def test_metric_mentioned_by_another_module_is_not_reported(tmp_path: Path) -> None:
    row = ami.Row(
        name="elsewhere_delta", in_bundle=False, files=["metric_registry.py", "other.py"]
    )
    assert row.status == "other module"
    assert row.other_modules == ["other.py"]


def test_analyse_separates_the_three_outcomes(tmp_path: Path) -> None:
    src = fake_package(tmp_path)
    audit = ami.analyse(dynamic=False, src=src, registry=src / "metric_registry.py")

    assert audit["registered"] == 4
    assert audit["in_bundle"] == 1
    assert audit["registry_only"] == ["grouped_delta", "lonely_delta"]
    assert audit["missing"] == []
    statuses = {row["name"]: row["status"] for row in audit["rows"]}
    assert statuses["elsewhere_delta"] == "other module"
    assert statuses["in_bundle_delta"] == "one-pass bundle"


def test_analyse_reports_a_name_absent_from_every_file(tmp_path: Path) -> None:
    src = fake_package(tmp_path)
    registry = write(
        tmp_path / "outside" / "registry.py",
        "METRIC_SPECS = [MetricSpec('ghost_delta', 'lower', 'x', 'y')]\n",
    )
    audit = ami.analyse(dynamic=False, src=src, registry=registry)
    assert audit["missing"] == ["ghost_delta"]


def test_nan_in_a_metric_with_another_home_is_a_placeholder() -> None:
    assert ami._finite(float("nan")) is False
    row = ami.Row(
        name="capacity_delta", in_bundle=True, files=["metrics.py", "importance.py"],
        emitted_finite=False,
    )
    assert "importance.py" in row.other_modules


def test_render_markdown_flags_registry_only(tmp_path: Path) -> None:
    src = fake_package(tmp_path)
    audit = ami.analyse(dynamic=False, src=src, registry=src / "metric_registry.py")
    report = ami.render_markdown(audit)
    assert "只有声明、没有实现痕迹" in report
    assert "`lonely_delta`" in report
    assert "「只有声明」才是问题" in report


def test_real_registry_has_no_unimplemented_metric() -> None:
    """Guards the released 86-row table: a name nobody computes would be silent.

    ``feature_adv_excess_centered_unit_l2`` is currently the one known
    "registry only" name (its formula lives in ``adv_excess_metric.py`` but no
    code path writes that key).  When that gap is closed, drop the second
    assertion -- the first one must hold either way.
    """
    audit = ami.analyse(dynamic=False)
    assert audit["missing"] == []
    assert audit["registry_only"] == ["feature_adv_excess_centered_unit_l2"]


