"""Tests for tools/run_audit_suite.py."""

from __future__ import annotations

import sys
from pathlib import Path

from tools import run_audit_suite as suite


def _inputs(tmp_path: Path) -> suite.Inputs:
    return suite.Inputs(
        table=tmp_path / "table.csv",
        rows=tmp_path / "rows.csv",
        repair=tmp_path / "repair.csv",
        ranking=tmp_path / "ranking.csv",
        redundancy=tmp_path / "redundancy.json",
    )


def test_step_names_are_unique_and_run_in_dependency_order(tmp_path):
    steps = suite.build_steps(_inputs(tmp_path))
    names = [step.name for step in steps]
    assert len(names) == len(set(names))
    assert names.index("redundancy") < names.index("phi_incremental")


def test_every_step_points_at_a_script_that_exists():
    steps = suite.build_steps(_inputs(Path(".")))
    for step in steps:
        assert (suite.ROOT / step.script).is_file(), step.script


def test_missing_input_skips_instead_of_failing(tmp_path):
    inputs = _inputs(tmp_path)  # nothing exists on disk
    step = next(s for s in suite.build_steps(inputs) if s.name == "signed_capacity")
    row = suite.run_step(step, inputs)
    assert row.status == "skipped"
    assert row.exit_code is None
    assert set(row.missing) == {"rows", "repair"}


def test_expected_nonzero_exit_counts_as_ok(tmp_path):
    script = tmp_path / "exit_one.py"
    script.write_text("raise SystemExit(1)\n", encoding="utf-8")
    step = suite.Step(
        name="tolerant",
        script=str(script),
        argv=(),
        needs=(),
        why="结论是退出码 1",
        expect_exit=(0, 1),
    )
    row = suite.run_step(step, _inputs(tmp_path))
    assert row.exit_code == 1
    assert row.status == "ok"


def test_unexpected_exit_is_a_failure(tmp_path):
    script = tmp_path / "exit_two.py"
    script.write_text("raise SystemExit(2)\n", encoding="utf-8")
    step = suite.Step(
        name="strict", script=str(script), argv=(), needs=(), why="只接受 0", expect_exit=(0,)
    )
    row = suite.run_step(step, _inputs(tmp_path))
    assert row.status == "failed"


def test_markdown_says_skipped_is_not_a_pass(tmp_path):
    inputs = _inputs(tmp_path)
    results = [
        suite.Result(name="a", why="x", status="ok", exit_code=0, seconds=1.0, tail="fine"),
        suite.Result(name="b", why="y", status="skipped", missing=["rows"], tail="缺少输入：rows"),
    ]
    text = suite.render_markdown(results, inputs)
    assert "skipped` 不是通过" in text
    assert "| `b` | skipped |" in text


def test_explicit_rows_root_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBUST_METRIC_OUTPUTS", str(tmp_path / "from_env"))
    assert suite.resolve_rows_root(str(tmp_path / "explicit")) == tmp_path / "explicit"


def test_environment_rows_root_is_used_when_no_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBUST_METRIC_OUTPUTS", str(tmp_path / "from_env"))
    assert suite.resolve_rows_root(None) == tmp_path / "from_env"


def test_python_executable_is_the_running_interpreter():
    # The suite must not depend on `python` being on PATH: the venv one is what
    # has torch and pandas.
    steps = suite.build_steps(_inputs(Path(".")))
    assert steps and sys.executable

def test_early_step_prefers_the_current_code_e0_table(tmp_path):
    inputs = suite.Inputs(
        table=tmp_path / "table.csv",
        rows=tmp_path / "rows.csv",
        repair=tmp_path / "repair.csv",
        ranking=tmp_path / "ranking.csv",
        redundancy=tmp_path / "redundancy.json",
        robbench=tmp_path / "robbench.json",
        e0_rows=tmp_path / "e0_rows.csv",
    )
    step = next(s for s in suite.build_steps(inputs) if s.name == "early_vs_robbench")
    assert step.argv[step.argv.index("--rows") + 1] == str(tmp_path / "e0_rows.csv")


def test_early_step_falls_back_to_rows_without_the_e0_table(tmp_path):
    inputs = suite.Inputs(
        table=tmp_path / "table.csv",
        rows=tmp_path / "rows.csv",
        repair=tmp_path / "repair.csv",
        ranking=tmp_path / "ranking.csv",
        redundancy=tmp_path / "redundancy.json",
        robbench=tmp_path / "robbench.json",
    )
    step = next(s for s in suite.build_steps(inputs) if s.name == "early_vs_robbench")
    assert step.argv[step.argv.index("--rows") + 1] == str(tmp_path / "rows.csv")