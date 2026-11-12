"""Run every audit in one command and write a single summary.

The repo now has ten independent audit scripts, each with its own arguments.  A
handover that lists ten commands in a doc is a handover where someone eventually
runs six of them, or runs one with the wrong CSV.  This suite fixes the argument
wiring in code, runs the audits in dependency order, and writes one table that
says which checks passed, which failed, and which could not run because the large
inputs live outside the repository.

Dependency order matters: the redundancy matrix feeds the Phi incremental test,
so redundancy must run first.  Steps that need the 47,600-row table are skipped
when that file is absent, and the summary says so explicitly instead of silently
reporting a shorter run.

Usage:

    python tools/run_audit_suite.py
    python tools/run_audit_suite.py --rows-root D:/Project/.../robust_metric/outputs
    python tools/run_audit_suite.py --only doc_claims,doc_links

Exit code 1 means at least one step returned an exit code outside its expected
set.  `signed_capacity` deliberately returns 1 when it concludes the two metrics
differ, so 1 is in its expected set; other steps expect 0 only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "audits"


@dataclass(frozen=True)
class Inputs:
    """Every large input the audits need, resolved once."""

    table: Path
    rows: Path
    repair: Path
    ranking: Path
    redundancy: Path

    def missing_for(self, needs: tuple[str, ...]) -> list[str]:
        missing = []
        for name in needs:
            path = getattr(self, name)
            if path is None or not Path(path).is_file():
                missing.append(name)
        return missing


@dataclass(frozen=True)
class Step:
    name: str
    script: str
    argv: tuple
    needs: tuple[str, ...]
    why: str
    expect_exit: tuple[int, ...] = (0,)


def _p(value: str) -> str:
    return str(value)


def build_steps(inputs: Inputs) -> list[Step]:
    table, rows, repair, ranking, redundancy = (
        inputs.table,
        inputs.rows,
        inputs.repair,
        inputs.ranking,
        inputs.redundancy,
    )
    return [
        Step(
            name="doc_claims",
            script="tools/verify_doc_claims.py",
            argv=(
                "--table", _p(table),
                "--json", _p(OUT / "doc-claims.json"),
                "--report", _p(OUT / "doc-claims.md"),
            ),
            needs=("table",),
            why="文档里引用的每个数字都能在发布表里对上",
        ),
        Step(
            name="doc_links",
            script="tools/check_doc_links.py",
            argv=(
                "--json", _p(OUT / "doc-links.json"),
                "--report", _p(OUT / "doc-links.md"),
                "--strict",
            ),
            needs=(),
            why="文档里引用的每个路径都存在",
        ),
        Step(
            name="metric_index",
            script="tools/build_metric_index.py",
            argv=(
                "--table", _p(table),
                "--rows", _p(rows),
                "--markdown", _p(ROOT / "docs" / "theory" / "METRIC_INDEX.md"),
                "--json", _p(OUT / "metric-index-20261112.json"),
            ),
            needs=("table",),
            why="86 个指标的注册、资格、名次、覆盖能拼成一张表",
        ),
        Step(
            name="setting_weights",
            script="tools/analyze_setting_weights.py",
            argv=(
                "--table", _p(table),
                "--json", _p(OUT / "setting-weights-20261112.json"),
                "--report", _p(OUT / "setting-weights-20261112.md"),
            ),
            needs=("table",),
            why="八设置平均里每个设置各占多少分量",
        ),
        Step(
            name="implementation",
            script="tools/audit_metric_implementation.py",
            argv=(
                "--json", _p(OUT / "metric-implementation-20261112.json"),
                "--report", _p(OUT / "metric-implementation-20261112.md"),
            ),
            needs=(),
            why="每个注册指标都有代码真的把它算出来",
            # registry_only=1 is the known adv-excess gap, recorded in D-024 and
            # in PROJECT_STATUS.  The tool exits 1 to force a human to look at it,
            # so 1 is expected here; the summary still prints the count.
            expect_exit=(0, 1),
        ),
        Step(
            name="invariants",
            script="tools/audit_metric_invariants.py",
            argv=(
                "--json", _p(OUT / "metric-invariants-20261112.json"),
                "--report", _p(OUT / "metric-invariants-20261112.md"),
            ),
            needs=(),
            why="零扰动为 0、处处有限、随扰动单调",
        ),
        Step(
            name="strict_ranking",
            script="tools/verify_strict_ranking.py",
            argv=(
                "--rows", _p(rows),
                "--repair", _p(repair),
                "--ranking", _p(ranking),
                "--output", _p(OUT / "strict-ranking-recompute-20261112.json"),
                "--report", _p(OUT / "strict-ranking-recompute-20261112.md"),
            ),
            needs=("rows", "repair", "ranking"),
            why="83 个名次能从原始行离线复算出来，误差为零",
        ),
        Step(
            name="redundancy",
            script="scripts/analyze_metric_redundancy.py",
            argv=(
                "--rows", _p(rows),
                "--json", _p(OUT / "metric-redundancy-20261112.json"),
                "--report", _p(OUT / "metric-redundancy-20261112.md"),
            ),
            needs=("rows",),
            why="85 个指标实际上聚成几个独立信号",
        ),
        Step(
            name="bootstrap_rb40",
            script="tools/bootstrap_rb40.py",
            argv=(
                "--rows", _p(rows),
                "--repair", _p(repair),
                "--table", _p(table),
                "--json", _p(OUT / "rb40-bootstrap-20261112.json"),
                "--report", _p(OUT / "rb40-bootstrap-20261112.md"),
            ),
            needs=("rows", "repair", "table"),
            why="RB40 上每个指标的相关性有没有 95% 区间",
        ),
        Step(
            name="phi_incremental",
            script="scripts/audit_phi_incremental.py",
            argv=(
                "--rows", _p(rows),
                "--repair", _p(repair),
                "--redundancy", _p(redundancy),
                "--json", _p(OUT / "phi-incremental-20261112.json"),
                "--report", _p(OUT / "phi-incremental-20261112.md"),
            ),
            needs=("rows", "repair", "redundancy"),
            why="Φ 相对 35 个漂移指标是否还有增量信息",
        ),
        Step(
            name="signed_capacity",
            script="tools/audit_signed_capacity.py",
            argv=(
                "--rows", _p(rows),
                "--repair", _p(repair),
                "--json", _p(OUT / "signed-capacity-20261112.json"),
                "--report", _p(OUT / "signed-capacity-20261112.md"),
            ),
            needs=("rows", "repair"),
            why="有符号与绝对 capacity 漂移是不是同一个指标",
            expect_exit=(0, 1),
        ),
    ]


@dataclass
class Result:
    name: str
    why: str
    status: str
    exit_code: int | None = None
    seconds: float = 0.0
    tail: str = ""
    missing: list[str] = field(default_factory=list)


def run_step(step: Step, inputs: Inputs) -> Result:
    missing = inputs.missing_for(step.needs)
    if missing:
        return Result(
            name=step.name,
            why=step.why,
            status="skipped",
            missing=missing,
            tail="缺少输入：" + "、".join(missing),
        )
    command = [sys.executable, step.script, *step.argv]
    started = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    seconds = time.perf_counter() - started
    output = (proc.stdout or "").strip().splitlines()
    tail = output[-1] if output else (proc.stderr or "").strip().splitlines()[-1:] or [""]
    if isinstance(tail, list):
        tail = tail[0] if tail else ""
    status = "ok" if proc.returncode in step.expect_exit else "failed"
    return Result(
        name=step.name,
        why=step.why,
        status=status,
        exit_code=proc.returncode,
        seconds=seconds,
        tail=tail,
    )


def render_markdown(results: list[Result], inputs: Inputs) -> str:
    counts = {"ok": 0, "failed": 0, "skipped": 0}
    for row in results:
        counts[row.status] += 1
    lines = [
        "# 交付前自检汇总",
        "",
        f"- 通过 **{counts['ok']}** / 失败 **{counts['failed']}** / 跳过 **{counts['skipped']}**",
        f"- 发布表：`{inputs.table}`",
        f"- 原始行：`{inputs.rows}`"
        + ("" if Path(inputs.rows).is_file() else "（不存在，依赖它的步骤被跳过）"),
        "",
        "| 步骤 | 状态 | 退出码 | 秒 | 这一步在查什么 | 最后一行输出 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in results:
        code = "—" if row.exit_code is None else str(row.exit_code)
        seconds = "—" if row.status == "skipped" else f"{row.seconds:.1f}"
        tail = row.tail.replace("|", "\\|")[:120]
        lines.append(
            f"| `{row.name}` | {row.status} | {code} | {seconds} | {row.why} | {tail} |"
        )
    if counts["skipped"]:
        lines += ["", "## 被跳过的步骤", ""]
        for row in results:
            if row.status == "skipped":
                lines.append(f"- `{row.name}`：{row.tail}")
    lines += [
        "",
        "## 怎么读",
        "",
        "- `skipped` 不是通过。它表示这台机器上没有那份大输入（通常因为原始行按策略不入库），",
        "  结论只能引用上次在完整数据上的运行结果，不能当成刚跑过。",
        "- `signed_capacity` 的退出码 1 是**结论**（两者不是同一指标），不是崩溃，",
        "  因此它的期望退出码是 0 或 1；其他步骤只接受 0。",
        "- 这个汇总不重算任何数字，它只负责把十个审计的**参数接线**固定下来并跑一遍。",
        "",
    ]
    return "\n".join(lines)


def resolve_rows_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    import os

    from_env = os.environ.get("ROBUST_METRIC_OUTPUTS")
    if from_env:
        return Path(from_env)
    # Fall back to the main working tree: `git worktree list --porcelain` prints
    # it first, and the large inputs live only there.
    try:
        listing = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).stdout
        for line in listing.splitlines():
            if line.startswith("worktree "):
                return Path(line[len("worktree ") :].strip()) / "outputs"
    except OSError:
        pass
    return ROOT / "outputs"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows-root", help="directory holding the large outputs tree")
    parser.add_argument("--table", type=Path, default=None)
    parser.add_argument("--only", help="comma separated step names")
    parser.add_argument("--json", type=Path, default=OUT / "audit-suite.json")
    parser.add_argument("--report", type=Path, default=OUT / "audit-suite.md")
    args = parser.parse_args(argv)

    release = "strict_adv_excess_release_20261024T110422Z"
    repaired = "strict_all_metrics_table_20261024_trnas_repaired"
    rows_root = resolve_rows_root(args.rows_root)
    table = args.table or (
        ROOT / "outputs" / release / "RobustMetric_严格全部指标总表_86项_三位小数.csv"
    )
    inputs = Inputs(
        table=table,
        rows=rows_root / "strict_all_metrics_table_20261024_trnas_repaired" / "sources" / "metric_values_original_strict.csv",
        repair=rows_root / "strict_all_metrics_table_20261024_trnas_repaired" / "sources" / "repair_rows.csv",
        ranking=ROOT / "outputs" / release / "strict_ranking.csv",
        redundancy=OUT / "metric-redundancy-20261112.json",
    )

    steps = build_steps(inputs)
    if args.only:
        wanted = {name.strip() for name in args.only.split(",") if name.strip()}
        unknown = wanted - {step.name for step in steps}
        if unknown:
            parser.error(f"unknown step(s): {', '.join(sorted(unknown))}")
        steps = [step for step in steps if step.name in wanted]

    results = []
    for step in steps:
        print(f"[{step.name}] ...", flush=True)
        row = run_step(step, inputs)
        print(f"[{step.name}] {row.status} ({row.exit_code}) {row.tail[:90]}", flush=True)
        results.append(row)

    payload = {
        "schema_version": 1,
        "rows_root": str(rows_root),
        "table": str(inputs.table),
        "counts": {
            "ok": sum(1 for r in results if r.status == "ok"),
            "failed": sum(1 for r in results if r.status == "failed"),
            "skipped": sum(1 for r in results if r.status == "skipped"),
        },
        "steps": [
            {
                "name": r.name,
                "status": r.status,
                "exit_code": r.exit_code,
                "seconds": round(r.seconds, 2),
                "missing": r.missing,
                "tail": r.tail,
                "why": r.why,
            }
            for r in results
        ],
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_markdown(results, inputs), encoding="utf-8")

    print(
        f"ok={payload['counts']['ok']} failed={payload['counts']['failed']} "
        f"skipped={payload['counts']['skipped']}"
    )
    return 1 if payload["counts"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
