"""Tests for tools/check_doc_links.py.

The point of these tests is the *classification*, not the regex: a checker that
reports everything as broken is as useless as one that reports nothing. Each
test pins one bucket so that a future edit cannot silently move a category.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_doc_links", ROOT / "tools" / "check_doc_links.py")
assert spec and spec.loader
cdl = importlib.util.module_from_spec(spec)
sys.modules["check_doc_links"] = cdl
spec.loader.exec_module(cdl)


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "docs" / "experiments").mkdir(parents=True)
    (tmp_path / "docs" / "superpowers" / "plans").mkdir(parents=True)
    (tmp_path / "results" / "audits").mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "tools").mkdir()
    (tmp_path / "configs").mkdir()
    (tmp_path / "README.md").write_text("# repo\n", encoding="utf-8")
    return tmp_path


def test_normalize_rejects_non_repo_targets() -> None:
    assert cdl.normalize("https://example.com/a.csv") is None
    assert cdl.normalize("/workspace/robust_metric_lab_nas/x.py") is None
    assert cdl.normalize("C:/Users/x/a.py") is None
    assert cdl.normalize("docs/has space.md") is None
    assert cdl.normalize("docs/glob_*.md") is None
    assert cdl.normalize("README") is None
    assert cdl.normalize("docs/a.png") is None


def test_normalize_strips_anchor_and_backslashes() -> None:
    assert cdl.normalize("`docs\\experiments\\e35.md#L12`") == "docs/experiments/e35.md"
    assert cdl.normalize("`src/robust_metric_lab/metrics.py`") == "src/robust_metric_lab/metrics.py"


def test_existing_paths_are_not_reported(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    (root / "docs" / "experiments" / "e35.md").write_text("x", encoding="utf-8")
    (root / "docs" / "experiments" / "report.md").write_text(
        "见 `docs/experiments/e35.md`。\n", encoding="utf-8"
    )
    payload = cdl.scan(root)
    assert payload["paths_checked"] == 1
    assert payload["missing_required"] == []


def test_missing_repo_path_is_required(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    (root / "docs" / "experiments" / "report.md").write_text(
        "见 `results/audits/nope.json`。\n", encoding="utf-8"
    )
    payload = cdl.scan(root)
    assert [item["path"] for item in payload["missing_required"]] == ["results/audits/nope.json"]


def test_both_backticks_and_markdown_links_are_scanned(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    (root / "docs" / "experiments" / "report.md").write_text(
        "反引号 `scripts/gone.py` 和链接 [x](tests/also_gone.py)。\n", encoding="utf-8"
    )
    payload = cdl.scan(root)
    assert {item["path"] for item in payload["missing_required"]} == {
        "scripts/gone.py",
        "tests/also_gone.py",
    }


def test_duplicate_mentions_count_once(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    (root / "docs" / "experiments" / "report.md").write_text(
        "`scripts/gone.py` 与 `scripts/gone.py` 又说一遍。\n", encoding="utf-8"
    )
    payload = cdl.scan(root)
    assert payload["paths_checked"] == 1
    assert len(payload["missing_required"]) == 1


def test_plan_documents_are_history_not_breakage(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    (root / "docs" / "superpowers" / "plans" / "plan.md").write_text(
        "将创建 `scripts/run_activation_holdout.py`。\n", encoding="utf-8"
    )
    payload = cdl.scan(root)
    assert payload["missing_required"] == []
    assert [item["path"] for item in payload["missing_planned"]] == ["scripts/run_activation_holdout.py"]


def test_outputs_are_optional(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    (root / "docs" / "experiments" / "report.md").write_text(
        "产物在 `outputs/e25/per_source.csv`。\n", encoding="utf-8"
    )
    payload = cdl.scan(root)
    assert payload["missing_required"] == []
    assert [item["path"] for item in payload["missing_optional"]] == ["outputs/e25/per_source.csv"]


def test_allowlisted_reference_carries_its_reason(tmp_path: Path, monkeypatch) -> None:
    root = _repo(tmp_path)
    (root / "docs" / "experiments" / "audit.md").write_text(
        "服务器文件 `scripts/train_v2.py`。\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        cdl,
        "ALLOWLIST",
        {("docs/experiments/audit.md", "scripts/train_v2.py"): "服务器上的同名新版本"},
    )
    payload = cdl.scan(root)
    assert payload["missing_required"] == []
    assert payload["missing_explained"][0]["reason"] == "服务器上的同名新版本"


def test_own_report_is_not_scanned(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    (root / "results" / "audits" / "doc-links-20261112.md").write_text(
        "| `docs/experiments/e25_k2_k8_k16_report.md` |\n", encoding="utf-8"
    )
    payload = cdl.scan(root)
    assert payload["docs_scanned"] == 1  # only README.md
    assert payload["missing_required"] == []


def test_render_markdown_reports_zero_as_plain_none(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    payload = cdl.scan(root)
    text = cdl.render_markdown(payload)
    assert "**0** 条" in text
    assert "无。" in text


def test_cli_writes_json_and_respects_strict(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    (root / "docs" / "experiments" / "report.md").write_text("`scripts/gone.py`\n", encoding="utf-8")
    out_json = tmp_path / "out.json"
    out_md = tmp_path / "out.md"
    argv = ["check_doc_links.py", "--root", str(root), "--json", str(out_json), "--report", str(out_md), "--strict"]
    sys.argv = argv
    assert cdl.main() == 1
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert out_md.read_text(encoding="utf-8").startswith("# 文档里的路径引用")
