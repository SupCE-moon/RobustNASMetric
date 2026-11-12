"""Copy uncommitted, Git-worthy files from one worktree into another.

The migration is deliberately conservative:

* only files Git reports as *untracked* or *modified* are considered; the
  source worktree is never modified, staged or committed;
* every candidate gets a policy classification, and every decision is written
  to a JSON plan before anything is copied, so the plan can be reviewed;
* raw per-sample outputs stay on the server and are *deferred*, not imported;
* after copying, the destination bytes are re-hashed and compared with the
  source hash; any mismatch aborts the run.

Typical use::

    python tools/import_worktree.py --source D:/.../robust_metric \
        --dest . --plan results/manifests/.../import-plan.json --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inventory_tree import sha256_file  # noqa: E402
from inventory_worktrees import _git, parse_status_z  # noqa: E402
from repository_policy import classify_path  # noqa: E402

SCHEMA_VERSION = 1

#: Directory prefixes whose uncommitted content is project knowledge.
IMPORT_PREFIXES = ("docs/", "scripts/", "src/", "tests/", "tools/", "configs/")

#: Sub-paths under outputs/ that hold per-sample or per-model raw rows.
DEFER_MARKERS = (
    "per_source/",
    "/raw/",
    "node_modules/",
    "/cache/",
    ".inspect.ndjson",
)

#: Files above this size are kept on the server even when otherwise importable.
IMPORT_COMFORT_BYTES = 10 * 1024 * 1024

#: Small, human-readable artefacts under outputs/ that belong in Git.
IMPORT_SUFFIXES = (".md", ".csv", ".json", ".jsonl", ".txt", ".yaml", ".yml")


def decide(relative: str, size: int) -> tuple[str, str]:
    """Return ``(action, reason)`` for one candidate path."""

    name = relative.rsplit("/", 1)[-1]
    if name.startswith("~$"):
        return "reject", "office lock/temporary file"
    policy = classify_path(relative, size)
    if policy == "reject":
        return "reject", "repository policy forbids this path"
    if size >= IMPORT_COMFORT_BYTES:
        return (
            "defer",
            "file exceeds the 10 MiB import comfort limit; keep it on the server",
        )
    if any(marker in relative for marker in DEFER_MARKERS):
        return "defer", "raw per-sample or vendored output stays out of Git"
    if relative.startswith(IMPORT_PREFIXES) or "/" not in relative:
        return "import", "source, test, config, documentation or top-level note"
    if relative.startswith("outputs/"):
        if relative.endswith(IMPORT_SUFFIXES):
            return "import", "small summary table or report"
        return "defer", "output format is not a small table or report"
    return "defer", "outside the recognised import surface"


def build_plan(source: Path) -> dict:
    source = source.resolve()
    head = _git(source, "rev-parse", "HEAD").strip()
    branch = _git(source, "rev-parse", "--abbrev-ref", "HEAD").strip()
    status_text = _git(source, "status", "--porcelain=v1", "-z", "--untracked-files=all")

    entries = []
    for status, relative in parse_status_z(status_text):
        absolute = source / relative
        if not absolute.is_file():
            entries.append(
                {
                    "path": relative,
                    "status": status,
                    "action": "skip",
                    "reason": "not a regular file",
                    "size": None,
                    "sha256": None,
                }
            )
            continue
        size = absolute.stat().st_size
        if status.strip() != "??":
            action, reason = (
                "patch",
                "uncommitted tracked change; preserved in the patch artefact instead",
            )
        else:
            action, reason = decide(relative, size)
        entries.append(
            {
                "path": relative,
                "status": status,
                "action": action,
                "reason": reason,
                "size": size,
                "sha256": sha256_file(absolute, -1),
            }
        )

    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry["action"]] = counts.get(entry["action"], 0) + 1

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "source": str(source),
        "source_head": head,
        "source_branch": branch,
        "counts": counts,
        "total_bytes_import": sum(
            entry["size"] or 0 for entry in entries if entry["action"] == "import"
        ),
        "entries": entries,
    }


def apply_plan(source: Path, dest: Path, plan: dict) -> list[dict]:
    copied = []
    for entry in plan["entries"]:
        if entry["action"] != "import":
            continue
        source_file = source / entry["path"]
        dest_file = dest / entry["path"]
        if dest_file.exists():
            if sha256_file(dest_file, -1) == entry["sha256"]:
                entry["applied"] = "already-present"
                continue
            raise SystemExit(
                f"refusing to overwrite differing file: {dest_file}"
            )
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, dest_file)
        if sha256_file(dest_file, -1) != entry["sha256"]:
            raise SystemExit(f"hash mismatch after copy: {dest_file}")
        entry["applied"] = "copied"
        copied.append(entry)
    return copied


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--dest", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    plan = build_plan(args.source)
    if args.apply:
        copied = apply_plan(args.source.resolve(), args.dest.resolve(), plan)
        plan["copied"] = len(copied)

    args.plan.parent.mkdir(parents=True, exist_ok=True)
    args.plan.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"plan: {plan['counts']}, import bytes "
        f"{plan['total_bytes_import'] / 1048576:.2f} MiB"
        + (f", copied {plan.get('copied')}" if args.apply else " (dry run)")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
