"""Compare two directory inventories by content hash.

The question this answers: "what does worktree A still hold that branch B does not?"
Answering it by commit graph is wrong here, because most of the interesting work never
got committed; answering it by mtime is wrong too, because checkout rewrites mtimes.
Only the file content hash is trustworthy.

Usage:
    python tools/compare_inventories.py --target target.json --other a.json b.json \
        --output comparison.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_inventory(path: Path) -> dict[str, dict[str, Any]]:
    """Return {relative path: entry} for one inventory file.

    `inventory_tree.py` writes its file list under "entries"; older hand-written
    inventories used "files". Accept both rather than silently comparing nothing.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    files = payload.get("entries")
    if files is None:
        files = payload.get("files")
    if files is None:
        raise ValueError(f"{path}: not an inventory (no 'entries' or 'files' key)")
    return {entry["path"]: entry for entry in files}


def compare(
    target: dict[str, dict[str, Any]],
    other: dict[str, dict[str, Any]],
    label: str,
) -> dict[str, Any]:
    """Describe what `other` holds relative to `target`, by content hash."""
    only_other = []
    differs = []
    unhashed = []
    for path, entry in other.items():
        if entry.get("classification") == "reject":
            continue
        match = target.get(path)
        if match is None:
            only_other.append({"path": path, "size": entry.get("size"), "sha256": entry.get("sha256")})
        elif entry.get("sha256") is None or match.get("sha256") is None:
            # Above the hash limit: neither side has content evidence, so a size
            # difference is the only defensible signal and "same" cannot be claimed.
            unhashed.append(
                {"path": path, "other_size": entry.get("size"), "target_size": match.get("size")}
            )
        elif match.get("sha256") != entry.get("sha256"):
            differs.append(
                {
                    "path": path,
                    "other_sha256": entry.get("sha256"),
                    "target_sha256": match.get("sha256"),
                    "other_size": entry.get("size"),
                    "target_size": match.get("size"),
                }
            )
    only_target = [
        {"path": path, "size": entry.get("size")}
        for path, entry in target.items()
        if path not in other and entry.get("classification") != "reject"
    ]
    return {
        "label": label,
        "other_file_count": len(other),
        "only_in_other": sorted(only_other, key=lambda item: item["path"]),
        "differs": sorted(differs, key=lambda item: item["path"]),
        "unhashed": sorted(unhashed, key=lambda item: item["path"]),
        "only_in_target_count": len(only_target),
        "only_in_other_bytes": sum(item["size"] or 0 for item in only_other),
    }


def render_markdown(comparisons: list[dict[str, Any]]) -> str:
    lines = [
        "| 工作树 | 独有文件 | 独有体积 | 内容不同 | 超限未哈希 | 目标独有 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in comparisons:
        lines.append(
            f"| {item['label']} | {len(item['only_in_other'])} | "
            f"{item['only_in_other_bytes'] / 1048576:.2f} MiB | "
            f"{len(item['differs'])} | {len(item['unhashed'])} | {item['only_in_target_count']} |"
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True, help="inventory to compare against")
    parser.add_argument("--other", type=Path, nargs="+", required=True, help="inventories to describe")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    target = load_inventory(args.target)
    comparisons = [
        compare(target, load_inventory(path), path.stem)
        for path in args.other
    ]
    payload = {"target": str(args.target), "comparisons": comparisons}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.markdown is not None:
        args.markdown.write_text(render_markdown(comparisons), encoding="utf-8")
    for item in comparisons:
        print(
            f"{item['label']}: only_in_other={len(item['only_in_other'])} "
            f"({item['only_in_other_bytes'] / 1048576:.2f} MiB) differs={len(item['differs'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
