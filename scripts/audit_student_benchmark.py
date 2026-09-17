"""Create auditable, non-verifying snapshots of a student benchmark bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

AUDIT_ARTIFACTS = {"source_snapshot.json", "manifest_sha256.csv", "file_inventory.csv"}

def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _files_under(root: Path, excluded: Path | None = None) -> list[Path]:
    result = []
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        if excluded == root and path.parent == root and path.name in AUDIT_ARTIFACTS:
            continue
        resolved = path.resolve()
        if excluded is not None and excluded != root:
            try:
                excluded.relative_to(root)
            except ValueError:
                excluded_inside = False
            else:
                excluded_inside = True
            if excluded_inside:
                try:
                    resolved.relative_to(excluded)
                except ValueError:
                    pass
                else:
                    continue
        result.append(path)
    return sorted(result, key=lambda p: p.relative_to(root).as_posix())


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _status(root: Path) -> tuple[object, str, str | None, bytes | None]:
    path = root / "TRANSFER_STATUS.json"
    if path.is_symlink():
        raise ValueError("TRANSFER_STATUS.json must not be a symlink")
    if not path.exists():
        return None, "MISSING", None, None
    try:
        raw_bytes = path.read_bytes()
        raw_text = raw_bytes.decode("utf-8")
        value = json.loads(raw_text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid TRANSFER_STATUS.json: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("invalid TRANSFER_STATUS.json: expected a JSON object")
    return value, "COMPLETE" if value.get("status") == "COMPLETE" else "NON_COMPLETE", raw_text, raw_bytes


def build_snapshot(root: str | Path, *, output_dir: str | Path | None = None) -> dict[str, object]:
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise ValueError(f"benchmark root is not a directory: {root}")
    excluded = Path(output_dir).resolve() if output_dir is not None else None
    raw_status, status_state, raw_status_text, raw_status_bytes = _status(root_path)
    files = _files_under(root_path, excluded)
    manifest_files = [p for p in files if p.is_relative_to(root_path / "manifests")]
    manifest_rows = [(_relative(root_path / "manifests", p), p.stat().st_size, sha256_file(p)) for p in manifest_files]
    manifest_digest = hashlib.sha256("".join(f"{name}\0{size}\0{digest}\n" for name, size, digest in manifest_rows).encode()).hexdigest()
    total_bytes = sum(p.stat().st_size for p in files)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "root": str(root_path),
        "raw_status": raw_status,
        "raw_status_text": raw_status_text,
        "raw_status_sha256": None if raw_status_bytes is None else hashlib.sha256(raw_status_bytes).hexdigest(),
        "status_state": status_state,
        "manifest_digest": manifest_digest,
        "file_count": len(files),
        "total_bytes": total_bytes,
        "audit_result": "RECORDED_NOT_VERIFIED",
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit student benchmark materials without verifying model results.")
    parser.add_argument("--root", required=True, type=Path, help="student benchmark bundle root")
    parser.add_argument("--output-dir", required=True, type=Path, help="directory for audit evidence files")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    snapshot = build_snapshot(root, output_dir=output)
    files = _files_under(root, output)
    manifests = [p for p in files if p.is_relative_to(root / "manifests")]
    manifest_rows = [{"relative_path": _relative(root / "manifests", p), "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in manifests]
    inventory_rows = [{"relative_path": _relative(root, p), "bytes": p.stat().st_size} for p in files]
    targets = [("source_snapshot.json", lambda p: _write_json(p, snapshot)),
               ("manifest_sha256.csv", lambda p: _write_csv(p, ["relative_path", "bytes", "sha256"], manifest_rows)),
               ("file_inventory.csv", lambda p: _write_csv(p, ["relative_path", "bytes"], inventory_rows))]
    temps: list[Path] = []
    backups: dict[Path, Path] = {}
    replaced: list[Path] = []
    backup_temps: list[Path] = []
    try:
        for name, writer in targets:
            temp = output / (name + ".tmp")
            temps.append(temp)
            writer(temp)
        for name, _ in targets:
            target = output / name
            if target.exists():
                backup = output / (name + ".bak.tmp")
                backup_temps.append(backup)
                backup.write_bytes(target.read_bytes())
                backup_final = output / (name + ".bak")
                backup.replace(backup_final)
                backups[target] = backup_final
        for name, _ in targets:
            target = output / name
            (output / (name + ".tmp")).replace(target)
            replaced.append(target)
    except Exception:
        for temp in temps:
            if temp.exists():
                temp.unlink()
        for temp in backup_temps:
            if temp.exists():
                temp.unlink()
        for target in replaced:
            backup = backups.get(target)
            if backup is None:
                if target.exists():
                    target.unlink()
            elif backup.exists():
                backup.replace(target)
        for backup in backups.values():
            if backup.exists():
                backup.unlink()
        raise
    for backup in backups.values():
        if backup.exists():
            backup.unlink()
    return 0


if __name__ == "__main__":
    main()
