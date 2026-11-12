"""Flatten immutable per-setting attempts only after the full 560-key gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.activation_survival_manifest import (  # noqa: E402
    SETTING_COUNTS, _canonical, require_project_path, sha256_file,
)
from scripts.verify_activation_survival import _strict_json, verify_records  # noqa: E402


def inspect_attempt_files(attempts_root: Path) -> list[tuple[Path, dict]]:
    if not attempts_root.is_dir():
        raise ValueError("attempt root missing")
    actual = {path.name for path in attempts_root.iterdir() if path.is_dir()}
    if actual != set(SETTING_COUNTS):
        raise ValueError("setting directories differ from exact eight-setting grid")
    if any(path.is_file() for path in attempts_root.iterdir()):
        raise ValueError("unexpected loose file in per-setting attempt root")
    files = []
    filenames = set()
    for setting in SETTING_COUNTS:
        directory = attempts_root / setting
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix != ".json":
                raise ValueError(f"unexpected attempt entry: {path}")
            if path.name in filenames:
                raise ValueError(f"duplicate filename across settings: {path.name}")
            filenames.add(path.name)
            record = _strict_json(path)
            if record.get("key", [None])[0] != setting:
                raise ValueError(f"attempt folder/key mismatch: {path}")
            files.append((path, record))
    return files


def collect(manifest_path: Path, attempts_root: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"flat attempt directory already exists: {output_dir}")
    manifest = _strict_json(manifest_path)
    if hashlib.sha256(_canonical(manifest["frozen_payload"])).hexdigest() != manifest["frozen_sha256"]:
        raise ValueError("frozen manifest digest mismatch")
    files = inspect_attempt_files(attempts_root)
    result = verify_records(manifest, [record for _, record in files])
    output_dir.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for source, _ in files:
        target = output_dir / source.name
        with source.open("rb") as incoming, target.open("xb") as outgoing:
            for chunk in iter(lambda: incoming.read(1024 * 1024), b""):
                outgoing.write(chunk)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        digest = sha256_file(source)
        if sha256_file(target) != digest:
            raise ValueError(f"copied attempt differs: {source}")
        hashes[f"{source.parent.name}/{source.name}"] = digest
    report = {"status": "PASS", "unique_success_keys": result["unique_success_keys"],
              "failed_attempts_retained": result["failed_attempts"],
              "attempt_files": len(files), "manifest_sha256": manifest["frozen_sha256"],
              "original_attempt_sha256": hashes}
    # The downstream verifier globs *.json as attempts, so the sidecar is .meta.
    with (output_dir / "collection_verification.meta").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--attempts-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    manifest, attempts, output = (require_project_path(path, ROOT) for path in
                                  (args.manifest, args.attempts_root, args.output_dir))
    report = collect(manifest, attempts, output)
    print(json.dumps({key: value for key, value in report.items()
                      if key != "original_attempt_sha256"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
