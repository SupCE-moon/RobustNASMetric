"""Repository content policy for the RobustNASMetric project.

The policy decides whether a file may enter Git.  It is deliberately small
and deterministic so that both the migration tooling and the test suite use
exactly the same rule.

Rationale: the server keeps datasets, model weights, checkpoints and raw
large outputs; Git keeps source, tests, configurations, documentation and
small result tables.  A file that is merely *large* is a warning, not a
rejection, because a few summary tables are legitimately a few megabytes.
"""

from __future__ import annotations

from pathlib import Path

#: Files at or above this size are reported but may still be committed.
WARN_BYTES = 10 * 1024 * 1024
#: Files above this size are rejected unless the user approves an exception.
REJECT_BYTES = 50 * 1024 * 1024

#: Binary formats that must never enter Git, even when small.
PROHIBITED_SUFFIXES = frozenset(
    {
        ".pt",
        ".pth",
        ".ckpt",
        ".safetensors",
        ".onnx",
        ".pb",
        ".h5",
        ".hdf5",
        ".pkl",
        ".pickle",
        ".joblib",
        ".npy",
        ".npz",
        ".mat",
        ".zip",
        ".7z",
        ".rar",
        ".tar",
        ".gz",
        ".tgz",
        ".pdf",
        ".pem",
        ".key",
        ".p12",
        ".pfx",
    }
)

#: Directory names that always hold machine-local or heavyweight material.
PROHIBITED_DIR_NAMES = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "checkpoints",
        "checkpoint",
        "secrets",
    }
)

#: Secret-bearing file names.
SECRET_NAMES = frozenset({".env", "id_rsa", "id_ed25519", ".netrc"})

#: Substrings that mark a file as credential material.
SECRET_MARKERS = ("secret", "credential", "password", "passwd", "api_key", "apikey")


def _parts_lower(path: Path) -> tuple[str, ...]:
    return tuple(part.lower() for part in path.parts)


def classify_path(path: str | Path, size_bytes: int) -> str:
    """Return ``"reject"``, ``"warn"`` or ``"allow"`` for one candidate file.

    The decision only depends on the path and the size, never on the file
    content, so callers can classify files without reading them.
    """

    p = Path(path)
    name = p.name.lower()
    parts = _parts_lower(p)

    if name in SECRET_NAMES or name.startswith(".env."):
        return "reject"
    if any(marker in name for marker in SECRET_MARKERS):
        return "reject"
    if any(part in PROHIBITED_DIR_NAMES for part in parts[:-1]):
        return "reject"
    if p.suffix.lower() in PROHIBITED_SUFFIXES:
        return "reject"
    if size_bytes > REJECT_BYTES:
        return "reject"
    if size_bytes >= WARN_BYTES:
        return "warn"
    return "allow"


def iter_violations(paths: list[tuple[str, int]]) -> list[tuple[str, str]]:
    """Return ``(path, reason)`` for every candidate classified ``reject``.

    ``paths`` is a list of ``(path, size_bytes)`` pairs, e.g. read from
    ``git ls-files -z`` plus ``os.stat``.  The helper exists so that CI and
    the migration script share one implementation.
    """

    violations: list[tuple[str, str]] = []
    for raw_path, size in paths:
        decision = classify_path(raw_path, size)
        if decision == "reject":
            violations.append((raw_path, "prohibited path or size"))
    return violations
