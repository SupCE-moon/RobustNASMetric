from __future__ import annotations

from dataclasses import dataclass, field
import csv
import json
from math import fsum
from pathlib import Path, PureWindowsPath
from typing import Iterable, Mapping

from .metric_registry import METRIC_BY_NAME


@dataclass(frozen=True)
class BenchmarkSetting:
    key: str
    benchmark: str
    checkpoint_epoch: int | None
    target_column: str
    expected_rows: int


BENCHMARK_SETTINGS = (
    BenchmarkSetting("rb40_mature", "rb40", None, "robust_accuracy", 40),
    BenchmarkSetting("nb201_e0", "nb201", 0, "official_final_robust_accuracy", 64),
    BenchmarkSetting("nb201_e5", "nb201", 5, "official_final_robust_accuracy", 64),
    BenchmarkSetting("nb201_e10", "nb201", 10, "official_final_robust_accuracy", 64),
    BenchmarkSetting("nb201_e25", "nb201", 25, "official_final_robust_accuracy", 64),
    BenchmarkSetting("nb201_e50", "nb201", 50, "official_final_robust_accuracy", 64),
    BenchmarkSetting("darts_e0", "darts", 0, "full_pgd20_accuracy", 100),
    BenchmarkSetting("darts_e100", "darts", 100, "full_pgd20_accuracy", 100),
)
SETTING_BY_KEY = {setting.key: setting for setting in BENCHMARK_SETTINGS}

SAMPLE_OFFSET = 0
NUM_SAMPLES = 64
SEED = 20260808
EPSILON = 8 / 255


@dataclass(frozen=True)
class BenchmarkManifestItem:
    """A single teacher-manifest row, resolved relative to a bundle root."""

    benchmark: str
    model_or_arch_id: str
    checkpoint_epoch: int | None
    checkpoint_path: Path
    target_value: float
    metadata: Mapping[str, object] = field(default_factory=dict)


def resolve_under_root(root: str | Path, relative_path: str | Path) -> Path:
    """Resolve an artifact path, rejecting absolute paths and root escapes."""
    root_path = Path(root).resolve()
    supplied = Path(relative_path)
    if supplied.is_absolute() or PureWindowsPath(str(relative_path)).is_absolute() or ".." in supplied.parts:
        raise ValueError(f"path is outside benchmark root: {relative_path}")
    candidate = (root_path / supplied).resolve()
    try:
        candidate.relative_to(root_path)
    except ValueError as exc:
        raise ValueError(f"path is outside benchmark root: {relative_path}") from exc
    return candidate


def _json_field(row: Mapping[str, str], column: str, row_number: int) -> object:
    try:
        return json.loads(row[column])
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON in {column} at manifest row {row_number}") from exc


def _load_items(path: str | Path, root: str | Path | None, *, benchmark: str,
                required: set[str], id_column: str, epoch_column: str | None,
                target_column: str, epochs: set[int | None], needs_genotype: bool) -> list[BenchmarkManifestItem]:
    manifest = Path(path)
    benchmark_root = Path(root) if root is not None else manifest.parent
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = sorted(required - columns)
        if missing:
            raise ValueError(f"missing required columns: {', '.join(missing)}")
        result: list[BenchmarkManifestItem] = []
        seen: set[tuple[str, int | None]] = set()
        epochs_by_id: dict[str, set[int | None]] = {}
        for row_number, row in enumerate(reader, 2):
            identifier = row[id_column].strip()
            epoch = None if epoch_column is None else int(row[epoch_column])
            if epoch not in epochs:
                raise ValueError(f"unsupported checkpoint epoch {epoch} at manifest row {row_number}")
            key = (identifier, epoch)
            if key in seen:
                raise ValueError(f"duplicate model/architecture ID: {identifier} (epoch {epoch})")
            seen.add(key)
            epochs_by_id.setdefault(identifier, set()).add(epoch)
            metadata_value = _json_field(row, "metadata_json", row_number)
            if not isinstance(metadata_value, dict):
                raise ValueError(f"metadata_json must be an object at manifest row {row_number}")
            metadata = dict(metadata_value)
            if needs_genotype:
                genotype = _json_field(row, "genotype_json", row_number)
                if benchmark == "nb201":
                    if not isinstance(genotype, dict):
                        raise ValueError(
                            f"genotype_json arch_string must be an object at manifest row {row_number}"
                        )
                    raw_arch = genotype.get("arch_string")
                    if not isinstance(raw_arch, str) or not raw_arch or raw_arch != raw_arch.strip():
                        raise ValueError(
                            f"genotype_json must contain a non-empty canonical arch_string "
                            f"at manifest row {row_number}"
                        )
                    # Keep the heavy torch-backed NAS helper lazy so CLI --help
                    # remains import-light.
                    from .nasrobbench201 import make_nb201_arch, parse_nb201_arch

                    operations = parse_nb201_arch(raw_arch)
                    if operations is None or make_nb201_arch(operations) != raw_arch:
                        raise ValueError(
                            f"genotype_json contains a noncanonical arch_string "
                            f"at manifest row {row_number}"
                        )
                    existing_arch = metadata.get("arch_string")
                    if existing_arch is not None and existing_arch != raw_arch:
                        raise ValueError(
                            f"metadata_json arch_string conflicts with genotype_json "
                            f"at manifest row {row_number}"
                        )
                    metadata["arch_string"] = raw_arch
                    metadata.pop("genotype", None)
                else:
                    if not isinstance(genotype, dict):
                        raise ValueError(
                            f"genotype_json must be an object at manifest row {row_number}"
                        )
                    metadata["genotype"] = genotype
            result.append(BenchmarkManifestItem(
                benchmark=benchmark,
                model_or_arch_id=identifier,
                checkpoint_epoch=epoch,
                checkpoint_path=resolve_under_root(benchmark_root, row["checkpoint_path"]),
                target_value=float(row[target_column]),
                metadata=metadata,
            ))
        if epoch_column is not None:
            # Compare against the frozen grid explicitly; this also makes a
            # truncated manifest fail before any runner can use it.
            incomplete = {
                identifier: sorted(set(epochs) - set(epochs_by_id[identifier]), key=lambda value: -1 if value is None else value)
                for identifier in epochs_by_id
                if set(epochs) - set(epochs_by_id[identifier])
            }
            if incomplete:
                details = ", ".join(f"{identifier}: {missing}" for identifier, missing in incomplete.items())
                raise ValueError(f"missing epochs for architecture IDs: {details}")
    return result


def load_rb40_items(path: str | Path, root: str | Path | None = None) -> list[BenchmarkManifestItem]:
    return _load_items(path, root, benchmark="rb40",
                       required={"model_name", "checkpoint_path", "robust_accuracy", "metadata_json"},
                       id_column="model_name", epoch_column=None, target_column="robust_accuracy",
                       epochs={None}, needs_genotype=False)


def load_nb201_items(path: str | Path, root: str | Path | None = None) -> list[BenchmarkManifestItem]:
    return _load_items(path, root, benchmark="nb201",
                       required={"arch_id", "epoch", "checkpoint_path", "official_final_robust_accuracy", "genotype_json", "metadata_json"},
                       id_column="arch_id", epoch_column="epoch", target_column="official_final_robust_accuracy",
                       epochs={0, 5, 10, 25, 50}, needs_genotype=True)


def load_darts_items(path: str | Path, root: str | Path | None = None) -> list[BenchmarkManifestItem]:
    return _load_items(path, root, benchmark="darts",
                       required={"arch_id", "epoch", "checkpoint_path", "full_pgd20_accuracy", "genotype_json", "metadata_json"},
                       id_column="arch_id", epoch_column="epoch", target_column="full_pgd20_accuracy",
                       epochs={0, 100}, needs_genotype=True)


@dataclass(frozen=True)
class MetricValueRow:
    """One metric observation under the shared student-benchmark protocol."""

    setting: str
    model_or_arch_id: str
    checkpoint_epoch: int | None
    metric_name: str
    metric_value: float | None
    target_value: float | None
    seed: int
    sample_offset: int
    num_samples: int
    epsilon: float
    direction: str
    status: str = "ok"
    error: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        setting = SETTING_BY_KEY.get(self.setting)
        if setting is None:
            raise ValueError(f"unknown benchmark setting: {self.setting}")
        if self.checkpoint_epoch != setting.checkpoint_epoch:
            raise ValueError(
                f"checkpoint epoch mismatch for {self.setting}: "
                f"expected {setting.checkpoint_epoch}, got {self.checkpoint_epoch}"
            )
        protocol_values = (
            ("seed", self.seed, SEED),
            ("sample_offset", self.sample_offset, SAMPLE_OFFSET),
            ("num_samples", self.num_samples, NUM_SAMPLES),
            ("epsilon", self.epsilon, EPSILON),
        )
        for name, value, expected in protocol_values:
            if value != expected:
                raise ValueError(f"{name} must be frozen at {expected!r}, got {value!r}")
        spec = METRIC_BY_NAME.get(self.metric_name)
        if spec is None:
            raise ValueError(f"unknown metric: {self.metric_name}")
        if self.direction != spec.direction:
            raise ValueError(
                f"metric direction mismatch for {self.metric_name}: "
                f"expected {spec.direction}, got {self.direction}"
            )


def validate_complete_setting_grid(
    rows: Iterable[Mapping[str, object]], expected_counts: Mapping[str, int]
) -> list[str]:
    """Return deterministic diagnostics for unknown, duplicate, and incomplete rows."""

    errors: list[str] = []
    counts: dict[str, int] = {}
    seen: set[tuple[str, object]] = set()
    known = set(expected_counts)
    for row in rows:
        setting = str(row.get("setting", ""))
        identifier = row.get("model_or_arch_id")
        if setting not in known:
            errors.append(f"unknown setting: {setting}")
            continue
        key = (setting, identifier)
        if key in seen:
            errors.append(
                f"duplicate row: setting={setting}, model_or_arch_id={identifier}"
            )
        seen.add(key)
        counts[setting] = counts.get(setting, 0) + 1
    for setting, expected in expected_counts.items():
        actual = counts.get(setting, 0)
        if actual < expected:
            errors.append(
                f"missing rows for setting {setting}: expected {expected}, found {actual}"
            )
    return errors


def all_setting_mean(values: Mapping[str, float]) -> float:
    keys = [setting.key for setting in BENCHMARK_SETTINGS]
    missing = [key for key in keys if key not in values]
    if missing:
        raise ValueError(f"missing settings: {', '.join(missing)}")
    return fsum(float(values[key]) for key in keys) / len(keys)
