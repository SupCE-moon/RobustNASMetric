from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, is_dataclass
import math
import os
from pathlib import Path
import re
import time
from typing import Callable, Iterable, Mapping, Sequence

from .metric_registry import METRIC_BY_NAME
from .student_benchmark import (
    EPSILON,
    NUM_SAMPLES,
    SAMPLE_OFFSET,
    SEED,
    SETTING_BY_KEY,
    BenchmarkManifestItem,
    MetricValueRow,
)


CSV_COLUMNS = (
    "run_id",
    "benchmark",
    "setting",
    "model_or_arch_id",
    "checkpoint_epoch",
    "metric_name",
    "direction",
    "metric_value",
    "target_value",
    "status",
    "error",
    "runtime_sec",
    "seed",
    "sample_offset",
    "num_samples",
    "epsilon",
    "checkpoint_path",
)

UNIQUE_KEY_COLUMNS = (
    "setting",
    "model_or_arch_id",
    "checkpoint_epoch",
    "metric_name",
    "seed",
    "sample_offset",
    "num_samples",
)


@dataclass(frozen=True)
class EvaluationResult:
    written: int
    failed: int
    skipped: int


def validate_run_id(run_id: object) -> str:
    value = str(run_id)
    if (
        not value
        or value != value.strip()
        or value in {".", ".."}
        or Path(value).is_absolute()
        or "/" in value
        or "\\" in value
        or "\n" in value
        or "\r" in value
    ):
        raise ValueError("run-id must be a safe single directory name")
    return value


def _required_text(row: Mapping[str, object], field: str) -> str:
    value = row[field]
    if not isinstance(value, str):
        value = str(value)
    if not value or value != value.strip() or "\n" in value or "\r" in value:
        raise ValueError(f"{field} must be a non-empty canonical identifier")
    return value


def _number(value: object, field: str, *, allow_empty: bool = False) -> float | None:
    if allow_empty and value in {None, ""}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _integer(value: object, field: str, *, allow_empty: bool = False) -> int | None:
    if allow_empty and value in {None, ""}:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if str(value).strip() not in {str(number), f"{number}.0"} and not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return number


def _normalise_row(row: Mapping[str, object]) -> dict[str, object]:
    if set(row) != set(CSV_COLUMNS):
        missing = sorted(set(CSV_COLUMNS) - set(row))
        extra = sorted(set(row) - set(CSV_COLUMNS))
        raise ValueError(f"row schema mismatch: missing={missing}, extra={extra}")

    run_id = validate_run_id(row["run_id"])
    benchmark = _required_text(row, "benchmark")
    setting_name = _required_text(row, "setting")
    identifier = _required_text(row, "model_or_arch_id")
    metric_name = _required_text(row, "metric_name")
    direction = _required_text(row, "direction")
    setting = SETTING_BY_KEY.get(setting_name)
    if setting is None:
        raise ValueError(f"unknown benchmark setting: {setting_name}")
    if benchmark != setting.benchmark:
        raise ValueError(
            f"benchmark mismatch for {setting_name}: expected {setting.benchmark}, got {benchmark}"
        )

    epoch = _integer(row["checkpoint_epoch"], "checkpoint_epoch", allow_empty=True)
    seed = _integer(row["seed"], "seed")
    sample_offset = _integer(row["sample_offset"], "sample_offset")
    num_samples = _integer(row["num_samples"], "num_samples")
    epsilon = _number(row["epsilon"], "epsilon")
    target = _number(row["target_value"], "target_value")
    runtime = _number(row["runtime_sec"], "runtime_sec")
    assert runtime is not None
    if runtime < 0:
        raise ValueError("runtime_sec must be finite and nonnegative")

    status = _required_text(row, "status")
    if status not in {"ok", "failed"}:
        raise ValueError("status must be 'ok' or 'failed'")
    metric_value = _number(row["metric_value"], "metric_value", allow_empty=status == "failed")
    if status == "ok" and metric_value is None:
        raise ValueError("ok row metric_value must be finite")

    error_value = row["error"]
    error = "" if error_value is None else str(error_value)
    if "\n" in error or "\r" in error:
        raise ValueError("error must be a single-line safe message")
    if status == "failed" and not error:
        raise ValueError("failed row must include an error")

    checkpoint_path = _required_text(row, "checkpoint_path")
    if not Path(checkpoint_path).is_absolute():
        raise ValueError("checkpoint_path must be an absolute manifest path")

    # Reuse the protocol object's frozen setting/epoch/protocol/direction checks.
    MetricValueRow(
        setting=setting_name,
        model_or_arch_id=identifier,
        checkpoint_epoch=epoch,
        metric_name=metric_name,
        metric_value=metric_value,
        target_value=target,
        seed=seed,
        sample_offset=sample_offset,
        num_samples=num_samples,
        epsilon=epsilon,
        direction=direction,
        status=status,
        error=error or None,
    )

    return {
        "run_id": run_id,
        "benchmark": benchmark,
        "setting": setting_name,
        "model_or_arch_id": identifier,
        "checkpoint_epoch": "" if epoch is None else epoch,
        "metric_name": metric_name,
        "direction": direction,
        "metric_value": "" if metric_value is None else metric_value,
        "target_value": target,
        "status": status,
        "error": error,
        "runtime_sec": runtime,
        "seed": seed,
        "sample_offset": sample_offset,
        "num_samples": num_samples,
        "epsilon": epsilon,
        "checkpoint_path": checkpoint_path,
    }


def _row_key(row: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(row[column] for column in UNIQUE_KEY_COLUMNS)


def _read_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(CSV_COLUMNS):
            raise ValueError("existing CSV schema mismatch")
        return [_normalise_row(row) for row in reader]


def append_rows_atomic(output: str | Path, rows: Iterable[Mapping[str, object]]) -> int:
    """Validate and atomically append rows without risking historical bytes."""

    output_path = Path(output)
    new_rows = [_normalise_row(row) for row in rows]
    if not new_rows:
        return 0

    new_keys = [_row_key(row) for row in new_rows]
    if len(new_keys) != len(set(new_keys)):
        raise ValueError("duplicate unique key in append batch")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_path.with_name(output_path.name + ".lock")
    temporary = output_path.with_name(output_path.name + ".tmp")
    lock_fd: int | None = None
    temporary_created = False
    try:
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RuntimeError(f"output lock already exists: {lock_path}") from exc
        os.write(lock_fd, f"pid={os.getpid()} time={time.time():.6f}\n".encode("utf-8"))
        os.fsync(lock_fd)

        old_rows = _read_rows(output_path)
        old_run_ids = {row["run_id"] for row in old_rows}
        new_run_ids = {row["run_id"] for row in new_rows}
        if old_run_ids and old_run_ids != new_run_ids:
            raise ValueError("run_id mismatch with existing CSV")
        old_keys = {_row_key(row) for row in old_rows}
        duplicate = next((key for key in new_keys if key in old_keys), None)
        if duplicate is not None:
            raise ValueError(f"duplicate unique key already exists: {duplicate}")
        if temporary.exists():
            raise RuntimeError(f"temporary output already exists: {temporary}")

        with temporary.open("x", newline="", encoding="utf-8") as handle:
            temporary_created = True
            writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS), extrasaction="raise")
            writer.writeheader()
            writer.writerows([*old_rows, *new_rows])
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
        temporary_created = False
        return len(new_rows)
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        if temporary_created:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        if lock_fd is not None:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass


def _safe_error(exc: Exception) -> str:
    message = " ".join(str(exc).splitlines()).strip()
    message = re.sub(
        r"(?i)\b(password|token|secret|api[_-]?key)\s*[:=]\s*[^\s,;]+",
        lambda match: f"{match.group(1)}=[redacted]",
        message,
    )
    value = f"{type(exc).__name__}: {message}" if message else type(exc).__name__
    return value[:1000]


def _protocol(run_metadata: Mapping[str, object]) -> tuple[str, str, int, int, int, float]:
    run_id = validate_run_id(run_metadata.get("run_id", ""))
    setting_name = str(run_metadata.get("setting", ""))
    if setting_name not in SETTING_BY_KEY:
        raise ValueError(f"unknown benchmark setting: {setting_name}")
    values = (
        ("seed", int(run_metadata.get("seed", SEED)), SEED),
        ("sample_offset", int(run_metadata.get("sample_offset", SAMPLE_OFFSET)), SAMPLE_OFFSET),
        ("num_samples", int(run_metadata.get("num_samples", NUM_SAMPLES)), NUM_SAMPLES),
        ("epsilon", float(run_metadata.get("epsilon", EPSILON)), EPSILON),
    )
    for name, value, expected in values:
        if value != expected:
            raise ValueError(f"{name} must be frozen at {expected!r}, got {value!r}")
    return run_id, setting_name, values[0][1], values[1][1], values[2][1], values[3][1]


def _result_mapping(value: object) -> Mapping[str, object]:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "__dict__"):
        return vars(value)
    raise TypeError(f"metric helper returned unsupported result type: {type(value).__name__}")


class _DefaultMetricComputer:
    def __init__(self, samples: object):
        self.samples = samples
        self._adversarial = None
        self._feature: Mapping[str, object] | None = None
        self._trnas: Mapping[str, object] | None = None
        self._racl: float | None = None

    def _inputs(self):
        if isinstance(self.samples, Mapping):
            return self.samples["inputs"], self.samples["labels"]
        try:
            clean, labels = self.samples  # type: ignore[misc]
        except (TypeError, ValueError) as exc:
            raise TypeError("loader must provide (inputs, labels) or a mapping") from exc
        return clean, labels

    def _adv(self, model):
        if self._adversarial is None:
            from .attacks import make_adversarial

            clean, labels = self._inputs()
            self._adversarial = make_adversarial(
                model, clean, labels, "fgsm", EPSILON, pgd_steps=1
            )
        return self._adversarial

    def __call__(self, metric: str, model: object, _samples: object, _item: BenchmarkManifestItem) -> float:
        if metric in {"racl_arch_lipschitz_log_upper", "racl_fixed_network_log_upper_proxy"}:
            if self._racl is None:
                from .nasrobbench201 import racl_fixed_network_lipschitz_stats

                self._racl = float(racl_fixed_network_lipschitz_stats(model).log_upper)
            return self._racl

        if metric.startswith("trnas_"):
            if self._trnas is None:
                from .trnas import compute_trnas_metrics

                clean, labels = self._inputs()
                result = compute_trnas_metrics(model, clean, self._adv(model), labels)
                self._trnas = {
                    "trnas_activation_psi": result.activation_psi,
                    "trnas_consistency_phi": result.consistency_phi,
                    "trnas_r_score": result.r_score,
                }
            if metric not in self._trnas:
                raise NotImplementedError(f"registered metric has no TRNAS dispatch: {metric}")
            return float(self._trnas[metric])

        if self._feature is None:
            from .metrics import compute_feature_metrics

            clean, labels = self._inputs()
            self._feature = _result_mapping(
                compute_feature_metrics(model, clean, self._adv(model), labels)
            )
        source_name = "input_grad_l2" if metric == "racl_input_gradient_lower" else metric
        if source_name not in self._feature:
            raise NotImplementedError(f"registered metric has no dispatch: {metric}")
        return float(self._feature[source_name])


def _existing_keys(path: Path) -> set[tuple[object, ...]]:
    return {_row_key(row) for row in _read_rows(path)}


def evaluate_items(
    items: Sequence[BenchmarkManifestItem],
    load_model: Callable[[BenchmarkManifestItem], object],
    loader: object,
    metrics: Sequence[str],
    output_path: str | Path,
    run_metadata: Mapping[str, object],
    *,
    compute_metric: Callable[[str, object, object, BenchmarkManifestItem], object] | None = None,
    compute: Callable[[str, object, object, BenchmarkManifestItem], object] | None = None,
    torch_module: object | None = None,
) -> EvaluationResult:
    """Evaluate manifest items in order, persisting one atomic batch per item."""

    if compute_metric is not None and compute is not None:
        raise ValueError("provide only one of compute_metric or compute")
    custom_compute = compute_metric or compute
    requested = list(metrics)
    if not requested:
        raise ValueError("at least one metric is required")
    if len(requested) != len(set(requested)):
        raise ValueError("duplicate requested metric")
    for metric in requested:
        if metric not in METRIC_BY_NAME:
            raise KeyError(f"unregistered metric: {metric}")

    run_id, setting_name, seed, sample_offset, num_samples, epsilon = _protocol(run_metadata)
    setting = SETTING_BY_KEY[setting_name]
    output = Path(output_path)
    existing = _existing_keys(output)
    pending_exists = any(
        (
            setting_name,
            item.model_or_arch_id,
            item.checkpoint_epoch,
            metric,
            seed,
            sample_offset,
            num_samples,
        )
        not in existing
        for item in items
        for metric in requested
    )
    samples = loader() if pending_exists and callable(loader) else loader
    if torch_module is None:
        import torch as torch_module  # type: ignore[no-redef]

    written = failed = skipped = 0
    for item in items:
        if item.benchmark != setting.benchmark:
            raise ValueError(
                f"manifest benchmark mismatch: expected {setting.benchmark}, got {item.benchmark}"
            )
        if item.checkpoint_epoch != setting.checkpoint_epoch:
            raise ValueError(
                f"manifest checkpoint epoch mismatch for {setting_name}: {item.checkpoint_epoch}"
            )
        pending_metrics = []
        for metric in requested:
            key = (
                setting_name,
                item.model_or_arch_id,
                item.checkpoint_epoch,
                metric,
                seed,
                sample_offset,
                num_samples,
            )
            if key in existing:
                skipped += 1
            else:
                pending_metrics.append(metric)
        if not pending_metrics:
            continue

        rows: list[dict[str, object]] = []
        model = None
        default_computer = None
        load_started = time.monotonic()
        try:
            try:
                model = load_model(item)
                model.eval()
            except Exception as exc:
                runtime = max(0.0, time.monotonic() - load_started)
                error = _safe_error(exc)
                for metric in pending_metrics:
                    rows.append(
                        _evaluation_row(
                            run_id, setting_name, item, metric, None, "failed", error,
                            runtime, seed, sample_offset, num_samples, epsilon,
                        )
                    )
                    failed += 1
            else:
                default_computer = _DefaultMetricComputer(samples)
                metric_computer = custom_compute or default_computer
                for metric in pending_metrics:
                    started = time.monotonic()
                    try:
                        value = float(metric_computer(metric, model, samples, item))
                        if not math.isfinite(value):
                            raise ValueError("metric result must be finite")
                    except Exception as exc:
                        rows.append(
                            _evaluation_row(
                                run_id, setting_name, item, metric, None, "failed",
                                _safe_error(exc), max(0.0, time.monotonic() - started),
                                seed, sample_offset, num_samples, epsilon,
                            )
                        )
                        failed += 1
                    else:
                        rows.append(
                            _evaluation_row(
                                run_id, setting_name, item, metric, value, "ok", "",
                                max(0.0, time.monotonic() - started), seed,
                                sample_offset, num_samples, epsilon,
                            )
                        )
            if rows:
                append_rows_atomic(output, rows)
                written += len(rows)
                existing.update(_row_key(_normalise_row(row)) for row in rows)
        finally:
            model = None
            default_computer = None
            try:
                cuda = getattr(torch_module, "cuda")
                if cuda.is_available():
                    cuda.empty_cache()
            except Exception:
                pass
    return EvaluationResult(written=written, failed=failed, skipped=skipped)


def _evaluation_row(
    run_id: str,
    setting_name: str,
    item: BenchmarkManifestItem,
    metric: str,
    value: float | None,
    status: str,
    error: str,
    runtime: float,
    seed: int,
    sample_offset: int,
    num_samples: int,
    epsilon: float,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "benchmark": item.benchmark,
        "setting": setting_name,
        "model_or_arch_id": item.model_or_arch_id,
        "checkpoint_epoch": "" if item.checkpoint_epoch is None else item.checkpoint_epoch,
        "metric_name": metric,
        "direction": METRIC_BY_NAME[metric].direction,
        "metric_value": "" if value is None else value,
        "target_value": item.target_value,
        "status": status,
        "error": error,
        "runtime_sec": runtime,
        "seed": seed,
        "sample_offset": sample_offset,
        "num_samples": num_samples,
        "epsilon": epsilon,
        "checkpoint_path": str(item.checkpoint_path.resolve()),
    }


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bundle-root", required=True, type=Path, metavar="PATH")
    parser.add_argument("--data-root", required=True, type=Path, metavar="PATH")
    parser.add_argument("--output-dir", required=True, type=Path, metavar="PATH")
    parser.add_argument("--run-id", required=True, metavar="TEXT")
    parser.add_argument("--sample-offset", type=int, default=SAMPLE_OFFSET, metavar="INT")
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES, metavar="INT")
    parser.add_argument("--seed", type=int, default=SEED, metavar="INT")
    parser.add_argument("--item-limit", type=int, default=None, metavar="INT")
    parser.add_argument("--metric", action="append", required=True, metavar="NAME")
    parser.add_argument("--device", default="cuda", metavar="TEXT")


def validate_common_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    try:
        validate_run_id(args.run_id)
    except ValueError as exc:
        parser.error(str(exc))
    frozen = (
        ("sample-offset", args.sample_offset, SAMPLE_OFFSET),
        ("num-samples", args.num_samples, NUM_SAMPLES),
        ("seed", args.seed, SEED),
    )
    for name, value, expected in frozen:
        if value != expected:
            parser.error(f"--{name} is frozen at {expected}")
    if args.item_limit is not None and args.item_limit < 1:
        parser.error("--item-limit must be positive")
    unknown = [metric for metric in args.metric if metric not in METRIC_BY_NAME]
    if unknown:
        parser.error(f"unknown --metric: {', '.join(unknown)}")
    if len(args.metric) != len(set(args.metric)):
        parser.error("duplicate --metric is not allowed")


def load_cifar10_samples(
    data_root: str | Path,
    device: str,
    *,
    sample_offset: int = SAMPLE_OFFSET,
    num_samples: int = NUM_SAMPLES,
) -> tuple[object, object]:
    """Load the frozen, ordered CIFAR-10 sample window exactly once."""

    if sample_offset != SAMPLE_OFFSET or num_samples != NUM_SAMPLES:
        raise ValueError("CIFAR-10 sample protocol is frozen")
    import torch
    from torchvision import datasets, transforms

    dataset = datasets.CIFAR10(
        root=str(data_root), train=False, download=False, transform=transforms.ToTensor()
    )
    stop = sample_offset + num_samples
    if stop > len(dataset):
        raise ValueError(f"CIFAR-10 dataset has only {len(dataset)} samples")
    selected = [dataset[index] for index in range(sample_offset, stop)]
    inputs = torch.stack([sample[0] for sample in selected]).to(device)
    labels = torch.tensor([int(sample[1]) for sample in selected], dtype=torch.long).to(device)
    return inputs, labels
