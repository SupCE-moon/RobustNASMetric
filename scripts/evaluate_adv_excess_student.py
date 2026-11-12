"""Evaluate the one frozen adversarial-excess metric on the student grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from robust_metric_lab.adv_excess_batched import (  # noqa: E402
    AdvExcessResult,
    compute_adv_excess_batched,
)
from robust_metric_lab.student_benchmark import (  # noqa: E402
    EPSILON,
    NUM_SAMPLES,
    SAMPLE_OFFSET,
    SEED,
    load_darts_items,
    load_nb201_items,
    load_rb40_items,
    resolve_under_root,
)
from robust_metric_lab.student_runner import (  # noqa: E402
    evaluate_items,
    load_cifar10_samples,
    validate_run_id,
)

from evaluate_darts100_student import make_model_loader as make_darts_loader  # noqa: E402
from evaluate_nb201_student import make_model_loader as make_nb201_loader  # noqa: E402
from evaluate_rb40_student import make_model_loader as make_rb40_loader  # noqa: E402


METRIC_NAME = "feature_adv_excess_centered_unit_l2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate frozen adversarial-excess feature drift")
    parser.add_argument("--benchmark", required=True, choices=("rb40", "nb201", "darts"))
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--bundle-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-offset", type=int, default=SAMPLE_OFFSET)
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--item-limit", type=int)
    return parser


def _setting_and_items(args):
    root = args.bundle_root.resolve()
    if args.benchmark == "rb40":
        if args.epoch is not None:
            raise ValueError("RB40 does not accept --epoch")
        items = load_rb40_items(resolve_under_root(root, "manifests/rb40_models.csv"), root)
        return "rb40_mature", items, make_rb40_loader(root, args.device)
    if args.benchmark == "nb201":
        if args.epoch not in {0, 5, 10, 25, 50}:
            raise ValueError("NB201 --epoch must be one of 0,5,10,25,50")
        all_items = load_nb201_items(
            resolve_under_root(root, "manifests/nb201_64_architectures.csv"), root
        )
        items = [item for item in all_items if item.checkpoint_epoch == args.epoch]
        return f"nb201_e{args.epoch}", items, make_nb201_loader(args.device)
    if args.epoch not in {0, 100}:
        raise ValueError("DARTS --epoch must be 0 or 100")
    all_items = load_darts_items(
        resolve_under_root(root, "manifests/darts100_architectures.csv"), root
    )
    items = [item for item in all_items if item.checkpoint_epoch == args.epoch]
    return f"darts_e{args.epoch}", items, make_darts_loader(args.device)


def write_diagnostics_atomically(
    run_dir: Path, setting: str, item, result: AdvExcessResult,
    *, seed: int, sample_offset: int, num_samples: int,
) -> Path:
    """Create one complete diagnostic JSON per exact raw key, never overwrite."""
    key = {
        "setting": setting,
        "model_or_arch_id": item.model_or_arch_id,
        "checkpoint_epoch": item.checkpoint_epoch,
        "metric_name": METRIC_NAME,
        "seed": seed,
        "sample_offset": sample_offset,
        "num_samples": num_samples,
    }
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()
    directory = run_dir / "diagnostics"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{digest}.json"
    payload = {
        **key,
        "target_value": item.target_value,
        "checkpoint_path": str(item.checkpoint_path.resolve()),
        "metric_value": result.value,
        **result.diagnostics,
    }
    data = (json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".diagnostic-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_run_id(args.run_id)
        if args.batch_size <= 0:
            raise ValueError("--batch-size must be positive")
        if args.sample_offset != SAMPLE_OFFSET or args.num_samples != NUM_SAMPLES or args.seed != SEED:
            raise ValueError("sample protocol is frozen")
        setting, items, load_model = _setting_and_items(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.item_limit is not None:
        if args.item_limit <= 0:
            parser.error("--item-limit must be positive")
        items = items[: args.item_limit]
    run_dir = args.output_dir / args.run_id

    def compute(metric, model, sample_value, item):
        if metric != METRIC_NAME:
            raise ValueError("unexpected metric")
        clean, labels = sample_value
        result = compute_adv_excess_batched(
            model, clean, labels, epsilon=EPSILON,
            batch_size=args.batch_size, sample_offset=args.sample_offset,
        )
        write_diagnostics_atomically(
            run_dir, setting, item, result, seed=args.seed,
            sample_offset=args.sample_offset, num_samples=args.num_samples,
        )
        return result.value

    output = run_dir / "metric_values.csv"
    result = evaluate_items(
        items,
        load_model,
        lambda: load_cifar10_samples(args.data_root, args.device),
        [METRIC_NAME],
        output,
        {
            "run_id": args.run_id,
            "setting": setting,
            "seed": args.seed,
            "sample_offset": args.sample_offset,
            "num_samples": args.num_samples,
            "epsilon": EPSILON,
        },
        compute_metric=compute,
    )
    print(
        f"output={output} written={result.written} failed={result.failed} "
        f"skipped={result.skipped} batch_size={args.batch_size} "
        f"peak_cuda_memory_mib={torch.cuda.max_memory_allocated() / (1024 ** 2):.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
