from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from robust_metric_lab.e36_batched import (  # noqa: E402
    E36_METRIC_NAMES,
    compute_e36_metrics_batched,
)
from robust_metric_lab.student_benchmark import (  # noqa: E402
    EPSILON,
    load_rb40_items,
    resolve_under_root,
)
from robust_metric_lab.student_runner import (  # noqa: E402
    add_common_arguments,
    evaluate_items,
    load_cifar10_samples,
    validate_common_arguments,
)
from evaluate_rb40_student import make_model_loader  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen E36 metrics with bounded GPU memory."
    )
    add_common_arguments(parser)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args(argv)
    validate_common_arguments(parser, args)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if set(args.metric) != set(E36_METRIC_NAMES):
        parser.error("this evaluator requires exactly the eight frozen E36 metrics")

    bundle_root = args.bundle_root.resolve()
    items = load_rb40_items(
        resolve_under_root(bundle_root, "manifests/rb40_models.csv"), bundle_root
    )
    if args.item_limit is not None:
        items = items[: args.item_limit]
    output = args.output_dir / args.run_id / "metric_values.csv"
    cache: dict[str, float] = {}
    cached_model_id: int | None = None

    def compute(metric, model, samples, _item):
        nonlocal cached_model_id
        if cached_model_id != id(model):
            clean, labels = samples
            cache.clear()
            cache.update(
                compute_e36_metrics_batched(
                    model,
                    clean,
                    labels,
                    epsilon=EPSILON,
                    batch_size=args.batch_size,
                )
            )
            cached_model_id = id(model)
        return cache[metric]

    result = evaluate_items(
        items,
        make_model_loader(bundle_root, args.device),
        lambda: load_cifar10_samples(args.data_root, args.device),
        args.metric,
        output,
        {
            "run_id": args.run_id,
            "setting": "rb40_mature",
            "seed": args.seed,
            "sample_offset": args.sample_offset,
            "num_samples": args.num_samples,
            "epsilon": EPSILON,
        },
        compute_metric=compute,
    )
    print(
        f"output={output} written={result.written} failed={result.failed} "
        f"skipped={result.skipped} batch_size={args.batch_size}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
