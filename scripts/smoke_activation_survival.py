"""Non-ranking 64-sample GPU smoke on one small frozen model per space."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from robust_metric_lab.activation_sign_survival import sign_survival  # noqa: E402
from robust_metric_lab.student_benchmark import (  # noqa: E402
    EPSILON, load_darts_items, load_nb201_items, load_rb40_items, resolve_under_root,
)
from robust_metric_lab.student_runner import load_cifar10_samples  # noqa: E402
from scripts.activation_survival_manifest import (  # noqa: E402
    _canonical, require_project_path, sha256_file, verify_source_inventory,
)
from scripts.evaluate_activation_survival_student import (  # noqa: E402
    make_darts_loader, make_nb201_loader, make_rb40_loader, write_attempt,
)


def select_smallest_item(items: Iterable[object], *, setting: str,
                         frozen_keys: set[tuple[str, str, int | None]]) -> object:
    """Use file size only to minimize smoke cost; never change the formal grid."""
    available = [item for item in items
                 if (setting, str(item.model_or_arch_id), item.checkpoint_epoch) in frozen_keys]
    if not available:
        raise ValueError(f"no frozen smoke item for {setting}")
    return min(available, key=lambda item: (Path(item.checkpoint_path).stat().st_size,
                                            str(item.model_or_arch_id)))


def run_smoke(bundle: Path, data: Path, manifest_path: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"smoke output already exists: {output_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = manifest["frozen_payload"]
    if hashlib.sha256(_canonical(payload)).hexdigest() != manifest["frozen_sha256"]:
        raise ValueError("frozen manifest digest mismatch")
    if payload.get("expected_rows") != 560 or len(payload.get("objects", [])) != 560:
        raise ValueError("frozen 560-key grid absent")
    verify_source_inventory(payload["source_sha256"], ROOT)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for preregistered GPU smoke")
    frozen = {(obj["setting"], str(obj["model_or_arch_id"]), obj["checkpoint_epoch"]): obj
              for obj in payload["objects"]}
    if len(frozen) != 560:
        raise ValueError("duplicate frozen key")
    rb = load_rb40_items(resolve_under_root(bundle, "manifests/rb40_models.csv"), bundle)
    nb = load_nb201_items(resolve_under_root(bundle, "manifests/nb201_64_architectures.csv"), bundle)
    darts = load_darts_items(resolve_under_root(bundle, "manifests/darts100_architectures.csv"), bundle)
    choices = [
        ("rb40_mature", select_smallest_item(rb, setting="rb40_mature", frozen_keys=set(frozen)),
         make_rb40_loader(bundle, "cuda")),
        ("nb201_e0", select_smallest_item(nb, setting="nb201_e0", frozen_keys=set(frozen)),
         make_nb201_loader("cuda")),
        ("darts_e0", select_smallest_item(darts, setting="darts_e0", frozen_keys=set(frozen)),
         make_darts_loader("cuda")),
    ]
    output_dir.mkdir(parents=True, exist_ok=False)
    attempts_dir = output_dir / "attempts"
    samples = None
    successful = []
    for setting, item, loader in choices:
        key = (setting, str(item.model_or_arch_id), item.checkpoint_epoch)
        attempt = {
            "purpose": "non-ranking 64-sample smoke", "key": list(key),
            "manifest_sha256": manifest["frozen_sha256"],
            "source_sha256": payload["source_sha256"],
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint_path": str(item.checkpoint_path),
        }
        model = None
        try:
            obj = frozen[key]
            if sha256_file(Path(item.checkpoint_path)) != obj["checkpoint_sha256"]:
                raise ValueError("selected checkpoint hash mismatch")
            if samples is None:
                samples = load_cifar10_samples(data, "cuda")
            torch.cuda.reset_peak_memory_stats("cuda")
            model = loader(item)
            score = sign_survival(model, *samples, epsilon=EPSILON)
            if (score["sample_count"] != 64 or score["call_count"] < 1
                    or not math.isfinite(float(score["metric_value"]))):
                raise ValueError("smoke protocol or finite score failed")
            attempt.update({"status": "ok", "metric_value": score["metric_value"],
                            "sample_count": score["sample_count"],
                            "call_count": score["call_count"],
                            "diagnostics": score["diagnostics"]})
            successful.append(setting)
        except Exception as exc:
            attempt.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc()})
        finally:
            attempt["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            write_attempt(attempts_dir, attempt)
            del model
            torch.cuda.empty_cache()
        print(json.dumps({"setting": setting, "status": attempt["status"],
                          "model_or_arch_id": str(item.model_or_arch_id)}, sort_keys=True), flush=True)
        if attempt["status"] != "ok":
            raise RuntimeError(f"non-ranking smoke failed at {setting}; evidence retained")
    report = {"status": "PASS", "purpose": "non-ranking smoke only",
              "manifest_sha256": manifest["frozen_sha256"],
              "successful_settings": successful, "attempts": 3}
    (output_dir / "smoke_verification.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = run_smoke(*(require_project_path(path, ROOT) for path in
                         (args.bundle_root, args.data_root, args.manifest, args.output_dir)))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
