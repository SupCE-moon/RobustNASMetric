# Robust Metric Lab

This project tests whether feature-space robustness metrics can predict CIFAR-10 adversarial robustness across many already-trained defense models.

The first target benchmark is RobustBench CIFAR-10, especially the `Linf` model zoo. RobustBench provides a standardized adversarial robustness benchmark and a unified model zoo API, which makes it useful for validating metric ideas before training new models.

## Why Start From Downloaded Defense Models?

Yes, it is worth starting with about 20 public CIFAR-10 defense models.

- It avoids spending weeks training baselines before the metric is even validated.
- It gives a diverse test set: standard model, adversarially trained models, extra-data models, WRN/RN18 variants, and fast/efficient defenses.
- It lets us test the core question directly: does the proposed metric correlate with known robust accuracy?
- It reduces the risk that a metric only works on one architecture or one training recipe.

The project is intentionally not a NAS project. It is a metric validation harness.

## Project Layout

```text
robust_metric_lab/
  configs/
    cifar10_linf_20.yaml      # representative RobustBench CIFAR-10 models
  scripts/
    smoke_test.py             # import + config smoke test
    find_cifar10.py           # locate existing CIFAR-10 data roots
    list_robustbench_models.py
    prepare_models.py         # load/download selected RobustBench models
    compute_metric_batch.py   # compute feature/logit robustness metrics
    generate_adv_cache.py     # generate reusable shared adversarial examples
    fetch_robustbench_scores.py
    join_robustbench_scores.py
  src/robust_metric_lab/
    attacks.py
    config.py
    datasets.py
    metrics.py
    robustbench_io.py
```

## Metric Candidates

The main metric family is feature perturbation stability. For a model `f`, clean input `x`, and adversarial input `x_adv`, the first implementation computes:

- `feature_delta`: distance between normalized clean/adversarial intermediate features.
- `perturbation_capacity_delta`: clean/adversarial change of centered feature activation volume.
- `logit_kl`: KL divergence between clean/adversarial output distributions.
- `clean_acc` and `adv_acc` on the sampled batch, for sanity checks.

Literature baseline metrics from the Robust NAS notes are also implemented:

- `input_loss_delta`: input loss-landscape sharpness proxy.
- `input_grad_l2`: input-gradient / Jacobian norm proxy.
- `local_logit_lipschitz`: local Lipschitz proxy in logit space.
- `margin_input_grad_ratio`: margin-over-gradient certified-bound proxy.
- `fsp_delta`: feature-flow/FSP matrix perturbation proxy.

Lower metric values should generally indicate higher robustness, but the validation target is rank correlation with RobustBench/PGD/AutoAttack-style robust accuracy.

See `docs/metric_baselines.md` for the mapping from literature families to implemented columns. The current protocol is white-box per model; shared-adversarial metrics are recorded as a deferred experiment.

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .
```

On Linux servers:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Data Policy

Do not redownload CIFAR-10 if a server already has it. Set one of:

```bash
export CIFAR10_ROOT=/path/to/existing/cifar10
export DATA_ROOT=/path/to/datasets
```

The helper below only searches likely paths; it does not download data:

```bash
python scripts/find_cifar10.py --roots /data /datasets /home /root
```

## First Validation Flow

1. Check environment:

```bash
python scripts/smoke_test.py --config configs/cifar10_linf_20.yaml
```

2. List available RobustBench models:

```bash
python scripts/list_robustbench_models.py --dataset cifar10 --threat-model Linf
```

3. Download/load a tiny subset first:

```bash
python scripts/prepare_models.py --config configs/cifar10_linf_20.yaml --limit 2
```

If the server cannot access Google Drive, download the RobustBench weights elsewhere and upload the `models/` directory to the server, then pass:

```bash
python scripts/prepare_models.py --config configs/cifar10_linf_20.yaml --limit 2 --model-dir /path/to/models
```

4. Compute metrics with a cheap attack:

```bash
python scripts/compute_metric_batch.py \
  --config configs/cifar10_linf_20.yaml \
  --model-limit 2 \
  --batch-size 64 \
  --attack fgsm \
  --output outputs/metrics_smoke.csv
```

After the smoke run works, increase `--model-limit` toward 20.

## Shared-Adversarial Validation

RobustBench is used here as the public model zoo and official robustness-score source. For shared adversarial examples, generate reproducible local caches with explicit metadata:

```bash
python scripts/generate_adv_cache.py \
  --config configs/cifar10_linf_20.yaml \
  --data-root /path/to/existing/cifar10 \
  --model-dir models \
  --source-model Standard \
  --attack pgd \
  --pgd-steps 10 \
  --sample-limit 128 \
  --output outputs/adv_cache/cifar10_linf_standard_pgd10_n128_seed20260704.pt
```

Then evaluate all models on the same cached adversarial examples:

```bash
python scripts/compute_metric_batch.py \
  --config configs/cifar10_linf_20.yaml \
  --model-dir models \
  --adv-cache outputs/adv_cache/cifar10_linf_standard_pgd10_n128_seed20260704.pt \
  --output outputs/metrics_20_shared_standard_pgd10_n128_seed20260704.csv
```

To compare against RobustBench official robust accuracy:

```bash
python scripts/fetch_robustbench_scores.py
python scripts/join_robustbench_scores.py \
  --metrics outputs/metrics_20_shared_standard_pgd10_n128_seed20260704.csv \
  --output outputs/metrics_20_shared_standard_pgd10_n128_with_rb_scores.csv
python scripts/analyze_metric_correlation.py \
  --input outputs/metrics_20_shared_standard_pgd10_n128_with_rb_scores.csv \
  --target rb_robust_acc
```

## Notes

- The default attack is intentionally cheap. For final labels, use stronger PGD/AutoAttack evaluations or RobustBench reported scores.
- Adversarial examples can be cached as tensors later; the current code supports generated FGSM/PGD-lite samples first because those are enough to test the metric pipeline.
