from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TrnasMetricResult:
    activation_psi: float
    consistency_phi: float
    r_score: float
    num_relu_units: int
    num_consistency_layers: int


class _OutputRecorder:
    def __init__(self, model: nn.Module, module_types: tuple[type[nn.Module], ...]):
        self.model = model
        self.module_types = module_types
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.outputs: list[tuple[str, torch.Tensor]] = []

    def __enter__(self) -> "_OutputRecorder":
        for name, module in self.model.named_modules():
            if isinstance(module, self.module_types):
                self.handles.append(module.register_forward_hook(self._make_hook(name)))
        return self

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            if torch.is_tensor(output):
                self.outputs.append((name, output.detach()))

        return hook

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in self.handles:
            handle.remove()


def _sample_wise_activation_score(model: nn.Module, inputs: torch.Tensor) -> tuple[float, int]:
    """Compute the unregularized SWAP cardinality used as TRNAS Psi."""
    model.eval()
    with torch.no_grad(), _OutputRecorder(model, (nn.ReLU,)) as recorder:
        model(inputs)
    if not recorder.outputs:
        return 0.0, 0

    activations = torch.cat([output.flatten(1) for _, output in recorder.outputs], dim=1)
    patterns = (activations > 0).to(torch.uint8).transpose(0, 1)
    psi = float(torch.unique(patterns, dim=0).shape[0])
    return psi, int(patterns.shape[0])


def _adversarial_weight_update(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    step_size: float,
) -> nn.Module:
    """Return a copied model after one cross-entropy loss-ascent step."""
    perturbed = copy.deepcopy(model)
    perturbed.eval()
    perturbed.zero_grad(set_to_none=True)
    loss = F.cross_entropy(perturbed(inputs), labels)
    parameters = [parameter for parameter in perturbed.parameters() if parameter.requires_grad]
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)

    with torch.no_grad():
        for parameter, gradient in zip(parameters, gradients):
            if gradient is None or step_size == 0:
                continue
            parameter.add_(gradient, alpha=float(step_size))
    perturbed.zero_grad(set_to_none=True)
    return perturbed


def _direct_parameter_vector(module: nn.Module) -> torch.Tensor | None:
    parameters = [parameter.detach().flatten() for parameter in module.parameters(recurse=False)]
    if not parameters:
        return None
    return torch.cat(parameters)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left.flatten().float()
    right = right.flatten().float()
    return F.cosine_similarity(left, right, dim=0, eps=1e-12)


def _feature_parameter_consistency(
    model: nn.Module,
    perturbed_model: nn.Module,
    clean_inputs: torch.Tensor,
    adversarial_inputs: torch.Tensor,
) -> tuple[float, int]:
    layer_types = (nn.Conv2d, nn.Linear)
    model.eval()
    perturbed_model.eval()
    with torch.no_grad(), _OutputRecorder(model, layer_types) as clean_recorder:
        model(clean_inputs)
    with torch.no_grad(), _OutputRecorder(perturbed_model, layer_types) as adv_recorder:
        perturbed_model(adversarial_inputs)

    clean_modules = dict(model.named_modules())
    perturbed_modules = dict(perturbed_model.named_modules())
    if len(clean_recorder.outputs) != len(adv_recorder.outputs):
        raise RuntimeError("Clean and perturbed networks produced different layer counts.")

    terms: list[torch.Tensor] = []
    for (clean_name, clean_feature), (adv_name, adv_feature) in zip(
        clean_recorder.outputs, adv_recorder.outputs
    ):
        if clean_name != adv_name:
            raise RuntimeError(f"Layer mismatch: {clean_name!r} != {adv_name!r}")
        clean_parameter = _direct_parameter_vector(clean_modules[clean_name])
        adv_parameter = _direct_parameter_vector(perturbed_modules[adv_name])
        if clean_parameter is None or adv_parameter is None:
            continue
        feature_similarity = 1.0 + _cosine(clean_feature, adv_feature)
        parameter_similarity = 1.0 + _cosine(clean_parameter, adv_parameter)
        terms.append(feature_similarity * parameter_similarity)

    if not terms:
        return 0.0, 0
    return float(torch.stack(terms).sum().item()), len(terms)


def _run_trnas_once(
    model: nn.Module,
    clean_inputs: torch.Tensor,
    adversarial_inputs: torch.Tensor,
    labels: torch.Tensor,
    weight_step_size: float = 1e-6,
) -> TrnasMetricResult:
    psi, num_relu_units = _sample_wise_activation_score(model, clean_inputs)
    perturbed_model = _adversarial_weight_update(
        model, clean_inputs, labels, step_size=weight_step_size
    )
    phi, num_consistency_layers = _feature_parameter_consistency(
        model, perturbed_model, clean_inputs, adversarial_inputs
    )
    r_score = psi**0.5 * phi
    del perturbed_model
    return TrnasMetricResult(
        activation_psi=psi,
        consistency_phi=phi,
        r_score=r_score,
        num_relu_units=num_relu_units,
        num_consistency_layers=num_consistency_layers,
    )


def compute_trnas_metrics_stable(
    model: nn.Module,
    clean_inputs: torch.Tensor,
    adversarial_inputs: torch.Tensor,
    labels: torch.Tensor,
    weight_step_size: float = 1e-6,
) -> TrnasMetricResult:
    """Run TRNAS once, retrying only the known float32 step underflow in float64."""
    try:
        return _run_trnas_once(
            model,
            clean_inputs,
            adversarial_inputs,
            labels,
            weight_step_size=weight_step_size,
        )
    except AssertionError as exc:
        if str(exc) != "underflow in dt 0.0":
            raise
        retry_model = copy.deepcopy(model).double()
        try:
            return _run_trnas_once(
                retry_model,
                clean_inputs.double(),
                adversarial_inputs.double(),
                labels,
                weight_step_size=weight_step_size,
            )
        finally:
            del retry_model


def compute_trnas_metrics(
    model: nn.Module,
    clean_inputs: torch.Tensor,
    adversarial_inputs: torch.Tensor,
    labels: torch.Tensor,
    weight_step_size: float = 1e-6,
) -> TrnasMetricResult:
    """Compute the two TRNAS components and their Eq. (6) R-Score.

    The ICCV 2025 paper does not release code or specify its parameter-update
    magnitude. The frozen default remains 1e-6. A known float32 scalar-step
    underflow is retried without changing that value, using float64 arithmetic.
    """
    return compute_trnas_metrics_stable(
        model,
        clean_inputs,
        adversarial_inputs,
        labels,
        weight_step_size=weight_step_size,
    )
