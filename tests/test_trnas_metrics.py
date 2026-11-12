import pytest
import torch
import torch.nn as nn

from robust_metric_lab.trnas import (
    _run_trnas_once,
    _sample_wise_activation_score,
    compute_trnas_metrics,
    compute_trnas_metrics_stable,
)


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(2, 3, bias=False),
            nn.ReLU(),
            nn.Linear(3, 2, bias=False),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(2, 2, bias=False)

    def forward(self, x):
        return self.classifier(self.features(x))


def test_sample_wise_activation_score_counts_unique_neuron_patterns():
    model = nn.Sequential(nn.Linear(2, 2, bias=False), nn.ReLU())
    with torch.no_grad():
        model[0].weight.copy_(torch.eye(2))
    inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    psi, units = _sample_wise_activation_score(model, inputs)

    assert psi == 2.0
    assert units == 2


def test_trnas_score_reduces_to_exact_consistency_without_perturbation():
    torch.manual_seed(7)
    model = TinyNet()
    inputs = torch.tensor([[1.0, 0.5], [0.2, 1.2], [0.7, 0.9]])
    labels = torch.tensor([0, 1, 0])
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}

    result = compute_trnas_metrics(
        model,
        inputs,
        inputs.clone(),
        labels,
        weight_step_size=0.0,
    )

    assert result.num_consistency_layers == 3
    assert result.consistency_phi == pytest.approx(12.0, abs=1e-5)
    assert result.r_score == pytest.approx(
        result.activation_psi**0.5 * result.consistency_phi
    )
    for name, value in model.state_dict().items():
        assert torch.equal(value, original[name])


def test_trnas_metrics_are_finite_after_weight_and_input_perturbation():
    torch.manual_seed(11)
    model = TinyNet()
    inputs = torch.rand(4, 2)
    labels = torch.tensor([0, 1, 0, 1])
    adversarial = (inputs + 0.01).clamp(0, 1)

    result = compute_trnas_metrics(model, inputs, adversarial, labels)

    assert result.activation_psi >= 0
    assert 0 <= result.consistency_phi <= 4 * result.num_consistency_layers
    assert torch.isfinite(torch.tensor(result.r_score))


class Float32UnderflowNet(TinyNet):
    def forward(self, x):
        if next(self.parameters()).dtype == torch.float32:
            raise AssertionError("underflow in dt 0.0")
        return super().forward(x)


class DifferentAssertionNet(TinyNet):
    def forward(self, x):
        raise AssertionError("different assertion")


def test_exact_underflow_retries_float64_and_matches_direct_float64():
    torch.manual_seed(29)
    model = Float32UnderflowNet()
    inputs = torch.rand(4, 2)
    labels = torch.tensor([0, 1, 0, 1])
    adversarial = (inputs + 0.01).clamp(0, 1)

    actual = compute_trnas_metrics_stable(model, inputs, adversarial, labels)
    expected = _run_trnas_once(
        model.double(),
        inputs.double(),
        adversarial.double(),
        labels,
        weight_step_size=1e-6,
    )

    assert actual == expected


def test_non_underflow_assertion_is_not_swallowed():
    model = DifferentAssertionNet()
    inputs = torch.rand(2, 2)
    with pytest.raises(AssertionError, match="different assertion"):
        compute_trnas_metrics_stable(
            model,
            inputs,
            inputs.clone(),
            torch.tensor([0, 1]),
        )
