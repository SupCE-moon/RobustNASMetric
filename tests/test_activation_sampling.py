import pytest
import torch
import torch.nn.functional as F

from robust_metric_lab.metrics import ActivationRecorder, compute_feature_metrics


class TinyFeatureNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 4, 3, padding=1),
            torch.nn.ReLU(),
        )
        self.classifier = torch.nn.Linear(4, 2)

    def forward(self, x):
        features = self.features(x)
        return self.classifier(features.mean(dim=(2, 3)))


class FunctionalReluNet(torch.nn.Module):
    def forward(self, x):
        return torch.relu(F.relu(x))


class Swish(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class NativeActivationNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.activations = torch.nn.Sequential(torch.nn.SiLU(), Swish())

    def forward(self, x):
        return self.activations(x)


def test_post_relu_recorder_only_captures_relu_outputs():
    model = TinyFeatureNet()
    x = torch.randn(3, 3, 8, 8)

    with ActivationRecorder(model, activation_mode="all_leaf") as all_leaf:
        model(x)
    with ActivationRecorder(model, activation_mode="post_relu") as post_relu:
        model(x)

    assert len(all_leaf.outputs) == 4
    assert len(post_relu.outputs) == 2
    assert all(torch.all(output >= 0) for output in post_relu.outputs)


def test_post_relu_recorder_captures_functional_and_torch_relu_calls():
    x = torch.randn(2, 3, 4, 4)

    with ActivationRecorder(FunctionalReluNet(), activation_mode="post_relu") as recorder:
        FunctionalReluNet()(x)

    assert len(recorder.outputs) == 2
    assert all(torch.all(output >= 0) for output in recorder.outputs)


def test_post_activation_recorder_captures_silu_and_custom_swish():
    model = NativeActivationNet()
    x = torch.randn(2, 3, 4, 4)

    with ActivationRecorder(model, activation_mode="post_activation") as recorder:
        model(x)

    assert len(recorder.outputs) == 2


def test_compute_feature_metrics_reports_sampling_audit_metadata():
    model = TinyFeatureNet()
    x_clean = torch.rand(4, 3, 8, 8)
    x_adv = (x_clean + 0.01).clamp(0, 1)
    labels = torch.tensor([0, 1, 0, 1])

    result = compute_feature_metrics(
        model,
        x_clean,
        x_adv,
        labels,
        max_layers=2,
        activation_mode="post_relu",
    )

    assert result.num_candidate_layers == 2
    assert result.num_layers == 2
    assert result.sampled_layer_indices == "0,1"
    assert torch.isfinite(torch.tensor(result.feature_measure_mmd))


def test_activation_recorder_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unsupported activation mode"):
        ActivationRecorder(TinyFeatureNet(), activation_mode="after_batch_norm")
