from __future__ import annotations

import pytest
import torch

from robust_metric_lab.adv_excess_batched import compute_adv_excess_batched


def _toy_model() -> torch.nn.Module:
    torch.manual_seed(20261024)
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(4, 2, 1),
        torch.nn.Flatten(),
        torch.nn.Linear(2 * 4 * 4, 2),
    )


def test_batch_one_and_full_agree_for_the_same_64_samples():
    model = _toy_model()
    clean = torch.linspace(0.05, 0.95, 64 * 3 * 4 * 4).reshape(64, 3, 4, 4)
    labels = torch.arange(64) % 2
    one = compute_adv_excess_batched(
        model, clean, labels, epsilon=8 / 255, batch_size=1
    )
    full = compute_adv_excess_batched(
        model, clean, labels, epsilon=8 / 255, batch_size=64
    )
    assert abs(one.value - full.value) < 1e-6
    assert one.diagnostics["samples"] == full.diagnostics["samples"] == 64
    assert one.diagnostics["random_linf_mean"] == pytest.approx(
        full.diagnostics["random_linf_mean"], abs=1e-7
    )


def test_final_short_chunk_has_equal_sample_weight():
    model = _toy_model()
    clean = torch.rand((5, 3, 4, 4), generator=torch.Generator().manual_seed(12))
    labels = torch.tensor([0, 1, 0, 1, 0])
    short = compute_adv_excess_batched(
        model, clean, labels, epsilon=8 / 255, batch_size=2
    )
    whole = compute_adv_excess_batched(
        model, clean, labels, epsilon=8 / 255, batch_size=5
    )
    assert short.value == pytest.approx(whole.value, abs=1e-6)


def test_invalid_input_lengths_and_batch_size_fail_before_model_use():
    model = _toy_model()
    clean = torch.empty((0, 3, 4, 4))
    with pytest.raises(ValueError):
        compute_adv_excess_batched(model, clean, torch.empty(0, dtype=torch.long),
                                   epsilon=8 / 255, batch_size=1)
    clean = torch.zeros((2, 3, 4, 4))
    with pytest.raises(ValueError):
        compute_adv_excess_batched(model, clean, torch.zeros(1, dtype=torch.long),
                                   epsilon=8 / 255, batch_size=1)
    with pytest.raises(ValueError):
        compute_adv_excess_batched(model, clean, torch.zeros(2, dtype=torch.long),
                                   epsilon=8 / 255, batch_size=0)


class _VaryingHooks(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Conv2d(3, 3, 1)
        self.optional = torch.nn.Conv2d(3, 3, 1)
        self.readout = torch.nn.Linear(3 * 4 * 4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first(x)
        if x.shape[0] == 1 and x.detach().mean().item() > 0:
            x = self.optional(x)
        return self.readout(x.flatten(1))


def test_changing_anchor_count_across_chunks_fails():
    model = _VaryingHooks()
    clean = torch.stack((torch.zeros((3, 4, 4)), torch.ones((3, 4, 4))))
    labels = torch.tensor([0, 1])
    with pytest.raises(ValueError, match="anchor|feature"):
        compute_adv_excess_batched(
            model, clean, labels, epsilon=8 / 255, batch_size=1
        )


class _NaNProbe(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x, float("nan"))


class _UnselectedCorruptFeature(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Conv2d(3, 3, 1)
        self.probe = _NaNProbe()
        self.second = torch.nn.Conv2d(3, 3, 1)
        self.relu = torch.nn.ReLU()
        self.third = torch.nn.Conv2d(3, 3, 1)
        self.readout = torch.nn.Linear(3 * 4 * 4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first(x)
        self.probe(x)
        x = self.relu(self.second(x))
        return self.readout(self.third(x).flatten(1))


def test_nonfinite_unselected_feature_fails_instead_of_being_ignored():
    model = _UnselectedCorruptFeature()
    clean = torch.full((2, 3, 4, 4), 0.5)
    labels = torch.tensor([0, 1])
    with pytest.raises(ValueError, match="nonfinite feature"):
        compute_adv_excess_batched(
            model, clean, labels, epsilon=8 / 255, batch_size=1, max_layers=2
        )
