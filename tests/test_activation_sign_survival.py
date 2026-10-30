import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from robust_metric_lab.activation_sign_survival import (
    call_survival,
    mean_sample_scores,
    pair_score,
    sign_survival,
)


def test_zero_is_nonpositive_and_calls_are_summed():
    clean = torch.tensor([[0.0, 2.0], [-1.0, 3.0]])
    adv = torch.tensor([[0.0, -2.0], [4.0, 3.0]])
    assert call_survival(clean, adv) == pytest.approx(0.5)
    assert call_survival(torch.tensor([0.0, 0.0]), torch.tensor([0.0, 1.0])) == pytest.approx(0.5)
    assert mean_sample_scores([[0.5, 1.0], [1.0, 1.0]]) == pytest.approx(1.75)


class SharedConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Conv2d(1, 1, 1, bias=False)
        self.shared.weight.data.fill_(1)

    def forward(self, x):
        return self.shared(self.shared(x))


def test_repeated_module_has_distinct_invocations():
    result = pair_score(SharedConv(), torch.ones(1, 1, 2, 2), -torch.ones(1, 1, 2, 2))
    assert result["call_ids"] == ["shared#1", "shared#2"]
    assert result["score"] == pytest.approx(0)
    assert result["call_count"] == 2


def test_no_registered_module_fails_even_with_functional_conv():
    class Functional(nn.Module):
        def forward(self, x):
            return F.conv2d(x, torch.ones(1, 1, 1, 1))
    with pytest.raises(ValueError, match="no Conv2d/Linear"):
        pair_score(Functional(), torch.ones(1, 1, 2, 2), torch.ones(1, 1, 2, 2))


def test_shape_mismatch_fails_and_cleans_hooks():
    model = SharedConv()
    with pytest.raises(ValueError, match="shape"):
        pair_score(model, torch.ones(1, 1, 2, 2), torch.ones(1, 1, 3, 3))
    assert not model.shared._forward_hooks


def test_nonfinite_activation_fails_and_cleans_hooks():
    model = SharedConv()
    with pytest.raises(ValueError, match="nonfinite"):
        pair_score(model, torch.full((1, 1, 2, 2), float("nan")), torch.ones(1, 1, 2, 2))
    assert not model.shared._forward_hooks


def test_empty_activation_fails():
    class EmptyLinear(nn.Linear):
        def forward(self, x):
            return super().forward(x)[:, :0]
    model = EmptyLinear(1, 1)
    with pytest.raises(ValueError, match="empty"):
        pair_score(model, torch.ones(1, 1), torch.ones(1, 1))


def test_call_order_mismatch_fails():
    class Conditional(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(1, 1)
            self.second = nn.Linear(1, 1)

        def forward(self, x):
            if x.item() > 0:
                return self.first(x)
            return self.second(x)
    with pytest.raises(ValueError, match="call"):
        pair_score(Conditional(), torch.ones(1, 1), -torch.ones(1, 1))


def test_full_metric_uses_fgsm_and_reports_finite_model_mean():
    model = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
    clean = torch.full((2, 1, 2, 2), 0.5)
    labels = torch.tensor([0, 1])
    result = sign_survival(model, clean, labels, epsilon=8 / 255)
    assert result["sample_count"] == 2
    assert result["call_count"] == 1
    assert math.isfinite(result["metric_value"])
    assert model.training is False
    assert not model[1]._forward_hooks
    assert not any(torch.is_tensor(value) for value in result["diagnostics"].values())


def test_many_reused_calls_keep_only_one_sample_masks_and_numeric_diagnostics():
    class Repeated(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(2, 2)

        def forward(self, x):
            for _ in range(20):
                x = self.layer(x)
            return x

    pair = pair_score(Repeated(), torch.ones(1, 2), -torch.ones(1, 2))
    assert pair["call_count"] == 20
    assert pair["call_ids"] == [f"layer#{i}" for i in range(1, 21)]
    assert pair["max_retained_masks"] <= pair["call_count"]
    assert not any(torch.is_tensor(value) for value in pair.values())
