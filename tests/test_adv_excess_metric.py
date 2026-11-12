from __future__ import annotations

import numpy as np
import pytest
import torch

from robust_metric_lab.adv_excess_metric import adv_excess_per_sample, fixed_random_linf


def test_fixed_random_is_chunk_invariant_and_uses_global_indices():
    clean = torch.full((4, 3, 2, 2), 0.5)
    full = fixed_random_linf(clean, 8 / 255, [0, 1, 2, 3])
    parts = torch.cat(
        (
            fixed_random_linf(clean[:2], 8 / 255, [0, 1]),
            fixed_random_linf(clean[2:], 8 / 255, [2, 3]),
        )
    )
    assert torch.equal(full, parts)
    bits = np.random.Generator(np.random.PCG64(20261024)).integers(
        0, 2, size=(3, 2, 2), dtype=np.int8
    )
    expected = clean[0] + (8 / 255) * torch.from_numpy((bits * 2 - 1).astype(np.float32))
    assert torch.equal(full[0], expected)


def test_random_control_clips_boundary_values_without_exceeding_budget():
    clean = torch.tensor([[[[0.0, 1.0], [0.5, 0.99]]]])
    result = fixed_random_linf(clean, 8 / 255, [7])
    assert torch.all(result >= 0) and torch.all(result <= 1)
    assert (result - clean).abs().max().item() <= 8 / 255 + 1e-7


def test_random_control_rejects_bad_shape_indices_and_budget():
    clean = torch.full((2, 1, 2, 2), 0.5)
    with pytest.raises(ValueError):
        fixed_random_linf(clean, 8 / 255, [0])
    with pytest.raises(ValueError):
        fixed_random_linf(clean, -1.0, [0, 1])
    with pytest.raises(ValueError):
        fixed_random_linf(clean, 8 / 255, [0, 0])


def test_identical_features_are_zero_and_positive_affine_rescale_invariant():
    clean = [torch.tensor([[[[0.0, 1.0]]], [[[1.0, 0.0]]]])]
    adversarial = [torch.tensor([[[[1.0, 0.0]]], [[[0.0, 1.0]]]])]
    random = [torch.tensor([[[[0.5, 1.0]]], [[[1.0, 0.5]]]])]
    zero, adv_zero, random_zero = adv_excess_per_sample(clean, clean, clean, [0])
    assert torch.equal(zero, torch.zeros(2, dtype=torch.float64))
    assert torch.equal(adv_zero, zero)
    assert torch.equal(random_zero, zero)
    reference = adv_excess_per_sample(clean, adversarial, random, [0])[0]
    scaled = adv_excess_per_sample(
        [value * 7 + 3 for value in clean],
        [value * 7 + 3 for value in adversarial],
        [value * 7 + 3 for value in random],
        [0],
    )[0]
    assert torch.allclose(reference, scaled, atol=1e-6)


def test_even_anchor_median_averages_central_two_and_odd_uses_middle():
    base = torch.tensor([[[[0.0, 1.0]]]])
    flipped = torch.tensor([[[[1.0, 0.0]]]])
    clean = [base.clone() for _ in range(4)]
    adversarial = [base.clone(), base.clone(), flipped.clone(), flipped.clone()]
    random = [base.clone() for _ in range(4)]
    even, adv_even, random_even = adv_excess_per_sample(
        clean, adversarial, random, [0, 1, 2, 3]
    )
    assert even.item() == pytest.approx(0.5)
    assert adv_even.item() == pytest.approx(0.5)
    assert random_even.item() == pytest.approx(0.0)
    odd = adv_excess_per_sample(clean, adversarial, random, [1, 2, 3])[0]
    assert odd.item() == pytest.approx(1.0)


def test_zero_and_one_by_one_maps_remain_finite():
    zero = [torch.zeros((2, 1, 1, 1))]
    main, adv, random = adv_excess_per_sample(zero, zero, zero, [0])
    assert torch.equal(main, torch.zeros(2, dtype=torch.float64))
    assert torch.isfinite(adv).all() and torch.isfinite(random).all()


def test_nonfinite_features_and_invalid_anchors_fail():
    clean = [torch.tensor([[[[0.0, 1.0]]]])]
    corrupt = [torch.tensor([[[[0.0, float("nan")]]]])]
    with pytest.raises(ValueError):
        adv_excess_per_sample(clean, corrupt, clean, [0])
    with pytest.raises(ValueError):
        adv_excess_per_sample(clean, clean, clean, [0, 0])
    with pytest.raises(ValueError):
        adv_excess_per_sample(clean, clean, clean, [1])
