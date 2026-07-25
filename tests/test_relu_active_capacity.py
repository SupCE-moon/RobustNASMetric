import torch

from robust_metric_lab.metrics import (
    _perturbation_capacity,
    _perturbation_capacity_above_mean,
    _perturbation_capacity_relu_active,
)


def test_relu_active_capacity_matches_plain_capacity_when_all_values_are_positive():
    feature = torch.tensor([[[[1.0, 2.0], [4.0, 7.0]]]])

    actual = _perturbation_capacity_relu_active(feature, feature)

    torch.testing.assert_close(actual, _perturbation_capacity(feature))


def test_relu_active_capacity_ignores_changes_outside_clean_active_support():
    clean = torch.tensor([[[[1.0, -2.0], [3.0, -4.0]]]])
    adv = torch.tensor([[[[1.0, 200.0], [3.0, -400.0]]]])

    clean_capacity = _perturbation_capacity_relu_active(clean, clean)
    adv_capacity = _perturbation_capacity_relu_active(clean, adv)

    torch.testing.assert_close(adv_capacity, clean_capacity)


def test_relu_active_capacity_detects_changes_on_clean_active_support():
    clean = torch.tensor([[[[1.0, -2.0], [3.0, -4.0]]]])
    adv = torch.tensor([[[[1.0, -2.0], [7.0, -4.0]]]])

    clean_capacity = _perturbation_capacity_relu_active(clean, clean)
    adv_capacity = _perturbation_capacity_relu_active(clean, adv)

    assert adv_capacity.item() > clean_capacity.item()


def test_above_mean_capacity_ignores_changes_below_clean_mean_support():
    clean = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    adv = torch.tensor([[[[-100.0, 200.0], [3.0, 4.0]]]])

    clean_capacity = _perturbation_capacity_above_mean(clean, clean)
    adv_capacity = _perturbation_capacity_above_mean(clean, adv)

    torch.testing.assert_close(adv_capacity, clean_capacity)


def test_above_mean_capacity_detects_changes_on_selected_support():
    clean = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    adv = torch.tensor([[[[1.0, 2.0], [3.0, 8.0]]]])

    clean_capacity = _perturbation_capacity_above_mean(clean, clean)
    adv_capacity = _perturbation_capacity_above_mean(clean, adv)

    assert adv_capacity.item() > clean_capacity.item()


def test_above_mean_capacity_is_invariant_to_per_channel_shift():
    clean = torch.tensor([[[[1.0, 2.0], [3.0, 7.0]]]])
    shifted = clean + 100.0

    original_capacity = _perturbation_capacity_above_mean(clean, clean)
    shifted_capacity = _perturbation_capacity_above_mean(shifted, shifted)

    torch.testing.assert_close(shifted_capacity, original_capacity)
