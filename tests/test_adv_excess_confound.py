"""Frozen architecture-control construction and coverage."""

import pandas as pd
import pytest

from scripts.build_adv_excess_confound import controls_from_manifests


def toy_manifests():
    rb = pd.DataFrame({
        "model_name": ["a", "b"], "parameter_count": [100, 1000],
        "architecture_family": ["wrn", "resnet"],
    })
    nb = pd.DataFrame({
        "arch_id": [1, 2], "op_none": [0, 1],
        "op_skip_connect": [1, 0], "op_nor_conv_1x1": [2, 1],
        "op_nor_conv_3x3": [1, 2], "op_avg_pool_3x3": [1, 1],
    })
    da = pd.DataFrame({
        "arch_id": [3, 4], "parameter_count": [100, 1000],
        "op_sep_conv_3x3": [1, 2], "op_skip_connect": [2, 1],
    })
    return rb, nb, da


def test_controls_preserve_ids_and_repeat_across_epochs():
    controls = controls_from_manifests(*toy_manifests())
    assert set(controls) == {
        "rb40_mature", "nb201_e0", "nb201_e5", "nb201_e10",
        "nb201_e25", "nb201_e50", "darts_e0", "darts_e100",
    }
    assert set(controls["rb40_mature"]) == {
        "model_or_arch_id", "log_parameter_count", "model_family",
    }
    assert controls["rb40_mature"]["model_or_arch_id"].tolist() == ["a", "b"]
    assert controls["nb201_e0"].equals(controls["nb201_e50"])
    assert controls["darts_e0"].equals(controls["darts_e100"])
    assert controls["darts_e0"]["log_parameter_count"].tolist()[0] == pytest.approx(4.605170186)


def test_nonpositive_parameter_count_rejected():
    rb, nb, da = toy_manifests()
    rb.loc[0, "parameter_count"] = 0
    with pytest.raises(ValueError):
        controls_from_manifests(rb, nb, da)
