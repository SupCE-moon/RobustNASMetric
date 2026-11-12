from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.repair_kang_trnas_direct_ode import direct_ode_context, select_repair_rows


def test_direct_ode_context_restores_original_solver_after_success_and_failure():
    original = object()
    direct = object()
    module = SimpleNamespace(odeint=original)
    with direct_ode_context(module, direct):
        assert module.odeint is direct
    assert module.odeint is original

    with pytest.raises(RuntimeError, match="boom"):
        with direct_ode_context(module, direct):
            raise RuntimeError("boom")
    assert module.odeint is original


def test_select_repair_rows_rejects_any_other_or_successful_metric():
    names = ["trnas_activation_psi", "trnas_consistency_phi", "trnas_r_score"]
    frame = pd.DataFrame(
        [
            {"metric_name": name, "setting": "rb40_mature", "model_or_arch_id": "Kang2021Stable", "status": "failed"}
            for name in names
        ]
        + [{"metric_name": "trnas_r_score", "setting": "nb201_e0", "model_or_arch_id": "1", "status": "ok"}]
    )
    assert set(select_repair_rows(frame).metric_name) == set(names)

    modified = frame.copy()
    modified.loc[0, "status"] = "ok"
    with pytest.raises(ValueError, match="exactly three failed"):
        select_repair_rows(modified)
