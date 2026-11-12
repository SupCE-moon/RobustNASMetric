from __future__ import annotations

import torch


def test_symmetric_eigvalsh_retries_failed_float32_in_float64(monkeypatch):
    from robust_metric_lab.metrics import _symmetric_eigvalsh_stable

    original = torch.linalg.eigvalsh
    attempted_dtypes: list[torch.dtype] = []

    def fail_float32_once(matrix: torch.Tensor) -> torch.Tensor:
        attempted_dtypes.append(matrix.dtype)
        if matrix.dtype == torch.float32:
            raise RuntimeError("synthetic eigensolver non-convergence")
        return original(matrix)

    monkeypatch.setattr(torch.linalg, "eigvalsh", fail_float32_once)
    matrix = torch.tensor(
        [[[1.0, 1.0], [1.0, 1.0 + 1e-7]]], dtype=torch.float32
    )

    result = _symmetric_eigvalsh_stable(matrix)

    assert attempted_dtypes == [torch.float32, torch.float64]
    assert result.dtype == torch.float32
    assert torch.isfinite(result).all()

