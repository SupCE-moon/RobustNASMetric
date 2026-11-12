from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from robust_metric_lab.release import repair_existing_metric_grid


KEY = ["benchmark", "setting", "model_or_arch_id", "checkpoint_epoch", "metric_name", "seed", "sample_offset", "num_samples"]


def _rows() -> pd.DataFrame:
    common = {"benchmark":"rb40","setting":"rb40_mature","checkpoint_epoch":"","metric_name":"feature_delta","direction":"lower","target_value":0.5,"runtime_sec":1.0,"seed":20260808,"sample_offset":0,"num_samples":64,"epsilon":8/255,"checkpoint_path":"a.pt"}
    return pd.DataFrame([
        {**common,"model_or_arch_id":"ok","metric_value":1.0,"status":"ok","error":""},
        {**common,"model_or_arch_id":"bad","metric_value":"","status":"failed","error":"oom"},
    ])


def _repair() -> pd.DataFrame:
    row = _rows().iloc[1].to_dict()
    row.update(metric_value=2.0,status="ok",error="",runtime_sec=2.0)
    return pd.DataFrame([row])


def test_repair_replaces_only_failed_exact_key_and_writes_manifest(tmp_path: Path) -> None:
    source, repair = tmp_path/"source.csv", tmp_path/"repair.csv"
    target, manifest = tmp_path/"target.csv", tmp_path/"repair_manifest.json"
    _rows().to_csv(source,index=False); _repair().to_csv(repair,index=False)
    repair_existing_metric_grid(source,repair,target,manifest)
    result=pd.read_csv(target)
    assert result.loc[result.model_or_arch_id=="ok","metric_value"].item()==1.0
    assert result.loc[result.model_or_arch_id=="bad","metric_value"].item()==2.0
    payload=json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["replaced_count"]==1
    assert payload["source_sha256"] and payload["repair_sha256"] and payload["target_sha256"]


@pytest.mark.parametrize("mutation,match",[
    (lambda r: r.assign(model_or_arch_id="ok"),"successful"),
    (lambda r: r.assign(target_value=0.6),"target_value"),
    (lambda r: r.assign(seed=1),"seed"),
    (lambda r: pd.concat([r,r],ignore_index=True),"duplicate"),
])
def test_repair_rejects_unsafe_rows(tmp_path: Path, mutation, match: str) -> None:
    source, repair = tmp_path/"source.csv", tmp_path/"repair.csv"
    _rows().to_csv(source,index=False); mutation(_repair()).to_csv(repair,index=False)
    with pytest.raises(ValueError,match=match):
        repair_existing_metric_grid(source,repair,tmp_path/"target.csv",tmp_path/"manifest.json")
    assert not (tmp_path/"target.csv").exists()
