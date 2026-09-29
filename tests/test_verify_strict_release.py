from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from robust_metric_lab.metric_eligibility import MetricEligibility, MetricEligibilityRecord
from robust_metric_lab.strict_ranking import ManifestExpectation
from robust_metric_lab.release import verify_strict_release
from robust_metric_lab.student_benchmark import BENCHMARK_SETTINGS, EPSILON, NUM_SAMPLES, SAMPLE_OFFSET, SEED


METRIC="feature_delta"


def _fixture():
    rows=[]; manifests={}
    for setting in BENCHMARK_SETTINGS:
        targets={}
        for i in range(setting.expected_rows):
            identifier=f"{setting.key}_{i:03d}"; target=i/setting.expected_rows; targets[identifier]=target
            rows.append({"benchmark":setting.benchmark,"setting":setting.key,"model_or_arch_id":identifier,"checkpoint_epoch":setting.checkpoint_epoch,"metric_name":METRIC,"direction":"lower","metric_value":float(setting.expected_rows-i),"target_value":target,"status":"ok","seed":SEED,"sample_offset":SAMPLE_OFFSET,"num_samples":NUM_SAMPLES,"epsilon":EPSILON})
        manifests[setting.key]=ManifestExpectation(setting.benchmark,setting.checkpoint_epoch,targets)
    eligibility=(MetricEligibilityRecord(METRIC,MetricEligibility.CROSS_SPACE,tuple(s.key for s in BENCHMARK_SETTINGS),"test"),)
    ranking=pd.DataFrame([{"rank":1,"metric_name":METRIC,**{s.key:1.0 for s in BENCHMARK_SETTINGS},"all_setting_mean":1.0}])
    return pd.DataFrame(rows),manifests,eligibility,ranking


def test_release_accepts_exact_complete_grid() -> None:
    frame,manifests,eligibility,ranking=_fixture()
    result=verify_strict_release(frame,manifests,eligibility,ranking)
    assert result["status"]=="PASS" and result["ranked_metrics"]==1


@pytest.mark.parametrize("mutation,match",[
    (lambda f,r:(f.iloc[:-1].copy(),r),"expected 100 rows, found 99"),
    (lambda f,r:(pd.concat([f,f.iloc[[0]]],ignore_index=True),r),"duplicate raw key"),
    (lambda f,r:(f.assign(metric_value=lambda x: x.metric_value.mask(x.index==0,np.inf)),r),"nonfinite metric_value"),
    (lambda f,r:(f.assign(target_value=lambda x: x.target_value.mask(x.index==0,0.9)),r),"target mismatch"),
    (lambda f,r:(f.assign(seed=lambda x: x.seed.mask(x.index==0,1)),r),"seed mismatch"),
    (lambda f,r:(f,r.assign(all_setting_mean=0.5)),"all_setting_mean mismatch"),
])
def test_release_rejects_mutated_evidence(mutation,match: str) -> None:
    frame,manifests,eligibility,ranking=_fixture(); frame,ranking=mutation(frame,ranking)
    with pytest.raises(ValueError,match=match): verify_strict_release(frame,manifests,eligibility,ranking)
