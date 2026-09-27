import math
from pathlib import Path

import pandas as pd
import pytest

from predict import RAW_FEATURES, add_features, predict_failure_risk, risk_level

ROOT = Path(__file__).parent.parent
needs_model = pytest.mark.skipif(
    not (ROOT / "predictive_maintenance_model.pkl").exists(), reason="run train.py first"
)

MACHINE = {"Type": 0, "Air temperature K": 300.0, "Process temperature K": 310.0,
           "Rotational speed rpm": 1500, "Torque Nm": 40.0, "Tool wear min": 100}


def test_add_features_physics():
    out = add_features(pd.DataFrame([MACHINE]))
    assert out["Power W"].iloc[0] == pytest.approx(40.0 * 1500 * 2 * math.pi / 60)
    assert out["Temp diff K"].iloc[0] == pytest.approx(10.0)
    assert out["Strain"].iloc[0] == pytest.approx(4000.0)


@pytest.mark.parametrize("prob,level", [(0.0, "Low"), (0.29, "Low"), (0.3, "Medium"),
                                        (0.69, "Medium"), (0.7, "High"), (1.0, "High")])
def test_risk_level_bands(prob, level):
    assert risk_level(prob) == level


@needs_model
def test_predict_shape_and_range():
    df = pd.DataFrame([MACHINE] * 3)
    out = predict_failure_risk(df)
    assert list(out.columns) == ["failure_probability", "risk_level", "alert"]
    assert len(out) == 3
    assert out["failure_probability"].between(0, 1).all()


@needs_model
def test_extreme_machine_scores_higher_than_normal():
    # Very high torque + worn tool = classic overstrain failure in this dataset.
    worn = {**MACHINE, "Torque Nm": 70.0, "Tool wear min": 240, "Rotational speed rpm": 1200}
    out = predict_failure_risk(pd.DataFrame([MACHINE, worn]))
    assert out["failure_probability"].iloc[1] > out["failure_probability"].iloc[0]
    assert out["failure_probability"].iloc[1] > 0.5


@needs_model
def test_recall_on_held_out_test_set():
    test = pd.read_csv(ROOT / "reports" / "test_set.csv")
    out = predict_failure_risk(test[RAW_FEATURES])
    failures = test["Machine failure"] == 1
    recall = out.loc[failures, "alert"].mean()
    assert recall >= 0.75, f"recall dropped to {recall:.2f}"
