"""Step 8: score new equipment with the saved model.

Run:  python predict.py   (scores 5 machines from the held-out test set)
"""
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).parent


@lru_cache(maxsize=1)
def load_model():
    model = joblib.load(ROOT / "predictive_maintenance_model.pkl")
    feature_names = joblib.load(ROOT / "feature_names.pkl")
    return model, feature_names


def risk_level(prob: float) -> str:
    if prob < 0.3:
        return "Low"
    if prob < 0.7:
        return "Medium"
    return "High"


def predict_failure_risk(equipment_data: pd.DataFrame) -> pd.DataFrame:
    """Predict failure probability for new equipment.

    Args:
        equipment_data: DataFrame with the training feature columns
            (Type encoded L=0, M=1, H=2).

    Returns:
        DataFrame with failure probability and risk level per row.
    """
    model, feature_names = load_model()
    probs = model.predict_proba(equipment_data[feature_names])[:, 1]
    return pd.DataFrame(
        {"failure_probability": probs.round(4), "risk_level": [risk_level(p) for p in probs]},
        index=equipment_data.index,
    )


if __name__ == "__main__":
    test = pd.read_csv(ROOT / "reports" / "test_set.csv")
    # Show 5 machines, including some that actually failed.
    sample = pd.concat([test[test["Machine failure"] == 1].head(2),
                        test[test["Machine failure"] == 0].head(3)])
    print(sample.join(predict_failure_risk(sample)).to_string())
