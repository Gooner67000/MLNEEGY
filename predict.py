"""Step 8: score new equipment with the saved model.

Run:  python predict.py   (scores 5 machines from the held-out test set)
"""
import json
import math
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).parent

# Raw sensor columns the model expects (units stripped: XGBoost rejects '[' and ']').
RAW_FEATURES = [
    "Type",
    "Air temperature K",
    "Process temperature K",
    "Rotational speed rpm",
    "Torque Nm",
    "Tool wear min",
]
TYPE_CODES = {"L": 0, "M": 1, "H": 2}
DEFAULT_THRESHOLD = 0.5


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Physics-based features that mirror the dataset's failure mechanisms.

    - Power W: power failure happens when torque x speed is too low or too high.
    - Temp diff K: heat-dissipation failure happens when process-air gap is small.
    - Strain: overstrain failure depends on tool wear x torque.
    """
    out = df.copy()
    out["Power W"] = out["Torque Nm"] * out["Rotational speed rpm"] * 2 * math.pi / 60
    out["Temp diff K"] = out["Process temperature K"] - out["Air temperature K"]
    out["Strain"] = out["Tool wear min"] * out["Torque Nm"]
    return out


@lru_cache(maxsize=1)
def load_model():
    """Returns (model, feature_names, threshold)."""
    model = joblib.load(ROOT / "predictive_maintenance_model.pkl")
    feature_names = joblib.load(ROOT / "feature_names.pkl")
    meta_path = ROOT / "model_meta.json"
    threshold = DEFAULT_THRESHOLD
    if meta_path.exists():
        threshold = json.loads(meta_path.read_text())["threshold"]
    return model, feature_names, threshold


def risk_level(prob: float) -> str:
    if prob < 0.3:
        return "Low"
    if prob < 0.7:
        return "Medium"
    return "High"


def predict_failure_risk(equipment_data: pd.DataFrame) -> pd.DataFrame:
    """Predict failure probability for new equipment.

    Args:
        equipment_data: DataFrame with the RAW_FEATURES columns
            (Type encoded L=0, M=1, H=2). Engineered features are added here.

    Returns:
        DataFrame with failure probability, risk level and a maintenance alert
        flag (probability >= the recall-tuned threshold saved at training time).
    """
    model, feature_names, threshold = load_model()
    X = add_features(equipment_data[RAW_FEATURES])[feature_names]
    probs = model.predict_proba(X)[:, 1]
    return pd.DataFrame(
        {
            "failure_probability": probs.round(4),
            "risk_level": [risk_level(p) for p in probs],
            "alert": probs >= threshold,
        },
        index=equipment_data.index,
    )


if __name__ == "__main__":
    test = pd.read_csv(ROOT / "reports" / "test_set.csv")
    # Show 5 machines, including some that actually failed.
    sample = pd.concat([test[test["Machine failure"] == 1].head(2),
                        test[test["Machine failure"] == 0].head(3)])
    print(sample[RAW_FEATURES + ["Machine failure"]]
          .join(predict_failure_risk(sample)).to_string())
