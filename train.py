"""Steps 2-7: load data, EDA, preprocessing, train/compare 3 models, evaluate, save.

Run:  python train.py
Outputs: predictive_maintenance_model.pkl, feature_names.pkl, reports/*.png, reports/metrics.json
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # save figures to files instead of opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "ai4i2020.csv"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

# Mirrors, used only if data/ai4i2020.csv is missing.
DATA_URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv",
    "https://www.openml.org/data/get_csv/22045521",  # OpenML id 42890, same file
]

# XGBoost rejects feature names containing '[', ']' or '<', so strip the units.
RENAME = {
    "Air temperature [K]": "Air temperature K",
    "Process temperature [K]": "Process temperature K",
    "Rotational speed [rpm]": "Rotational speed rpm",
    "Torque [Nm]": "Torque Nm",
    "Tool wear [min]": "Tool wear min",
}
# Explicit, human-readable encoding (LabelEncoder would give H=0, L=1, M=2).
TYPE_CODES = {"L": 0, "M": 1, "H": 2}
# Failure-mode flags: each one is a *cause* of "Machine failure", so they leak the
# target. They aren't sensor readings and aren't known before a failure happens.
LEAKY = ["TWF", "HDF", "PWF", "OSF", "RNF"]
TARGET = "Machine failure"


# ---------------------------------------------------------------- Step 2: load
def load_data() -> pd.DataFrame:
    if not DATA.exists():
        DATA.parent.mkdir(exist_ok=True)
        for url in DATA_URLS:
            try:
                pd.read_csv(url).to_csv(DATA, index=False)
                break
            except Exception as exc:  # try the next mirror
                print(f"Download failed from {url}: {exc}")
    df = pd.read_csv(DATA)
    print(df.head())
    print("Shape:", df.shape)  # (10000, 14)
    return df


# ----------------------------------------------------------------- Step 3: EDA
def eda(df: pd.DataFrame) -> None:
    print("\nMissing values:\n", df.isnull().sum())
    print(f"\nFailure rate: {df[TARGET].mean() * 100:.2f}%")

    numeric = df.select_dtypes(include=[np.number]).drop(columns=["UDI"])
    numeric.hist(figsize=(12, 8), bins=30)
    plt.tight_layout()
    plt.savefig(REPORTS / "distributions.png", dpi=110)
    plt.close()

    # numeric_only: df.corr() fails on the text columns in pandas 2+
    corr = numeric.corr()[TARGET].drop(TARGET).sort_values()
    print("\nCorrelation with target:\n", corr.sort_values(ascending=False))
    plt.figure(figsize=(10, 6))
    corr.plot(kind="barh")
    plt.xlabel("Correlation with Machine Failure")
    plt.title("Feature Correlation Analysis")
    plt.tight_layout()
    plt.savefig(REPORTS / "correlations.png", dpi=110)
    plt.close()


# ------------------------------------------------------- Step 4: preprocessing
def preprocess(df: pd.DataFrame):
    df_clean = df.drop(columns=["UDI", "Product ID"] + LEAKY).rename(columns=RENAME)
    df_clean["Type"] = df_clean["Type"].map(TYPE_CODES)

    X = df_clean.drop(columns=TARGET)
    y = df_clean[TARGET]

    # Split FIRST so the test set keeps the real ~3.4% failure rate, then
    # undersample only the training data.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_bal, y_train_bal = RandomUnderSampler(random_state=42).fit_resample(X_train, y_train)

    print(f"\nOriginal class distribution:\n{y.value_counts()}")
    print(f"Balanced training distribution:\n{y_train_bal.value_counts()}")
    print(f"Training set: {len(X_train_bal)} rows | Test set: {len(X_test)} rows "
          f"({y_test.mean() * 100:.2f}% failures)")
    return X_train_bal, X_test, y_train_bal, y_test


# ------------------------------------------------------ Step 5: compare models
def compare_models(X_train, y_train, X_test, y_test):
    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)
        ),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, eval_metric="logloss",
        ),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    print("\n" + "=" * 60 + "\nMODEL COMPARISON (ROC AUC)\n" + "=" * 60)
    for name, model in models.items():
        cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc").mean()
        model.fit(X_train, y_train)
        test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        results[name] = {"cv_auc": round(cv_auc, 4), "test_auc": round(test_auc, 4)}
        print(f"{name:<22} CV AUC = {cv_auc:.4f}   Test AUC = {test_auc:.4f}")
    print("=" * 60)
    return models, results


# ---------------------------------------------------- Step 6: evaluate XGBoost
def evaluate(model, X_train, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    print(f"\nCONFUSION MATRIX\nTN={tn}  FP={fp}\nFN={fn}  TP={tp}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f} (avoid false alarms)")
    print(f"Recall:    {metrics['recall']:.4f} (catch actual failures)")
    print(f"F1 Score:  {metrics['f1']:.4f}")

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - XGBoost Model")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(REPORTS / "roc_curve.png", dpi=110)
    plt.close()

    importance = pd.DataFrame(
        {"Feature": X_train.columns, "Importance": model.feature_importances_}
    ).sort_values("Importance", ascending=False)
    plt.figure(figsize=(10, 6))
    plt.barh(importance["Feature"], importance["Importance"])
    plt.xlabel("Importance Score")
    plt.title("Feature Importance - XGBoost")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(REPORTS / "feature_importance.png", dpi=110)
    plt.close()
    print("\nFeature Importance:\n", importance.to_string(index=False))

    # Threshold trade-off (see "Common Pitfalls" in the guide)
    print("\nThreshold  Precision  Recall")
    metrics["thresholds"] = {}
    for t in (0.3, 0.5, 0.7):
        p = (y_proba >= t).astype(int)
        pr, rc = precision_score(y_test, p), recall_score(y_test, p)
        metrics["thresholds"][str(t)] = {"precision": round(pr, 4), "recall": round(rc, 4)}
        print(f"  {t:.1f}      {pr:.4f}    {rc:.4f}")
    return metrics


def main():
    df = load_data()
    eda(df)
    X_train, X_test, y_train, y_test = preprocess(df)
    models, results = compare_models(X_train, y_train, X_test, y_test)
    xgb_model = models["XGBoost"]
    metrics = evaluate(xgb_model, X_train, X_test, y_test)

    # ---------------------------------------------------------- Step 7: save
    joblib.dump(xgb_model, ROOT / "predictive_maintenance_model.pkl")
    joblib.dump(X_train.columns.tolist(), ROOT / "feature_names.pkl")
    X_test.assign(**{TARGET: y_test}).to_csv(REPORTS / "test_set.csv", index=False)
    with open(REPORTS / "metrics.json", "w") as f:
        json.dump({"comparison": results, "xgboost_test": metrics}, f, indent=2, default=float)
    print("\nModel saved as 'predictive_maintenance_model.pkl'")


if __name__ == "__main__":
    main()
