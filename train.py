"""Steps 2-7: load data, EDA, preprocessing, train/compare models, evaluate, save.

Run:  python train.py
Outputs:
  predictive_maintenance_model.pkl, feature_names.pkl, model_meta.json
  reports/*.png, reports/metrics.json, reports/results.md, reports/test_set.csv
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
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from predict import TYPE_CODES, add_features

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
# Failure-mode flags: each one is a *cause* of "Machine failure", so they leak the
# target. They aren't sensor readings and aren't known before a failure happens.
LEAKY = ["TWF", "HDF", "PWF", "OSF", "RNF"]
TARGET = "Machine failure"
TARGET_RECALL = 0.85  # guide: "Recall >= 0.85 (catch most failures)"
SEED = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def save_fig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(REPORTS / name, dpi=110)
    plt.close()


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

    numeric = add_features(df.rename(columns=RENAME)).select_dtypes(include=[np.number])
    numeric = numeric.drop(columns=["UDI"] + LEAKY)
    numeric.hist(figsize=(14, 9), bins=30)
    save_fig("distributions.png")

    # numeric only: df.corr() fails on the text columns in pandas 2+
    corr = numeric.corr()[TARGET].drop(TARGET).sort_values()
    print("\nCorrelation with target:\n", corr.sort_values(ascending=False))
    plt.figure(figsize=(10, 6))
    corr.plot(kind="barh")
    plt.xlabel("Correlation with Machine Failure")
    plt.title("Feature Correlation Analysis")
    save_fig("correlations.png")


# ------------------------------------------------------- Step 4: preprocessing
def preprocess(df: pd.DataFrame):
    df_clean = df.drop(columns=["UDI", "Product ID"] + LEAKY).rename(columns=RENAME)
    df_clean["Type"] = df_clean["Type"].map(TYPE_CODES)

    X = add_features(df_clean.drop(columns=TARGET))
    y = df_clean[TARGET]

    # Split FIRST so the test set keeps the real ~3.4% failure rate. Only the
    # training data is undersampled (for the guide's baseline models).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    X_bal, y_bal = RandomUnderSampler(random_state=SEED).fit_resample(X_train, y_train)

    print(f"\nOriginal class distribution:\n{y.value_counts()}")
    print(f"Undersampled training distribution:\n{y_bal.value_counts()}")
    print(f"Train: {len(X_train)} rows ({len(X_bal)} undersampled) | "
          f"Test: {len(X_test)} rows ({y_test.mean() * 100:.2f}% failures)")
    return X_train, X_test, y_train, y_test, X_bal, y_bal


# ---------------------------------------------- Step 5: compare the guide's 3
def baseline_models():
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED)
        ),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=SEED, eval_metric="logloss",
        ),
    }


# ------------------------------------ Beyond the guide: tuned, class-weighted
def tune_xgboost(X_train, y_train):
    """Train on ALL training rows (no data thrown away by undersampling) and
    handle imbalance with scale_pos_weight instead. Tune for average precision,
    which rewards ranking the rare failures at the top."""
    spw = float((y_train == 0).sum() / (y_train == 1).sum())
    search = RandomizedSearchCV(
        XGBClassifier(eval_metric="logloss", scale_pos_weight=spw,
                      random_state=SEED, n_jobs=1),
        param_distributions={
            "n_estimators": [200, 300, 500],
            "max_depth": [3, 4, 5, 6],
            "learning_rate": [0.03, 0.05, 0.1],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
            "min_child_weight": [1, 3, 5],
            "reg_lambda": [1, 3, 10],
        },
        n_iter=25, scoring="average_precision", cv=CV, random_state=SEED, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"\nTuned XGBoost: CV average precision = {search.best_score_:.4f}")
    print("Best params:", search.best_params_)
    return search.best_estimator_, search.best_params_, spw


def choose_threshold(model, X_train, y_train) -> float:
    """Highest threshold whose out-of-fold recall on TRAINING data is >= target.
    Chosen without touching the test set, so test metrics stay honest."""
    oof = cross_val_predict(clone(model), X_train, y_train, cv=CV,
                            method="predict_proba")[:, 1]
    _, recall, thresholds = precision_recall_curve(y_train, oof)
    ok = recall[:-1] >= TARGET_RECALL
    threshold = float(thresholds[ok].max()) if ok.any() else 0.5
    print(f"Chosen threshold for recall >= {TARGET_RECALL}: {threshold:.4f}")
    return threshold


def test_metrics(model, X_test, y_test, threshold: float) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return {
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


# ------------------------------------------------------------ Step 6: evaluate
def plots(models: dict, final_name: str, X_test, y_test) -> pd.DataFrame:
    plt.figure(figsize=(8, 6))
    for name, m in models.items():
        fpr, tpr, _ = roc_curve(y_test, m.predict_proba(X_test)[:, 1])
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC {roc_auc_score(y_test, m.predict_proba(X_test)[:, 1]):.3f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (test set)")
    plt.legend(loc="lower right")
    save_fig("roc_curve.png")

    plt.figure(figsize=(8, 6))
    for name, m in models.items():
        proba = m.predict_proba(X_test)[:, 1]
        p, r, _ = precision_recall_curve(y_test, proba)
        plt.plot(r, p, lw=2, label=f"{name} (AP {average_precision_score(y_test, proba):.3f})")
    plt.axhline(y_test.mean(), color="gray", lw=1, linestyle="--", label="Random")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves (test set, 3.4% failures)")
    plt.legend(loc="upper right")
    save_fig("pr_curve.png")

    final = models[final_name]
    gain = pd.Series(final.feature_importances_, index=X_test.columns).sort_values()
    plt.figure(figsize=(10, 6))
    gain.plot(kind="barh")
    plt.xlabel("XGBoost importance")
    plt.title(f"Feature Importance - {final_name}")
    save_fig("feature_importance.png")

    perm = permutation_importance(final, X_test, y_test, scoring="average_precision",
                                  n_repeats=10, random_state=SEED, n_jobs=-1)
    perm_df = pd.DataFrame({"Feature": X_test.columns, "Importance": perm.importances_mean,
                            "Std": perm.importances_std}).sort_values("Importance")
    plt.figure(figsize=(10, 6))
    plt.barh(perm_df["Feature"], perm_df["Importance"], xerr=perm_df["Std"])
    plt.xlabel("Drop in average precision when feature is shuffled")
    plt.title(f"Permutation Importance - {final_name} (test set)")
    save_fig("permutation_importance.png")
    return perm_df.sort_values("Importance", ascending=False)


def write_results(comparison: dict, final_name: str, final: dict, best_params: dict,
                  perm_df: pd.DataFrame, n_test: int, n_fail: int) -> None:
    lines = [
        "# Results",
        "",
        f"Test set: {n_test} machines, {n_fail} failures (real failure rate, never resampled).",
        "",
        "## Model comparison",
        "",
        "| Model | Trained on | CV ROC AUC | Test ROC AUC | Test PR AUC | Precision @0.5 | Recall @0.5 |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in comparison.items():
        lines.append(f"| {name} | {r['trained_on']} | {r['cv_auc']:.3f} | {r['roc_auc']:.3f} | "
                     f"{r['pr_auc']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} |")
    cm = final["confusion_matrix"]
    target_auc = "✅" if final["roc_auc"] >= 0.90 else "❌"
    target_rec = "✅" if final["recall"] >= TARGET_RECALL else "❌"
    lines += [
        "",
        f"## Final model: {final_name}",
        "",
        f"Alert threshold **{final['threshold']:.3f}**, chosen by cross-validation on the "
        f"training data to reach recall ≥ {TARGET_RECALL}.",
        "",
        "| Metric | Test value | Guide target |",
        "|---|---|---|",
        f"| ROC AUC | {final['roc_auc']:.3f} | ≥ 0.90 {target_auc} |",
        f"| Recall | {final['recall']:.3f} | ≥ 0.85 {target_rec} |",
        f"| Precision | {final['precision']:.3f} | — |",
        f"| F1 | {final['f1']:.3f} | — |",
        f"| PR AUC | {final['pr_auc']:.3f} | — (random = {n_fail / n_test:.3f}) |",
        "",
        f"Confusion matrix: caught **{cm['tp']}** of {cm['tp'] + cm['fn']} failures, "
        f"missed {cm['fn']}, with {cm['fp']} false alarms across {cm['tn'] + cm['fp']} healthy machines.",
        "",
        "Best hyperparameters: `" + json.dumps(best_params) + "`",
        "",
        "## What drives the predictions (permutation importance)",
        "",
        "| Feature | Drop in PR AUC when shuffled |",
        "|---|---|",
    ]
    lines += [f"| {r.Feature} | {r.Importance:.3f} |" for r in perm_df.itertuples()]
    lines += [
        "",
        "## Plots",
        "",
        "![ROC](roc_curve.png)",
        "![PR](pr_curve.png)",
        "![Permutation importance](permutation_importance.png)",
        "![Feature importance](feature_importance.png)",
        "![Correlations](correlations.png)",
        "",
    ]
    (REPORTS / "results.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    df = load_data()
    eda(df)
    X_train, X_test, y_train, y_test, X_bal, y_bal = preprocess(df)

    print("\n" + "=" * 70 + "\nMODEL COMPARISON\n" + "=" * 70)
    fitted, comparison = {}, {}
    for name, model in baseline_models().items():
        cv_auc = cross_val_score(model, X_bal, y_bal, cv=CV, scoring="roc_auc").mean()
        model.fit(X_bal, y_bal)
        fitted[name] = model
        comparison[name] = {"trained_on": "undersampled", "cv_auc": cv_auc,
                            **test_metrics(model, X_test, y_test, 0.5)}

    final_name = "XGBoost tuned (class-weighted)"
    tuned, best_params, spw = tune_xgboost(X_train, y_train)
    fitted[final_name] = tuned
    cv_auc = cross_val_score(clone(tuned), X_train, y_train, cv=CV, scoring="roc_auc").mean()
    comparison[final_name] = {"trained_on": "all rows", "cv_auc": cv_auc,
                              **test_metrics(tuned, X_test, y_test, 0.5)}

    for name, r in comparison.items():
        print(f"{name:<32} CV AUC {r['cv_auc']:.4f} | Test AUC {r['roc_auc']:.4f} | "
              f"PR AUC {r['pr_auc']:.4f} | P {r['precision']:.3f} R {r['recall']:.3f}")

    threshold = choose_threshold(tuned, X_train, y_train)
    final = test_metrics(tuned, X_test, y_test, threshold)
    cm = final["confusion_matrix"]
    print(f"\nFINAL MODEL @ threshold {threshold:.3f}")
    print(f"TN={cm['tn']}  FP={cm['fp']}\nFN={cm['fn']}  TP={cm['tp']}")
    for k in ("roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1"):
        print(f"{k:<10} {final[k]:.4f}")

    print("\nThreshold  Precision  Recall")
    proba = tuned.predict_proba(X_test)[:, 1]
    threshold_table = {}
    for t in sorted({0.3, 0.5, 0.7, round(threshold, 3)}):
        p = (proba >= t).astype(int)
        threshold_table[str(t)] = {"precision": precision_score(y_test, p, zero_division=0),
                                   "recall": recall_score(y_test, p)}
        print(f"  {t:.3f}    {threshold_table[str(t)]['precision']:.4f}    "
              f"{threshold_table[str(t)]['recall']:.4f}")

    perm_df = plots(fitted, final_name, X_test, y_test)
    print("\nPermutation importance:\n", perm_df.to_string(index=False))

    # ---------------------------------------------------------- Step 7: save
    joblib.dump(tuned, ROOT / "predictive_maintenance_model.pkl")
    joblib.dump(X_train.columns.tolist(), ROOT / "feature_names.pkl")
    (ROOT / "model_meta.json").write_text(json.dumps({
        "model": final_name,
        "threshold": threshold,
        "target_recall": TARGET_RECALL,
        "scale_pos_weight": spw,
        "best_params": best_params,
        "features": X_train.columns.tolist(),
    }, indent=2))
    X_test.assign(**{TARGET: y_test}).to_csv(REPORTS / "test_set.csv", index=False)
    with open(REPORTS / "metrics.json", "w") as f:
        json.dump({"comparison": comparison, "final": final,
                   "thresholds": threshold_table}, f, indent=2, default=float)
    write_results(comparison, final_name, final, best_params, perm_df,
                  len(y_test), int(y_test.sum()))
    print("\nModel saved as 'predictive_maintenance_model.pkl'; see reports/results.md")


if __name__ == "__main__":
    main()
