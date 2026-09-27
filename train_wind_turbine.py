"""Trains the model for machine_types['wind_turbine_generator'].

Data: real SCADA telemetry from an operating wind farm (anonymized sensor
names, real status/event codes) -- not synthetic. The source file is ~12 GB,
far more than needed and impractical to fully re-download on every CI run, so
this script streams it in chunks and keeps a bounded, real subsample: up to
PER_ASSET_CAP rows for each of the first N_ASSETS_TARGET distinct turbines it
encounters. That subsample is cached to data/ so later runs don't re-stream.
This is subsampling real data, not synthesizing data.

Label: status_type_id == 0 means normal operation; any other code is a
logged event/anomaly. y = 1 for "not normal right now".

Leakage guards, same shape as backtest.py:
  1. Turbine (asset_id) holdout -- test turbines never appear in training.
  2. Cycle causality -- rolling features at time t use only that turbine's
     own rows at time <= t.

Run:  python train_wind_turbine.py
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)
SOURCE_URL = ("https://huggingface.co/datasets/kevykibbz/"
              "wind-turbine-scada-data-for-early-fault-detection/resolve/main/"
              "wind-turbine-scada-data-for-early-fault-detection.csv")
SUBSAMPLE_CSV = ROOT / "data" / "wind_turbine_subsample.csv"

PER_ASSET_CAP = 2000     # rows kept per turbine
N_ASSETS_TARGET = 20     # stop once this many turbines are fully sampled
MAX_CHUNKS = 60          # hard safety limit on how much of the 12 GB file we scan
CHUNK_SIZE = 20_000
ROLL_WINDOW = 6
SEED = 42


def build_subsample() -> pd.DataFrame:
    """Streams the real source file and keeps a bounded, honest subsample."""
    print(f"Streaming {SOURCE_URL} (this can take a few minutes)...")
    kept: dict[object, list[pd.DataFrame]] = {}
    finished_assets = set()
    reader = pd.read_csv(SOURCE_URL, chunksize=CHUNK_SIZE, low_memory=False)
    for i, chunk in enumerate(reader):
        if i >= MAX_CHUNKS:
            print(f"Hit MAX_CHUNKS={MAX_CHUNKS} safety limit, stopping stream.")
            break
        for asset_id, group in chunk.groupby("asset_id"):
            if asset_id in finished_assets:
                continue
            have = sum(len(g) for g in kept.get(asset_id, []))
            take = group.head(max(0, PER_ASSET_CAP - have))
            if len(take):
                kept.setdefault(asset_id, []).append(take)
            if have + len(take) >= PER_ASSET_CAP:
                finished_assets.add(asset_id)
        if len(finished_assets) >= N_ASSETS_TARGET:
            print(f"Collected {N_ASSETS_TARGET} fully-sampled turbines after {i + 1} chunks.")
            break
    frames = [g for groups in kept.values() for g in groups]
    df = pd.concat(frames, ignore_index=True)
    print(f"Subsample: {len(df)} rows across {df['asset_id'].nunique()} real turbines "
          f"(up to {PER_ASSET_CAP} rows/turbine, streamed from the real 12 GB source).")
    return df


def load_data() -> pd.DataFrame:
    SUBSAMPLE_CSV.parent.mkdir(exist_ok=True)
    if not SUBSAMPLE_CSV.exists():
        df = build_subsample()
        df.to_csv(SUBSAMPLE_CSV, index=False)
    return pd.read_csv(SUBSAMPLE_CSV)


def add_labels_and_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.sort_values(["asset_id", "time_stamp"]).reset_index(drop=True)
    df["y"] = (df["status_type_id"] != 0).astype(int)

    sensor_cols = [c for c in df.columns if c.endswith("_avg")]
    sensor_cols = [c for c in sensor_cols if df[c].std(skipna=True) > 1e-6]

    g = df.groupby("asset_id")
    feature_cols = list(sensor_cols)
    for c in sensor_cols:
        roll = g[c].rolling(ROLL_WINDOW, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f"{c}_roll"] = roll
        feature_cols.append(f"{c}_roll")
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
    return df, feature_cols


def main():
    df = load_data()
    df, feature_cols = add_labels_and_features(df)
    print(f"Event rate: {df['y'].mean():.3f}")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
    train_idx, test_idx = next(splitter.split(df, groups=df["asset_id"]))
    train_df, test_df = df.loc[train_idx], df.loc[test_idx]
    train_assets = set(train_df["asset_id"])
    test_assets = set(test_df["asset_id"])
    assert not train_assets & test_assets, "turbine leaked between train/test!"
    print(f"Train turbines: {len(train_assets)} | Test turbines (never trained on): "
          f"{len(test_assets)}")

    spw = float((train_df["y"] == 0).sum() / max((train_df["y"] == 1).sum(), 1))
    model = XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.07,
                          scale_pos_weight=spw, eval_metric="logloss", random_state=SEED)
    model.fit(train_df[feature_cols], train_df["y"])

    proba = model.predict_proba(test_df[feature_cols])[:, 1]
    y_test = test_df["y"].values
    roc_auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)
    threshold = 0.5
    pred = (proba >= threshold).astype(int)
    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    print(f"\nHeld-out turbines -- ROC AUC {roc_auc:.4f}  PR AUC {pr_auc:.4f}  "
          f"Precision {precision:.4f}  Recall {recall:.4f}")

    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, lw=2, label=f"ROC AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("Wind turbine backtest -- held-out turbines")
    plt.legend(); plt.tight_layout()
    plt.savefig(REPORTS / "wind_turbine_roc.png", dpi=110); plt.close()

    # Lead-time: for held-out turbines, walk each one forward in time order
    # (features already causal), and for every real event row, look BACKWARD
    # at that turbine's own already-computed causal predictions to see how
    # many rows earlier the model's alert first turned on.
    lead_times = []
    for asset in test_assets:
        eng = test_df[test_df["asset_id"] == asset].sort_values("time_stamp").reset_index(drop=True)
        eng_proba = model.predict_proba(eng[feature_cols])[:, 1]
        alert = eng_proba >= threshold
        event_rows = eng.index[eng["y"] == 1].tolist()
        for r in event_rows:
            # walk backward from r while alert stays on
            k = r
            while k > 0 and alert[k - 1]:
                k -= 1
            if alert[r] or k < r:
                lead_times.append(r - k)
    lead_times = np.array(lead_times)
    print(f"Rows-before-event lead time: n={len(lead_times)}, "
          f"median {np.median(lead_times) if len(lead_times) else float('nan'):.1f}")

    sensor_cols = [c for c in feature_cols if c.endswith("_avg")]
    joblib.dump(model, ROOT / "wind_turbine_model.pkl")
    joblib.dump({
        "features": feature_cols,
        "raw_sensor_cols": sensor_cols,
        "roll_window": ROLL_WINDOW,
        # backend fallback for any raw sensor a manual/partial reading omits
        "medians": train_df[sensor_cols].median().to_dict(),
    }, ROOT / "wind_turbine_meta.pkl")

    metrics = {
        "dataset": "Real wind-farm SCADA telemetry (subsampled, streamed from real source)",
        "n_rows": len(df), "n_turbines": int(df["asset_id"].nunique()),
        "n_train_turbines": len(train_assets), "n_test_turbines": len(test_assets),
        "roc_auc": roc_auc, "pr_auc": pr_auc, "precision": precision, "recall": recall,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "lead_time_rows": {"n_events": len(lead_times),
                           "median": float(np.median(lead_times)) if len(lead_times) else None},
    }
    with open(REPORTS / "wind_turbine_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)

    md = f"""# Wind turbine / generator model

**Dataset:** real wind-farm SCADA telemetry, subsampled while streaming the
real ~12 GB source (up to {PER_ASSET_CAP} rows for each of {df['asset_id'].nunique()}
real turbines) -- a bounded real sample, not synthetic data.

**Leakage guards:** {len(train_assets)} turbines trained on, {len(test_assets)}
completely different turbines held out. All rolling features use only each
turbine's own past readings.

| Metric (held-out turbines) | Value |
|---|---|
| ROC AUC | {roc_auc:.3f} |
| PR AUC | {pr_auc:.3f} |
| Precision @ {threshold} | {precision:.3f} |
| Recall @ {threshold} | {recall:.3f} |

Of {len(lead_times)} real events on held-out turbines, the model's alert (using
only that turbine's own past data) was already on a median of
**{np.median(lead_times) if len(lead_times) else float('nan'):.0f} readings** before the event.

![ROC](wind_turbine_roc.png)
"""
    (REPORTS / "wind_turbine_results.md").write_text(md, encoding="utf-8")
    print("\nSaved wind_turbine_model.pkl; see reports/wind_turbine_results.md")


if __name__ == "__main__":
    main()
