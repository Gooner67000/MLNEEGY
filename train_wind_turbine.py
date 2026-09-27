"""Trains the model for machine_types['wind_turbine_generator'].

Data: real SCADA telemetry from an operating wind farm (anonymized sensor
names, real status/event codes) -- not synthetic. The source file is ~12 GB
and, importantly, laid out SEQUENTIALLY BY TURBINE (confirmed by probing it:
a single contiguous slice anywhere in the file belongs to one turbine, not
a mix), so this script issues N_ASSETS_TARGET small HTTP Range requests at
points spread evenly across the whole file -- each one lands in a different
turbine's block -- and keeps up to PER_ASSET_CAP consecutive rows from each.
That subsample is cached to data/ so later runs don't re-fetch. This is
subsampling real data at several real points in time, not synthesizing data.

Two earlier approaches failed before this one:
  1. pandas' own chunked CSV reader directly against the URL (chunksize=...)
     stalled unpredictably in CI with no output and got killed by the
     runner (exit 143).
  2. A single large bounded prefix (one contiguous slice from the start of
     the file) was fast and reliable, but -- because of the sequential
     layout above -- only ever captured ONE turbine, useless for a
     multi-turbine holdout.
This version's small, spread-out, explicitly-timed-out requests are both
fast/reliable AND turbine-diverse.

Label: status_type_id == 0 means normal operation; any other code is a
logged event/anomaly. y = 1 for "not normal right now".

Leakage guards, same shape as backtest.py:
  1. Turbine (asset_id) holdout -- test turbines never appear in training.
  2. Cycle causality -- rolling features at time t use only that turbine's
     own rows at time <= t.

Run:  python train_wind_turbine.py
"""
import io
import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
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
# Only the columns the model uses, gzipped: the full 957-column subsample was
# 95 MB -- right at GitHub's 100 MB per-file limit.
SUBSAMPLE_CSV = ROOT / "data" / "wind_turbine_subsample.csv.gz"

PER_ASSET_CAP = 1500          # rows kept per turbine
N_ASSETS_TARGET = 15          # number of spread-out points sampled across the file
OFFSET_CHUNK_BYTES = 10_000_000  # ~10MB per sample point -> comfortably > PER_ASSET_CAP rows
REQUEST_TIMEOUT = (15, 60)    # (connect, read) seconds -- fail loud instead of hanging
ROLL_WINDOW = 6                # backward-only rolling window, in readings
SEED = 42


def fetch_range(start: int, length: int) -> bytes:
    resp = requests.get(SOURCE_URL, headers={"Range": f"bytes={start}-{start + length - 1}"},
                        timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.content


def build_subsample() -> pd.DataFrame:
    """N_ASSETS_TARGET small, explicitly-timed-out Range requests at points
    spread evenly across the real ~12GB file -- fast and reliable (each
    request is a few MB with its own timeout), and turbine-diverse (the
    file's sequential-by-turbine layout means spread-out points land in
    different turbines' blocks)."""
    total_size = int(requests.head(SOURCE_URL, allow_redirects=True,
                                   timeout=REQUEST_TIMEOUT).headers.get("Content-Length", 0))
    if not total_size:  # HEAD didn't give a size (e.g. chunked encoding) -- ask via Range instead
        resp = requests.get(SOURCE_URL, headers={"Range": "bytes=0-1"}, timeout=REQUEST_TIMEOUT)
        total_size = int(resp.headers["Content-Range"].split("/")[-1])
    print(f"Real source file is {total_size / 1e9:.1f}GB total.", flush=True)

    # The real header turned out to be 957 columns wide (far more than an
    # early truncated preview suggested) -- fetch generously so it's never
    # cut mid-header, which would silently misalign every parsed row.
    header = fetch_range(0, 200_000).decode("utf-8", errors="ignore").split("\n")[0]
    columns = header.split(",")
    print(f"Header has {len(columns)} columns.", flush=True)

    start = time.time()
    kept, seen_assets = [], set()
    offsets = [int(total_size * i / N_ASSETS_TARGET) for i in range(N_ASSETS_TARGET)]
    for i, off in enumerate(offsets):
        print(f"  Sampling point {i + 1}/{len(offsets)} (~{off / 1e9:.2f}GB into the file, "
              f"{time.time() - start:.0f}s elapsed)...", flush=True)
        raw = fetch_range(off, OFFSET_CHUNK_BYTES)
        text = raw.decode("utf-8", errors="ignore")
        lines = text.split("\n")[1:-1]  # drop the partial first & last line (mid-file landing)
        if not lines:
            continue
        chunk = pd.read_csv(io.StringIO("\n".join(lines)), names=columns, header=None,
                            low_memory=False)
        for asset_id, group in chunk.groupby("asset_id"):
            if asset_id in seen_assets:
                continue
            seen_assets.add(asset_id)
            kept.append(group.head(PER_ASSET_CAP))
    if not kept:
        raise RuntimeError("Sampled 0 rows from the real source -- check SOURCE_URL / Range support")

    out = pd.concat(kept, ignore_index=True)
    print(f"Subsample: {len(out)} rows across {out['asset_id'].nunique()} real turbines, "
          f"fetched in {time.time() - start:.0f}s.", flush=True)
    return out


def load_data() -> pd.DataFrame:
    SUBSAMPLE_CSV.parent.mkdir(exist_ok=True)
    if not SUBSAMPLE_CSV.exists():
        df = build_subsample()
        keep = ["time_stamp", "asset_id", "status_type_id"] + [c for c in df.columns
                                                               if c.endswith("_avg")]
        df[keep].to_csv(SUBSAMPLE_CSV, index=False, compression="gzip")
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
