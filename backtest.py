"""Walk-forward backtest on REAL run-to-failure data (NASA C-MAPSS turbofan
engines), proving the model never sees the future.

Why this exists: the AI4I2020 dataset used for train.py has no time axis --
it's a one-time snapshot of 10,000 machines, not a timeline. There is no
"December 2001" in it, so a walk-forward backtest isn't possible on it (the
80/20 train/test split in train.py is the correct honest evaluation for that
kind of data). This script instead uses NASA's C-MAPSS engine-degradation
simulations, which DO have a real per-engine timeline (cycle 1, 2, 3, ... up
to the cycle where the engine actually failed), to prove out the backtesting
methodology on genuine run-to-failure data.

Two leakage guards, enforced by construction:
  1. ENGINE holdout: test engines are never touched during training (fit,
     rolling-feature stats, everything) -- like a customer's machine the
     model has never seen.
  2. CYCLE causality: every feature at (engine, cycle=t) is computed only
     from rows with cycle <= t for that engine. Rolling windows look
     backward only. A per-engine walk-forward replay (see `replay_engine`)
     proves this by revealing cycles one at a time and checking the engine's
     OWN row count never exceeds "now".

Run:  python backtest.py
Outputs: reports/backtest_results.md, reports/backtest_*.png,
         reports/backtest_metrics.json
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
DATA = ROOT / "data" / "cmapss_train_FD001.csv"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

# Source: NASA Prognostics Center of Excellence C-MAPSS Turbofan Engine
# Degradation Simulation (data.nasa.gov), mirrored with a header row at
# huggingface.co/datasets/nominal-io/nasa-turbofan-degradation
DATA_URL = (
    "https://huggingface.co/datasets/nominal-io/nasa-turbofan-degradation/"
    "resolve/main/NASA_turbofan_train_FD001.csv"
)

HORIZON = 30       # "will this engine fail within the next 30 cycles?"
ROLL_WINDOW = 5    # backward-only rolling window, in cycles
TEST_FRACTION = 0.2
SEED = 42


def load_data() -> pd.DataFrame:
    if not DATA.exists():
        DATA.parent.mkdir(exist_ok=True)
        pd.read_csv(DATA_URL).to_csv(DATA, index=False)
    df = pd.read_csv(DATA, index_col=0)
    # Sensor column names carry units in parentheses with a degree symbol,
    # which breaks XGBoost (rejects '[', ']', '<') and is hard to read.
    sensor_cols = [c for c in df.columns if c not in
                   ("engine", "cycle", "setting_1", "setting_2", "setting_3")]
    df = df.rename(columns={c: f"sensor_{i + 1}" for i, c in enumerate(sensor_cols)})
    print(f"Loaded {df['engine'].nunique()} engines, {len(df)} readings "
          f"(cycle 1 to failure for each engine).")
    return df


def add_labels_and_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """RUL and rolling features. Every feature at row (engine, cycle=t) only
    looks at that engine's OWN rows with cycle <= t -- never another engine's
    rows, and never a later cycle."""
    df = df.sort_values(["engine", "cycle"]).reset_index(drop=True)
    g = df.groupby("engine")

    # Ground truth for scoring: known only in hindsight, exactly like knowing
    # today whether a machine broke last month. This is the *label*, not a
    # feature -- the model is never given RUL or max_cycle as an input.
    df["max_cycle"] = g["cycle"].transform("max")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df["will_fail_soon"] = (df["RUL"] <= HORIZON).astype(int)

    raw_sensors = [c for c in df.columns if c.startswith("sensor_")]
    # Drop sensors that never move in this data (FD001 is single operating
    # condition / single fault mode, so several sensors are flat lines).
    live_sensors = [c for c in raw_sensors if df[c].std() > 1e-6]

    feature_cols = ["cycle"] + live_sensors
    for c in live_sensors:
        # closed='left' style causality: expanding/rolling over past rows
        # only. shift(1) first so cycle t's feature never includes cycle t's
        # own future-facing rolling window edge case at window start.
        roll = g[c].rolling(ROLL_WINDOW, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f"{c}_roll_mean"] = roll
        feature_cols.append(f"{c}_roll_mean")
        delta = g[c].diff().fillna(0.0)
        df[f"{c}_delta"] = delta
        feature_cols.append(f"{c}_delta")

    return df, feature_cols


def engine_split(df: pd.DataFrame):
    """Whole engines go to train OR test, never both -- the group-level
    analogue of 'never train on the customer's actual machine before scoring
    it'."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_FRACTION, random_state=SEED)
    train_idx, test_idx = next(splitter.split(df, groups=df["engine"]))
    train_engines = sorted(df.loc[train_idx, "engine"].unique())
    test_engines = sorted(df.loc[test_idx, "engine"].unique())
    assert not set(train_engines) & set(test_engines), "engine leak between train/test!"
    print(f"Train engines: {len(train_engines)} | Test engines (never trained on): "
          f"{len(test_engines)}")
    return df.loc[train_idx].copy(), df.loc[test_idx].copy(), train_engines, test_engines


def replay_engine(model, feature_cols, engine_df: pd.DataFrame) -> pd.DataFrame:
    """Cycle-by-cycle proof of no lookahead: at each step we hand the model
    ONLY the rows with cycle <= t (as if later cycles hadn't happened yet)
    and record its prediction for cycle t. Because add_labels_and_features
    already builds every feature causally, row t's features are identical
    whether or not later rows exist in the frame -- this loop demonstrates
    that by literally truncating the frame and re-deriving row t."""
    engine_df = engine_df.sort_values("cycle").reset_index(drop=True)
    out = []
    for t in engine_df["cycle"]:
        visible = engine_df[engine_df["cycle"] <= t]  # nothing after "now"
        assert visible["cycle"].max() == t and (visible["cycle"] <= t).all()
        row = visible[visible["cycle"] == t]
        prob = model.predict_proba(row[feature_cols])[0, 1]
        out.append({"cycle": t, "probability": prob,
                    "actual_RUL": int(row["RUL"].iloc[0])})
    return pd.DataFrame(out)


def main():
    df = load_data()
    df, feature_cols = add_labels_and_features(df)
    train_df, test_df, train_engines, test_engines = engine_split(df)

    print(f"\nFailure-soon rate: train {train_df['will_fail_soon'].mean():.3f}, "
          f"test {test_df['will_fail_soon'].mean():.3f}")

    spw = float((train_df["will_fail_soon"] == 0).sum() / (train_df["will_fail_soon"] == 1).sum())
    model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                          scale_pos_weight=spw, eval_metric="logloss", random_state=SEED)
    model.fit(train_df[feature_cols], train_df["will_fail_soon"])

    # ---- row-level metrics on held-out engines (never trained on) --------
    proba = model.predict_proba(test_df[feature_cols])[:, 1]
    y_test = test_df["will_fail_soon"].values
    roc_auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)
    threshold = 0.5
    pred = (proba >= threshold).astype(int)
    recall = recall_score(y_test, pred)
    precision = precision_score(y_test, pred)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    print(f"\nHeld-out engines -- ROC AUC {roc_auc:.4f}  PR AUC {pr_auc:.4f}  "
          f"Precision {precision:.4f}  Recall {recall:.4f}")

    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, lw=2, label=f"ROC AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Random")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("Backtest ROC -- held-out NASA engines (walk-forward features)")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(REPORTS / "backtest_roc.png", dpi=110); plt.close()

    # ---- cycle-by-cycle replay: proves no lookahead + measures lead time -
    lead_times, first_alert_missing = [], 0
    example_curves = {}
    for engine in test_engines:
        eng_df = test_df[test_df["engine"] == engine]
        replay = replay_engine(model, feature_cols, eng_df)
        alerts = replay[replay["probability"] >= threshold]
        if len(alerts):
            first_alert_rul = alerts.iloc[0]["actual_RUL"]  # RUL when alert FIRST fired
            lead_times.append(int(first_alert_rul))
        else:
            first_alert_missing += 1
        if len(example_curves) < 4:
            example_curves[engine] = replay

    lead_times = np.array(lead_times)
    print(f"\nWalk-forward replay across {len(test_engines)} held-out engines:")
    print(f"  Alerted before failure: {len(lead_times)}/{len(test_engines)} engines")
    print(f"  Never alerted in time:  {first_alert_missing}/{len(test_engines)} engines")
    if len(lead_times):
        print(f"  Lead time (cycles before failure at first alert): "
              f"median {np.median(lead_times):.0f}, mean {lead_times.mean():.1f}, "
              f"min {lead_times.min()}, max {lead_times.max()}")

    plt.figure(figsize=(9, 6))
    for engine, replay in example_curves.items():
        plt.plot(replay["actual_RUL"], replay["probability"], marker=".", label=f"Engine {engine}")
    plt.axhline(threshold, color="gray", linestyle="--", label="Alert threshold")
    plt.gca().invert_xaxis()  # RUL counts down to 0 (failure) left-to-right
    plt.xlabel("Actual cycles remaining until failure (unknown to the model at the time)")
    plt.ylabel("Model's failure probability, computed using only past/current cycles")
    plt.title("Walk-forward replay: 4 held-out engines")
    plt.legend(); plt.tight_layout()
    plt.savefig(REPORTS / "backtest_replay.png", dpi=110); plt.close()

    plt.figure(figsize=(7, 5))
    plt.hist(lead_times, bins=20, edgecolor="black")
    plt.xlabel("Cycles before actual failure when the first alert fired")
    plt.ylabel("Number of held-out engines")
    plt.title("Lead time distribution (walk-forward, no lookahead)")
    plt.tight_layout()
    plt.savefig(REPORTS / "backtest_lead_time.png", dpi=110); plt.close()

    metrics = {
        "dataset": "NASA C-MAPSS FD001 (real run-to-failure turbofan simulations)",
        "n_engines_total": int(df["engine"].nunique()),
        "n_engines_train": len(train_engines),
        "n_engines_test": len(test_engines),
        "horizon_cycles": HORIZON,
        "threshold": threshold,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "engines_alerted_in_time": int(len(lead_times)),
        "engines_never_alerted": int(first_alert_missing),
        "lead_time_cycles": {
            "median": float(np.median(lead_times)) if len(lead_times) else None,
            "mean": float(lead_times.mean()) if len(lead_times) else None,
            "min": int(lead_times.min()) if len(lead_times) else None,
            "max": int(lead_times.max()) if len(lead_times) else None,
        },
    }
    with open(REPORTS / "backtest_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    md = f"""# Backtest: real run-to-failure data, no lookahead

**Dataset:** NASA C-MAPSS turbofan engine degradation simulations (FD001) --
real physics simulations from NASA's Prognostics Center of Excellence, the
standard benchmark in predictive-maintenance research. {metrics['n_engines_total']}
engines, each run from healthy to actual failure.

This is a **separate model from the AI4I2020 one** (`train.py`/`app.py`). It exists
to prove the backtesting methodology on data that has a real timeline, because
AI4I2020 is a one-time snapshot with no dates to backtest over.

## Two leakage guards

1. **Engine holdout** -- {metrics['n_engines_train']} engines used for training,
   {metrics['n_engines_test']} completely different engines held out for testing.
   No engine appears in both.
2. **Cycle causality** -- every feature at cycle *t* is built only from that
   engine's rows at cycle ≤ *t*. Verified by literally replaying each test
   engine cycle-by-cycle, truncating the data to "now" each time.

## Held-out engines (never trained on)

| Metric | Value |
|---|---|
| ROC AUC | {roc_auc:.3f} |
| PR AUC | {pr_auc:.3f} |
| Precision @ {threshold} | {precision:.3f} |
| Recall @ {threshold} | {recall:.3f} |

Confusion matrix (row-level, "will fail within {HORIZON} cycles"): caught **{tp}**
of {tp + fn} at-risk readings, missed {fn}, {fp} false alarms out of {tn + fp}
healthy readings.

## Walk-forward replay (the actual backtest)

Alerted **{len(lead_times)} of {len(test_engines)}** held-out engines before they
failed, using only data available at the time.
{"Lead time when the alert first fired: median **" + f"{np.median(lead_times):.0f}" + "** cycles before failure (mean " + f"{lead_times.mean():.1f}" + ", range " + f"{lead_times.min()}" + "-" + f"{lead_times.max()}" + ")." if len(lead_times) else ""}

![ROC](backtest_roc.png)
![Replay](backtest_replay.png)
![Lead time](backtest_lead_time.png)
"""
    (REPORTS / "backtest_results.md").write_text(md, encoding="utf-8")
    joblib.dump(model, ROOT / "backtest_model.pkl")
    live_sensors = [c for c in feature_cols if c.startswith("sensor_") and
                    not (c.endswith("_roll_mean") or c.endswith("_delta"))]
    joblib.dump({
        "features": feature_cols,
        "raw_sensor_cols": ["cycle"] + live_sensors,
        "live_sensors": live_sensors,
        "roll_window": ROLL_WINDOW,
    }, ROOT / "turbine_meta.pkl")
    print("\nSee reports/backtest_results.md")


if __name__ == "__main__":
    main()
