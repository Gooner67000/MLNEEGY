"""Trains the model for machine_types['robotic_arm'].

Data: UR3 CobotOps, UCI Machine Learning Repository (dataset 963),
https://archive.ics.uci.edu/dataset/963/ur3+cobotops -- real telemetry from a
Universal Robots UR3 collaborative robot (MODBUS/RTDE): per-joint current,
temperature and speed for all 6 joints, gripper current, operating cycle,
and the robot's own logged protective stops and grip losses. ~1 reading/s.

Target: "will a protective stop or grip loss START within the next HORIZON
readings (~10 s)?" -- a genuine look-ahead prediction, not detection of a stop
that's already happening (rows where the robot is already stopped are
excluded; you can't "predict" something that's in progress).

Leakage guard -- CHRONOLOGICAL WALK-FORWARD: the model trains on the robot's
earlier timeline and is scored on the later part it never saw, with a gap of
HORIZON readings between them so no label straddles the boundary. Rolling
features only look backward.

Run:  python train_robot.py
"""
import io
import json
import warnings
import zipfile
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
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier

from pm_features import ROBOT_FEATURES, ROBOT_RAW, robot_features

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)
SOURCE_URL = "https://archive.ics.uci.edu/static/public/963/ur3+cobotops.zip"
CACHE = ROOT / "data" / "robot_ur3_cobotops.csv.gz"
HORIZON = 10       # readings (~10 s)
TEST_FRACTION = 0.3
SEED = 42


def load_data() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_csv(CACHE, parse_dates=["Timestamp"])
    print(f"Downloading {SOURCE_URL} ...", flush=True)
    resp = requests.get(SOURCE_URL, timeout=(15, 120))
    resp.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    xlsx = [n for n in zf.namelist() if n.endswith(".xlsx")][0]
    df = pd.read_excel(io.BytesIO(zf.read(xlsx)))
    df = df.loc[:, [c for c in df.columns if isinstance(c, str) and not c.startswith("Unnamed")]]
    df = df.rename(columns={"Temperature_T0": "Temperature_J0", "cycle ": "cycle"})
    # Some source timestamps are wrapped in an extra pair of quote marks.
    df["Timestamp"] = pd.to_datetime(df["Timestamp"].astype(str).str.strip('"'),
                                     utc=True, format="ISO8601")
    df.to_csv(CACHE, index=False, compression="gzip")
    return df


def main():
    df = load_data().sort_values("Timestamp").reset_index(drop=True)
    for c in ["Robot_ProtectiveStop", "grip_lost"]:
        df[c] = df[c].astype(str).str.lower().eq("true")
    df["event"] = df["Robot_ProtectiveStop"] | df["grip_lost"]
    df["onset"] = df["event"] & ~df["event"].shift(1, fill_value=False)
    df["onset_type"] = np.where(df["onset"] & df["Robot_ProtectiveStop"], "protective_stop",
                                np.where(df["onset"], "grip_lost", ""))
    # y = an event starts in (t, t + HORIZON]; this row itself must be "running".
    # (int, not bool: pandas can't roll over a bool column)
    future = (df["onset"].astype(int)[::-1].rolling(HORIZON, min_periods=1).max()[::-1]
              .shift(-1, fill_value=0))
    df["y"] = future.astype(int)
    df = robot_features(df)
    usable = ~df["event"]
    print(f"{len(df)} readings, {int(df['onset'].sum())} event onsets "
          f"({int((df['onset_type'] == 'protective_stop').sum())} protective stops, "
          f"{int((df['onset_type'] == 'grip_lost').sum())} grip losses)", flush=True)

    split = int(len(df) * (1 - TEST_FRACTION))
    train = df.iloc[: split - HORIZON][usable.iloc[: split - HORIZON]]  # gap: no label straddles
    test = df.iloc[split:][usable.iloc[split:]]
    print(f"Train: readings 0-{split - HORIZON} (earlier timeline) | "
          f"Test: readings {split}-{len(df)} (later timeline, never seen)", flush=True)

    spw = float((train["y"] == 0).sum() / max((train["y"] == 1).sum(), 1))
    clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9,
                        colsample_bytree=0.9, scale_pos_weight=spw, eval_metric="logloss",
                        random_state=SEED)
    clf.fit(train[ROBOT_FEATURES], train["y"])
    proba = clf.predict_proba(test[ROBOT_FEATURES])[:, 1]
    y = test["y"].values
    roc = roc_auc_score(y, proba) if y.min() != y.max() else float("nan")
    pr = average_precision_score(y, proba) if y.max() else float("nan")
    base = float(y.mean())
    thr = 0.5
    rec = recall_score(y, proba >= thr, zero_division=0)
    prec = precision_score(y, proba >= thr, zero_division=0)

    # Event-level: for each onset in the test period, was the alert on during
    # the HORIZON readings before it, and how early did it first come on?
    full_p = pd.Series(np.nan, index=df.index)
    full_p.loc[test.index] = proba
    onsets = df.index[(df.index >= split) & df["onset"]]
    caught, leads, by_type = 0, [], {"protective_stop": [0, 0], "grip_lost": [0, 0]}
    for o in onsets:
        window = full_p.loc[max(split, o - HORIZON): o - 1].dropna()
        hit = (window >= thr)
        t = df.loc[o, "onset_type"]
        by_type[t][1] += 1
        if hit.any():
            caught += 1
            by_type[t][0] += 1
            leads.append(int(o - hit[hit].index[0]))
    print(f"\nWalk-forward test: ROC AUC {roc:.3f}, PR AUC {pr:.3f} (random = {base:.3f})")
    print(f"Events warned before they started: {caught}/{len(onsets)}; "
          f"median warning {np.median(leads) if leads else float('nan'):.0f} readings ahead")

    fpr, tpr, _ = roc_curve(y, proba)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, lw=2, label=f"ROC AUC = {roc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("UR3 robot -- walk-forward (later timeline, never trained on)")
    plt.legend(); plt.tight_layout(); plt.savefig(REPORTS / "robot_roc.png", dpi=110); plt.close()

    final = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9,
                          colsample_bytree=0.9,
                          scale_pos_weight=float((df.loc[usable, "y"] == 0).sum()
                                                 / max((df.loc[usable, "y"] == 1).sum(), 1)),
                          eval_metric="logloss", random_state=SEED)
    final.fit(df.loc[usable, ROBOT_FEATURES], df.loc[usable, "y"])
    joblib.dump(final, ROOT / "robot_model.pkl")
    joblib.dump({"features": ROBOT_FEATURES, "raw": ROBOT_RAW, "horizon": HORIZON,
                 "threshold": thr, "medians": df.loc[usable, ROBOT_RAW].median().to_dict()},
                ROOT / "robot_meta.pkl")

    metrics = {
        "dataset": "UR3 CobotOps (UCI #963), real Universal Robots UR3 telemetry",
        "validation": f"chronological walk-forward: first {1 - TEST_FRACTION:.0%} train, "
                      f"last {TEST_FRACTION:.0%} test, {HORIZON}-reading gap",
        "target": f"protective stop or grip loss starts within next {HORIZON} readings",
        "roc_auc": roc, "pr_auc": pr, "pr_auc_random_baseline": base,
        "precision": prec, "recall": rec, "threshold": thr,
        "events_in_test": int(len(onsets)), "events_warned": int(caught),
        "median_warning_readings": float(np.median(leads)) if leads else None,
        "by_type": {k: f"{v[0]}/{v[1]}" for k, v in by_type.items()},
    }
    (REPORTS / "robot_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    md = f"""# Robotic arm model

**Dataset:** [UR3 CobotOps](https://archive.ics.uci.edu/dataset/963/ur3+cobotops), UCI Machine
Learning Repository -- real telemetry from a Universal Robots UR3 collaborative robot:
current, temperature and speed of all 6 joints, gripper current, and the robot's own logged
**protective stops** and **grip losses**. {len(df):,} readings (~1/s).

**Task:** predict that a protective stop or grip loss will *start* within the next
{HORIZON} readings (~{HORIZON} s) -- a look-ahead warning, not detecting a stop already underway.

**Leakage guard -- chronological walk-forward:** trained on the first
{1 - TEST_FRACTION:.0%} of the robot's timeline, scored on the last {TEST_FRACTION:.0%}, with a
{HORIZON}-reading gap between them. Every feature uses only past readings.

| Metric (later timeline, never trained on) | Value |
|---|---|
| ROC AUC | {roc:.3f} |
| PR AUC | {pr:.3f} (random = {base:.3f}) |
| Precision / recall @ {thr} | {prec:.2f} / {rec:.2f} |
| Events warned before they started | **{caught} / {len(onsets)}** |
| Median warning lead | {f"{np.median(leads):.0f} readings" if leads else "n/a"} |
| Protective stops warned | {by_type['protective_stop'][0]} / {by_type['protective_stop'][1]} |
| Grip losses warned | {by_type['grip_lost'][0]} / {by_type['grip_lost'][1]} |

**Caveats.** One robot, one workcell, one program. Protective stops can be caused by things
no sensor sees coming (a person bumping the arm), so a perfect score isn't possible. Retrain
on a customer's own robot logs before relying on it.

![ROC](robot_roc.png)
"""
    (REPORTS / "robot_results.md").write_text(md, encoding="utf-8")
    print("\nSaved robot_model.pkl; see reports/robot_results.md")


if __name__ == "__main__":
    main()
