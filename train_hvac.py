"""Trains the model for machine_types['hvac'] -- packaged rooftop units (RTUs),
the most common HVAC equipment in small commercial buildings.

Data: LBNL Fault Detection and Diagnostics Data Sets (U.S. DOE; LBNL, ORNL,
NREL), DOI 10.25984/1881324 -- the EXPERIMENTAL rooftop-unit subset: a real
Trane YCD150 12.5-ton RTU serving a 3,200 sq ft two-story test building at Oak
Ridge National Laboratory. Faults were physically imposed one day at a time
(4 fault types x 4 intensities x 4 seasons = 48 fault-days) and the unit was
also run fault-free in every season. 1-minute data.

Faults: outdoor-air damper stuck (5/10/50/100%), incorrect economizer setpoint
(+/-2, +/-4 C), supply-air temperature sensor bias (+/-2, +/-4 C).

Leakage guard -- LEAVE-ONE-SEASON-OUT: train on three seasons' test days,
score the fourth season's days, rotate through all four. Every reported
number comes from days (and weather) the model never trained on. Rolling
features only look backward within each day's file.

Run:  python train_hvac.py
"""
import io
import json
import re
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
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from pm_features import HVAC_FEATURES, HVAC_RAW, hvac_features

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)
SOURCE_URL = ("https://fdddata.lbl.gov/data/Simulated_LBNL_FDD_Data_Sets_RTU/"
              "LBNL_FDD_Data_Sets_RTU.zip")  # folder name says "Simulated" but the zip
                                              # also holds the ORNL experimental set used here
CACHE = ROOT / "data" / "hvac_ornl_rtu.csv.gz"
SEASONS = ["Fall_2020", "Spring_2021", "Summer_2021", "Winter_2022"]
FAULT_NAMES = {"OA_damper_stuck": "oa_damper_stuck", "Inc_Eco_SP": "economizer_setpoint",
               "SA_temp_bias": "sa_temp_sensor_bias"}
CLASSES = ["fault_free", "oa_damper_stuck", "economizer_setpoint", "sa_temp_sensor_bias"]
MAX_FALSE_ALARM = 0.2   # alert budget: at most ~1 in 5 fault-free days flagged
SMOOTH_MINUTES = 15     # the app averages this many recent minutes before alerting
SEED = 42


def load_data() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_csv(CACHE, parse_dates=["Datetime"])
    print(f"Downloading {SOURCE_URL} ...", flush=True)
    resp = requests.get(SOURCE_URL, timeout=(15, 300))
    resp.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    frames = []
    for name in zf.namelist():
        m = re.fullmatch(r"ORNL_RTU/(.+)_(Fall_2020|Spring_2021|Summer_2021|Winter_2022)\.csv", name)
        if not m:
            continue
        stem, season = m.group(1), m.group(2)
        if stem == "ERTU":
            fault, intensity = "fault_free", "0"
        else:
            fm = re.fullmatch(r"(OA_damper_stuck|Inc_Eco_SP|SA_temp_bias)_(-?\d+)", stem)
            fault, intensity = FAULT_NAMES[fm.group(1)], fm.group(2)
        df = pd.read_csv(zf.open(name), na_values=["NAN", "NaN"])
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df["hour"] = df["Datetime"].dt.hour + df["Datetime"].dt.minute / 60
        keep = ["Datetime"] + [c for c in HVAC_RAW if c in df.columns]
        df = df[keep].copy()
        df["source_file"] = name.split("/")[-1]
        df["season"] = season
        df["fault"] = fault
        df["intensity"] = intensity
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data.to_csv(CACHE, index=False, compression="gzip")
    print(f"Cached {len(data)} rows from {data['source_file'].nunique()} real test files", flush=True)
    return data


def build(data: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, g in data.sort_values("Datetime").groupby("source_file", sort=False):
        f = hvac_features(g)
        f["day"] = f["source_file"] + "|" + f["Datetime"].dt.date.astype(str)
        parts.append(f)
    df = pd.concat(parts, ignore_index=True)
    df["y"] = (df["fault"] != "fault_free").astype(int)
    df["y_class"] = df["fault"].map({c: i for i, c in enumerate(CLASSES)})
    return df


def make_models(y_train):
    spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    binary = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.9,
                           colsample_bytree=0.9, scale_pos_weight=spw, eval_metric="logloss",
                           random_state=SEED)
    diag = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.9,
                         colsample_bytree=0.9, eval_metric="mlogloss", random_state=SEED)
    return binary, diag


def fit_predict(train: pd.DataFrame, test: pd.DataFrame):
    binary, diag = make_models(train["y"])
    binary.fit(train[HVAC_FEATURES], train["y"])
    diag.fit(train[HVAC_FEATURES], train["y_class"])
    return (binary.predict_proba(test[HVAC_FEATURES])[:, 1],
            diag.predict(test[HVAC_FEATURES]))


def day_scores(df: pd.DataFrame, prob: np.ndarray) -> pd.DataFrame:
    """One score per unit-day: mean fault probability over occupied minutes."""
    occ = (df["OCCU_MOD"] == 1).values
    d = df.loc[occ, ["day", "y", "fault", "season"]].assign(prob=prob[occ])
    return d.groupby("day").agg(y=("y", "first"), fault=("fault", "first"),
                                season=("season", "first"), score=("prob", "mean"))


def choose_threshold(days: pd.DataFrame, max_false_alarm: float = MAX_FALSE_ALARM) -> float:
    """Lowest day-score threshold whose false-alarm rate on fault-free days is
    within budget (lowest = catches the most fault days)."""
    ok = np.sort(days.loc[days["y"] == 0, "score"].values)
    if len(ok) == 0:
        return 0.5
    allowed = int(np.floor(max_false_alarm * len(ok)))  # fault-free days we may flag
    return float(ok[len(ok) - 1 - allowed] + 1e-9)


def detection_minutes(day_df: pd.DataFrame, prob: np.ndarray, threshold: float) -> float | None:
    """Minutes after the building's occupied period starts (07:00) until the
    15-minute rolling fault probability first crosses the threshold."""
    s = pd.Series(prob, index=day_df.index).rolling(15, min_periods=1).mean()
    occ = day_df["OCCU_MOD"] == 1
    hit = s[occ & (s >= threshold)]
    if hit.empty:
        return None
    first = day_df.loc[hit.index[0], "Datetime"]
    start = first.normalize() + pd.Timedelta(hours=7)
    return max(0.0, (first - start).total_seconds() / 60)


def main():
    raw = load_data()
    df = build(raw)
    print(f"{len(df)} minutes of real RTU data, {df['day'].nunique()} unit-days, "
          f"fault-minute share {df['y'].mean():.2f}", flush=True)

    oof = np.full(len(df), np.nan)
    oof_cls = np.full(len(df), -1)
    season_thr = {}
    for season in SEASONS:
        tr, te = df["season"] != season, df["season"] == season
        # Nested: choose the alert threshold using ONLY the training seasons,
        # by scoring each of them with a model trained on the other two.
        inner = []
        for inner_season in [s for s in SEASONS if s != season]:
            itr = tr & (df["season"] != inner_season)
            ite = df["season"] == inner_season
            p, _ = fit_predict(df[itr], df[ite])
            inner.append(day_scores(df[ite], p))
        season_thr[season] = choose_threshold(pd.concat(inner))
        p, c = fit_predict(df[tr], df[te])
        oof[te.values], oof_cls[te.values] = p, c
        print(f"  held out {season}: {te.sum()} minutes scored by a model that never saw it "
              f"(threshold {season_thr[season]:.3f}, chosen on training seasons only)", flush=True)
    df["prob"], df["pred_class"] = oof, oof_cls

    occ = df["OCCU_MOD"] == 1
    roc_occ = roc_auc_score(df.loc[occ, "y"], df.loc[occ, "prob"])
    pr_occ = average_precision_score(df.loc[occ, "y"], df.loc[occ, "prob"])
    roc_all = roc_auc_score(df["y"], df["prob"])

    # Day-level: what an operator actually sees -- "was this unit-day flagged?"
    days = day_scores(df, df["prob"].values)
    days["diag"] = (df[occ].groupby("day")["pred_class"]
                    .agg(lambda s: CLASSES[int(s.mode().iloc[0])]))
    days["threshold"] = days["season"].map(season_thr)
    days["flagged"] = days["score"] >= days["threshold"]
    day_auc = roc_auc_score(days["y"], days["score"])
    fault_days, ok_days = days[days["y"] == 1], days[days["y"] == 0]
    day_recall = float(fault_days["flagged"].mean())
    day_false_alarm = float(ok_days["flagged"].mean()) if len(ok_days) else float("nan")
    per_fault = fault_days.groupby("fault")["flagged"].agg(["mean", "count"])
    diag_acc_days = float((fault_days["diag"] == fault_days["fault"]).mean())

    delays = []
    for day, g in df[df["y"] == 1].groupby("day"):
        d = detection_minutes(g, g["prob"].values, season_thr[g["season"].iloc[0]])
        if d is not None:
            delays.append(d)
    delays = np.array(delays)

    cls_mask = occ
    macro_f1 = f1_score(df.loc[cls_mask, "y_class"], df.loc[cls_mask, "pred_class"], average="macro")
    cm = confusion_matrix(df.loc[cls_mask, "y_class"], df.loc[cls_mask, "pred_class"],
                          labels=range(len(CLASSES)))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(cmap="Blues", xticks_rotation=25)
    plt.title("HVAC RTU diagnosis -- leave-one-season-out (occupied minutes)")
    plt.tight_layout(); plt.savefig(REPORTS / "hvac_confusion_matrix.png", dpi=110); plt.close()

    print(f"\nMinute-level ROC AUC (occupied hours) {roc_occ:.3f}, PR AUC {pr_occ:.3f}; "
          f"all hours {roc_all:.3f}")
    print(f"Day-level: flagged {fault_days['flagged'].sum()}/{len(fault_days)} fault days "
          f"({day_recall:.0%}); false alarms on {ok_days['flagged'].sum()}/{len(ok_days)} "
          f"fault-free days")
    print(per_fault)

    # Final models on everything, for the app.
    binary, diag = make_models(df["y"])
    binary.fit(df[HVAC_FEATURES], df["y"])
    diag.fit(df[HVAC_FEATURES], df["y_class"])
    joblib.dump(binary, ROOT / "hvac_model.pkl")
    joblib.dump(diag, ROOT / "hvac_diag_model.pkl")
    medians = df[[c for c in HVAC_RAW if c in df.columns]].median().to_dict()
    # Deployed threshold: chosen on the out-of-fold day scores above (every one
    # produced by a model that never saw that day), same false-alarm budget.
    final_thr = choose_threshold(days)
    joblib.dump({"features": HVAC_FEATURES, "raw": HVAC_RAW, "classes": CLASSES,
                 "medians": medians, "threshold": final_thr, "smooth_minutes": SMOOTH_MINUTES},
                ROOT / "hvac_meta.pkl")

    metrics = {
        "dataset": "LBNL FDD Data Sets - ORNL experimental RTU (real Trane YCD150), DOI 10.25984/1881324",
        "validation": "leave-one-season-out (4 folds); alert threshold chosen by nested "
                      "leave-one-season-out inside the training seasons only",
        "max_false_alarm_budget": MAX_FALSE_ALARM,
        "per_season_threshold": season_thr, "day_level_roc_auc": day_auc,
        "n_minutes": int(len(df)), "n_unit_days": int(df["day"].nunique()),
        "minute_roc_auc_occupied": roc_occ, "minute_pr_auc_occupied": pr_occ,
        "minute_roc_auc_all_hours": roc_all,
        "fault_days": int(len(fault_days)), "fault_days_flagged": int(fault_days["flagged"].sum()),
        "fault_free_days": int(len(ok_days)), "false_alarm_days": int(ok_days["flagged"].sum()),
        "per_fault_day_recall": {k: float(v) for k, v in per_fault["mean"].items()},
        "diagnosis_macro_f1_minutes": macro_f1, "diagnosis_accuracy_fault_days": diag_acc_days,
        "median_detection_minutes_after_0700": float(np.median(delays)) if len(delays) else None,
    }
    (REPORTS / "hvac_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    rows = "\n".join(f"| {k} | {int(r['count'])} | {r['mean']:.0%} |" for k, r in per_fault.iterrows())
    md = f"""# HVAC rooftop-unit model

**Dataset:** LBNL Fault Detection & Diagnostics Data Sets (U.S. DOE; LBNL, ORNL, NREL),
[DOI 10.25984/1881324](https://dx.doi.org/10.25984/1881324) -- the **experimental**
subset: a real Trane YCD150 12.5-ton rooftop unit at Oak Ridge National Laboratory's
Flexible Research Platform, with faults physically imposed one day at a time across four
seasons. {len(df):,} minutes, {df['day'].nunique()} unit-days.

**Leakage guard -- leave-one-season-out:** every number below comes from days scored by a
model trained only on the *other three seasons*. The alert threshold was also chosen without
the held-out season: by a second, nested leave-one-season-out inside the training seasons,
targeting at most {MAX_FALSE_ALARM:.0%} false alarms on fault-free days.

| Metric | Value |
|---|---|
| Day-level ROC AUC (fault day vs. fault-free day) | {day_auc:.3f} |
| Minute-level ROC AUC (occupied hours) | {roc_occ:.3f} |
| Minute-level PR AUC (occupied hours) | {pr_occ:.3f} |
| Fault days flagged | **{int(fault_days['flagged'].sum())} / {len(fault_days)}** ({day_recall:.0%}) |
| False alarms on fault-free days | {int(ok_days['flagged'].sum())} / {len(ok_days)} |
| Fault type correctly named (fault days) | {diag_acc_days:.0%} |
| Median detection time after occupied hours begin | {f"{np.median(delays):.0f} min" if len(delays) else "n/a"} |

Per fault type (day level):

| Fault | Fault days | Flagged |
|---|---|---|
{rows}

**Caveats.** In this experiment the damper and supply-air faults were imposed by changing
the unit's *control program*, so part of their signature is in the commanded signals
themselves. On a real building the same model sees commanded vs. measured values, which is
the stronger signal -- but it has only been validated on this one unit. Faults only show up
while the unit is running, so unoccupied night hours are excluded from the headline metrics.

![Confusion matrix](hvac_confusion_matrix.png)
"""
    (REPORTS / "hvac_results.md").write_text(md, encoding="utf-8")
    print("\nSaved hvac_model.pkl, hvac_diag_model.pkl; see reports/hvac_results.md")


if __name__ == "__main__":
    main()
