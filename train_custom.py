"""Train a model on a CUSTOMER'S OWN data -- the step that turns a pilot into a
product. Given the two files almost any plant can export, it builds labels,
validates honestly, and registers a new machine type the app can score
immediately (it shows up in the machine-type dropdown after a restart).

Inputs (see pilot/DATA_REQUEST.md and pilot/templates/):
  --sensors  CSV with columns: timestamp, machine_id, <one column per sensor>
  --events   CSV with columns: machine_id, timestamp[, event_type]
             -- every breakdown / unplanned stop / failure-driven repair

Target: "will this machine have a failure event within the next --horizon?"
(e.g. 24h, 7d). Rows too close to the end of the data to know the answer
are dropped rather than guessed.

Leakage guards:
  * 4+ machines: LEAVE-MACHINES-OUT (grouped folds) -- scored on machines the
    model never trained on.
  * fewer machines: CHRONOLOGICAL -- each machine's last 30% of history is
    scored by a model trained on everyone's earlier history, with a
    --horizon gap so no label straddles the split.
  * Features only look backward (pm_features.generic_features).

Refuses to train (and says why) with fewer than MIN_EVENTS failure events --
a model built on 2 breakdowns is noise, and it's better to say so.

Example:
  python train_custom.py --key packaging_line --name "Acme packaging line" \\
      --sensors acme_sensors.csv --events acme_failures.csv --horizon 24h
"""
import argparse
import json
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

from pm_features import generic_feature_names, generic_features

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
MIN_EVENTS = 5
THRESHOLD = 0.5
SEED = 42


def load(sensors_csv: Path, events_csv: Path):
    s = pd.read_csv(sensors_csv)
    e = pd.read_csv(events_csv)
    for name, df, need in [("sensors", s, {"timestamp", "machine_id"}),
                           ("events", e, {"timestamp", "machine_id"})]:
        missing = need - set(df.columns)
        if missing:
            raise SystemExit(f"{name} CSV is missing required column(s): {sorted(missing)}")
    s["timestamp"] = pd.to_datetime(s["timestamp"])
    e["timestamp"] = pd.to_datetime(e["timestamp"])
    s["machine_id"] = s["machine_id"].astype(str)
    e["machine_id"] = e["machine_id"].astype(str)
    sensors = [c for c in s.columns if c not in ("timestamp", "machine_id")
               and pd.to_numeric(s[c], errors="coerce").notna().mean() > 0.5]
    if not sensors:
        raise SystemExit("No numeric sensor columns found in the sensors CSV.")
    return s.sort_values(["machine_id", "timestamp"]), e.sort_values("timestamp"), sensors


def label(s: pd.DataFrame, e: pd.DataFrame, horizon: pd.Timedelta) -> pd.DataFrame:
    """y = 1 if the machine's next failure event is within (t, t + horizon]."""
    parts = []
    for mid, g in s.groupby("machine_id", sort=False):
        ev = e.loc[e["machine_id"] == mid, "timestamp"].sort_values().values
        t = g["timestamp"].values
        if len(ev) == 0:  # a machine that never broke down: every reading is "no failure ahead"
            nxt = np.full(len(t), np.datetime64("NaT"), dtype="datetime64[ns]")
        else:
            nxt_idx = np.searchsorted(ev, t, side="right")
            nxt = np.where(nxt_idx < len(ev), ev[np.minimum(nxt_idx, len(ev) - 1)],
                           np.datetime64("NaT"))
        g = g.copy()
        g["time_to_event"] = pd.to_datetime(nxt) - g["timestamp"]
        g["y"] = (g["time_to_event"] <= horizon).astype(int)
        # No later event: only "no failure" if we can see a full horizon ahead.
        censored = g["time_to_event"].isna() & (g["timestamp"] > g["timestamp"].max() - horizon)
        g = g[~censored]
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key", required=True, help="short id, e.g. packaging_line")
    ap.add_argument("--name", required=True, help="display name in the app")
    ap.add_argument("--sensors", required=True, type=Path)
    ap.add_argument("--events", required=True, type=Path)
    ap.add_argument("--horizon", default="24h", help="look-ahead window, e.g. 12h, 24h, 7d")
    ap.add_argument("--roll", type=int, default=10, help="short rolling window, in readings")
    ap.add_argument("--long-roll", type=int, default=100, help="baseline window, in readings")
    ap.add_argument("--source", default="customer-provided sensor and maintenance logs",
                    help="how the data was obtained, shown in the app")
    ap.add_argument("--out", type=Path, default=ROOT / "custom_models")
    args = ap.parse_args()

    key = "custom_" + re.sub(r"[^a-z0-9_]", "_", args.key.lower()).strip("_")
    horizon = pd.Timedelta(args.horizon)
    s, e, sensors = load(args.sensors, args.events)
    n_events = int(e["machine_id"].isin(s["machine_id"].unique()).sum())
    print(f"{len(s)} readings from {s['machine_id'].nunique()} machines, {len(sensors)} sensors, "
          f"{n_events} failure events", flush=True)
    if n_events < MIN_EVENTS:
        raise SystemExit(f"Only {n_events} failure events -- need at least {MIN_EVENTS} to learn "
                         f"anything real. Collect more history (or more machines) first.")

    parts = [generic_features(g, sensors, args.roll, args.long_roll)
             for _, g in s.groupby("machine_id", sort=False)]
    df = label(pd.concat(parts, ignore_index=True), e, horizon)
    feats = generic_feature_names(sensors)
    df[feats] = df[feats].replace([np.inf, -np.inf], np.nan)
    print(f"{len(df)} labelled readings, {df['y'].mean():.1%} within {horizon} of a failure",
          flush=True)

    machines = df["machine_id"].unique()
    oof = np.full(len(df), np.nan)
    if len(machines) >= 4:
        scheme = f"leave-machines-out, {min(5, len(machines))} folds"
        for tr, te in GroupKFold(n_splits=min(5, len(machines))).split(df, groups=df["machine_id"]):
            m = _model(df.iloc[tr]["y"])
            m.fit(df.iloc[tr][feats], df.iloc[tr]["y"])
            oof[te] = m.predict_proba(df.iloc[te][feats])[:, 1]
    else:
        scheme = "chronological: last 30% of each machine's history, with a horizon gap"
        cut = df.groupby("machine_id")["timestamp"].transform(lambda t: t.quantile(0.7))
        tr = df["timestamp"] <= cut - horizon
        te = df["timestamp"] > cut
        m = _model(df.loc[tr, "y"])
        m.fit(df.loc[tr, feats], df.loc[tr, "y"])
        oof[te.values] = m.predict_proba(df.loc[te, feats])[:, 1]
    scored = ~np.isnan(oof)
    y = df.loc[scored, "y"].values
    p = oof[scored]
    both = y.min() != y.max()
    roc = roc_auc_score(y, p) if both else float("nan")
    pr = average_precision_score(y, p) if both else float("nan")

    # Event level: was an alert raised in the horizon before each scored failure?
    warned, leads, total = 0, [], 0
    df["prob"] = oof
    for _, ev in e.iterrows():
        g = df[(df["machine_id"] == ev["machine_id"]) & df["prob"].notna()
               & (df["timestamp"] < ev["timestamp"])
               & (df["timestamp"] >= ev["timestamp"] - horizon)]
        if g.empty:
            continue
        total += 1
        hit = g[g["prob"] >= THRESHOLD]
        if len(hit):
            warned += 1
            leads.append((ev["timestamp"] - hit["timestamp"].min()) / pd.Timedelta(hours=1))
    print(f"\nValidation ({scheme}): ROC AUC {roc:.3f}, PR AUC {pr:.3f} "
          f"(random = {y.mean():.3f}); warned before {warned}/{total} failures", flush=True)

    final = _model(df["y"])
    final.fit(df[feats], df["y"])
    out = args.out / key
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, out / "model.pkl")
    medians = df.loc[df["y"] == 0, sensors].median().to_dict()
    joblib.dump({"features": feats, "raw": sensors, "roll": args.roll, "long_roll": args.long_roll,
                 "medians": medians, "threshold": THRESHOLD, "horizon": str(horizon)},
                out / "meta.pkl")
    ranges = {c: (float(df[c].quantile(0.01)), float(df[c].quantile(0.99))) for c in sensors}
    # Only call it "validated" if the held-out results would survive a customer's scrutiny.
    maturity = ("validated" if roc == roc and roc >= 0.8 and total and warned / total >= 0.5
                else "experimental")
    (out / "type.json").write_text(json.dumps({
        "key": key, "name": args.name, "maturity": maturity,
        "description": f"Custom model: warns of a failure within {horizon}.",
        "dataset": f"{args.source} -- {s['machine_id'].nunique()} machines, "
                   f"{n_events} failure events",
        "note": (f"Trained on this business's own history. Validation ({scheme}): ROC AUC "
                 f"{roc:.2f}, warned before {warned}/{total} failures. Feed readings "
                 f"continuously; rolling features use this machine's own recent readings."),
        "sensors": [{"key": c, "label": c, "unit": "", "min": lo, "max": hi,
                     "default": float(medians[c])} for c, (lo, hi) in ranges.items()],
    }, indent=2))
    metrics = {"key": key, "validation": scheme, "roc_auc": roc, "pr_auc": pr,
               "positive_rate": float(y.mean()), "failures_warned": f"{warned}/{total}",
               "median_warning_hours": float(np.median(leads)) if leads else None,
               "n_machines": int(s["machine_id"].nunique()), "n_events": n_events}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(f"\nRegistered machine type '{key}' in {out}. Restart the backend to use it.")
    return metrics


def _model(y) -> XGBClassifier:
    spw = float((y == 0).sum() / max((y == 1).sum(), 1))
    return XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.06, subsample=0.9,
                         colsample_bytree=0.9, scale_pos_weight=spw, eval_metric="logloss",
                         random_state=SEED)


if __name__ == "__main__":
    main()
