"""Trains the model for machine_types['rotating_equipment'] -- motors, pumps,
fans, compressors, gearboxes: anything whose dominant failure mode is bearing
wear.

Data: University of Ottawa Rolling-element Dataset (UORED-VAFCLS), Mendeley
Data doi:10.17632/y2px5tg92h.2, CC BY 4.0. 20 physical bearings (5 each with
an inner-race, outer-race, ball, or cage defect), each recorded in three
states: healthy -> developing fault -> faulty. Accelerometer at 42 kHz.
Features were extracted from the raw recordings by
tools/extract_mendeley_features.py (provenance + SHA-256 of every raw file in
data/provenance/bearing_uored.json).

This replaces the earlier CWRU-based model, which had only 10 independent
recordings and scored 38.8% on its 3 held-out ones.

Leakage guard -- LEAVE-BEARINGS-OUT: 5 folds, each holding out 4 whole
physical bearings (all three of their states). Every reported number is for
bearings the model never saw in any state.

Only accelerometer features are used -- the most common industrial vibration
sensor. The recordings' temperature channel is excluded: it has corrupted
values in some files and rises through each test sequence, so it would let a
model learn *when* a recording was made rather than *what state* the bearing
is in.

Order-tracking: this rig recorded at ~1700-1820 RPM (it varies by bearing --
confirmed by checking the raw files, not assumed). A model that only reads
fixed-Hz frequency bands implicitly assumes every customer's machine runs at
that same speed; a motor at 900 or 3600 RPM would have its fault energy land
in different Hz bins than what the model learned. ORDER_FEATURES express
energy as multiples of shaft speed instead (order_band_features() in
tools/extract_mendeley_features.py) -- the standard fix in real vibration
analysis. This dataset only covers ~1700-1820 RPM, so cross-speed
generalization to e.g. 900 or 3600 RPM is a methodological improvement, NOT a
validated result -- there's no data here to test it against. Order features
require the customer's shaft speed (rpm); when it isn't provided, the API
falls back to this dataset's median.

Run:  python train_bearing.py
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
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)
FEATURES_CSV = ROOT / "data" / "bearing_uored_features.csv.gz"
STATES = ["healthy", "developing", "faulty"]
TYPES = ["inner_race", "outer_race", "ball", "cage"]
FIXED_BAND_FEATURES = ["rms", "std", "peak", "p2p", "crest", "kurtosis", "skewness", "shape",
                       "impulse", "spec_cent", "spec_bw", "spec_ent",
                       "band0", "band1", "band2", "band3", "band4"]
ORDER_FEATURES = ["order0", "order1", "order2", "order3", "order4"]  # multiples of shaft speed
FEATURES = FIXED_BAND_FEATURES + ORDER_FEATURES + ["rpm"]
SEED = 42


def model(n_classes: int) -> XGBClassifier:
    return XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.08, subsample=0.9,
                         colsample_bytree=0.9, eval_metric="mlogloss", random_state=SEED)


def main():
    df = pd.read_csv(FEATURES_CSV)
    df["y_state"] = df["state"].map({s: i for i, s in enumerate(STATES)})
    df["y_type"] = df["bearing_fault_type"].map({t: i for i, t in enumerate(TYPES)})
    print(f"{len(df)} windows from {df['source_file'].nunique()} recordings of "
          f"{df['bearing_id'].nunique()} physical bearings", flush=True)

    oof_state = np.zeros((len(df), len(STATES)))
    oof_type = np.full(len(df), -1)
    folds = GroupKFold(n_splits=5)
    for k, (tr, te) in enumerate(folds.split(df, groups=df["bearing_id"])):
        held = sorted(df.iloc[te]["bearing_id"].unique())
        assert not set(held) & set(df.iloc[tr]["bearing_id"]), "bearing leaked across folds!"
        m = model(3)
        m.fit(df.iloc[tr][FEATURES], df.iloc[tr]["y_state"])
        oof_state[te] = m.predict_proba(df.iloc[te][FEATURES])
        # Fault type is only meaningful once a bearing is damaged.
        tr_bad = df.iloc[tr][df.iloc[tr]["state"] != "healthy"]
        te_bad_idx = [i for i in te if df.iloc[i]["state"] != "healthy"]
        mt = model(4)
        mt.fit(tr_bad[FEATURES], tr_bad["y_type"])
        oof_type[te_bad_idx] = mt.predict(df.iloc[te_bad_idx][FEATURES])
        print(f"  fold {k + 1}: held out bearings {held}", flush=True)

    p_not_healthy = 1 - oof_state[:, 0]
    pred_state = oof_state.argmax(axis=1)
    is_h, is_dev, is_bad = (df["state"] == s for s in STATES)
    auc = roc_auc_score(~is_h, p_not_healthy)
    flagged = p_not_healthy >= 0.5
    early = float(flagged[is_dev].mean())
    late = float(flagged[is_bad].mean())
    false_alarm = float(flagged[is_h].mean())
    macro_f1 = f1_score(df["y_state"], pred_state, average="macro")

    # Per recording (20 half-second windows each): majority vote.
    df["pred_state"] = pred_state
    rec = df.groupby("source_file").agg(true=("y_state", "first"),
                                        pred=("pred_state", lambda s: s.mode().iloc[0]),
                                        flagged=("pred_state", lambda s: (s != 0).mean() >= 0.5))
    rec_acc = float((rec["true"] == rec["pred"]).mean())
    rec_dev = rec[rec["true"] == 1]
    rec_h = rec[rec["true"] == 0]

    bad = df["state"] != "healthy"
    type_acc = float((oof_type[bad.values] == df.loc[bad, "y_type"].values).mean())

    cm = confusion_matrix(df["y_state"], pred_state, labels=range(3))
    ConfusionMatrixDisplay(cm, display_labels=STATES).plot(cmap="Blues")
    plt.title("Bearing health state -- leave-bearings-out")
    plt.tight_layout(); plt.savefig(REPORTS / "bearing_confusion_matrix.png", dpi=110); plt.close()

    print(f"\nHealthy-vs-damaged ROC AUC {auc:.3f} | developing faults caught {early:.0%} | "
          f"faulty caught {late:.0%} | false alarms on healthy {false_alarm:.0%}")
    print(f"Recordings: {int((rec['true'] == rec['pred']).sum())}/{len(rec)} states correct; "
          f"fault type correct on {type_acc:.0%} of damaged windows")

    # Final models, for the app.
    m = model(3); m.fit(df[FEATURES], df["y_state"])
    mt = model(4); mt.fit(df.loc[bad, FEATURES], df.loc[bad, "y_type"])
    perm = permutation_importance(m, df[FEATURES], df["y_state"], n_repeats=5,
                                  random_state=SEED, n_jobs=-1)
    imp = pd.Series(perm.importances_mean, index=FEATURES).sort_values()
    plt.figure(figsize=(8, 6)); imp.plot(kind="barh")
    plt.xlabel("Drop in accuracy when shuffled"); plt.title("Bearing model -- feature importance")
    plt.tight_layout(); plt.savefig(REPORTS / "bearing_feature_importance.png", dpi=110); plt.close()

    joblib.dump(m, ROOT / "bearing_model.pkl")
    joblib.dump(mt, ROOT / "bearing_type_model.pkl")
    joblib.dump({"features": FEATURES, "classes": STATES, "type_classes": TYPES,
                 "medians": df.loc[df["state"] == "healthy", FEATURES].median().to_dict()},
                ROOT / "bearing_meta.pkl")

    metrics = {
        "dataset": "University of Ottawa UORED-VAFCLS, doi:10.17632/y2px5tg92h.2",
        "validation": "leave-bearings-out, 5 folds (4 physical bearings held out per fold)",
        "n_windows": int(len(df)), "n_recordings": int(df["source_file"].nunique()),
        "n_bearings": int(df["bearing_id"].nunique()),
        "healthy_vs_damaged_roc_auc": auc,
        "developing_fault_windows_flagged": early, "faulty_windows_flagged": late,
        "healthy_windows_false_alarm": false_alarm, "state_macro_f1": macro_f1,
        "recording_state_accuracy": rec_acc,
        "developing_recordings_flagged": f"{int(rec_dev['flagged'].sum())}/{len(rec_dev)}",
        "healthy_recordings_false_alarm": f"{int(rec_h['flagged'].sum())}/{len(rec_h)}",
        "fault_type_accuracy_damaged_windows": type_acc,
    }
    (REPORTS / "bearing_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    md = f"""# Bearing / rotating-equipment model

Covers motors, pumps, fans, compressors, gearboxes -- anything whose dominant failure mode
is bearing wear.

**Dataset:** University of Ottawa Rolling-element Dataset (UORED-VAFCLS),
[doi:10.17632/y2px5tg92h.2](https://data.mendeley.com/datasets/y2px5tg92h/2), CC BY 4.0 --
**20 real bearings**, each recorded **healthy -> developing fault -> faulty** (60 recordings,
42 kHz accelerometer). Replaces the earlier CWRU model (only 10 recordings; 38.8% on held-out).

**Leakage guard -- leave-bearings-out:** each of 5 folds holds out 4 whole physical bearings
(all three of their states). Accelerometer only; the temperature channel is excluded because
it has corrupted values and drifts through each test sequence.

**Order-tracking:** features now include vibration energy expressed as multiples of shaft
speed (not just fixed Hz bands), the standard way to make a reading comparable across
machines running at different speeds -- bearing fault frequencies scale with shaft speed,
not absolute Hz. This rig ran at ~1700-1820 RPM; generalizing to a real customer motor at,
say, 900 or 3600 RPM is a methodological improvement, not a validated result -- there's no
data here outside that range to test it against.

| Metric (bearings never seen in training) | Value |
|---|---|
| Healthy vs. damaged ROC AUC | {auc:.3f} |
| **Developing faults caught** (early warning) | **{early:.0%}** of windows · {int(rec_dev['flagged'].sum())}/{len(rec_dev)} recordings |
| Fully faulty caught | {late:.0%} of windows |
| False alarms on healthy bearings | {false_alarm:.0%} of windows · {int(rec_h['flagged'].sum())}/{len(rec_h)} recordings |
| Health state correct (per recording, 3 states) | {int((rec['true'] == rec['pred']).sum())}/{len(rec)} |
| Fault type correct (damaged windows, 4 types) | {type_acc:.0%} |

![Confusion matrix](bearing_confusion_matrix.png)
![Feature importance](bearing_feature_importance.png)
"""
    (REPORTS / "bearing_results.md").write_text(md, encoding="utf-8")
    print("\nSaved bearing_model.pkl, bearing_type_model.pkl; see reports/bearing_results.md")


if __name__ == "__main__":
    main()
