"""Trains the model for machine_types['conveyor'] -- the conveyor's DRIVE train
(motor -> coupling -> shaft -> bearings), where most unplanned conveyor
downtime starts.

Data: KAIST rotating-machine dataset (Jung et al., Data in Brief 2023),
Mendeley Data doi:10.17632/ztmf3m7h5x.6, CC BY 4.0 -- a real test rig run
normal and with bearing inner-race / outer-race defects, shaft misalignment
and rotor unbalance at several severities, under 0 / 2 / 4 Nm load. Four
accelerometers at 25.6 kHz. Order-tracked features were extracted from the
raw recordings by tools/extract_mendeley_features.py (provenance + SHA-256
in data/provenance/conveyor_kaist.json).

What this does NOT cover: belt-surface damage, belt mistracking and idler
failures. No reliable, openly downloadable dataset for those exists (the one
real one -- 135 idlers recorded in a working mine -- is access-restricted).

Why vibration, not the dataset's motor-current files: those were checked
first and rejected -- the "normal" and "unbalance" recordings used a 48 Hz
supply while most bearing/misalignment recordings used 49 Hz, and the
current features tracked that session difference instead of the faults.

Leakage guard -- LEAVE-ONE-LOAD-OUT: train on two load levels, score the
third, rotate. Every number is for an operating load the model never saw.
Windows from one recording never appear on both sides.

Run:  python train_conveyor.py
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
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)
FEATURES_CSV = ROOT / "data" / "conveyor_kaist_features.csv.gz"
CLASSES = ["normal", "unbalance", "misalignment", "bearing_inner", "bearing_outer"]
FEATURES = ["rms", "peak", "crest", "kurtosis", "skewness", "shape", "impulse",
            "spec_cent", "spec_bw", "spec_ent", "band0", "band1", "band2", "band3",
            "order1_share", "order2_share", "order3_share", "order1_amp", "order2_amp"]
SEED = 42


def make_model() -> XGBClassifier:
    return XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.06, subsample=0.9,
                         colsample_bytree=0.9, eval_metric="mlogloss", random_state=SEED)


def main():
    df = pd.read_csv(FEATURES_CSV)
    df["y"] = df["label"].map({c: i for i, c in enumerate(CLASSES)})
    print(f"{len(df)} windows x accelerometers from {df['source_file'].nunique()} real recordings",
          flush=True)

    proba = np.zeros((len(df), len(CLASSES)))
    for load in sorted(df["load_nm"].unique()):
        tr, te = df["load_nm"] != load, df["load_nm"] == load
        assert not set(df.loc[tr, "source_file"]) & set(df.loc[te, "source_file"])
        m = make_model()
        m.fit(df.loc[tr, FEATURES], df.loc[tr, "y"],
              sample_weight=compute_sample_weight("balanced", df.loc[tr, "y"]))
        proba[te.values] = m.predict_proba(df.loc[te, FEATURES])
        print(f"  held out {load} Nm load: {te.sum()} rows scored by a model that never saw it",
              flush=True)

    p_fault = 1 - proba[:, 0]
    pred = proba.argmax(axis=1)
    is_normal = df["label"] == "normal"
    auc = roc_auc_score(~is_normal, p_fault)
    flagged = p_fault >= 0.5
    fault_recall = float(flagged[~is_normal].mean())
    false_alarm = float(flagged[is_normal].mean())
    macro_f1 = f1_score(df["y"], pred, average="macro")
    per_class = {c: float((pred[df["label"] == c] == i).mean()) for i, c in enumerate(CLASSES)}
    per_sev = (df.assign(flag=flagged).groupby(["label", "severity"])["flag"].mean())

    df["pred"] = pred
    rec = df.groupby("source_file").agg(true=("y", "first"),
                                        pred=("pred", lambda s: s.mode().iloc[0]))
    rec_ok = int((rec["true"] == rec["pred"]).sum())

    cm = confusion_matrix(df["y"], pred, labels=range(len(CLASSES)))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(cmap="Blues", xticks_rotation=25)
    plt.title("Conveyor drive -- leave-one-load-out")
    plt.tight_layout(); plt.savefig(REPORTS / "conveyor_confusion_matrix.png", dpi=110); plt.close()

    print(f"\nFault vs normal ROC AUC {auc:.3f} | faults flagged {fault_recall:.0%} | "
          f"false alarms on normal {false_alarm:.0%} | recordings diagnosed {rec_ok}/{len(rec)}")
    print(per_sev.to_string())

    final = make_model()
    final.fit(df[FEATURES], df["y"], sample_weight=compute_sample_weight("balanced", df["y"]))
    joblib.dump(final, ROOT / "conveyor_model.pkl")
    joblib.dump({"features": FEATURES, "classes": CLASSES,
                 "medians": df.loc[is_normal, FEATURES].median().to_dict()},
                ROOT / "conveyor_meta.pkl")

    metrics = {
        "dataset": "KAIST rotating machine (Jung et al. 2023), doi:10.17632/ztmf3m7h5x.6, vibration",
        "validation": "leave-one-load-out (0/2/4 Nm)",
        "fault_vs_normal_roc_auc": auc, "fault_windows_flagged": fault_recall,
        "normal_windows_false_alarm": false_alarm, "diagnosis_macro_f1": macro_f1,
        "per_class_window_accuracy": per_class,
        "recordings_correctly_diagnosed": f"{rec_ok}/{len(rec)}",
        "flag_rate_by_severity": {f"{a}/{b}": float(v) for (a, b), v in per_sev.items()},
    }
    (REPORTS / "conveyor_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    sev_rows = "\n".join(f"| {a} | {b} | {v:.0%} |" for (a, b), v in per_sev.items())
    cls_rows = "\n".join(f"| {c} | {v:.0%} |" for c, v in per_class.items())
    md = f"""# Conveyor drive model

Covers the conveyor's **drive train** -- motor, coupling, shaft and bearings -- where most
unplanned conveyor downtime starts. **Not covered:** belt tears, belt mistracking, idler
failures (no reliable open dataset exists for those; see README).

**Dataset:** KAIST rotating-machine dataset (Jung et al., *Data in Brief* 2023),
[doi:10.17632/ztmf3m7h5x.6](https://data.mendeley.com/datasets/ztmf3m7h5x/6), CC BY 4.0 --
a real test rig with bearing inner/outer-race defects, shaft misalignment and rotor unbalance
at several severities, under three loads. Vibration (4 accelerometers, 25.6 kHz),
order-tracked. The same dataset's motor-current files were examined first and **rejected**:
their features tracked a 48 vs 49 Hz supply difference between recording sessions, not the
faults.

**Leakage guard -- leave-one-load-out:** each load level is scored by a model trained only on
the other two.

| Metric (load never seen in training) | Value |
|---|---|
| Fault vs. normal ROC AUC | {auc:.3f} |
| Fault windows flagged | {fault_recall:.0%} |
| False alarms on normal running | {false_alarm:.0%} |
| Recordings correctly diagnosed (majority vote) | {rec_ok} / {len(rec)} |
| Diagnosis macro F1 (5 classes) | {macro_f1:.3f} |

Per class (window accuracy):

| Class | Correct |
|---|---|
{cls_rows}

Flag rate by fault severity -- the honest picture of what it can and can't catch:

| Fault | Severity | Flagged |
|---|---|---|
{sev_rows}

Physically, small unbalance barely changes vibration (1x amplitude only rises clearly from
~2.2 g of added mass up), so light unbalance is expected to be missed.

![Confusion matrix](conveyor_confusion_matrix.png)
"""
    (REPORTS / "conveyor_results.md").write_text(md, encoding="utf-8")
    print("\nSaved conveyor_model.pkl; see reports/conveyor_results.md")


if __name__ == "__main__":
    main()
