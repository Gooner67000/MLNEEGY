"""Trains the model for machine_types['rotating_equipment'] -- motors, pumps,
fans, compressors, gearboxes: anything whose dominant failure mode is bearing
wear.

Data: the CWRU (Case Western Reserve University) bearing-fault dataset --
real induced-fault vibration experiments, not synthetic. 9,488 vibration
segments, statistical/spectral features precomputed, 4 classes: Normal,
Ball_Fault, Inner_Race, Outer_Race.

No time axis here (these are independent lab recordings, not one continuous
machine running to failure like AI4I2020/CMAPSS), so the leakage guard is
different: segments are grouped by SOURCE RECORDING FILE (a .mat file is one
continuous run under one fault condition). All segments from the same file go
entirely to train or entirely to test, so the model is never scored on a
near-duplicate of something it trained on.

Run:  python train_bearing.py
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)
FEATURES_URL = ("https://huggingface.co/datasets/mdsajjadullah/"
                "bearing-fault-diagnosis-cwru-results/resolve/main/features_all_segments.csv")
LOG_URL = ("https://huggingface.co/datasets/mdsajjadullah/"
           "bearing-fault-diagnosis-cwru-results/resolve/main/dataset_file_log.csv")
FEATURES_CSV = ROOT / "data" / "cwru_features_all_segments.csv"
LOG_CSV = ROOT / "data" / "cwru_dataset_file_log.csv"
SEED = 42
CLASSES = ["Normal", "Ball_Fault", "Inner_Race", "Outer_Race"]


def load_data() -> pd.DataFrame:
    FEATURES_CSV.parent.mkdir(exist_ok=True)
    if not FEATURES_CSV.exists():
        pd.read_csv(FEATURES_URL).to_csv(FEATURES_CSV, index=False)
    if not LOG_CSV.exists():
        pd.read_csv(LOG_URL).to_csv(LOG_CSV, index=False)
    df = pd.read_csv(FEATURES_CSV)
    log = pd.read_csv(LOG_CSV)

    # Recover which source .mat file each segment came from, in the same
    # per-label order the segments were generated in, so we can group by it.
    log = log.sort_values(["Label", "File"]).reset_index(drop=True)
    df = df.sort_values("label", kind="stable").reset_index(drop=True)
    file_for_row = []
    for label, group in log.groupby("Label"):
        for _, row in group.iterrows():
            file_for_row.extend([row["File"]] * int(row["Segments"]))
    # Fallback if segment counts don't line up exactly with the CSV row order:
    # fail loud rather than silently mislabel the grouping key.
    assert len(file_for_row) == len(df), (
        f"segment/file count mismatch: {len(file_for_row)} vs {len(df)} -- "
        "grouping key would be wrong, refusing to guess"
    )
    df["source_file"] = file_for_row
    print(f"Loaded {len(df)} segments from {df['source_file'].nunique()} recordings.")
    print(df["label"].value_counts())
    return df


def main():
    df = load_data()
    feature_cols = [c for c in df.columns if c not in ("label", "source_file")]
    df["y"] = df["label"].map({c: i for i, c in enumerate(CLASSES)})

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    train_idx, test_idx = next(splitter.split(df, groups=df["source_file"]))
    train_files = set(df.loc[train_idx, "source_file"])
    test_files = set(df.loc[test_idx, "source_file"])
    assert not train_files & test_files, "recording leaked between train/test!"
    train_df, test_df = df.loc[train_idx], df.loc[test_idx]
    print(f"\nTrain: {len(train_df)} segments from {len(train_files)} recordings")
    print(f"Test:  {len(test_df)} segments from {len(test_files)} recordings (never trained on)")

    model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                          random_state=SEED, eval_metric="mlogloss")
    model.fit(train_df[feature_cols], train_df["y"])

    pred = model.predict(test_df[feature_cols])
    acc = accuracy_score(test_df["y"], pred)
    bal_acc = balanced_accuracy_score(test_df["y"], pred)
    f1 = f1_score(test_df["y"], pred, average="macro")
    fault_recall = ((pred != 0) & (test_df["y"] != 0)).sum() / max((test_df["y"] != 0).sum(), 1)
    print(f"\nHeld-out recordings -- accuracy {acc:.4f}  balanced accuracy {bal_acc:.4f}  "
          f"macro F1 {f1:.4f}  any-fault recall {fault_recall:.4f}")

    cm = confusion_matrix(test_df["y"], pred, labels=list(range(len(CLASSES))))
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASSES)
    disp.plot(cmap="Blues", xticks_rotation=30)
    plt.title("Bearing fault diagnosis -- held-out recordings")
    plt.tight_layout()
    plt.savefig(REPORTS / "bearing_confusion_matrix.png", dpi=110)
    plt.close()

    perm = permutation_importance(model, test_df[feature_cols], test_df["y"],
                                  scoring="accuracy", n_repeats=8, random_state=SEED, n_jobs=-1)
    perm_df = pd.DataFrame({"Feature": feature_cols, "Importance": perm.importances_mean}
                           ).sort_values("Importance", ascending=False)
    plt.figure(figsize=(9, 6))
    plt.barh(perm_df["Feature"][:12][::-1], perm_df["Importance"][:12][::-1])
    plt.xlabel("Drop in accuracy when feature is shuffled")
    plt.title("Bearing model -- top features")
    plt.tight_layout()
    plt.savefig(REPORTS / "bearing_feature_importance.png", dpi=110)
    plt.close()

    joblib.dump(model, ROOT / "bearing_model.pkl")
    joblib.dump({
        "features": feature_cols,
        "classes": CLASSES,
        # backend fallback for any feature a manual/partial reading omits
        "medians": train_df[feature_cols].median().to_dict(),
    }, ROOT / "bearing_meta.pkl")
    with open(REPORTS / "bearing_metrics.json", "w") as f:
        json.dump({"dataset": "CWRU bearing fault (real induced-fault vibration experiments)",
                   "n_segments": len(df), "n_recordings": df["source_file"].nunique(),
                   "n_train_recordings": len(train_files), "n_test_recordings": len(test_files),
                   "accuracy": acc, "balanced_accuracy": bal_acc, "macro_f1": f1,
                   "any_fault_recall": float(fault_recall),
                   "top_features": perm_df.head(10).to_dict("records")}, f, indent=2, default=float)

    md = f"""# Bearing / rotating-equipment model

Covers motors, pumps, fans, compressors, gearboxes -- anything whose dominant
failure mode is bearing wear.

**Dataset:** CWRU bearing fault dataset -- real induced-fault vibration
experiments (not synthetic), {len(df)} segments from {df['source_file'].nunique()}
recordings, 4 classes (Normal / Ball fault / Inner-race fault / Outer-race fault).

**Leakage guard:** segments are grouped by source recording file. All segments
from one recording go entirely to train ({len(train_files)} recordings) or
entirely to test ({len(test_files)} recordings, {len(test_df)} segments) --
never split across both.

| Metric (held-out recordings) | Value |
|---|---|
| Accuracy | {acc:.3f} |
| Balanced accuracy | {bal_acc:.3f} |
| Macro F1 | {f1:.3f} |
| Recall on any real fault (vs Normal) | {fault_recall:.3f} |

![Confusion matrix](bearing_confusion_matrix.png)
![Feature importance](bearing_feature_importance.png)
"""
    (REPORTS / "bearing_results.md").write_text(md, encoding="utf-8")
    print("\nSaved bearing_model.pkl, bearing_meta.pkl; see reports/bearing_results.md")


if __name__ == "__main__":
    main()
