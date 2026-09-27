# Bearing / rotating-equipment model

Covers motors, pumps, fans, compressors, gearboxes -- anything whose dominant
failure mode is bearing wear.

**Dataset:** CWRU bearing fault dataset -- real induced-fault vibration
experiments (not synthetic), 9487 segments from 10
recordings, 4 classes (Normal / Ball fault / Inner-race fault / Outer-race fault).

**Leakage guard:** segments are grouped by source recording file. All segments
from one recording go entirely to train (7 recordings) or
entirely to test (3 recordings, 2848 segments) --
never split across both.

| Metric (held-out recordings) | Value |
|---|---|
| Accuracy | 0.388 |
| Balanced accuracy | 0.388 |
| Macro F1 | 0.312 |
| Recall on any real fault (vs Normal) | 1.000 |

![Confusion matrix](bearing_confusion_matrix.png)
![Feature importance](bearing_feature_importance.png)
