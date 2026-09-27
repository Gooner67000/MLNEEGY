# Predictive Maintenance

Predicts machine failure from sensor readings (type, air/process temperature, rotational speed, torque, tool wear) using the [AI4I 2020 dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) (10,000 machines, about 3.4% of them failed). The project compares Logistic Regression, Random Forest and XGBoost, and serves the XGBoost model in a Streamlit app.

## Quick start

```bash
python -m venv maintenance_env
maintenance_env\Scripts\activate        # macOS/Linux: source maintenance_env/bin/activate
pip install -r requirements.txt
python train.py        # EDA, training, evaluation -> model .pkl + reports/
python predict.py      # score 5 held-out machines
streamlit run app.py   # http://localhost:8501
```

## Files

| File | Guide step | Purpose |
|---|---|---|
| `train.py` | 2–7 | Load, EDA, preprocess, compare models, evaluate, save |
| `predict.py` | 8 | `predict_failure_risk()` → probability + Low/Medium/High |
| `app.py` | 9 | Streamlit app (single machine + CSV batch scoring) |
| `notebooks/01_eda_modeling.ipynb` | 2–8 | The same analysis as a notebook |
| `reports/` | — | Plots, `metrics.json`, and the held-out test set |

## Methodology notes

- **Leakage removed:** the `TWF/HDF/PWF/OSF/RNF` columns record *which* failure mode happened, so they give away the target. Keeping them makes any model look perfect.
- **Honest test set:** the data is split 80/20 first. Only the training data is undersampled, so test metrics reflect the real ~3.4% failure rate.
- **Thresholds:** `reports/metrics.json` lists precision and recall at thresholds 0.3, 0.5 and 0.7. Lower the threshold to catch more failures; raise it to get fewer false alarms.
- The dataset is a snapshot with no timestamps. The model estimates *current* failure risk from the readings, not "hours in advance."

## Deploy (Streamlit Community Cloud)

Push this repo to GitHub, including the `.pkl` files. Then go to share.streamlit.io → **New app** → pick the repo → main file `app.py` → **Deploy**.
