# 🔧 Predictive Maintenance

Predicts industrial machine failure from sensor readings using the [AI4I 2020 dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) (10,000 machines, about 3.4% of them failed). The project compares Logistic Regression, Random Forest and XGBoost, adds a tuned class-weighted XGBoost with physics-based features, and serves it in a Streamlit app.

**📊 Latest results: [reports/results.md](reports/results.md)** (regenerated automatically on every push by GitHub Actions)

## Quick start

```bash
python -m venv maintenance_env
maintenance_env\Scripts\activate        # macOS/Linux: source maintenance_env/bin/activate
pip install -r requirements-dev.txt
python train.py        # EDA, training, tuning, evaluation -> model + reports/
python predict.py      # score sample held-out machines
python -m pytest       # unit tests + headless app test
streamlit run app.py   # http://localhost:8501
```

## Files

| File | Guide step | Purpose |
|---|---|---|
| `train.py` | 2–7 | Load, EDA, preprocess, compare models, tune, choose threshold, evaluate, save |
| `predict.py` | 8 | `add_features()` and `predict_failure_risk()` → probability, Low/Medium/High, alert flag |
| `app.py` | 9 | Streamlit app: single machine, batch CSV scoring, model performance tab |
| `notebooks/01_eda_modeling.ipynb` | 2–8 | The same pipeline as a narrated notebook |
| `tests/` | — | pytest: feature math, risk bands, recall regression check, headless app test |
| `.github/workflows/train.yml` | — | CI: retrain → test → execute notebook → commit model and reports |
| `model_meta.json` | — | Alert threshold, hyperparameters, and feature list for the saved model |

## Methodology

- **No target leakage:** the `TWF/HDF/PWF/OSF/RNF` columns record *which* failure mode happened, so they give away the target and are dropped.
- **Honest test set:** the data is split 80/20 first. The test set is never resampled, so it keeps the real ~3.4% failure rate.
- **Engineered features:** each maps to one of the dataset's failure mechanisms.
  - Power = torque × speed (power failure)
  - Process − air temperature gap (heat-dissipation failure)
  - Wear × torque (overstrain failure)
- **Imbalance:** the guide's baselines use undersampling. The final model uses every training row with `scale_pos_weight` and is tuned for average precision (PR AUC). PR AUC is the metric that matters when failures are rare.
- **Alert threshold:** chosen by 5-fold cross-validation on the training data to reach recall ≥ 0.85, then checked on the untouched test set.
- **Explainability:** permutation importance on the test set shows which readings drive predictions.
- **Limitation:** the dataset is a snapshot with no timestamps. The model estimates *current* failure risk from the readings; it cannot forecast "hours in advance."

## Deploy (Streamlit Community Cloud, free)

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **Create app**, pick this repo and branch `main`, and set the main file to `app.py`.
3. Click **Deploy**. The trained model files are committed by CI, so the app works immediately.
