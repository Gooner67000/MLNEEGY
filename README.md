# 🛠️ Predictive Maintenance Platform

Two things live in this repo:

1. **The models** — trained, backtested predictive-maintenance models for several real machine categories (see table below).
2. **The platform** — a multi-tenant app businesses can run themselves (`docker compose up`, nothing leaves their network) that lets them register machines, pick a type from a dropdown, feed it readings (typed in, CSV, or their own live automation), and see current failure risk per machine.

**📊 Model results:** [reports/results.md](reports/results.md) · [reports/backtest_results.md](reports/backtest_results.md) · [reports/bearing_results.md](reports/bearing_results.md) · [reports/wind_turbine_results.md](reports/wind_turbine_results.md) — all regenerated automatically on every push by GitHub Actions, which also builds and smoke-tests the Docker stack below on every push.

## Which machines this actually covers

| Machine type | Status | Trained/validated on |
|---|---|---|
| CNC / lathe / milling machine | ✅ Trained & tested | AI4I2020 (10,000-machine snapshot) |
| Gas / jet turbine | ✅ Trained & backtested | NASA C-MAPSS (real run-to-failure engine simulations) |
| Motor / pump / fan / compressor / gearbox | ✅ Trained & tested | CWRU bearing-fault dataset (real induced-fault experiments) |
| Wind turbine / generator | ✅ Trained & backtested | Real wind-farm SCADA telemetry |
| HVAC / chiller | ⚠️ Schema only | No real public failure dataset exists — needs a business's own history |
| Robotic arm | ⚠️ Schema only | Same — no real public dataset found |
| Conveyor / belt | ⚠️ Schema only | Same — no real public dataset found |

The ⚠️ types have input forms in the app already, so adding a real model later is just training on that business's data and dropping the file in — no schema changes. The app tells the user plainly when a type isn't trained yet instead of returning a fake number (`backend/scoring.py` → `_not_trained`).

## Run it (downloadable — this is what a business would actually run)

```bash
git clone https://github.com/Gooner67000/MLNEEGY.git
cd MLNEEGY
docker compose up --build
```

- Frontend (register your business, add machines, enter readings): **http://localhost:8501**
- Backend API + interactive docs: **http://localhost:8000/docs**

Everything — the database, the models, both containers — runs on your own machine/network. Nothing is sent anywhere. Data persists in `./data/app.db` between restarts.

### Automatic live data import

There's one ingestion endpoint, `POST /machines/{id}/readings`, and manual entry, CSV upload, and live automation all call it — "automatic" means pointing your own PLC/SCADA/IoT script at that URL on a schedule (the "Live/API import" tab in the app shows the exact `curl` call for each of your machines, with your real auth token filled in). This repo does not, and honestly cannot, reach into a factory's OPC-UA/Modbus network on its own — that always needs a small script on the business's side, because every plant's setup is different.

### Running it as a hosted, multi-tenant service instead

The same backend works unchanged as a hosted service — set `DATABASE_URL` to a real Postgres instance and `SECRET_KEY` to a real secret (a `.env` file next to `docker-compose.yml` is enough), and every business's data is isolated by account automatically. **This repo does not stand up or manage that hosting for you** — you'd need your own server/domain and to handle accounts, billing, ToS, and data-privacy obligations before treating it as a real product with paying customers.

## Local development (no Docker)

```bash
python -m venv maintenance_env
maintenance_env\Scripts\activate        # macOS/Linux: source maintenance_env/bin/activate
pip install -r requirements-dev.txt

python train.py              # AI4I2020 model (steps 2-7 of the original guide)
python backtest.py           # NASA C-MAPSS backtest
python train_bearing.py      # CWRU bearing model
python train_wind_turbine.py # wind turbine SCADA model (streams real data, capped)
python -m pytest             # unit tests + backend API tests + headless app test

uvicorn backend.main:app --reload   # API on :8000
streamlit run frontend/app.py       # platform UI on :8501, talking to the API above
streamlit run app.py                # the original single-model (CNC-only) demo app
```

## Files

| Path | Purpose |
|---|---|
| `machine_types.py` | Single source of truth: every machine type's sensors, dataset, trained/untrained status. Both training scripts and the app read from here. |
| `train.py`, `predict.py`, `app.py` | The original AI4I2020 (CNC) model: steps 2–9 of the guide, plus a standalone single-model Streamlit demo |
| `backtest.py` | Walk-forward, no-lookahead backtest on real NASA C-MAPSS turbine data |
| `train_bearing.py` | Bearing/rotating-equipment model (real CWRU data), grouped by source recording |
| `train_wind_turbine.py` | Wind turbine/generator model (real SCADA data, streamed and subsampled), grouped by turbine |
| `backend/` | FastAPI app: auth, machine registry, one ingestion endpoint, risk scoring dispatch per machine type |
| `frontend/` | Streamlit app for the platform: register, add machines, enter/upload readings, dashboard |
| `docker-compose.yml`, `Dockerfile.*` | The whole downloadable deployment |
| `tests/` | Feature-math tests, leakage-guard tests, backend API tests (auth, isolation, ingestion, untrained-type handling), headless single-model app test |
| `.github/workflows/train.yml` | CI: trains all 4 real models → runs all tests → executes the notebook → **builds and smoke-tests the actual Docker stack** → commits models/reports |

## Methodology (AI4I2020 / CNC model)

- **No target leakage:** the `TWF/HDF/PWF/OSF/RNF` columns record *which* failure mode happened, so they give away the target and are dropped.
- **Honest test set:** the data is split 80/20 first. The test set is never resampled, so it keeps the real ~3.4% failure rate.
- **Engineered features:** power (torque × speed), process−air temperature gap, wear × torque — each maps to one of the dataset's failure mechanisms.
- **Imbalance:** the final model uses every training row with `scale_pos_weight`, tuned for average precision (PR AUC) — the metric that matters when failures are rare.
- **Alert threshold:** chosen by 5-fold cross-validation on the training data to reach recall ≥ 0.85, then checked on the untouched test set.
- **Limitation:** the dataset is a one-time snapshot with no timestamps — it estimates *current* risk, not "hours in advance." See the backtest below for a model that does have a real timeline.

## Does it actually predict failure in advance? (backtest)

AI4I2020 has no dates in it — there's no timeline to walk forward through, so the 80/20 held-out-machines split above is the correct honest evaluation for *that* dataset, but it is not a walk-forward backtest.

`backtest.py` (turbines) and `train_wind_turbine.py` (wind turbines) run genuine walk-forward backtests on data that **does** have a real timeline — NASA C-MAPSS run-to-failure simulations and real wind-farm SCADA telemetry — with two guards enforced in code and checked by `tests/test_backtest.py`:

1. **Machine holdout** — test machines are 100% unseen during training.
2. **Time causality** — every feature at time *t* uses only that machine's own data up to time *t*. `backtest.py`'s replay literally truncates each test engine's data to "now" before asking the model for a prediction.

The result that actually answers "does this predict failure in advance": on 20 held-out NASA engines, the model alerted **20/20** before failure, a median of **36 cycles** early (see [reports/backtest_results.md](reports/backtest_results.md) for current numbers). The bearing model (CWRU) has no time axis — it's independent lab recordings — so its leakage guard is grouping by source recording file instead; see [reports/bearing_results.md](reports/bearing_results.md).

## Deploy the single-model demo (Streamlit Community Cloud, free)

The original CNC-only `app.py` (not the platform) can also run as a free public demo:

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **Create app**, pick this repo and branch `main`, and set the main file to `app.py`.
3. Click **Deploy**. The trained model files are committed by CI, so the app works immediately.
