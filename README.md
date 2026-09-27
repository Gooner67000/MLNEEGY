# 🛠️ Predictive Maintenance Platform

Two things live in this repo:

1. **Models** — one per machine type, each trained on a real, cited dataset and scored
   only on machines, seasons, loads or time periods it never trained on.
2. **The platform** — an app businesses can download and run themselves
   (`docker compose up`; nothing leaves their network) or that you can host. Businesses
   register machines of any type, feed readings (typed in, CSV, or their own automation
   via the API), and see live failure risk. It also includes a pipeline that trains a new
   model on a customer's own history.

Every push retrains all models, runs the test suite, and scores a machine of **every
type** through the live Docker stack in CI.

## Which machines it covers, and how well

Results below are all on held-out data. Full reports are in [`reports/`](reports/).

| Machine type | Trained on (real data) | Held-out result | Status |
|---|---|---|---|
| **Motor / pump / fan / compressor / gearbox** (bearings) | Univ. of Ottawa UORED — 20 real bearings, healthy → developing → faulty | ROC AUC 0.94; **84% of developing faults caught early**; 1/20 healthy bearings falsely flagged | ✅ validated |
| **Gas / jet turbine** | NASA C-MAPSS run-to-failure | 20/20 engines alerted, median 36 cycles early | ✅ validated |
| **Wind turbine / generator** | Real wind-farm SCADA | ROC AUC 0.999 on held-out turbines | ✅ validated |
| **CNC / lathe / mill** | AI4I2020 (modeled snapshot, no timeline) | ROC AUC 0.986, recall 85% | ✅ validated* |
| **Conveyor drive** (motor, shaft, bearings) | KAIST rig (*Data in Brief* 2023), vibration | At an unseen load with 1% false alarms: bearing faults ~90%, moderate/heavy misalignment 91–100%; light misalignment and unbalance mostly missed | ✅ validated† |
| **HVAC rooftop unit** | DOE/LBNL — real Trane 12.5-ton unit at Oak Ridge Nat'l Lab | Only ~1 in 5 fault days caught at a 20% false-alarm cap; economizer faults missed | ⚠️ experimental |
| **Robotic arm / cobot** | UCI UR3 CobotOps — real UR3 telemetry | Warned before 17/24 grip losses, but only ~1 in 5 alerts real | ⚠️ experimental |

\* AI4I2020 is a well-known benchmark modeled on real machines, not a live plant log.
† Drive train only. Belt tears, mistracking and idler faults are **not** covered: no
reliable open dataset exists for them. The one real one (135 idlers in a working mine) is
access-restricted.

"Experimental" types are labeled that way inside the app too. Offer them only as a pilot,
validated on the customer's own equipment first.

### How the data was checked, not just used

- **Leakage guards match each dataset.** Leave-bearings-out, leave-one-season-out,
  leave-one-load-out, leave-turbines-out, chronological walk-forward with a label gap.
  Alert thresholds are chosen by *nested* validation, never on the held-out data.
- **Confounds were looked for, and they turned up.** The conveyor dataset's motor-current
  files were **rejected**: their features tracked a 48 vs 49 Hz difference between
  recording sessions, not the faults. The vibration features that replaced them are
  order-tracked, and they rise with fault severity as physics predicts. The bearing
  dataset's temperature channel was excluded: it has corrupted values and drifts through
  each test.
- **Provenance.** Raw data behind a bot check (Mendeley) was downloaded once and reduced
  to feature tables by [`tools/extract_mendeley_features.py`](tools/extract_mendeley_features.py).
  The SHA-256 of every raw file is in [`data/provenance/`](data/provenance/).

## Run it

**Download & run (on-prem):** needs Docker.
```bash
git clone https://github.com/Gooner67000/MLNEEGY.git
cd MLNEEGY
docker compose up --build
```
Web app on http://localhost:8501, API docs on http://localhost:8000/docs.

**Hosted:** one-click Render blueprint plus one paste. See **[DEPLOY.md](DEPLOY.md)**,
including the free-tier limits (read them before onboarding real customers).

**Live data:** point any PLC, SCADA or historian script at
`POST /machines/{id}/readings`. The app's "Live/API import" tab shows the exact call.
Manual entry and CSV upload use the same endpoint.

## Onboarding a real customer

No public dataset replaces a customer's own machines, so the path is built in:

1. Send them [`pilot/DATA_REQUEST.md`](pilot/DATA_REQUEST.md) plus the CSV templates: a
   sensor log and a breakdown log.
2. `python train_custom.py --key their_line --name "Their line" --sensors s.csv --events e.csv --horizon 24h`
   validates on machines it never saw, refuses to train on fewer than 5 failures, and
   registers a new machine type in the app, labeled validated or experimental by its own
   results. Tested end-to-end on real run-to-failure data in CI.
3. Follow [`pilot/PILOT_PLAN.md`](pilot/PILOT_PLAN.md): backtest → shadow mode → review.
   Get a data agreement first ([checklist](pilot/DATA_AGREEMENT_CHECKLIST.md)).
   Customer data and models are gitignored, since this repo is public.

Outreach materials are in [`outreach/`](outreach/).

## Develop locally (no Docker)

```bash
python -m venv maintenance_env
maintenance_env\Scripts\activate        # macOS/Linux: source maintenance_env/bin/activate
pip install -r requirements-dev.txt
python train.py && python backtest.py && python train_bearing.py && python train_wind_turbine.py
python train_hvac.py && python train_robot.py && python train_conveyor.py
python -m pytest
uvicorn backend.main:app --reload       # API on :8000
streamlit run frontend/app.py           # platform UI on :8501
```

## Files

| Path | Purpose |
|---|---|
| `machine_types.py` | Registry of every machine type: sensors, data source, maturity. Customer types load from `custom_models/`. |
| `train*.py`, `backtest.py` | One training/validation script per model; each writes `reports/<name>_results.md` |
| `train_custom.py` | Train on a customer's own sensor + breakdown logs |
| `pm_features.py` | Feature code shared by training **and** the live backend (tested to never look ahead) |
| `backend/`, `frontend/` | FastAPI API (auth, per-business isolation, scoring) and Streamlit app |
| `tools/` | Raw-data feature extraction; end-to-end smoke test for a running stack |
| `docker-compose.yml`, `Dockerfile.*`, `render.yaml` | On-prem and hosted deployment |
| `pilot/`, `outreach/` | Customer data request, pilot plan, data-agreement checklist; outreach materials |
