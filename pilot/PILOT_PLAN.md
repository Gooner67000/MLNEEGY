# Pilot plan: first real customer

The goal of a pilot isn't to sell software. It's to **prove the model on their own
machines**, which is the one thing no public dataset can do. Keep it small, free or
low-cost, and time-boxed.

## Before you start (week 0)

- [ ] Pick **one machine type** they have several of (4+ units is ideal) and where
      unplanned downtime clearly costs money.
- [ ] Agree on success criteria **in writing, before seeing any results**, for example:
      *"Warns before at least half of the breakdowns in the held-out period, with no more
      than one false alarm per machine per month."*
- [ ] Sign a data agreement (see `DATA_AGREEMENT_CHECKLIST.md`).
- [ ] Send `DATA_REQUEST.md` and the two CSV templates.

## Phase 1: backtest on their history (weeks 1–2)

- [ ] Run `train_custom.py` on their export.
- [ ] Share the validation report as-is, including what it misses. The validation is
      scored on machines or time periods the model never trained on.
- [ ] **Go / no-go.** If it doesn't beat the agreed criteria, say so and stop, or ask for
      more data. A pilot that honestly fails early still earns trust.

## Phase 2: shadow mode (weeks 3–8)

- [ ] Install the downloadable version on their site (`docker compose up`).
- [ ] Point their historian/SCADA at the API: a small script that POSTs readings every
      minute. The app's "Live/API import" tab shows the exact call.
- [ ] The model scores live, but **no one acts on it yet**. Log every alert.
- [ ] Each week, compare the alerts against what actually broke.

## Phase 3: review and decide (week 9)

- [ ] Present the shadow-mode scorecard: alerts vs. real failures, false alarms, and
      warning lead time.
- [ ] Estimate the value in *their* numbers: breakdowns caught × their cost of an
      unplanned stop.
- [ ] If it met the criteria, propose paid monitoring. If not, say what data would fix it.

## What NOT to promise

- No accuracy number before Phase 1 is done on their data.
- No "hours in advance" claim unless their data shows it.
- Nothing about equipment types the model wasn't validated on.
