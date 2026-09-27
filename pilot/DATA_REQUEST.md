# What to ask a pilot customer for

Send this to the person who owns the equipment data. That's usually a maintenance
manager, plant or controls engineer, or facilities manager. Two exports are all that's
needed. Both are routine for any site with a PLC historian, SCADA, building-automation
system (BMS), or CMMS.

## 1. Sensor log (from the historian / SCADA / BMS / robot controller)

One row per reading, one column per sensor. See `templates/sensor_readings_template.csv`.

| Column | Required | Example | Notes |
|---|---|---|---|
| `timestamp` | yes | `2026-03-04 14:05:00` | Any standard date-time format |
| `machine_id` | yes | `PRESS-03` | Their own tag or asset ID |
| one column per sensor | at least 1 | `motor_current_a`, `bearing_temp_c`, `vibration_rms_g` | Numbers only; units in the column name help |

- **How much:** as much history as they have. **Aim for 6+ months.** The more machines of
  the same type, the better (4+ lets us test on machines the model never saw).
- **How often:** whatever they log. 1/minute is plenty; 1/hour works for slow-developing
  faults.
- Gaps and missing values are fine.

## 2. Breakdown log (from the CMMS / maintenance records / work orders)

One row per unplanned failure or breakdown. See `templates/failure_log_template.csv`.

| Column | Required | Example |
|---|---|---|
| `machine_id` | yes | `PRESS-03` (same IDs as the sensor log) |
| `timestamp` | yes | when it failed or was taken down for the fault |
| `event_type` | optional | `bearing failure`, `motor trip`, `jam` |

- **Only unplanned events.** Leave out scheduled maintenance, otherwise the model learns
  the maintenance calendar instead of the failures.
- **At least 5 events in total, ideally 20+.** With fewer, `train_custom.py` refuses to
  train, because any "accuracy" would be noise, and it's better to say so up front.

## What happens next

```bash
python train_custom.py --key press_line --name "Customer press line" \
    --sensors their_sensors.csv --events their_failures.csv --horizon 24h
```

It prints an honest validation, scored on machines or time periods the model never
trained on, and registers a new machine type in the app. Show the customer that result
**before** any claims.

## Handling their data

- Keep customer files in `pilot/customer_data/`. That folder and `custom_models/custom_*`
  are in `.gitignore`, so they can't be pushed to this public repo by accident.
- For the pilot, run the **downloadable** version (`docker compose up`) on their premises,
  or on a machine they approve. Their data never has to leave their site.
- See `DATA_AGREEMENT_CHECKLIST.md` before receiving any files.
