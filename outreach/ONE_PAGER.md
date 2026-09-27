# Catch equipment failures before they stop your line

**What it is:** software that watches the sensor data your machines already produce and
warns you before a failure, so you can fix it on your schedule instead of in the middle of
a shift.

**Runs on your site.** Install it on your own PC or server; your data never leaves.
A hosted version is available if you prefer.

## Proven on real equipment data, not demos

Every result below was measured on machines the model had **never seen** during training:

- **Pumps, fans, motors, compressors, gearboxes:** on 20 real bearings, it flagged
  **84% of developing faults early**, while they were still developing, with only 1 of 20
  healthy bearings falsely flagged.
- **Turbines:** in NASA run-to-failure data, it warned before **20 of 20** engines failed,
  a median of **36 operating cycles** ahead.
- **Wind turbines / generators:** 0.999 ROC AUC on real wind-farm data from turbines it
  never trained on.
- **Conveyor drives:** on a real test rig, at an operating load it had never seen, caught
  about 90% of bearing faults with just 1% false alarms on normal running.

## How a pilot works (no commitment)

1. **Two files from you:** your sensor history and your breakdown log. Exports most
   historian, SCADA and maintenance systems can produce.
2. **Two weeks:** we test the model on *your* past breakdowns and show you honestly how
   many it would have caught, and how many false alarms it would have raised.
3. **Only if the numbers are good:** it runs alongside your team for a few weeks before
   anyone acts on it.

We agree on what "good" means **before** we look at the results.

## What we won't claim

- Anything about your equipment before we've tested on your data.
- Coverage of faults it wasn't validated on. For example, belt tears on conveyors aren't
  covered.

**Contact:** [your name] · [email] · [phone]
