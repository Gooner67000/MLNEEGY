# Interview pitch (EPC Power, or any hardware/energy company)

## Know the company (as of Sept 2026)

EPC Power (Poway, CA, founded 2010) makes **utility-scale power conversion**: inverters
for battery energy storage, solar and microgrids, and more recently power systems for AI
data centers (M System, CAB1000). On **Sept 3, 2026, Flex announced it will acquire EPC
Power for about $4.4B**, with closing expected in Q4 2026, as part of Flex's Cloud & Power
Infrastructure segment. Double-check the latest news the day before the interview.

**Don't claim** your models predict their inverters. None was trained on inverter data.
The honest pitch is the *method* and the *pipeline*.

## 60-second pitch

> "I built a predictive-maintenance platform end to end: seven machine types, each trained
> on real, cited equipment data, and each scored only on units the model never saw. That
> means held-out bearings, seasons, operating loads, or later time periods. On 20 real
> bearings it caught 84% of faults while they were still developing. On NASA run-to-failure
> data it warned before all 20 held-out engines failed, a median of 36 cycles ahead.
>
> The part I'm proudest of is what I rejected. One dataset's motor-current signals looked
> great, until I found they were tracking a 48 vs 49 Hz difference between recording
> sessions rather than the faults. I threw that out and used order-tracked vibration
> instead, and checked that the new features rise with fault severity the way physics says
> they should. I also label two of the seven models 'experimental' because their held-out
> results aren't good enough yet.
>
> For a company like EPC Power, the relevant piece is the onboarding pipeline: give it a
> fleet's telemetry and failure log and it trains, validates on units it never saw, and
> refuses to train if there aren't enough failures to be meaningful. That's how I'd
> approach predicting inverter faults: cooling fans, IGBT thermal cycling, capacitor
> aging, from field data."

## Likely questions and honest answers

**"How accurate is it?"**
Depends on the equipment and I'll give the real range: strong on bearings, turbines and
wind turbines; weak on rooftop HVAC, where I caught only about 1 in 5 fault days at a 20%
false-alarm cap. I can explain why: many of those imposed faults are physically invisible
on days the economizer doesn't run.

**"How do you know it's not overfitting?"**
Every score comes from data held out *by unit*: bearings, turbines, seasons, loads, or
later time. Alert thresholds are chosen by nested validation, never on the test data. CI
retrains everything and checks, on every commit, that no feature can see the future.

**"What would you do with our inverter fleet data?"**
Start with the breakdown log. How many real failures, of which types? If there are too
few, say so. Then train with leave-sites-out validation, agree on success criteria with
the service team, and run in shadow mode before anyone acts on an alert.

**"What was the hardest part?"**
Not fooling myself: finding the recording-session confound, and being willing to label
two models experimental instead of rounding the numbers up.

## Bring

- The GitHub repo, with the `reports/` folder open.
- One chart: the bearing early-warning result or the turbine lead-time histogram.
- A live demo if deployed (DEPLOY.md), or a 2-minute screen recording of the app.
