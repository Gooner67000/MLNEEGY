# Who to contact, and in what order

## Match the prospect to what's actually validated

Lead with the machine types where the evidence is strongest. Don't pitch equipment the
models weren't validated on.

| If they run... | Lead with | Evidence |
|---|---|---|
| Pumps, fans, motors, compressors, gearboxes | Bearing model | 20 real bearings, early-warning rate (see `reports/bearing_results.md`) |
| Wind turbines / generators | Wind turbine model | Real wind-farm SCADA, held-out turbines |
| CNC mills / lathes | CNC model | AI4I2020 (note: a modeled snapshot, not a live-plant timeline) |
| Rooftop HVAC (retail, restaurants, offices, warehouses) | HVAC model | Real ORNL rooftop unit — check the current numbers in `reports/hvac_results.md` first |
| Collaborative robots | Robot model | One real UR3. Pitch as a **pilot**, not a finished product |
| Conveyors | Conveyor-drive model | Drive train only. Say up front that belt and idler faults aren't covered |
| **Anything else** | `train_custom.py` + the pilot | "We'll validate on your own history first" |

## Best first customers

1. **Small and mid-size manufacturers (20–500 employees)** with several of the same
   machine and no predictive maintenance yet. Big plants already buy from large
   vendors; small ones often only run scheduled maintenance.
2. **Facilities teams** with many identical rooftop units, such as multi-site retail,
   restaurants, schools and warehouses. It's the same equipment many times over, which
   is ideal for validating on held-out units.
3. **Food and beverage, packaging, plastics**: conveyors, motors and pumps running around
   the clock, where downtime is very visible.

## Who inside the company

- **Maintenance manager / reliability engineer.** Feels the pain, owns the breakdown log.
- **Plant / operations manager.** Owns the downtime cost, often signs off on a pilot.
- **Controls / automation engineer.** Owns the SCADA/historian export and the API hookup.
- **Facilities manager** for HVAC.

## Where to find them (without buying contact lists)

- Your state's **Manufacturing Extension Partnership (MEP)** center (nist.gov/mep). They
  run events and introduce small manufacturers to new technology.
- Local **manufacturing associations** and **industrial park** business groups.
- Trade shows and meetups: SMRP (Society for Maintenance & Reliability Professionals)
  chapters, ASHRAE chapters for HVAC, automation meetups.
- LinkedIn: search job titles like "maintenance manager" or "reliability engineer" plus
  your region, then connect with a short, specific note. Don't bulk-message.
- Your own network: anyone who works at a plant, a facility, or an HVAC service company.

## EPC Power: honest fit

EPC Power (Poway, CA) makes **utility-scale inverters and power conversion** for battery
storage, solar and AI data centers. Flex announced on Sept 3, 2026 that it will acquire it
for about $4.4B, with closing expected in Q4 2026. None of the 7 models here is trained on
inverters, so **don't claim they predict EPC Power's equipment**. The relevant angles:

- **Their fleet:** inverters in the field stream telemetry (temperatures, currents,
  fan speeds, faults). `train_custom.py` plus leave-units-out validation is exactly the
  pipeline you'd use on that. You've shown it working on real run-to-failure data.
- **Their data-center customers:** cooling is critical there, and the HVAC work is directly
  relevant.
- **Their factory:** motors, pumps, conveyors, robots and HVAC on the production floor.

Best as a **job/interview conversation** (see `INTERVIEW_PITCH.md`) rather than a cold sale.
