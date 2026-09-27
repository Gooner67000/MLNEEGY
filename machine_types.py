"""Single source of truth for every machine type the platform supports: which
raw sensors it needs, which of those a human can reasonably type into a form,
where its trained model lives, and whether it's actually trained on real data
or still a placeholder.

Both the training scripts and the backend/frontend import this file, so the
app can never drift out of sync with what was actually trained.
"""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).parent


@dataclass
class SensorField:
    key: str            # dict key used everywhere (payload, DataFrame column)
    label: str           # shown in the UI
    unit: str = ""
    min: float = 0.0
    max: float = 100.0
    default: float = 0.0
    manual_entry: bool = True  # False = only realistic via CSV/API (e.g. raw vibration stats)


@dataclass
class MachineType:
    key: str
    name: str
    description: str
    dataset: str          # what it was trained/validated on, for honesty in the UI
    trained: bool          # False = schema exists, no real model behind it yet
    model_path: str = ""
    sensors: list[SensorField] = field(default_factory=list)
    note: str = ""         # shown in the UI, esp. for untrained types


MACHINE_TYPES: dict[str, MachineType] = {

    "cnc_machine_tool": MachineType(
        key="cnc_machine_tool", name="CNC / Lathe / Milling Machine",
        description="Machine tools that cut material: spindle, torque, tool wear sensors.",
        dataset="AI4I2020 (10,000 machines, real-world-modeled synthetic snapshot)",
        trained=True, model_path="predictive_maintenance_model.pkl",
        sensors=[
            SensorField("Type", "Product quality type (0=L, 1=M, 2=H)", "", 0, 2, 0),
            SensorField("Air temperature K", "Air temperature", "K", 295, 305, 300),
            SensorField("Process temperature K", "Process temperature", "K", 305, 315, 310),
            SensorField("Rotational speed rpm", "Spindle speed", "rpm", 1150, 2900, 1500),
            SensorField("Torque Nm", "Torque", "Nm", 3, 77, 40),
            SensorField("Tool wear min", "Tool wear", "min", 0, 260, 100),
        ],
    ),

    "turbine": MachineType(
        key="turbine", name="Gas / Jet Turbine",
        description="Multi-stage turbines with gas-path sensor arrays (temps, pressures, speeds).",
        dataset="NASA C-MAPSS turbofan degradation simulation (real physics, run-to-failure)",
        trained=True, model_path="backtest_model.pkl",
        sensors=[
            SensorField("cycle", "Operating cycle (age)", "cycles", 1, 400, 50),
            SensorField("sensor_2", "LPC outlet temperature", "R", 600, 650, 642),
            SensorField("sensor_3", "HPC outlet temperature", "R", 1570, 1620, 1590),
            SensorField("sensor_4", "LPT outlet temperature", "R", 1380, 1430, 1400),
            SensorField("sensor_7", "Total HPC outlet pressure", "psia", 550, 560, 554),
            SensorField("sensor_11", "HPC outlet static pressure", "psia", 46, 49, 47.5),
            SensorField("sensor_12", "Ratio fuel flow / Ps30", "pps/psia", 518, 524, 521.5),
            SensorField("sensor_15", "Bypass ratio", "", 8.3, 8.6, 8.44),
        ],
        note="Manual entry uses raw readings only; rolling-average/delta features "
             "(computed from an engine's own history) are added automatically once "
             "readings accumulate for a machine.",
    ),

    "rotating_equipment": MachineType(
        key="rotating_equipment", name="Motor / Pump / Fan / Compressor / Gearbox",
        description="Any rotating machine whose dominant failure mode is bearing wear.",
        dataset="University of Ottawa UORED-VAFCLS (doi:10.17632/y2px5tg92h.2): 20 real "
                "bearings, each recorded healthy -> developing fault -> faulty",
        trained=True, model_path="bearing_model.pkl",
        sensors=[
            SensorField("rms", "Vibration RMS", "accel. units", 0.5, 120.0, 3.3),
            SensorField("peak", "Vibration peak", "accel. units", 5.0, 900.0, 15.5),
            SensorField("crest", "Crest factor (peak/RMS)", "", 1.0, 20.0, 4.6),
            SensorField("kurtosis", "Kurtosis (impulsiveness, 0 = smooth)", "", -1.0, 40.0, 0.2),
            SensorField("spec_cent", "Spectral centroid", "Hz", 500, 10000, 2900),
        ],
        note="Trained on features of an accelerometer waveform (42 kHz), which usually come "
             "from a vibration sensor + edge device rather than being typed in. The 5 fields "
             "above are the most useful for a quick manual check; a full CSV/API reading "
             "(rms, std, peak, p2p, crest, kurtosis, skewness, shape, impulse, spec_cent, "
             "spec_bw, spec_ent, band0-4) scores more accurately. Reports health state "
             "(healthy / developing fault / faulty) and, once damaged, the likely defect "
             "(inner race, outer race, ball, cage). Absolute RMS/peak depend on the sensor, "
             "so calibrate against a known-healthy reading from the same sensor first.",
    ),

    "wind_turbine_generator": MachineType(
        key="wind_turbine_generator", name="Wind Turbine / Generator",
        description="Large rotating generation equipment with SCADA sensor arrays.",
        dataset="Real wind-farm SCADA telemetry (operating turbines, real status/fault codes)",
        trained=True, model_path="wind_turbine_model.pkl",
        sensors=[
            SensorField("sensor_0_avg", "Sensor 0 average (e.g. wind speed)", "", -50, 50, 0),
            SensorField("sensor_1_avg", "Sensor 1 average", "", -50, 50, 0),
            SensorField("power_2_avg", "Power output average", "kW", -50, 2500, 500),
            SensorField("sensor_9_avg", "Sensor 9 average", "", -50, 50, 0),
            SensorField("sensor_13_avg", "Sensor 13 average", "", -50, 50, 0),
        ],
        note="SCADA sensor names in the source data are anonymized (sensor_N); the "
             "labels above are best-effort guesses at what they represent. A full "
             "CSV/API reading with the original avg/max/min/std columns scores more "
             "accurately. See reports/wind_turbine_results.md for which columns "
             "actually drive the prediction.",
    ),

    "hvac": MachineType(
        key="hvac", name="HVAC Rooftop Unit (RTU)",
        description="Packaged rooftop heating/cooling units -- the most common HVAC equipment "
                    "in small commercial buildings. Detects economizer/damper and "
                    "supply-air control faults from standard building-automation points.",
        dataset="LBNL FDD Data Sets (U.S. DOE; LBNL/ORNL/NREL, DOI 10.25984/1881324) -- real "
                "Trane 12.5-ton RTU at Oak Ridge National Laboratory with faults physically "
                "imposed across four seasons",
        trained=True, model_path="hvac_model.pkl",
        sensors=[
            SensorField("RTU_OA_TEMP", "Outside air temperature", "F", 0, 110, 69),
            SensorField("RTU_MA_TEMP", "Mixed air temperature", "F", 40, 95, 68),
            SensorField("RTU_RA_TEMP", "Return air temperature", "F", 55, 90, 70),
            SensorField("RTU_SA_TEMP", "Supply air temperature", "F", 40, 90, 56),
            SensorField("RTU_OA_DMPR_DM", "Outdoor-air damper command", "% open", 0, 100, 10),
            SensorField("RTU_SA_FAN_WATT", "Supply fan power", "W", 0, 4000, 1780),
            SensorField("RTU_COMP_WATT_1", "Compressor 1 power", "W", 0, 8000, 4200),
            SensorField("RTU_COMP_WATT_2", "Compressor 2 power", "W", 0, 8000, 0),
            SensorField("OCCU_MOD", "Occupied schedule (1 = yes, 0 = no)", "", 0, 1, 1),
            SensorField("hour", "Hour of day", "h", 0, 23, 12),
        ],
        note="Faults only show while the unit is running, so readings during occupied "
             "hours matter most. Rolling features build up from this unit's own recent "
             "readings, so a steady feed (e.g. one reading per minute via the API) scores "
             "far better than a single manual entry. Validated on one real unit; a "
             "customer's own units should be spot-checked before relying on it.",
    ),

    "robotic_arm": MachineType(
        key="robotic_arm", name="Industrial Robotic Arm / Cobot",
        description="6-axis robot arms. Warns ~10 seconds before a protective stop or "
                    "gripper loss, from joint currents, temperatures and speeds.",
        dataset="UR3 CobotOps, UCI Machine Learning Repository #963 -- real telemetry and "
                "logged protective stops/grip losses from a Universal Robots UR3 cobot",
        trained=True, model_path="robot_model.pkl",
        sensors=(
            [SensorField(f"Current_J{j}", f"Joint {j} current", "A", -6.0, 6.0, 0.0)
             for j in range(6)]
            + [SensorField(f"Temperature_J{j}", f"Joint {j} temperature", "C", 20, 70, 40)
               for j in range(6)]
            + [SensorField(f"Speed_J{j}", f"Joint {j} speed", "rad/s", -3.0, 3.0, 0.0)
               for j in range(6)]
            + [SensorField("Tool_current", "Gripper/tool current", "A", 0.0, 1.0, 0.085)]
        ),
        note="Expects about one reading per second from the robot controller (UR robots "
             "expose these over RTDE/MODBUS). Rolling features use this robot's own last "
             "~10 readings, so feed it continuously via the API. Validated on one UR3 "
             "running one program -- retrain on a customer's own robot logs before relying on it.",
    ),

    "conveyor": MachineType(
        key="conveyor", name="Conveyor Drive (motor, shaft, bearings)",
        description="The conveyor's drive train -- where most unplanned conveyor downtime "
                    "starts. Detects bearing defects, shaft misalignment and rotor unbalance "
                    "from one accelerometer on the drive.",
        dataset="KAIST rotating-machine dataset (Jung et al., Data in Brief 2023, "
                "doi:10.17632/ztmf3m7h5x.6) -- real test rig, faults at several severities "
                "under 3 loads, order-tracked vibration",
        trained=True, model_path="conveyor_model.pkl",
        sensors=[
            SensorField("rms", "Vibration RMS", "g", 0.02, 1.5, 0.13),
            SensorField("peak", "Vibration peak", "g", 0.05, 8.0, 0.45),
            SensorField("crest", "Crest factor (peak/RMS)", "", 1.0, 15.0, 3.4),
            SensorField("kurtosis", "Kurtosis (impulsiveness, 0 = smooth)", "", -1.5, 10.0, -0.2),
            SensorField("order1_amp", "Vibration at 1x shaft speed", "g", 0.0, 0.003, 0.0004),
            SensorField("band3", "High-frequency share (5-12.8 kHz)", "0-1", 0.0, 1.0, 0.2),
        ],
        note="Covers the drive train only -- NOT belt tears, belt mistracking or idler "
             "failures (no reliable open dataset exists for those; the one real one, 135 "
             "idlers recorded in a working mine, is access-restricted). Light unbalance "
             "(under ~2 g of added mass on the test rig) is physically hard to see and is "
             "usually missed. Validated on one test rig at three loads.",
    ),
}


CUSTOM_DIR = ROOT / "custom_models"


def _load_custom_types() -> None:
    """Machine types trained on a customer's own data by train_custom.py."""
    import json
    for spec_path in sorted(CUSTOM_DIR.glob("*/type.json")):
        spec = json.loads(spec_path.read_text())
        MACHINE_TYPES[spec["key"]] = MachineType(
            key=spec["key"], name=spec["name"], description=spec["description"],
            dataset=spec["dataset"], trained=True,
            model_path=str(spec_path.parent.relative_to(ROOT) / "model.pkl"),
            sensors=[SensorField(**s) for s in spec["sensors"]], note=spec.get("note", ""))


_load_custom_types()


def get(key: str) -> MachineType:
    if key not in MACHINE_TYPES:
        raise KeyError(f"Unknown machine type '{key}'. Known: {list(MACHINE_TYPES)}")
    return MACHINE_TYPES[key]
