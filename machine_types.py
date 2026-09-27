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
        dataset="CWRU bearing fault dataset (Case Western Reserve University, real induced-fault vibration experiments)",
        trained=True, model_path="bearing_model.pkl",
        sensors=[
            SensorField("rms", "Vibration RMS", "g", 0.2, 3.0, 0.6),
            SensorField("peak", "Vibration peak", "g", 0.5, 8.0, 2.0),
            SensorField("crest", "Crest factor (peak/RMS)", "", 1.0, 6.0, 3.0),
            SensorField("kurtosis", "Kurtosis (impulsiveness)", "", -1.0, 15.0, 0.0),
            SensorField("spec_cent", "Spectral centroid", "Hz", 100, 1500, 700),
        ],
        note="Trained on statistical features of a vibration waveform, which usually "
             "come from an accelerometer + edge device, not a hand-typed value. The "
             "5 fields above are the most predictive ones for a quick manual check; "
             "a full CSV/API reading (rms, std, kurtosis, skewness, peak, p2p, crest, "
             "shape, energy, spec_cent, spec_bw, spec_ent, band0-3, wE0-4, wS0-4) "
             "scores more accurately. Diagnoses Normal vs Ball/Inner-race/Outer-race fault.",
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
        key="hvac", name="HVAC / Chiller / Compressor (climate)",
        description="Not yet trained on real data.",
        dataset="No real, publicly available failure-labeled dataset found. The one public "
                "HVAC fault dataset is explicitly synthetic, so it isn't used here.",
        trained=False,
        sensors=[
            SensorField("suction_pressure", "Suction pressure", "psi", 50, 150, 100),
            SensorField("discharge_pressure", "Discharge pressure", "psi", 150, 400, 250),
            SensorField("suction_temp", "Suction temperature", "F", 30, 60, 45),
            SensorField("compressor_current", "Compressor current", "A", 5, 60, 20),
        ],
        note="Schema only. Needs a business's own maintenance history to train a real model.",
    ),

    "robotic_arm": MachineType(
        key="robotic_arm", name="Industrial Robotic Arm",
        description="Not yet trained on real data.",
        dataset="No public failure-labeled dataset found for robotic arms.",
        trained=False,
        sensors=[
            SensorField("joint_torque", "Joint torque", "Nm", 0, 200, 50),
            SensorField("motor_current", "Motor current", "A", 0, 40, 10),
            SensorField("servo_temp", "Servo temperature", "C", 20, 90, 45),
        ],
        note="Schema only. Needs a business's own maintenance history to train a real model.",
    ),

    "conveyor": MachineType(
        key="conveyor", name="Conveyor / Belt System",
        description="Not yet trained on real data.",
        dataset="No public failure-labeled dataset found for conveyor systems.",
        trained=False,
        sensors=[
            SensorField("motor_current", "Drive motor current", "A", 0, 50, 15),
            SensorField("belt_tension", "Belt tension", "N", 0, 5000, 2000),
            SensorField("vibration_rms", "Vibration RMS", "g", 0, 3, 0.4),
        ],
        note="Schema only. Needs a business's own maintenance history to train a real model.",
    ),
}


def get(key: str) -> MachineType:
    if key not in MACHINE_TYPES:
        raise KeyError(f"Unknown machine type '{key}'. Known: {list(MACHINE_TYPES)}")
    return MACHINE_TYPES[key]
