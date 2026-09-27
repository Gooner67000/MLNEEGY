"""Feature engineering shared by training AND the live backend, for the machine
types whose features depend on a machine's recent history (HVAC, robot arm).

Each function takes ONE machine's readings as a DataFrame in time order and
returns it with feature columns added. Every rolling/diff feature looks only
backward (the row itself and earlier rows), so the feature for reading t never
uses anything after t. The backend calls the same function on
[that machine's stored history + the new reading] and keeps the last row, so a
live score is built exactly the way the model was validated.
"""
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ HVAC (RTU)
HVAC_RAW = [
    "RTU_OA_TEMP",      # outside air temperature, degF
    "RTU_MA_TEMP",      # mixed air temperature, degF
    "RTU_RA_TEMP",      # return air temperature, degF
    "RTU_SA_TEMP",      # supply air temperature, degF
    "RTU_OA_DMPR_DM",   # outdoor-air damper command, % open (0-100)
    "RTU_SA_FAN_WATT",  # supply fan power, W
    "RTU_COMP_WATT_1",  # compressor 1 power, W
    "RTU_COMP_WATT_2",  # compressor 2 power, W
    "RTU_GAS_CSUM",     # gas heat use (0 when not heating)
    "OCCU_MOD",         # 1 = occupied schedule, 0 = unoccupied
    "hour",             # hour of day, 0-23
]
HVAC_SA_SETPOINT_F = 55.0  # supply-air setpoint used by the test unit
HVAC_ROLL = 15             # minutes


def hvac_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in HVAC_RAW:
        out[c] = pd.to_numeric(out.get(c, np.nan), errors="coerce")
    dmpr = out["RTU_OA_DMPR_DM"]
    dmpr = np.where(dmpr > 1.0, dmpr / 100.0, dmpr)  # accept % or 0-1
    out["dmpr_frac"] = np.clip(dmpr, 0, 1)
    oa, ma, ra = out["RTU_OA_TEMP"], out["RTU_MA_TEMP"], out["RTU_RA_TEMP"]
    # Mixing-box energy balance: with the damper where it's commanded, mixed air
    # should sit between outdoor and return air in that proportion. A stuck or
    # mis-set damper breaks this.
    out["mix_resid"] = ma - (out["dmpr_frac"] * oa + (1 - out["dmpr_frac"]) * ra)
    gap = oa - ra
    out["oa_frac_est"] = np.where(np.abs(gap) > 3, (ma - ra) / gap.replace(0, np.nan), np.nan)
    out["oa_frac_est"] = out["oa_frac_est"].clip(-1, 2)
    out["oa_frac_gap"] = out["oa_frac_est"] - out["dmpr_frac"]
    out["sa_dev"] = out["RTU_SA_TEMP"] - HVAC_SA_SETPOINT_F
    out["comp_total"] = out["RTU_COMP_WATT_1"].fillna(0) + out["RTU_COMP_WATT_2"].fillna(0)
    # Economizer should be open when it's cool outside during occupied hours.
    out["econ_should_run"] = ((oa < 50) & (ma > 45) & (out["OCCU_MOD"] == 1)).astype(float)
    out["econ_dmpr"] = out["econ_should_run"] * out["dmpr_frac"]
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    # Control-sequence checks: what the damper SHOULD do given the conditions,
    # vs. what it's doing. (A damper fault imposed through the controller shows
    # up here, not as a command-vs-measurement mismatch.)
    running = (out["RTU_SA_FAN_WATT"] > 200).astype(float)
    out["dmpr_below_min"] = ((out["dmpr_frac"] < 0.095) & (running > 0)).astype(float)
    out["dmpr_open_when_hot"] = out["dmpr_frac"] * (oa > 55).astype(float) * running
    out["dmpr_shut_when_cool"] = (1 - out["dmpr_frac"]) * out["econ_should_run"]
    cooling = (out["comp_total"] > 100).astype(float)
    out["sa_dev_cooling"] = out["sa_dev"] * cooling  # SA is only held at setpoint while cooling
    for c in ["mix_resid", "oa_frac_gap", "sa_dev", "comp_total", "RTU_SA_FAN_WATT", "dmpr_frac",
              "dmpr_open_when_hot", "dmpr_shut_when_cool", "sa_dev_cooling"]:
        out[f"{c}_roll"] = out[c].rolling(HVAC_ROLL, min_periods=1).mean()
    return out


HVAC_FEATURES = [
    "RTU_OA_TEMP", "RTU_MA_TEMP", "RTU_RA_TEMP", "RTU_SA_TEMP", "dmpr_frac",
    "RTU_SA_FAN_WATT", "comp_total", "RTU_GAS_CSUM", "OCCU_MOD",
    "mix_resid", "oa_frac_est", "oa_frac_gap", "sa_dev", "econ_should_run", "econ_dmpr",
    "hour_sin", "hour_cos",
    "mix_resid_roll", "oa_frac_gap_roll", "sa_dev_roll", "comp_total_roll",
    "RTU_SA_FAN_WATT_roll", "dmpr_frac_roll",
    "dmpr_below_min", "dmpr_open_when_hot", "dmpr_shut_when_cool", "sa_dev_cooling",
    "dmpr_open_when_hot_roll", "dmpr_shut_when_cool_roll", "sa_dev_cooling_roll",
]

# ------------------------------------------------------------ robot arm (UR3)
ROBOT_JOINTS = range(6)
ROBOT_RAW = (
    [f"Current_J{j}" for j in ROBOT_JOINTS]
    + [f"Temperature_J{j}" for j in ROBOT_JOINTS]
    + [f"Speed_J{j}" for j in ROBOT_JOINTS]
    + ["Tool_current"]
)
ROBOT_ROLL = 10  # readings (~10 s at 1 Hz)


def robot_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ROBOT_RAW:
        out[c] = pd.to_numeric(out.get(c, np.nan), errors="coerce")
    cur = out[[f"Current_J{j}" for j in ROBOT_JOINTS]].abs()
    tmp = out[[f"Temperature_J{j}" for j in ROBOT_JOINTS]]
    spd = out[[f"Speed_J{j}" for j in ROBOT_JOINTS]].abs()
    out["cur_total"] = cur.sum(axis=1)
    out["cur_max"] = cur.max(axis=1)
    out["temp_max"] = tmp.max(axis=1)
    out["temp_mean"] = tmp.mean(axis=1)
    out["temp_spread"] = out["temp_max"] - tmp.min(axis=1)
    out["speed_total"] = spd.sum(axis=1)
    # Current needed per unit of motion: rises when a joint is loaded/obstructed.
    out["cur_per_speed"] = out["cur_total"] / (out["speed_total"] + 0.05)
    r = out["cur_total"].rolling(ROBOT_ROLL, min_periods=1)
    out["cur_total_roll"] = r.mean()
    out["cur_total_std"] = r.std().fillna(0)
    out["cur_jump"] = out["cur_total"] - out["cur_total_roll"]
    out["tool_cur_roll"] = out["Tool_current"].rolling(ROBOT_ROLL, min_periods=1).mean()
    out["temp_rate"] = (out["temp_max"] - out["temp_max"].shift(ROBOT_ROLL)).fillna(0)
    return out


def generic_features(df: pd.DataFrame, sensors: list[str], roll: int, long_roll: int) -> pd.DataFrame:
    """For a customer's own machine type (train_custom.py): per sensor, the
    reading itself, its short rolling mean/std, its change since the last
    reading, and how far it sits from this machine's own longer-run baseline.
    All backward-looking."""
    out = df.copy()
    for c in sensors:
        x = pd.to_numeric(out.get(c, np.nan), errors="coerce")
        out[c] = x
        r = x.rolling(roll, min_periods=1)
        out[f"{c}__mean"] = r.mean()
        out[f"{c}__std"] = r.std().fillna(0)
        out[f"{c}__diff"] = x.diff().fillna(0)
        base = x.rolling(long_roll, min_periods=1)
        out[f"{c}__z"] = ((x - base.mean()) / base.std().replace(0, np.nan)).fillna(0)
    return out


def generic_feature_names(sensors: list[str]) -> list[str]:
    return [f for c in sensors for f in (c, f"{c}__mean", f"{c}__std", f"{c}__diff", f"{c}__z")]


ROBOT_FEATURES = ROBOT_RAW + [
    "cur_total", "cur_max", "temp_max", "temp_mean", "temp_spread", "speed_total",
    "cur_per_speed", "cur_total_roll", "cur_total_std", "cur_jump", "tool_cur_roll", "temp_rate",
]
