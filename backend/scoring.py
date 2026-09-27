"""Turns a raw sensor reading (+ that machine's own reading history) into a
risk score, dispatching per machine type. Every trained type's causal/rolling
features are built the SAME way they were during training and backtesting --
only that machine's own past readings, never another machine's, never a
future reading -- so a live score has the same no-lookahead guarantee the
backtest proved.
"""
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

import machine_types as mt

ROOT = Path(__file__).parent.parent


def risk_level(prob: float) -> str:
    if prob < 0.3:
        return "Low"
    if prob < 0.7:
        return "Medium"
    return "High"


@lru_cache(maxsize=None)
def _load(model_file: str, meta_file: str | None):
    model = joblib.load(ROOT / model_file)
    meta = joblib.load(ROOT / meta_file) if meta_file else None
    return model, meta


def _not_trained(type_def: mt.MachineType) -> dict:
    return {"trained": False, "probability": None, "risk_level": None, "alert": None,
            "diagnosis": None, "note": type_def.note or "No trained model for this machine type yet."}


def _score_cnc(payload: dict) -> dict:
    import predict as cnc  # local import: only needed for this branch
    model, feature_names, threshold = cnc.load_model()
    row = {k: payload.get(k, 0) for k in cnc.RAW_FEATURES}
    X = cnc.add_features(pd.DataFrame([row]))[feature_names]
    prob = float(model.predict_proba(X)[0, 1])
    return {"trained": True, "probability": round(prob, 4), "risk_level": risk_level(prob),
            "alert": prob >= threshold, "diagnosis": None, "note": None}


def _score_turbine(payload: dict, history: list[dict]) -> dict:
    model, meta = _load("backtest_model.pkl", "turbine_meta.pkl")
    live_sensors, window = meta["live_sensors"], meta["roll_window"]
    row = {k: payload.get(k, 0.0) for k in meta["raw_sensor_cols"]}
    past = [{k: h.get(k, 0.0) for k in live_sensors} for h in history[-(window - 1):]]
    for c in live_sensors:
        recent = [p[c] for p in past] + [row[c]]
        row[f"{c}_roll_mean"] = sum(recent) / len(recent)
        row[f"{c}_delta"] = row[c] - (past[-1][c] if past else row[c])
    X = pd.DataFrame([row])[meta["features"]]
    prob = float(model.predict_proba(X)[0, 1])
    return {"trained": True, "probability": round(prob, 4), "risk_level": risk_level(prob),
            "alert": prob >= 0.5, "diagnosis": None, "note": None}


def _score_bearing(payload: dict) -> dict:
    model, meta = _load("bearing_model.pkl", "bearing_meta.pkl")
    row = {f: payload.get(f, meta["medians"][f]) for f in meta["features"]}
    X = pd.DataFrame([row])[meta["features"]]
    proba = model.predict_proba(X)[0]
    classes = meta["classes"]
    normal_idx = classes.index("Normal")
    prob_fault = float(1 - proba[normal_idx])
    diagnosis = classes[int(proba.argmax())]
    missing = [f for f in meta["features"] if f not in payload]
    note = (f"{len(missing)} of {len(meta['features'])} features defaulted to training "
            f"medians (only provided: {sorted(set(meta['features']) - set(missing))[:5]}...)"
            if missing else None)
    return {"trained": True, "probability": round(prob_fault, 4), "risk_level": risk_level(prob_fault),
            "alert": diagnosis != "Normal", "diagnosis": diagnosis, "note": note}


def _score_wind_turbine(payload: dict, history: list[dict]) -> dict:
    model, meta = _load("wind_turbine_model.pkl", "wind_turbine_meta.pkl")
    sensors, window, medians = meta["raw_sensor_cols"], meta["roll_window"], meta["medians"]
    row = {c: payload.get(c, medians[c]) for c in sensors}
    past = [{c: h.get(c, medians[c]) for c in sensors} for h in history[-(window - 1):]]
    for c in sensors:
        recent = [p[c] for p in past] + [row[c]]
        row[f"{c}_roll"] = sum(recent) / len(recent)
    X = pd.DataFrame([row])[meta["features"]]
    prob = float(model.predict_proba(X)[0, 1])
    return {"trained": True, "probability": round(prob, 4), "risk_level": risk_level(prob),
            "alert": prob >= 0.5, "diagnosis": None, "note": None}


def score(machine_type: str, payload: dict, history: list[dict]) -> dict:
    """history: that machine's past readings' payloads, oldest first, NOT
    including the current one -- comes straight from the Reading table."""
    type_def = mt.get(machine_type)
    if not type_def.trained:
        return _not_trained(type_def)
    if machine_type == "cnc_machine_tool":
        return _score_cnc(payload)
    if machine_type == "turbine":
        return _score_turbine(payload, history)
    if machine_type == "rotating_equipment":
        return _score_bearing(payload)
    if machine_type == "wind_turbine_generator":
        return _score_wind_turbine(payload, history)
    raise ValueError(f"No scoring function wired up for trained type '{machine_type}'")
