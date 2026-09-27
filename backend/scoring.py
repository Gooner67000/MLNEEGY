"""Turns a raw sensor reading (+ that machine's own reading history) into a
risk score, dispatching per machine type. Every trained type's causal/rolling
features are built the SAME way they were during training and backtesting --
only that machine's own past readings, never another machine's, never a
future reading -- so a live score has the same no-lookahead guarantee the
backtest proved.
"""
from datetime import datetime
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


def _defaulted_note(provided: dict, expected: list[str]) -> str | None:
    missing = [f for f in expected if f not in provided]
    if not missing:
        return None
    return (f"{len(missing)} of {len(expected)} inputs weren't provided and were filled with "
            f"typical healthy values; send the full set via CSV/API for the most accurate score.")


def _score_bearing(payload: dict) -> dict:
    model, meta = _load("bearing_model.pkl", "bearing_meta.pkl")
    type_model, _ = _load("bearing_type_model.pkl", None)
    feats = meta["features"]
    row = {f: payload.get(f, meta["medians"][f]) for f in feats}
    X = pd.DataFrame([row])[feats]
    proba = model.predict_proba(X)[0]
    states = meta["classes"]  # healthy / developing / faulty
    prob_damaged = float(1 - proba[states.index("healthy")])
    state = states[int(proba.argmax())]
    diagnosis = state
    if state != "healthy":
        defect = meta["type_classes"][int(type_model.predict(X)[0])]
        diagnosis = f"{state} fault (likely {defect.replace('_', ' ')})"
    return {"trained": True, "probability": round(prob_damaged, 4),
            "risk_level": risk_level(prob_damaged), "alert": state != "healthy",
            "diagnosis": diagnosis, "note": _defaulted_note(payload, feats)}


def _history_frame(payload: dict, history: list[dict], raw: list[str], medians: dict) -> pd.DataFrame:
    """This machine's recent readings + the new one, oldest first, with any
    missing raw inputs filled from typical values."""
    rows = [{c: h.get(c, medians.get(c)) for c in raw} for h in history]
    rows.append({c: payload.get(c, medians.get(c)) for c in raw})
    return pd.DataFrame(rows)


def _score_hvac(payload: dict, history: list[dict]) -> dict:
    from pm_features import hvac_features
    model, meta = _load("hvac_model.pkl", "hvac_meta.pkl")
    diag_model, _ = _load("hvac_diag_model.pkl", None)
    payload = dict(payload)
    payload.setdefault("hour", float(datetime.now().hour))
    feats = hvac_features(_history_frame(payload, history, meta["raw"], meta["medians"]))
    X = feats.iloc[[-1]][meta["features"]]
    # Validated as an average over recent running minutes, so alert on the same
    # smoothed score (a single minute on its own is too noisy to act on).
    recent = feats.iloc[-meta.get("smooth_minutes", 15):]
    probs = model.predict_proba(recent[meta["features"]])[:, 1]
    prob = float(probs.mean())
    alert = prob >= meta["threshold"]
    diagnosis = None
    if alert:
        p = diag_model.predict_proba(X)[0]
        classes = meta["classes"]
        p[classes.index("fault_free")] = 0  # we already know it's a fault; name the likeliest one
        diagnosis = classes[int(p.argmax())].replace("_", " ")
    return {"trained": True, "probability": round(prob, 4), "risk_level": risk_level(prob),
            "alert": alert, "diagnosis": diagnosis,
            "note": _defaulted_note(payload, [c for c in meta["raw"] if c != "hour"])}


def _score_conveyor(payload: dict) -> dict:
    model, meta = _load("conveyor_model.pkl", "conveyor_meta.pkl")
    feats = meta["features"]
    X = pd.DataFrame([{f: payload.get(f, meta["medians"][f]) for f in feats}])[feats]
    proba = model.predict_proba(X)[0]
    classes = meta["classes"]
    prob_fault = float(1 - proba[classes.index("normal")])
    top = classes[int(proba.argmax())]
    return {"trained": True, "probability": round(prob_fault, 4),
            "risk_level": risk_level(prob_fault), "alert": prob_fault >= 0.5,
            "diagnosis": None if top == "normal" else top.replace("_", " "),
            "note": _defaulted_note(payload, feats)}


def _score_custom(machine_type: str, payload: dict, history: list[dict]) -> dict:
    from pm_features import generic_features
    folder = f"custom_models/{machine_type}"
    model, meta = _load(f"{folder}/model.pkl", f"{folder}/meta.pkl")
    frame = _history_frame(payload, history, meta["raw"], meta["medians"])
    feats = generic_features(frame, meta["raw"], meta["roll"], meta["long_roll"])
    X = feats.iloc[[-1]][meta["features"]]
    prob = float(model.predict_proba(X)[0, 1])
    return {"trained": True, "probability": round(prob, 4), "risk_level": risk_level(prob),
            "alert": prob >= meta["threshold"],
            "diagnosis": (f"failure likely within {meta['horizon']}"
                          if prob >= meta["threshold"] else None),
            "note": _defaulted_note(payload, meta["raw"])}


def _score_robot(payload: dict, history: list[dict]) -> dict:
    from pm_features import robot_features
    model, meta = _load("robot_model.pkl", "robot_meta.pkl")
    feats = robot_features(_history_frame(payload, history, meta["raw"], meta["medians"]))
    X = feats.iloc[[-1]][meta["features"]]
    prob = float(model.predict_proba(X)[0, 1])
    return {"trained": True, "probability": round(prob, 4), "risk_level": risk_level(prob),
            "alert": prob >= meta["threshold"],
            "diagnosis": ("protective stop / grip loss likely within ~10 s"
                          if prob >= meta["threshold"] else None),
            "note": _defaulted_note(payload, meta["raw"])}


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
    if machine_type == "hvac":
        return _score_hvac(payload, history)
    if machine_type == "robotic_arm":
        return _score_robot(payload, history)
    if machine_type == "conveyor":
        return _score_conveyor(payload)
    if machine_type.startswith("custom_"):
        return _score_custom(machine_type, payload, history)
    raise ValueError(f"No scoring function wired up for trained type '{machine_type}'")
