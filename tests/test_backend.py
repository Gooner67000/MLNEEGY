"""Backend API tests against an isolated in-memory SQLite DB (never the real
data/app.db). Model-dependent assertions are skipped if the .pkl files
haven't been trained yet in this environment (see tests/test_predict.py for
the same pattern)."""
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database import Base, get_db
from backend.main import app

ROOT = Path(__file__).parent.parent
needs_model = pytest.mark.skipif(
    not (ROOT / "predictive_maintenance_model.pkl").exists(), reason="run train.py first"
)
needs_bearing = pytest.mark.skipif(
    not (ROOT / "bearing_type_model.pkl").exists(), reason="run train_bearing.py first"
)
needs_hvac = pytest.mark.skipif(
    not (ROOT / "hvac_model.pkl").exists(), reason="run train_hvac.py first"
)
needs_robot = pytest.mark.skipif(
    not (ROOT / "robot_model.pkl").exists(), reason="run train_robot.py first"
)
needs_conveyor = pytest.mark.skipif(
    not (ROOT / "conveyor_model.pkl").exists(), reason="run train_conveyor.py first"
)

engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                       poolclass=StaticPool)
TestingSessionLocal = sessionmaker(bind=engine)
Base.metadata.create_all(bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


def register(email="a@example.com") -> str:
    resp = client.post("/auth/register", json={"name": "Acme", "email": email, "password": "password123"})
    assert resp.status_code == 200
    return resp.json()["access_token"]


def auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_machine_types_lists_all_categories():
    types = client.get("/machine-types").json()
    for key in ("cnc_machine_tool", "turbine", "rotating_equipment",
               "wind_turbine_generator", "hvac", "robotic_arm", "conveyor"):
        assert key in types
    for key in types:
        assert types[key]["trained"] is True, f"{key} should be backed by a trained model"
        assert types[key]["dataset"], f"{key} must name its training data source"


def test_register_login_and_reject_duplicate_email():
    token = register("dup@example.com")
    assert token
    resp = client.post("/auth/login", json={"email": "dup@example.com", "password": "password123"})
    assert resp.status_code == 200
    dupe = client.post("/auth/register", json={"name": "Acme2", "email": "dup@example.com",
                                               "password": "password123"})
    assert dupe.status_code == 409


def test_unauthenticated_requests_are_rejected():
    resp = client.get("/machines")
    assert resp.status_code in (401, 403)


def test_businesses_cannot_see_each_others_machines():
    token_a = register("iso-a@example.com")
    token_b = register("iso-b@example.com")
    client.post("/machines", json={"name": "A's mill", "machine_type": "cnc_machine_tool"},
               headers=auth(token_a))
    machines_b = client.get("/machines", headers=auth(token_b)).json()
    assert machines_b == []


def test_unknown_machine_type_rejected():
    token = register("bad-type@example.com")
    resp = client.post("/machines", json={"name": "X", "machine_type": "nonexistent"},
                       headers=auth(token))
    assert resp.status_code == 400


def test_untrained_machine_type_reports_not_trained_instead_of_faking_a_score(monkeypatch):
    # Every shipped type is trained now, so register a stand-in untrained one --
    # the guarantee still matters for any type a customer adds before training.
    import machine_types as mt
    monkeypatch.setitem(mt.MACHINE_TYPES, "custom_untrained", mt.MachineType(
        key="custom_untrained", name="Custom", description="", dataset="none yet",
        trained=False, note="Needs this business's own data."))
    token = register("untrained@example.com")
    m = client.post("/machines", json={"name": "Press-1", "machine_type": "custom_untrained"},
                    headers=auth(token)).json()
    resp = client.post(f"/machines/{m['id']}/readings",
                       json={"payload": {"x": 1}, "source": "manual"},
                       headers=auth(token))
    assert resp.status_code == 200
    body = resp.json()
    assert body["trained"] is False
    assert body["probability"] is None


@needs_model
def test_cnc_machine_scores_a_reading():
    token = register("cnc@example.com")
    m = client.post("/machines", json={"name": "Mill-1", "machine_type": "cnc_machine_tool"},
                    headers=auth(token)).json()
    payload = {"Type": 0, "Air temperature K": 300, "Process temperature K": 310,
              "Rotational speed rpm": 1500, "Torque Nm": 40, "Tool wear min": 100}
    resp = client.post(f"/machines/{m['id']}/readings", json={"payload": payload, "source": "manual"},
                       headers=auth(token))
    assert resp.status_code == 200
    body = resp.json()
    assert body["trained"] is True
    assert 0 <= body["probability"] <= 1
    assert body["risk_level"] in ("Low", "Medium", "High")

    latest = client.get(f"/machines/{m['id']}/risk/latest", headers=auth(token)).json()
    assert latest["reading_id"] == body["reading_id"]


@needs_model
def test_bulk_csv_style_ingestion():
    token = register("bulk@example.com")
    m = client.post("/machines", json={"name": "Mill-2", "machine_type": "cnc_machine_tool"},
                    headers=auth(token)).json()
    base = {"Type": 0, "Air temperature K": 300, "Process temperature K": 310,
           "Rotational speed rpm": 1500, "Torque Nm": 40, "Tool wear min": 100}
    readings = [{"payload": {**base, "Tool wear min": w}, "source": "csv"} for w in (50, 100, 200)]
    resp = client.post(f"/machines/{m['id']}/readings/bulk", json={"readings": readings},
                       headers=auth(token))
    assert resp.status_code == 200
    assert len(resp.json()) == 3
    history = client.get(f"/machines/{m['id']}/risk/history", headers=auth(token)).json()
    assert len(history) == 3


def _new_machine(email: str, machine_type: str):
    token = register(email)
    m = client.post("/machines", json={"name": f"{machine_type}-1", "machine_type": machine_type},
                    headers=auth(token)).json()
    return token, m["id"]


def _score(token, machine_id, payload) -> dict:
    resp = client.post(f"/machines/{machine_id}/readings",
                       json={"payload": payload, "source": "api"}, headers=auth(token))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["trained"] is True
    assert 0 <= body["probability"] <= 1
    assert body["risk_level"] in ("Low", "Medium", "High")
    return body


@needs_bearing
def test_bearing_partial_reading_is_scored_and_flagged_as_defaulted():
    token, mid = _new_machine("bearing@example.com", "rotating_equipment")
    body = _score(token, mid, {"rms": 3.3, "peak": 15.5, "crest": 4.6})
    assert body["diagnosis"].split(" ")[0] in ("healthy", "developing", "faulty")
    assert body["note"] is not None  # tells the user some inputs were defaulted


@needs_bearing
def test_bearing_damaged_signature_scores_riskier_than_healthy():
    # Typical values from the training data: healthy vs. developing-fault bearings.
    token, mid = _new_machine("bearing2@example.com", "rotating_equipment")
    healthy = _score(token, mid, {"rms": 3.3, "peak": 15.5, "crest": 4.6, "kurtosis": 0.2,
                                  "spec_cent": 2900})
    damaged = _score(token, mid, {"rms": 18.5, "peak": 195, "crest": 7.7, "kurtosis": 4.5,
                                  "spec_cent": 3200})
    assert damaged["probability"] > healthy["probability"]


@needs_hvac
def test_hvac_scores_a_stream_of_readings():
    token, mid = _new_machine("hvac@example.com", "hvac")
    reading = {"RTU_OA_TEMP": 69, "RTU_MA_TEMP": 68, "RTU_RA_TEMP": 70, "RTU_SA_TEMP": 56,
               "RTU_OA_DMPR_DM": 10, "RTU_SA_FAN_WATT": 1780, "RTU_COMP_WATT_1": 4200,
               "RTU_COMP_WATT_2": 3, "OCCU_MOD": 1, "hour": 12}
    for minute in range(5):  # history builds up the rolling features
        body = _score(token, mid, {**reading, "hour": 12 + minute / 60})
    assert body["diagnosis"] is None or isinstance(body["diagnosis"], str)


@needs_robot
def test_robot_scores_a_stream_of_readings():
    token, mid = _new_machine("robot@example.com", "robotic_arm")
    reading = {**{f"Current_J{j}": 0.5 for j in range(6)},
               **{f"Temperature_J{j}": 40.0 for j in range(6)},
               **{f"Speed_J{j}": 0.1 for j in range(6)}, "Tool_current": 0.085}
    for _ in range(12):
        _score(token, mid, reading)


@needs_conveyor
def test_conveyor_bearing_signature_scores_riskier_than_normal():
    token, mid = _new_machine("conveyor@example.com", "conveyor")
    normal = _score(token, mid, {"rms": 0.13, "peak": 0.45, "crest": 3.4, "kurtosis": -0.2})
    faulty = _score(token, mid, {"rms": 0.6, "peak": 4.0, "crest": 7.0, "kurtosis": 3.0})
    assert faulty["probability"] > normal["probability"]
