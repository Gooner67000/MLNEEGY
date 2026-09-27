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
    not (ROOT / "bearing_model.pkl").exists(), reason="run train_bearing.py first"
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
    assert types["cnc_machine_tool"]["trained"] is True
    assert types["hvac"]["trained"] is False


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


def test_untrained_machine_type_reports_not_trained_instead_of_faking_a_score():
    token = register("hvac@example.com")
    m = client.post("/machines", json={"name": "Chiller-1", "machine_type": "hvac"},
                    headers=auth(token)).json()
    resp = client.post(f"/machines/{m['id']}/readings",
                       json={"payload": {"suction_pressure": 100}, "source": "manual"},
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


@needs_bearing
def test_bearing_machine_defaults_missing_features_to_medians():
    token = register("bearing@example.com")
    m = client.post("/machines", json={"name": "Pump-1", "machine_type": "rotating_equipment"},
                    headers=auth(token)).json()
    # deliberately incomplete: only the manual-entry subset
    resp = client.post(f"/machines/{m['id']}/readings",
                       json={"payload": {"rms": 0.6, "peak": 2.0, "crest": 3.0}, "source": "manual"},
                       headers=auth(token))
    assert resp.status_code == 200
    body = resp.json()
    assert body["trained"] is True
    assert body["diagnosis"] in ("Normal", "Ball_Fault", "Inner_Race", "Outer_Race")
    assert body["note"] is not None  # tells the user features were defaulted
