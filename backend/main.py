"""The one backend that serves both deployment modes:

  - Downloadable / on-prem:  DATABASE_URL unset -> local SQLite file, a
    business runs `docker compose up` inside their own network and nothing
    ever leaves it.
  - Hosted / multi-tenant:   DATABASE_URL set to a Postgres URL -> the exact
    same code and endpoints, many businesses' data isolated by business_id.

Manual entry, CSV upload, and a business's own automation all call the same
POST /machines/{id}/readings(/bulk) endpoint -- there is only one ingestion
path, so "automatic" live data is just this endpoint called by a script/cron
on their side instead of a person clicking a button.
"""
import sys
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).parent.parent))  # import machine_types, predict, backtest
import machine_types as mt
from backend import models, scoring, security
from backend.database import Base, SessionLocal, engine, get_db
from backend.schemas import (
    LoginRequest, MachineCreate, MachineOut, ReadingBulkIn, ReadingIn,
    RegisterRequest, RiskOut, TokenResponse,
)

Base.metadata.create_all(bind=engine)
app = FastAPI(title="Predictive Maintenance Platform", version="1.0")
bearer = HTTPBearer()

HISTORY_WINDOW = 20  # readings looked back at for rolling/causal features


def current_business(creds: HTTPAuthorizationCredentials = Depends(bearer),
                     db: Session = Depends(get_db)) -> models.Business:
    business_id = security.decode_token(creds.credentials)
    business = db.get(models.Business, business_id) if business_id else None
    if not business:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")
    return business


def owned_machine(machine_id: int, db: Session, business: models.Business) -> models.Machine:
    machine = db.get(models.Machine, machine_id)
    if not machine or machine.business_id != business.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Machine not found")
    return machine


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/machine-types")
def list_machine_types():
    """Drives the frontend's type picker and dynamic sensor-input forms --
    the same registry the training scripts read from, so it never drifts."""
    return {
        key: {
            "name": t.name, "description": t.description, "dataset": t.dataset,
            "trained": t.trained, "note": t.note,
            "sensors": [s.__dict__ for s in t.sensors],
        }
        for key, t in mt.MACHINE_TYPES.items()
    }


@app.post("/auth/register", response_model=TokenResponse)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(models.Business).filter_by(email=req.email).first():
        raise HTTPException(status.HTTP_409_CONFLICT, "Email already registered")
    business = models.Business(name=req.name, email=req.email,
                               password_hash=security.hash_password(req.password))
    db.add(business); db.commit(); db.refresh(business)
    return TokenResponse(access_token=security.create_token(business.id), business_name=business.name)


@app.post("/auth/login", response_model=TokenResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    business = db.query(models.Business).filter_by(email=req.email).first()
    if not business or not security.verify_password(req.password, business.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Wrong email or password")
    return TokenResponse(access_token=security.create_token(business.id), business_name=business.name)


@app.post("/machines", response_model=MachineOut)
def create_machine(req: MachineCreate, db: Session = Depends(get_db),
                   business: models.Business = Depends(current_business)):
    try:
        mt.get(req.machine_type)
    except KeyError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))
    machine = models.Machine(business_id=business.id, name=req.name, machine_type=req.machine_type)
    db.add(machine); db.commit(); db.refresh(machine)
    return machine


@app.get("/machines", response_model=list[MachineOut])
def list_machines(db: Session = Depends(get_db), business: models.Business = Depends(current_business)):
    # a business can register as many machines, of as many types, as it wants
    return db.query(models.Machine).filter_by(business_id=business.id).all()


@app.delete("/machines/{machine_id}")
def delete_machine(machine_id: int, db: Session = Depends(get_db),
                   business: models.Business = Depends(current_business)):
    machine = owned_machine(machine_id, db, business)
    db.delete(machine); db.commit()
    return {"deleted": machine_id}


def _ingest(machine: models.Machine, reading_in: ReadingIn, db: Session) -> models.RiskScore:
    history_rows = (
        db.query(models.Reading).filter_by(machine_id=machine.id)
        .order_by(models.Reading.created_at.desc()).limit(HISTORY_WINDOW).all()
    )
    history = [r.payload for r in reversed(history_rows)]  # oldest first, BEFORE this new one

    reading = models.Reading(machine_id=machine.id, payload=reading_in.payload,
                             source=reading_in.source)
    db.add(reading); db.commit(); db.refresh(reading)

    result = scoring.score(machine.machine_type, reading_in.payload, history)
    risk = models.RiskScore(reading_id=reading.id, **result)
    db.add(risk); db.commit(); db.refresh(risk)
    risk.reading_id = reading.id
    return risk


@app.post("/machines/{machine_id}/readings", response_model=RiskOut)
def add_reading(machine_id: int, reading_in: ReadingIn, db: Session = Depends(get_db),
                business: models.Business = Depends(current_business)):
    """Manual entry, a business's own automation, and a live API push all
    call this same endpoint -- that's the whole 'automatic import' feature:
    point your PLC/SCADA script at this URL on a schedule."""
    machine = owned_machine(machine_id, db, business)
    return _ingest(machine, reading_in, db)


@app.post("/machines/{machine_id}/readings/bulk", response_model=list[RiskOut])
def add_readings_bulk(machine_id: int, body: ReadingBulkIn, db: Session = Depends(get_db),
                      business: models.Business = Depends(current_business)):
    """What a CSV upload turns into: many readings in original chronological
    order, scored one at a time so causal/rolling features stay correct."""
    machine = owned_machine(machine_id, db, business)
    return [_ingest(machine, r, db) for r in body.readings]


@app.get("/machines/{machine_id}/risk/latest", response_model=RiskOut)
def latest_risk(machine_id: int, db: Session = Depends(get_db),
                business: models.Business = Depends(current_business)):
    machine = owned_machine(machine_id, db, business)
    risk = (db.query(models.RiskScore).join(models.Reading)
           .filter(models.Reading.machine_id == machine.id)
           .order_by(models.RiskScore.computed_at.desc()).first())
    if not risk:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No readings yet for this machine")
    return risk


@app.get("/machines/{machine_id}/risk/history", response_model=list[RiskOut])
def risk_history(machine_id: int, limit: int = 200, db: Session = Depends(get_db),
                 business: models.Business = Depends(current_business)):
    machine = owned_machine(machine_id, db, business)
    return (db.query(models.RiskScore).join(models.Reading)
           .filter(models.Reading.machine_id == machine.id)
           .order_by(models.RiskScore.computed_at.desc()).limit(limit).all())
