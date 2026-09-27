import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base


def now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


class Business(Base):
    __tablename__ = "businesses"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    email: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=now)

    machines: Mapped[list["Machine"]] = relationship(back_populates="business",
                                                      cascade="all, delete-orphan")


class Machine(Base):
    __tablename__ = "machines"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    business_id: Mapped[int] = mapped_column(ForeignKey("businesses.id"), nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    machine_type: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=now)

    business: Mapped["Business"] = relationship(back_populates="machines")
    readings: Mapped[list["Reading"]] = relationship(back_populates="machine",
                                                      cascade="all, delete-orphan",
                                                      order_by="Reading.created_at")


class Reading(Base):
    __tablename__ = "readings"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    machine_id: Mapped[int] = mapped_column(ForeignKey("machines.id"), nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)  # raw sensor values
    source: Mapped[str] = mapped_column(String, default="manual")  # manual | api | csv
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=now, index=True)

    machine: Mapped["Machine"] = relationship(back_populates="readings")
    risk_score: Mapped["RiskScore"] = relationship(back_populates="reading", uselist=False,
                                                    cascade="all, delete-orphan")


class RiskScore(Base):
    __tablename__ = "risk_scores"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    reading_id: Mapped[int] = mapped_column(ForeignKey("readings.id"), unique=True, nullable=False)
    trained: Mapped[bool] = mapped_column(Boolean, default=True)
    probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    risk_level: Mapped[str | None] = mapped_column(String, nullable=True)
    alert: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    diagnosis: Mapped[str | None] = mapped_column(String, nullable=True)  # e.g. bearing fault class
    note: Mapped[str | None] = mapped_column(String, nullable=True)
    computed_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=now)

    reading: Mapped["Reading"] = relationship(back_populates="risk_score")
