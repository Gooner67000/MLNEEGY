"""SQLite by default (zero-setup for a downloaded/local install). Point
DATABASE_URL at Postgres (e.g. postgresql://user:pass@host/db) to run this
same code as a hosted, multi-tenant service -- nothing else in the app needs
to change."""
import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
def normalize_db_url(url: str) -> str:
    """Hosting providers (Render, Heroku) hand out "postgres://" or
    "postgresql://" URLs. Name the driver explicitly: newer SQLAlchemy defaults
    to psycopg (v3), but the installed driver is psycopg2 -- the first Render
    deploy crashed on exactly that."""
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg2://" + url[len(prefix):]
    return url


DATABASE_URL = normalize_db_url(os.environ.get("DATABASE_URL")
                                or f"sqlite:///{DATA_DIR / 'app.db'}")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
