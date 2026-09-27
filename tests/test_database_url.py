"""The hosted deployment uses Postgres, but every other test uses SQLite -- so
this checks the Postgres path directly. The first Render deploy crashed because
SQLAlchemy picked a Postgres driver that wasn't installed."""
import pytest

pytest.importorskip("sqlalchemy")
from sqlalchemy import create_engine

from backend.database import normalize_db_url


@pytest.mark.parametrize("url", [
    "postgres://user:pw@host:5432/db",     # Heroku-style
    "postgresql://user:pw@host:5432/db",   # Render-style
])
def test_hosted_postgres_urls_use_the_installed_driver(url):
    fixed = normalize_db_url(url)
    assert fixed.startswith("postgresql+psycopg2://")
    # create_engine imports the DB driver without connecting -- the exact step
    # that failed on Render.
    engine = create_engine(fixed)
    assert engine.dialect.driver == "psycopg2"


def test_sqlite_and_explicit_driver_urls_are_left_alone():
    assert normalize_db_url("sqlite:///x.db") == "sqlite:///x.db"
    assert normalize_db_url("postgresql+psycopg2://h/db") == "postgresql+psycopg2://h/db"
