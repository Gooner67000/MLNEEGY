"""Minimal auth: bcrypt password hashing + signed, expiring tokens (JWT/HS256).

SECRET_KEY must be overridden via the environment for any real deployment --
the default here is only for local/dev use and is intentionally obvious.
"""
import datetime
import os

import bcrypt
import jwt

SECRET_KEY = os.environ.get("SECRET_KEY", "dev-only-insecure-secret-change-me")
# On a public deployment a guessable key would let anyone forge login tokens,
# so refuse to start rather than run insecurely.
if os.environ.get("PM_ENV") == "production" and (
        len(SECRET_KEY) < 32 or "change-me" in SECRET_KEY or "insecure" in SECRET_KEY):
    raise RuntimeError("PM_ENV=production requires SECRET_KEY to be a random string of "
                       "32+ characters (e.g. `python -c \"import secrets; print(secrets.token_urlsafe(48))\"`).")
ALGORITHM = "HS256"
TOKEN_TTL_HOURS = 24 * 7


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), password_hash.encode())
    except ValueError:
        return False


def create_token(business_id: int) -> str:
    payload = {
        "sub": str(business_id),
        "exp": datetime.datetime.now(datetime.timezone.utc)
        + datetime.timedelta(hours=TOKEN_TTL_HOURS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> int | None:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return int(payload["sub"])
    except jwt.PyJWTError:
        return None
