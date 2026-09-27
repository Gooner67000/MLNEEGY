import datetime

from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str = Field(min_length=8)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    business_name: str


class MachineCreate(BaseModel):
    name: str
    machine_type: str


class MachineOut(BaseModel):
    id: int
    name: str
    machine_type: str
    created_at: datetime.datetime

    class Config:
        from_attributes = True


class ReadingIn(BaseModel):
    payload: dict[str, float]
    source: str = "manual"  # manual | api | csv


class ReadingBulkIn(BaseModel):
    readings: list[ReadingIn]


class RiskOut(BaseModel):
    reading_id: int
    trained: bool
    probability: float | None
    risk_level: str | None
    alert: bool | None
    diagnosis: str | None
    note: str | None
    computed_at: datetime.datetime

    class Config:
        from_attributes = True
