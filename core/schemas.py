from datetime import datetime

from pydantic import BaseModel, EmailStr


# ── Auth ──────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    email: EmailStr
    name: str
    password: str


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: int
    email: str
    name: str
    role: str
    created_at: datetime

    model_config = {"from_attributes": True}


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenRefresh(BaseModel):
    refresh_token: str


# ── Meetings ──────────────────────────────────────────────────────────

class MeetingListItem(BaseModel):
    id: int
    title: str
    date_time: str | None
    location: str | None
    participants: str | None
    status: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class MeetingDetail(BaseModel):
    id: int
    title: str
    date_time: str | None
    location: str | None
    participants: str | None
    agenda: list | None
    status: str
    error_message: str | None
    raw_transcript: str | None
    refined_transcript: str | None
    meeting_json: dict | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class MeetingUpdate(BaseModel):
    title: str | None = None
    date_time: str | None = None
    location: str | None = None
    participants: str | None = None
    agenda: list | None = None
    meeting_json: dict | None = None
    refined_transcript: str | None = None
