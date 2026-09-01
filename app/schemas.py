from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class LoginBody(StrictModel):
    username: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=1, max_length=256)


class ClientEvent(StrictModel):
    event_id: str = Field(min_length=1, max_length=128)
    event_type: Literal["click", "like", "favorite", "not_interested"]
    request_id: str = Field(min_length=1, max_length=128)
    item_id: str = Field(min_length=1, max_length=128)
    position: int = Field(ge=0, le=9999)
    client_timestamp: datetime


class EventBatch(StrictModel):
    events: list[ClientEvent] = Field(min_length=1, max_length=100)


class ItemStatusBody(StrictModel):
    status: Literal["online", "offline"]
    reason: str = Field(min_length=1, max_length=500)


class BoostBody(StrictModel):
    item_id: str = Field(min_length=1, max_length=128)
    audience: Literal["all", "users"] = "all"
    user_ids: list[str] = Field(default_factory=list, max_length=100)
    feed_types: list[Literal["personalized", "popular", "explore"]] = Field(
        default_factory=list,
        max_length=3,
    )
    position: int = Field(default=0, ge=0, le=49)
    priority: int = Field(default=0, ge=-1000, le=1000)
    starts_at: datetime
    ends_at: datetime
    reason: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_scope_and_time(self) -> "BoostBody":
        if self.audience == "users" and not self.user_ids:
            raise ValueError("user_ids is required when audience is users")
        if self.audience == "all" and self.user_ids:
            raise ValueError("user_ids must be empty when audience is all")
        if self.ends_at <= self.starts_at:
            raise ValueError("ends_at must be later than starts_at")
        self.user_ids = list(dict.fromkeys(self.user_ids))
        self.feed_types = list(dict.fromkeys(self.feed_types))
        return self


class BoostStatusBody(StrictModel):
    active: bool
    reason: str = Field(min_length=1, max_length=500)

