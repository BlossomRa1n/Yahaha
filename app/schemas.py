from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class LoginBody(StrictModel):
    username: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=1, max_length=256)


class RegisterBody(StrictModel):
    username: str = Field(
        min_length=3,
        max_length=32,
        pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$",
    )
    password: str = Field(min_length=10, max_length=128)

    @model_validator(mode="after")
    def validate_password_strength(self) -> "RegisterBody":
        if not any(character.isalpha() for character in self.password):
            raise ValueError("password must contain at least one letter")
        if not any(character.isdigit() for character in self.password):
            raise ValueError("password must contain at least one digit")
        return self


class ClientEvent(StrictModel):
    event_id: str = Field(min_length=1, max_length=128)
    event_type: Literal[
        "impression", "click", "like", "favorite", "not_interested", "dwell", "share"
    ]
    request_id: str = Field(min_length=1, max_length=128)
    item_id: str = Field(min_length=1, max_length=128)
    position: int = Field(ge=0, le=9999)
    client_timestamp: datetime
    dwell_ms: int | None = Field(default=None, ge=750, le=600_000)
    visit_index: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_engagement_payload(self) -> "ClientEvent":
        if self.event_type == "dwell" and self.dwell_ms is None:
            raise ValueError("dwell_ms is required for dwell events")
        if self.event_type != "dwell" and self.dwell_ms is not None:
            raise ValueError("dwell_ms is only valid for dwell events")
        return self


class EventBatch(StrictModel):
    events: list[ClientEvent] = Field(min_length=1, max_length=100)


class ItemStatusBody(StrictModel):
    status: Literal["online", "offline"]
    reason: str = Field(min_length=1, max_length=500)


class BatchItemStatusBody(StrictModel):
    item_ids: list[str] = Field(min_length=1, max_length=100)
    status: Literal["online", "offline"]
    reason: str = Field(min_length=1, max_length=500)
    idempotency_key: str = Field(min_length=8, max_length=128)

    @model_validator(mode="after")
    def deduplicate_items(self) -> "BatchItemStatusBody":
        self.item_ids = list(dict.fromkeys(self.item_ids))
        return self


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


class AlertRuleBody(StrictModel):
    name: str = Field(min_length=1, max_length=120)
    metric: str = Field(min_length=1, max_length=64)
    operator: Literal[">", "<", ">=", "<="]
    threshold: float
    severity: Literal["info", "warn", "critical"] = "warn"


class AlertRuleUpdateBody(StrictModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    operator: Literal[">", "<", ">=", "<="] | None = None
    threshold: float | None = None
    severity: Literal["info", "warn", "critical"] | None = None
    enabled: bool | None = None


class TrainingJobBody(StrictModel):
    start_time: datetime
    end_time: datetime
    mode: Literal["smoke", "full"] = "smoke"
    max_users: int | None = Field(default=None, ge=1)
    max_eval_users: int | None = Field(default=None, ge=1)
    rank: int = Field(default=32, ge=1, le=256)
    seed: int = 20260901
    base_processed_dir: str = Field(default="data/processed", min_length=1, max_length=500)
    output_root: str = Field(default="data/retraining", min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_window(self) -> "TrainingJobBody":
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be later than start_time")
        return self
