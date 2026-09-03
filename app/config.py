from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    app_env: str
    app_secret: str
    database_path: Path
    model_pointer: Path
    session_hours: int
    session_cookie: str
    feed_snapshot_ttl_minutes: int = 60
    feed_snapshot_max_items: int = 500
    experiment_model_pointer: Path | None = None
    multimodal_model_pointer: Path | None = None
    alert_window_minutes: int = 60
    redis_url: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        # The deep/experiment pointer defaults to ``experiment-current.json``, which
        # ``train_deep_experiment`` always writes. ``deep-current.json`` is only written
        # when an explicit ``--publish`` request completes its publication checks, so
        # pointing the default at it would leave the online consumer empty otherwise. Set
        # ``DEEP_MODEL_POINTER`` to ``artifacts/deep-current.json`` to serve only the
        # explicitly published artifact.
        experiment_pointer = os.getenv(
            "DEEP_MODEL_POINTER",
            os.getenv("EXPERIMENT_MODEL_POINTER", "artifacts/experiment-current.json"),
        ).strip()
        multimodal_pointer = os.getenv(
            "MULTIMODAL_MODEL_POINTER", "artifacts/multimodal-current.json"
        ).strip()
        return cls(
            app_env=os.getenv("APP_ENV", "development"),
            app_secret=os.getenv("APP_SECRET", "development-only-change-me"),
            database_path=Path(os.getenv("APP_DATABASE", "data/app.db")).resolve(),
            model_pointer=Path(os.getenv("MODEL_POINTER", "artifacts/current.json")).resolve(),
            session_hours=max(1, int(os.getenv("SESSION_HOURS", "12"))),
            session_cookie=os.getenv("SESSION_COOKIE", "microlens_session"),
            feed_snapshot_ttl_minutes=max(
                1,
                int(os.getenv("FEED_SNAPSHOT_TTL_MINUTES", "60")),
            ),
            feed_snapshot_max_items=max(
                50,
                min(2000, int(os.getenv("FEED_SNAPSHOT_MAX_ITEMS", "500"))),
            ),
            experiment_model_pointer=(
                Path(experiment_pointer).resolve() if experiment_pointer else None
            ),
            multimodal_model_pointer=(
                Path(multimodal_pointer).resolve() if multimodal_pointer else None
            ),
            alert_window_minutes=max(
                1, int(os.getenv("ALERT_WINDOW_MINUTES", "60"))
            ),
            redis_url=(os.getenv("REDIS_URL") or "").strip() or None,
        )

    @property
    def secure_cookie(self) -> bool:
        return self.app_env not in {"development", "test", "testing"}
