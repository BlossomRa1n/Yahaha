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

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            app_env=os.getenv("APP_ENV", "development"),
            app_secret=os.getenv("APP_SECRET", "development-only-change-me"),
            database_path=Path(os.getenv("APP_DATABASE", "data/app.db")).resolve(),
            model_pointer=Path(os.getenv("MODEL_POINTER", "artifacts/current.json")).resolve(),
            session_hours=max(1, int(os.getenv("SESSION_HOURS", "12"))),
            session_cookie=os.getenv("SESSION_COOKIE", "microlens_session"),
        )

    @property
    def secure_cookie(self) -> bool:
        return self.app_env not in {"development", "test", "testing"}
