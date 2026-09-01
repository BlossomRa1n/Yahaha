from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
    dataset_user_id TEXT UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'admin')),
    is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS auth_sessions (
    id TEXT PRIMARY KEY,
    token_hash TEXT NOT NULL UNIQUE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    revoked_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON auth_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_expiry ON auth_sessions(expires_at);

CREATE TABLE IF NOT EXISTS items (
    item_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    likes INTEGER NOT NULL DEFAULT 0 CHECK (likes >= 0),
    views INTEGER NOT NULL DEFAULT 0 CHECK (views >= 0),
    popularity_score REAL NOT NULL DEFAULT 0,
    cover_url TEXT,
    status TEXT NOT NULL DEFAULT 'online' CHECK (status IN ('online', 'offline')),
    status_version INTEGER NOT NULL DEFAULT 1 CHECK (status_version >= 1),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_items_status_popularity
    ON items(status, popularity_score DESC, item_id);

CREATE TABLE IF NOT EXISTS model_versions (
    model_version TEXT PRIMARY KEY,
    data_version TEXT,
    algorithm TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    metrics_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL CHECK (status IN ('active', 'inactive', 'failed')),
    created_at TEXT NOT NULL,
    activated_at TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_one_active_model
    ON model_versions((1)) WHERE status = 'active';

CREATE TABLE IF NOT EXISTS profiles (
    user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    version INTEGER NOT NULL DEFAULT 0 CHECK (version >= 0),
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS recommendation_requests (
    request_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    feed_type TEXT NOT NULL CHECK (feed_type IN ('personalized', 'popular', 'explore')),
    model_version TEXT,
    profile_version INTEGER NOT NULL,
    cursor TEXT,
    fallback_reason TEXT,
    returned_count INTEGER NOT NULL CHECK (returned_count >= 0),
    latency_ms REAL NOT NULL CHECK (latency_ms >= 0),
    created_at TEXT NOT NULL,
    UNIQUE(request_id, user_id)
);
CREATE INDEX IF NOT EXISTS idx_requests_user_time
    ON recommendation_requests(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_requests_feed_time
    ON recommendation_requests(feed_type, created_at DESC);

CREATE TABLE IF NOT EXISTS exposures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    item_id TEXT NOT NULL REFERENCES items(item_id),
    position INTEGER NOT NULL CHECK (position >= 0),
    source TEXT NOT NULL,
    score REAL NOT NULL,
    explanation TEXT NOT NULL,
    model_version TEXT,
    is_forced INTEGER NOT NULL DEFAULT 0 CHECK (is_forced IN (0, 1)),
    created_at TEXT NOT NULL,
    FOREIGN KEY(request_id, user_id)
        REFERENCES recommendation_requests(request_id, user_id) ON DELETE CASCADE,
    UNIQUE(request_id, position),
    UNIQUE(request_id, item_id),
    UNIQUE(request_id, user_id, item_id, position)
);
CREATE INDEX IF NOT EXISTS idx_exposures_user_item
    ON exposures(user_id, item_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_exposures_item_time
    ON exposures(item_id, created_at DESC);

CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL CHECK (
        event_type IN ('impression', 'click', 'like', 'not_interested')
    ),
    request_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    position INTEGER NOT NULL CHECK (position >= 0),
    client_timestamp TEXT,
    received_at TEXT NOT NULL,
    FOREIGN KEY(request_id, user_id, item_id, position)
        REFERENCES exposures(request_id, user_id, item_id, position),
    UNIQUE(request_id, user_id, item_id, position, event_type)
);
CREATE INDEX IF NOT EXISTS idx_events_user_time
    ON events(user_id, received_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_type_time
    ON events(event_type, received_at DESC);

CREATE TABLE IF NOT EXISTS user_item_state (
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    item_id TEXT NOT NULL REFERENCES items(item_id),
    exposure_count INTEGER NOT NULL DEFAULT 0 CHECK (exposure_count >= 0),
    click_count INTEGER NOT NULL DEFAULT 0 CHECK (click_count >= 0),
    like_count INTEGER NOT NULL DEFAULT 0 CHECK (like_count >= 0),
    not_interested INTEGER NOT NULL DEFAULT 0 CHECK (not_interested IN (0, 1)),
    affinity REAL NOT NULL DEFAULT 0,
    last_event_at TEXT NOT NULL,
    PRIMARY KEY(user_id, item_id)
);

CREATE TABLE IF NOT EXISTS boost_campaigns (
    id TEXT PRIMARY KEY,
    item_id TEXT NOT NULL REFERENCES items(item_id),
    audience TEXT NOT NULL CHECK (audience IN ('all', 'users')),
    user_ids_json TEXT NOT NULL DEFAULT '[]',
    feed_types_json TEXT NOT NULL DEFAULT '[]',
    position INTEGER NOT NULL DEFAULT 0 CHECK (position >= 0),
    priority INTEGER NOT NULL DEFAULT 0,
    starts_at TEXT NOT NULL,
    ends_at TEXT NOT NULL,
    reason TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
    created_by TEXT NOT NULL REFERENCES users(id),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (starts_at < ends_at)
);
CREATE INDEX IF NOT EXISTS idx_boosts_active_time
    ON boost_campaigns(active, starts_at, ends_at, priority DESC);

CREATE TABLE IF NOT EXISTS operations (
    operation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    admin_user_id TEXT NOT NULL REFERENCES users(id),
    action TEXT NOT NULL,
    item_id TEXT REFERENCES items(item_id),
    target_id TEXT,
    reason TEXT NOT NULL,
    before_json TEXT NOT NULL,
    after_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_operations_time
    ON operations(created_at DESC);
"""


class ClosingConnection(sqlite3.Connection):
    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        try:
            return bool(super().__exit__(exc_type, exc_value, traceback))
        finally:
            self.close()


class Database:
    def __init__(self, path: Path | str):
        self.path = Path(path).resolve()

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            self.path,
            timeout=5.0,
            isolation_level=None,
            factory=ClosingConnection,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    def reset(self) -> None:
        if self.path.exists():
            self.path.unlink()
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(self.path) + suffix)
            if sidecar.exists():
                sidecar.unlink()
        self.initialize()

    @contextmanager
    def transaction(self, *, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
