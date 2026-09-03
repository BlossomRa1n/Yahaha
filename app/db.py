from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


SCHEMA = """
CREATE TABLE IF NOT EXISTS app_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
    dataset_user_id TEXT UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'analyst', 'operator', 'admin')),
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
    stats_snapshot_version TEXT,
    stats_available_at TEXT,
    cover_url TEXT,
    status TEXT NOT NULL DEFAULT 'online' CHECK (status IN ('online', 'offline')),
    status_version INTEGER NOT NULL DEFAULT 1 CHECK (status_version >= 1),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_items_status_popularity
    ON items(status, popularity_score DESC, item_id);

CREATE TABLE IF NOT EXISTS item_stats_snapshots (
    snapshot_version TEXT PRIMARY KEY,
    available_at TEXT,
    source_file_hash TEXT NOT NULL,
    source_file_name TEXT NOT NULL,
    ingest_time TEXT NOT NULL,
    row_count INTEGER NOT NULL CHECK (row_count >= 0),
    quality_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_versions (
    model_version TEXT PRIMARY KEY,
    data_version TEXT,
    algorithm TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    metrics_json TEXT NOT NULL DEFAULT '{}',
    training_window_start TEXT,
    training_window_end TEXT,
    sample_count INTEGER,
    event_count INTEGER,
    training_status TEXT NOT NULL DEFAULT 'succeeded',
    publish_status TEXT NOT NULL DEFAULT 'published',
    evaluation_protocol_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL CHECK (status IN ('active', 'inactive', 'failed')),
    created_at TEXT NOT NULL,
    activated_at TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_one_active_model
    ON model_versions((1)) WHERE status = 'active';

CREATE TABLE IF NOT EXISTS training_runs (
    run_id TEXT PRIMARY KEY,
    data_version TEXT,
    model_version TEXT,
    window_start TEXT NOT NULL,
    window_end TEXT NOT NULL,
    event_count INTEGER NOT NULL DEFAULT 0 CHECK (event_count >= 0),
    sample_count INTEGER NOT NULL DEFAULT 0 CHECK (sample_count >= 0),
    config_json TEXT NOT NULL DEFAULT '{}',
    metrics_json TEXT NOT NULL DEFAULT '{}',
    artifact_path TEXT,
    training_status TEXT NOT NULL CHECK (
        training_status IN ('running', 'succeeded', 'failed')
    ),
    publish_status TEXT NOT NULL CHECK (
        publish_status IN ('not_published', 'published', 'failed')
    ),
    error TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_training_runs_created
    ON training_runs(created_at DESC);

CREATE TABLE IF NOT EXISTS profiles (
    user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    version INTEGER NOT NULL DEFAULT 0 CHECK (version >= 0),
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feed_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    feed_type TEXT NOT NULL CHECK (feed_type IN ('personalized', 'popular', 'explore')),
    model_version TEXT,
    profile_version INTEGER NOT NULL,
    ops_revision INTEGER NOT NULL DEFAULT 0,
    seed INTEGER NOT NULL,
    fallback_reason TEXT,
    diversity_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'expired')),
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_feed_snapshots_expiry
    ON feed_snapshots(status, expires_at);
CREATE INDEX IF NOT EXISTS idx_feed_snapshots_user
    ON feed_snapshots(user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS feed_snapshot_items (
    snapshot_id TEXT NOT NULL REFERENCES feed_snapshots(snapshot_id) ON DELETE CASCADE,
    snapshot_position INTEGER NOT NULL CHECK (snapshot_position >= 0),
    item_id TEXT NOT NULL REFERENCES items(item_id),
    source TEXT NOT NULL,
    score REAL NOT NULL,
    raw_score REAL,
    normalized_score REAL,
    rank_in_source INTEGER,
    explanation TEXT NOT NULL,
    model_version TEXT,
    is_forced INTEGER NOT NULL DEFAULT 0 CHECK (is_forced IN (0, 1)),
    invalidated_at TEXT,
    PRIMARY KEY(snapshot_id, snapshot_position),
    UNIQUE(snapshot_id, item_id)
);
CREATE INDEX IF NOT EXISTS idx_feed_snapshot_items_item
    ON feed_snapshot_items(item_id, invalidated_at);

CREATE TABLE IF NOT EXISTS recommendation_requests (
    request_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    feed_type TEXT NOT NULL CHECK (feed_type IN ('personalized', 'popular', 'explore')),
    model_version TEXT,
    profile_version INTEGER NOT NULL,
    cursor TEXT,
    fallback_reason TEXT,
    snapshot_id TEXT,
    snapshot_offset INTEGER,
    returned_count INTEGER NOT NULL CHECK (returned_count >= 0),
    latency_ms REAL NOT NULL CHECK (latency_ms >= 0),
    created_at TEXT NOT NULL,
    UNIQUE(request_id, user_id)
);
CREATE INDEX IF NOT EXISTS idx_requests_user_time
    ON recommendation_requests(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_requests_feed_time
    ON recommendation_requests(feed_type, created_at DESC);

CREATE TABLE IF NOT EXISTS candidate_manifests (
    snapshot_id TEXT NOT NULL REFERENCES feed_snapshots(snapshot_id) ON DELETE CASCADE,
    item_id TEXT NOT NULL REFERENCES items(item_id),
    primary_source TEXT NOT NULL,
    source_scores_json TEXT NOT NULL,
    source_calibrated_scores_json TEXT NOT NULL,
    source_mask_json TEXT NOT NULL,
    source_raw_scores_json TEXT NOT NULL,
    source_evidence_json TEXT NOT NULL DEFAULT '{}',
    ranker_score REAL,
    ranker_rank INTEGER CHECK (ranker_rank IS NULL OR ranker_rank >= 0),
    model_version TEXT,
    feature_schema_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(snapshot_id, item_id)
);
CREATE INDEX IF NOT EXISTS idx_candidate_manifests_item
    ON candidate_manifests(item_id, created_at DESC);

CREATE TABLE IF NOT EXISTS exposures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    item_id TEXT NOT NULL REFERENCES items(item_id),
    position INTEGER NOT NULL CHECK (position >= 0),
    source TEXT NOT NULL,
    score REAL NOT NULL,
    raw_score REAL,
    normalized_score REAL,
    rank_in_source INTEGER,
    explanation TEXT NOT NULL,
    model_version TEXT,
    is_forced INTEGER NOT NULL DEFAULT 0 CHECK (is_forced IN (0, 1)),
    snapshot_id TEXT,
    snapshot_position INTEGER,
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
        event_type IN (
            'impression', 'click', 'like', 'not_interested',
            'dwell', 'share', 'revisit'
        )
    ),
    request_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    position INTEGER NOT NULL CHECK (position >= 0),
    client_timestamp TEXT,
    dwell_ms INTEGER CHECK (dwell_ms IS NULL OR dwell_ms BETWEEN 750 AND 600000),
    visit_index INTEGER CHECK (visit_index IS NULL OR visit_index >= 1),
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
    dwell_ms_total INTEGER NOT NULL DEFAULT 0 CHECK (dwell_ms_total >= 0),
    dwell_event_count INTEGER NOT NULL DEFAULT 0 CHECK (dwell_event_count >= 0),
    share_count INTEGER NOT NULL DEFAULT 0 CHECK (share_count >= 0),
    revisit_count INTEGER NOT NULL DEFAULT 0 CHECK (revisit_count >= 0),
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
    batch_id TEXT,
    reason TEXT NOT NULL,
    before_json TEXT NOT NULL,
    after_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_operations_time
    ON operations(created_at DESC);

CREATE TABLE IF NOT EXISTS operation_batches (
    batch_id TEXT PRIMARY KEY,
    admin_user_id TEXT NOT NULL REFERENCES users(id),
    idempotency_key TEXT NOT NULL,
    request_json TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(admin_user_id, idempotency_key)
);

CREATE TABLE IF NOT EXISTS alert_rules (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    metric TEXT NOT NULL,
    operator TEXT NOT NULL CHECK (operator IN ('>', '<', '>=', '<=')),
    threshold REAL NOT NULL,
    severity TEXT NOT NULL CHECK (severity IN ('info', 'warn', 'critical')),
    enabled INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
    created_by TEXT NOT NULL REFERENCES users(id),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_alert_rules_enabled ON alert_rules(enabled, created_at);

CREATE TABLE IF NOT EXISTS alert_events (
    id TEXT PRIMARY KEY,
    rule_id TEXT NOT NULL REFERENCES alert_rules(id) ON DELETE CASCADE,
    metric TEXT NOT NULL,
    operator TEXT NOT NULL,
    threshold REAL NOT NULL,
    severity TEXT NOT NULL,
    observed_value REAL,
    status TEXT NOT NULL CHECK (status IN ('open', 'resolved')),
    triggered_at TEXT NOT NULL,
    resolved_at TEXT,
    acknowledged_by TEXT REFERENCES users(id),
    acknowledged_at TEXT,
    last_evaluated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_alert_events_rule
    ON alert_events(rule_id, triggered_at DESC);
CREATE INDEX IF NOT EXISTS idx_alert_events_status
    ON alert_events(status, triggered_at DESC);

CREATE TABLE IF NOT EXISTS training_jobs (
    job_id TEXT PRIMARY KEY,
    run_id TEXT,
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'succeeded', 'failed')),
    window_start TEXT NOT NULL,
    window_end TEXT NOT NULL,
    config_json TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    created_by TEXT NOT NULL REFERENCES users(id),
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_training_jobs_created ON training_jobs(created_at DESC);
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
            self._migrate_user_roles(conn)
            request_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(recommendation_requests)")
            }
            if "snapshot_id" not in request_columns:
                conn.execute("ALTER TABLE recommendation_requests ADD COLUMN snapshot_id TEXT")
            if "snapshot_offset" not in request_columns:
                conn.execute("ALTER TABLE recommendation_requests ADD COLUMN snapshot_offset INTEGER")
            exposure_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(exposures)")
            }
            if "snapshot_id" not in exposure_columns:
                conn.execute("ALTER TABLE exposures ADD COLUMN snapshot_id TEXT")
            if "snapshot_position" not in exposure_columns:
                conn.execute("ALTER TABLE exposures ADD COLUMN snapshot_position INTEGER")
            candidate_columns = {
                "raw_score": "REAL",
                "normalized_score": "REAL",
                "rank_in_source": "INTEGER",
            }
            snapshot_item_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(feed_snapshot_items)")
            }
            for column, definition in candidate_columns.items():
                if column not in snapshot_item_columns:
                    conn.execute(
                        f"ALTER TABLE feed_snapshot_items ADD COLUMN {column} {definition}"
                    )
            for column, definition in candidate_columns.items():
                if column not in exposure_columns:
                    conn.execute(f"ALTER TABLE exposures ADD COLUMN {column} {definition}")
            manifest_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(candidate_manifests)")
            }
            if "source_evidence_json" not in manifest_columns:
                conn.execute(
                    "ALTER TABLE candidate_manifests "
                    "ADD COLUMN source_evidence_json TEXT NOT NULL DEFAULT '{}'"
                )
            item_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(items)")
            }
            if "stats_snapshot_version" not in item_columns:
                conn.execute("ALTER TABLE items ADD COLUMN stats_snapshot_version TEXT")
            if "stats_available_at" not in item_columns:
                conn.execute("ALTER TABLE items ADD COLUMN stats_available_at TEXT")
            model_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(model_versions)")
            }
            model_migrations = {
                "training_window_start": "TEXT",
                "training_window_end": "TEXT",
                "sample_count": "INTEGER",
                "event_count": "INTEGER",
                "training_status": "TEXT NOT NULL DEFAULT 'succeeded'",
                "publish_status": "TEXT NOT NULL DEFAULT 'published'",
                "evaluation_protocol_json": "TEXT NOT NULL DEFAULT '{}'",
            }
            for column, definition in model_migrations.items():
                if column not in model_columns:
                    conn.execute(f"ALTER TABLE model_versions ADD COLUMN {column} {definition}")
            snapshot_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(feed_snapshots)")
            }
            if "diversity_json" not in snapshot_columns:
                conn.execute(
                    "ALTER TABLE feed_snapshots ADD COLUMN diversity_json TEXT NOT NULL DEFAULT '{}'"
                )
            operation_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(operations)")
            }
            if "batch_id" not in operation_columns:
                conn.execute("ALTER TABLE operations ADD COLUMN batch_id TEXT")
            self._migrate_events(conn)
            state_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(user_item_state)")
            }
            state_migrations = {
                "dwell_ms_total": "INTEGER NOT NULL DEFAULT 0 CHECK (dwell_ms_total >= 0)",
                "dwell_event_count": (
                    "INTEGER NOT NULL DEFAULT 0 CHECK (dwell_event_count >= 0)"
                ),
                "share_count": "INTEGER NOT NULL DEFAULT 0 CHECK (share_count >= 0)",
                "revisit_count": "INTEGER NOT NULL DEFAULT 0 CHECK (revisit_count >= 0)",
            }
            for column, definition in state_migrations.items():
                if column not in state_columns:
                    conn.execute(
                        f"ALTER TABLE user_item_state ADD COLUMN {column} {definition}"
                    )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_operations_batch ON operations(batch_id, operation_id)"
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_requests_snapshot
                ON recommendation_requests(snapshot_id, snapshot_offset)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_exposures_snapshot
                ON exposures(snapshot_id, snapshot_position)
                """
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO app_metadata(key, value, updated_at)
                VALUES (
                    'viewable_impression_semantics_started_at',
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                )
                """
            )

    @staticmethod
    def _migrate_events(conn: sqlite3.Connection) -> None:
        table = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'events'"
        ).fetchone()
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(events)")}
        sql = str(table["sql"] or "") if table else ""
        if {"dwell_ms", "visit_index"} <= columns and "'revisit'" in sql:
            return

        before = int(conn.execute("SELECT COUNT(*) FROM events").fetchone()[0])
        conn.execute("ALTER TABLE events RENAME TO events_legacy")
        conn.execute(
            """
            CREATE TABLE events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL CHECK (
                    event_type IN (
                        'impression', 'click', 'like', 'not_interested',
                        'dwell', 'share', 'revisit'
                    )
                ),
                request_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                item_id TEXT NOT NULL,
                position INTEGER NOT NULL CHECK (position >= 0),
                client_timestamp TEXT,
                dwell_ms INTEGER CHECK (
                    dwell_ms IS NULL OR dwell_ms BETWEEN 750 AND 600000
                ),
                visit_index INTEGER CHECK (visit_index IS NULL OR visit_index >= 1),
                received_at TEXT NOT NULL,
                FOREIGN KEY(request_id, user_id, item_id, position)
                    REFERENCES exposures(request_id, user_id, item_id, position),
                UNIQUE(request_id, user_id, item_id, position, event_type)
            )
            """
        )
        conn.execute(
            """
            INSERT INTO events(
                event_id, event_type, request_id, user_id, item_id, position,
                client_timestamp, dwell_ms, visit_index, received_at
            )
            SELECT event_id, event_type, request_id, user_id, item_id, position,
                   client_timestamp, NULL, NULL, received_at
            FROM events_legacy
            """
        )
        after = int(conn.execute("SELECT COUNT(*) FROM events").fetchone()[0])
        if after != before:
            raise sqlite3.IntegrityError("events migration row-count mismatch")
        conn.execute("DROP TABLE events_legacy")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_user_time ON events(user_id, received_at DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_type_time ON events(event_type, received_at DESC)"
        )

    @staticmethod
    def _migrate_user_roles(conn: sqlite3.Connection) -> None:
        table = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'users'"
        ).fetchone()
        sql = str(table["sql"] or "") if table else ""
        if "'analyst'" in sql and "'operator'" in sql:
            return

        before = int(conn.execute("SELECT COUNT(*) FROM users").fetchone()[0])
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("PRAGMA legacy_alter_table = ON")
        try:
            conn.execute("ALTER TABLE users RENAME TO users_legacy")
            conn.execute(
                """
                CREATE TABLE users (
                    id TEXT PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    dataset_user_id TEXT UNIQUE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'analyst', 'operator', 'admin')),
                    is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                INSERT INTO users(
                    id, username, dataset_user_id, password_hash, role, is_active, created_at
                )
                SELECT id, username, dataset_user_id, password_hash, role, is_active, created_at
                FROM users_legacy
                """
            )
            after = int(conn.execute("SELECT COUNT(*) FROM users").fetchone()[0])
            if after != before:
                raise sqlite3.IntegrityError("users role migration row-count mismatch")
            conn.execute("DROP TABLE users_legacy")
        finally:
            conn.execute("PRAGMA legacy_alter_table = OFF")
            conn.execute("PRAGMA foreign_keys = ON")

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
