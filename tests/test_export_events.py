from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

from app.cli import _iso_to_epoch_ms, export_events

SCHEMA = """
CREATE TABLE users (
    id TEXT PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    dataset_user_id TEXT UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);
CREATE TABLE events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    request_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    client_timestamp TEXT,
    received_at TEXT NOT NULL
);
"""


def _conn(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def test_export_events_stages_explicit_weight_without_duplicate_rows(tmp_path: Path) -> None:
    conn = _conn(tmp_path / "test.db")
    conn.executemany(
        "INSERT INTO users(id, username, dataset_user_id, password_hash, role, created_at) "
        "VALUES (?,?,?,?,?,?)",
        [
            ("u1", "alice", "5", "x", "user", "2026-09-01T00:00:00Z"),
            ("u2", "bob", "7", "x", "user", "2026-09-01T00:00:00Z"),
            ("u3", "carol", None, "x", "user", "2026-09-01T00:00:00Z"),
            ("u4", "admin", None, "x", "admin", "2026-09-01T00:00:00Z"),
        ],
    )
    conn.executemany(
        "INSERT INTO events(event_id, event_type, request_id, user_id, item_id, position, received_at) "
        "VALUES (?,?,?,?,?,?,?)",
        [
            ("e1", "click", "r1", "u1", "101", 0, "2026-09-01T10:00:00.000Z"),
            ("e2", "like", "r1", "u1", "102", 1, "2026-09-01T10:05:00.000Z"),
            ("e3", "impression", "r1", "u1", "101", 2, "2026-09-01T10:06:00.000Z"),
            ("e4", "not_interested", "r1", "u1", "103", 3, "2026-09-01T10:07:00.000Z"),
            ("e5", "click", "r2", "u2", "201", 0, "2026-09-01T11:00:00.000Z"),
            ("e6", "click", "r3", "u3", "301", 0, "2026-09-01T12:00:00.000Z"),
        ],
    )
    conn.commit()

    out = tmp_path / "online_events.csv"
    result = export_events(conn, out)
    conn.close()

    assert result["events"] == 3
    assert result["rows"] == 3
    assert result["skipped"] == 0
    assert result["weights"] == {"click": 1, "like": 3}
    assert result["consumed_by_training"] is False
    assert result["usage"] == "staging_only_requires_new_chronological_split"

    with out.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0] == {
        "user": "5",
        "item": "101",
        "timestamp": str(_iso_to_epoch_ms("2026-09-01T10:00:00.000Z")),
        "event_type": "click",
        "weight": "1",
    }

    like_rows = [r for r in rows if r["item"] == "102"]
    assert len(like_rows) == 1
    assert all(
        r["user"] == "5"
        and r["timestamp"] == str(_iso_to_epoch_ms("2026-09-01T10:05:00.000Z"))
        and r["event_type"] == "like"
        and r["weight"] == "3"
        for r in like_rows
    )

    assert {"user": "7", "item": "201"} in [{"user": r["user"], "item": r["item"]} for r in rows]
    # carol（dataset_user_id 为空）与 impression / not_interested 均被排除
    assert not any(r["item"] == "301" for r in rows)
    assert not any(r["item"] == "103" for r in rows)


def test_export_events_no_real_user_events(tmp_path: Path) -> None:
    conn = _conn(tmp_path / "test.db")
    conn.execute(
        "INSERT INTO users(id, username, dataset_user_id, password_hash, role, created_at) "
        "VALUES ('u1', 'alice', '5', 'x', 'user', '2026-09-01T00:00:00Z')"
    )
    conn.commit()

    out = tmp_path / "online_events.csv"
    result = export_events(conn, out)
    conn.close()

    assert result["events"] == 0
    assert result["rows"] == 0
    with out.open("r", encoding="utf-8", newline="") as handle:
        assert [line.strip() for line in handle] == [
            "user,item,timestamp,event_type,weight"
        ]
