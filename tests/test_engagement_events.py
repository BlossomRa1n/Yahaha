from __future__ import annotations

import csv
import io
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from tests.test_api import _login, api_env


def _event(feed: dict, event_type: str, event_id: str, *, dwell_ms: int | None = None) -> dict:
    item = feed["items"][0]
    payload = {
        "event_id": event_id,
        "event_type": event_type,
        "request_id": feed["request_id"],
        "item_id": item["item_id"],
        "position": item["position"],
        "client_timestamp": datetime.now(UTC).isoformat(),
    }
    if dwell_ms is not None:
        payload["dwell_ms"] = dwell_ms
    return payload


def test_dwell_revisit_share_are_linked_idempotent_and_observable(api_env) -> None:
    app, database, _ = api_env
    with TestClient(app) as alice, TestClient(app) as bob, TestClient(app) as admin:
        alice_user = _login(alice, "alice")
        _login(bob, "bob")
        _login(admin, "admin", "admin-pass")

        first = alice.get("/api/v1/feeds/popular?limit=3").json()
        item_id = first["items"][0]["item_id"]
        first_impression = _event(
            first,
            "impression",
            f"imp:{first['request_id']}:{item_id}:0",
        )
        assert alice.post("/api/v1/events/batch", json={"events": [first_impression]}).status_code == 200

        now = datetime.now(UTC)
        boost = admin.post(
            "/api/v1/admin/boosts",
            json={
                "item_id": item_id,
                "audience": "users",
                "user_ids": [alice_user["id"]],
                "feed_types": ["popular"],
                "position": 0,
                "priority": 100,
                "starts_at": (now - timedelta(minutes=1)).isoformat(),
                "ends_at": (now + timedelta(hours=1)).isoformat(),
                "reason": "revisit verification",
            },
        )
        assert boost.status_code == 200, boost.text
        second = alice.get("/api/v1/feeds/popular?limit=3").json()
        assert second["items"][0]["item_id"] == item_id

        second_impression = _event(
            second,
            "impression",
            f"imp:{second['request_id']}:{item_id}:0",
        )
        second_impression["visit_index"] = 999
        dwell = _event(second, "dwell", "dwell-stable", dwell_ms=10_000)
        share = _event(second, "share", "share-stable")
        created = alice.post(
            "/api/v1/events/batch",
            json={"events": [second_impression, dwell, share]},
        )
        assert created.status_code == 200, created.text
        assert created.json()["accepted"] == 3

        retry = alice.post("/api/v1/events/batch", json={"events": [dwell, share]})
        assert retry.status_code == 200
        assert retry.json()["accepted"] == 0
        assert retry.json()["duplicates"] == 2
        extended_dwell = {**dwell, "event_id": "dwell-extended", "dwell_ms": 30_000}
        merged = alice.post("/api/v1/events/batch", json={"events": [extended_dwell]})
        assert merged.status_code == 200
        assert merged.json()["duplicates"] == 1

        forged = bob.post(
            "/api/v1/events/batch",
            json={"events": [{**dwell, "event_id": "forged-dwell"}]},
        )
        assert forged.status_code == 422
        assert forged.json()["error"]["code"] == "exposure_mismatch"
        assert alice.post(
            "/api/v1/events/batch",
            json={"events": [{**share, "event_id": str(uuid.uuid4()), "dwell_ms": 900}]},
        ).status_code == 422
        assert alice.post(
            "/api/v1/events/batch",
            json={"events": [{**dwell, "event_id": str(uuid.uuid4()), "dwell_ms": 700}]},
        ).status_code == 422

        with database.connect() as conn:
            visits = conn.execute(
                """
                SELECT event_type, request_id, dwell_ms, visit_index
                FROM events WHERE user_id = ? AND item_id = ?
                ORDER BY received_at, event_type
                """,
                (alice_user["id"], item_id),
            ).fetchall()
            impressions = [row for row in visits if row["event_type"] == "impression"]
            revisits = [row for row in visits if row["event_type"] == "revisit"]
            dwells = [row for row in visits if row["event_type"] == "dwell"]
            assert [row["visit_index"] for row in impressions] == [1, 2]
            assert len(revisits) == 1 and revisits[0]["visit_index"] == 2
            assert len(dwells) == 1 and dwells[0]["dwell_ms"] == 30_000
            state = conn.execute(
                "SELECT * FROM user_item_state WHERE user_id = ? AND item_id = ?",
                (alice_user["id"], item_id),
            ).fetchone()
            assert state["dwell_ms_total"] == 30_000
            assert state["dwell_event_count"] == 1
            assert state["share_count"] == 1
            assert state["revisit_count"] == 1
            assert state["affinity"] <= 8.0

        overview = admin.get("/api/v1/admin/dashboard/overview").json()
        assert overview["shares"] == 1
        assert overview["revisits"] == 1
        assert overview["revisit_users"] == 1
        assert overview["dwell"]["count"] == 1
        assert overview["dwell"]["average"] == 30_000
        popular = next(row for row in overview["feed_breakdown"] if row["feed_type"] == "popular")
        assert popular["shares"] == 1
        assert popular["revisits"] == 1
        assert popular["average_dwell_ms"] == 30_000

        export = admin.get("/api/v1/admin/dashboard/export.csv")
        rows = list(csv.DictReader(io.StringIO(export.content.decode("utf-8-sig"))))
        summary = next(row for row in rows if row["record_type"] == "overview")
        assert int(summary["shares"]) == overview["shares"]
        assert int(summary["revisits"]) == overview["revisits"]
        assert float(summary["average_dwell_ms"]) == overview["dwell"]["average"]

        recent = alice.get("/api/v1/me/events").json()["events"]
        assert any(row["event_type"] == "revisit" and row["visit_index"] == 2 for row in recent)
        assert any(row["event_type"] == "dwell" and row["dwell_ms"] == 30_000 for row in recent)


def test_not_interested_dominates_later_positive_profile_events(api_env) -> None:
    app, database, _ = api_env
    with TestClient(app) as alice:
        user = _login(alice, "alice")
        feed = alice.get("/api/v1/feeds/popular?limit=1").json()
        negative = _event(feed, "not_interested", "negative-first")
        share = _event(feed, "share", "share-after-negative")
        assert alice.post(
            "/api/v1/events/batch", json={"events": [negative, share]}
        ).status_code == 200
        with database.connect() as conn:
            state = conn.execute(
                "SELECT affinity, not_interested FROM user_item_state WHERE user_id = ? AND item_id = ?",
                (user["id"], feed["items"][0]["item_id"]),
            ).fetchone()
            assert state["not_interested"] == 1
            assert state["affinity"] == -4.0


def test_legacy_event_table_migration_preserves_history(api_env) -> None:
    app, database, _ = api_env
    with TestClient(app) as alice:
        _login(alice, "alice")
        feed = alice.get("/api/v1/feeds/popular?limit=1").json()
        event = _event(feed, "click", "legacy-click")
        assert alice.post("/api/v1/events/batch", json={"events": [event]}).status_code == 200

    with database.transaction(immediate=True) as conn:
        conn.execute("ALTER TABLE events RENAME TO events_current")
        conn.execute(
            """
            CREATE TABLE events (
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
            )
            """
        )
        conn.execute(
            """
            INSERT INTO events(
                event_id, event_type, request_id, user_id, item_id, position,
                client_timestamp, received_at
            )
            SELECT event_id, event_type, request_id, user_id, item_id, position,
                   client_timestamp, received_at
            FROM events_current
            """
        )
        conn.execute("DROP TABLE events_current")

    database.initialize()
    with database.connect() as conn:
        row = conn.execute("SELECT * FROM events WHERE event_id = 'legacy-click'").fetchone()
        assert row is not None
        assert row["event_type"] == "click"
        assert row["dwell_ms"] is None
        assert row["visit_index"] is None
        sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'events'"
        ).fetchone()["sql"]
        assert "'dwell'" in sql and "'share'" in sql and "'revisit'" in sql
