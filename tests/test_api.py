from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.cli import cleanup_snapshots, load_items, seed_accounts
from app.config import Settings
from app.db import Database
from app.main import create_app
from app.security import decode_cursor, encode_cursor, session_digest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_artifact(root: Path) -> Path:
    version_dir = root / "model-v1"
    version_dir.mkdir(parents=True)
    arrays = {
        "user_ids.npy": np.array(["dataset-alice", "dataset-bob"]),
        "item_ids.npy": np.array([str(value) for value in range(1, 9)]),
        "user_factors.npy": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        "item_factors.npy": np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.8, 0.2],
                [0.1, 0.9],
                [0.0, 1.0],
                [0.2, 0.8],
                [0.5, 0.5],
                [-0.2, 0.1],
            ],
            dtype=np.float64,
        ),
    }
    for name, value in arrays.items():
        np.save(version_dir / name, value, allow_pickle=False)
    popularity_path = version_dir / "popularity.json"
    popularity_path.write_text(
        json.dumps(
            {"items": [{"item_id": str(value), "score": 9 - value} for value in range(1, 9)]}
        ),
        encoding="utf-8",
    )
    files = {
        name: {"sha256": _sha256(version_dir / name)}
        for name in (*arrays.keys(), "popularity.json")
    }
    manifest = {
        "schema_version": 1,
        "model_version": "model-v1",
        "data_version": "fixture-v1",
        "algorithm": "truncated_svd",
        "metrics": {"recall@5": 0.5, "ndcg@5": 0.4},
        "files": files,
    }
    (version_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    pointer = root / "current.json"
    pointer.write_text(json.dumps({"manifest": "model-v1/manifest.json"}), encoding="utf-8")
    return pointer


def _write_items(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id", "title", "likes", "views"])
        writer.writeheader()
        for value in range(1, 21):
            writer.writerow(
                {
                    "item_id": str(value),
                    "title": f"Item {value}",
                    "likes": (9 - value) * 10,
                    "views": (9 - value) * 100,
                }
            )


@pytest.fixture
def api_env(tmp_path: Path):
    pointer = _write_artifact(tmp_path / "artifacts")
    items_path = tmp_path / "items.csv"
    _write_items(items_path)
    settings = Settings(
        app_env="test",
        app_secret="test-secret-with-enough-entropy",
        database_path=tmp_path / "app.db",
        model_pointer=pointer,
        session_hours=12,
        session_cookie="test_session",
    )
    database = Database(settings.database_path)
    database.reset()
    with database.transaction(immediate=True) as conn:
        load_items(conn, items_path)
        seed_accounts(conn, ["dataset-alice", "dataset-bob"])
    app = create_app(settings)
    return app, database, pointer


def _login(client: TestClient, username: str, password: str = "demo-pass") -> dict:
    response = client.post("/api/v1/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200, response.text
    return response.json()["user"]


def _client_event(feed: dict, event_type: str, event_id: str) -> dict:
    item = feed["items"][0]
    return {
        "event_id": event_id,
        "event_type": event_type,
        "request_id": feed["request_id"],
        "item_id": item["item_id"],
        "position": item["position"],
        "client_timestamp": datetime.now(UTC).isoformat(),
    }


def _dashboard(client: TestClient) -> dict:
    response = client.get(
        "/api/v1/admin/dashboard/overview",
        params={"from": "2026-01-01T00:00:00Z", "to": "2026-12-31T00:00:00Z"},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_auth_logout_admin_forbidden_and_sqlite_pragmas(api_env) -> None:
    app, database, _ = api_env
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/web/app.js").status_code == 200
        assert client.post(
            "/api/v1/auth/login",
            json={"username": "alice", "password": "wrong"},
        ).status_code == 401
        user = _login(client, "alice")
        assert user["role"] == "user"
        assert "HttpOnly" in client.post(
            "/api/v1/auth/login",
            json={"username": "alice", "password": "demo-pass"},
        ).headers["set-cookie"]
        assert client.get("/api/v1/auth/me").json()["user"]["username"] == "alice"
        forbidden = client.get("/api/v1/admin/dashboard/overview")
        assert forbidden.status_code == 403
        assert forbidden.json()["error"]["code"] == "forbidden"
        assert client.post("/api/v1/auth/logout").json() == {"ok": True}
        assert client.get("/api/v1/auth/me").status_code == 401
    with database.connect() as conn:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000


def test_registration_creates_cold_user_profile_and_session(api_env) -> None:
    app, database, _ = api_env
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/auth/register",
            json={"username": "new.user", "password": "strong-pass-123"},
        )
        assert response.status_code == 201, response.text
        user = response.json()["user"]
        assert user["username"] == "new.user"
        assert user["role"] == "user"
        assert user["dataset_user_id"] is None
        cookie = response.headers["set-cookie"]
        assert "HttpOnly" in cookie
        assert "SameSite=strict" in cookie
        assert client.get("/api/v1/auth/me").json()["user"] == user
        assert client.get("/api/v1/me/profile").json()["version"] == 0
        feed = client.get("/api/v1/feeds/personalized?limit=3")
        assert feed.status_code == 200, feed.text
        assert feed.json()["fallback_reason"] == "cold_start"

        assert client.post("/api/v1/auth/logout").status_code == 200
        assert client.get("/api/v1/auth/me").status_code == 401

    with database.connect() as conn:
        stored = conn.execute(
            "SELECT dataset_user_id, role FROM users WHERE username = ?",
            ("new.user",),
        ).fetchone()
        assert stored is not None
        assert stored["dataset_user_id"] is None
        assert stored["role"] == "user"
        assert conn.execute(
            "SELECT COUNT(*) FROM profiles WHERE user_id = ?",
            (user["id"],),
        ).fetchone()[0] == 1


@pytest.mark.parametrize(
    ("payload", "status_code"),
    [
        ({"username": "alice", "password": "strong-pass-123"}, 409),
        ({"username": "ALICE", "password": "strong-pass-123"}, 409),
        ({"username": "1invalid", "password": "strong-pass-123"}, 422),
        ({"username": "ab", "password": "strong-pass-123"}, 422),
        ({"username": "a" * 33, "password": "strong-pass-123"}, 422),
        ({"username": "valid-name", "password": "onlyletters"}, 422),
        ({"username": "valid-name", "password": "1234567890"}, 422),
        ({"username": "valid-name", "password": "a1short"}, 422),
        ({"username": "valid-name", "password": "a1" * 65}, 422),
        (
            {
                "username": "valid-name",
                "password": "strong-pass-123",
                "role": "admin",
            },
            422,
        ),
        (
            {
                "username": "valid-name",
                "password": "strong-pass-123",
                "dataset_user_id": "dataset-alice",
            },
            422,
        ),
    ],
)
def test_registration_rejects_duplicates_weak_input_and_privilege_fields(
    api_env, payload: dict, status_code: int
) -> None:
    app, database, _ = api_env
    with TestClient(app) as client:
        response = client.post("/api/v1/auth/register", json=payload)
        assert response.status_code == status_code, response.text
    with database.connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM users WHERE role = 'admin'"
        ).fetchone()[0] == 1


def test_expired_session_is_rejected_cleaned_on_relogin_and_cannot_be_reused(api_env) -> None:
    app, database, _ = api_env
    expired_at = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
    with TestClient(app) as client:
        _login(client, "alice")
        old_cookie = client.cookies.get("test_session")
        assert old_cookie
        old_digest = session_digest(old_cookie)
        with database.transaction(immediate=True) as conn:
            updated = conn.execute(
                "UPDATE auth_sessions SET expires_at = ? WHERE token_hash = ?",
                (expired_at, old_digest),
            ).rowcount
        assert updated == 1

        assert client.get("/api/v1/auth/me").status_code == 401
        assert client.get("/api/v1/feeds/personalized?limit=1").status_code == 401
        assert client.get("/api/v1/admin/dashboard/overview").status_code == 401

        _login(client, "alice")
        new_cookie = client.cookies.get("test_session")
        assert new_cookie and new_cookie != old_cookie
        assert client.get("/api/v1/auth/me").status_code == 200
        assert client.get("/api/v1/admin/dashboard/overview").status_code == 403
        with database.connect() as conn:
            assert conn.execute(
                "SELECT COUNT(*) FROM auth_sessions WHERE token_hash = ?",
                (old_digest,),
            ).fetchone()[0] == 0

    with TestClient(app) as stale:
        stale.cookies.set("test_session", old_cookie)
        assert stale.get("/api/v1/auth/me").status_code == 401
        assert stale.get("/api/v1/feeds/personalized?limit=1").status_code == 401

    with TestClient(app) as admin:
        _login(admin, "admin", "admin-pass")
        assert admin.get("/api/v1/admin/dashboard/overview").status_code == 200


def test_admin_model_comparison_uses_real_metadata_and_protocol(api_env) -> None:
    app, database, _ = api_env
    protocol = {
        "k": 5,
        "candidate_universe": "test-items",
        "cohort_aggregation": "macro",
    }
    metrics = {
        "evaluation_protocol": protocol,
        "test": {
            "models": {
                "svd": {
                    "recall@5": 0.6,
                    "ndcg@5": 0.5,
                    "hitrate@5": 0.7,
                }
            }
        },
    }
    with TestClient(app) as user, TestClient(app) as admin:
        _login(user, "alice")
        _login(admin, "admin", "admin-pass")
        assert admin.get("/api/v1/admin/models").status_code == 200
        with database.transaction(immediate=True) as conn:
            conn.execute(
                "UPDATE model_versions SET metrics_json = ?, evaluation_protocol_json = ? WHERE model_version = 'model-v1'",
                (
                    json.dumps(
                        {
                            "evaluation_protocol": protocol,
                            "test": {
                                "models": {
                                    "svd": {
                                        "recall@5": 0.5,
                                        "ndcg@5": 0.4,
                                        "hitrate@5": 0.6,
                                    }
                                }
                            },
                        }
                    ),
                    json.dumps(protocol, sort_keys=True),
                ),
            )
            conn.execute(
                """
                INSERT INTO model_versions(
                    model_version, data_version, algorithm, artifact_path, metrics_json,
                    training_window_start, training_window_end, sample_count, event_count,
                    evaluation_protocol_json, status, created_at
                ) VALUES ('model-v2', 'data-v2', 'truncated_svd', 'artifact-v2', ?,
                          '2026-01-01T00:00:00Z', '2026-02-01T00:00:00Z', 42, 50,
                          ?, 'inactive', '2026-02-02T00:00:00Z')
                """,
                (json.dumps(metrics), json.dumps(protocol, sort_keys=True)),
            )
        params = [("versions", "model-v1"), ("versions", "model-v2")]
        assert user.get("/api/v1/admin/models/compare", params=params).status_code == 403
        response = admin.get("/api/v1/admin/models/compare", params=params)
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["protocol_compatible"] is True
        assert payload["models"][1]["training_window"]["start"] == "2026-01-01T00:00:00Z"
        assert payload["models"][1]["sample_count"] == 42
        assert payload["models"][1]["deltas_from_baseline"]["recall@5"] == pytest.approx(0.1)
        missing = admin.get(
            "/api/v1/admin/models/compare",
            params=[("versions", "model-v1"), ("versions", "missing")],
        )
        assert missing.status_code == 404
        assert admin.get(
            "/api/v1/admin/models/compare", params={"versions": "model-v1"}
        ).status_code == 422


def test_personalized_feed_events_profile_dashboard_and_isolation(api_env) -> None:
    app, database, _ = api_env
    with TestClient(app) as alice, TestClient(app) as bob, TestClient(app) as admin:
        alice_user = _login(alice, "alice")
        _login(bob, "bob")
        _login(admin, "admin", "admin-pass")
        alice_feed = alice.get("/api/v1/feeds/personalized?limit=2").json()
        bob_feed = bob.get("/api/v1/feeds/personalized?limit=2").json()
        # The unified 7-source + DeepFM path is the sole personalized path. This
        # fixture provisions no deep/multimodal artifacts, so warm dataset users
        # fall back to the deterministic popular feed (not a per-user SVD mix).
        assert alice_feed["fallback_reason"] == "unified_model_unavailable"
        assert bob_feed["fallback_reason"] == "unified_model_unavailable"
        assert [item["item_id"] for item in alice_feed["items"]] == [
            item["item_id"] for item in bob_feed["items"]
        ]
        assert {
            "request_id",
            "snapshot_id",
            "feed_type",
            "model_version",
            "profile_version",
            "fallback_reason",
            "diversity",
            "next_cursor",
            "has_more",
            "items",
        } == set(alice_feed)
        assert alice_feed["diversity"]["strategy"] == "title_token_mmr"
        assert alice_feed["model_version"] == "model-v1"
        assert alice_feed["items"]
        for position, item in enumerate(alice_feed["items"]):
            assert item["position"] == position
            assert {"source", "score", "explanation", "model_version", "is_forced"} <= set(item)

        with database.connect() as conn:
            assert conn.execute(
                "SELECT COUNT(*) FROM recommendation_requests WHERE request_id = ?",
                (alice_feed["request_id"],),
            ).fetchone()[0] == 1
            assert conn.execute(
                "SELECT COUNT(*) FROM exposures WHERE request_id = ?",
                (alice_feed["request_id"],),
            ).fetchone()[0] == 2
            assert conn.execute(
                "SELECT COUNT(*) FROM events WHERE request_id = ? AND event_type = 'impression'",
                (alice_feed["request_id"],),
            ).fetchone()[0] == 0

        profile_before = alice.get("/api/v1/me/profile").json()
        impression = _client_event(
            alice_feed,
            "impression",
            "imp:" + alice_feed["request_id"] + ":" + alice_feed["items"][0]["item_id"] + ":0",
        )
        event = _client_event(alice_feed, "favorite", "alice-like-1")
        accepted = alice.post("/api/v1/events/batch", json={"events": [impression, event]})
        assert accepted.status_code == 200
        assert accepted.json() == {
            "accepted": 2,
            "duplicates": 0,
            "profile_version": profile_before["version"] + 1,
        }
        duplicate = alice.post("/api/v1/events/batch", json={"events": [event]}).json()
        assert duplicate["accepted"] == 0
        assert duplicate["duplicates"] == 1
        own_events = alice.get("/api/v1/me/events").json()["events"]
        linked = next(row for row in own_events if row["event_id"] == event["event_id"])
        assert linked["feed_type"] == "personalized"
        assert linked["source"] == alice_feed["items"][0]["source"]

        cross_user = bob.post(
            "/api/v1/events/batch",
            json={"events": [{**event, "event_id": "bob-forgery"}]},
        )
        assert cross_user.status_code == 422
        assert cross_user.json()["error"]["code"] == "exposure_mismatch"
        assert bob.get("/api/v1/me/events").json()["events"] == []
        assert all(
            row["request_id"] != alice_feed["request_id"]
            for row in bob.get("/api/v1/me/events").json()["events"]
        )

        next_feed = alice.get("/api/v1/feeds/personalized?limit=2").json()
        assert next_feed["profile_version"] == profile_before["version"] + 1
        assert alice_feed["items"][0]["item_id"] not in {
            item["item_id"] for item in next_feed["items"]
        }
        assert alice_feed["items"][1]["item_id"] in {
            item["item_id"] for item in next_feed["items"]
        }
        assert next_feed["fallback_reason"] == "unified_model_unavailable"
        assert all(item["source"] == "popular" for item in next_feed["items"])

        overview = _dashboard(admin)
        assert overview["requests"] >= 3
        assert overview["exposures"] >= 6
        assert overview["served_exposures"] == overview["exposures"]
        assert overview["viewable_impressions"] == 1
        assert overview["ctr_denominator"] == "served_exposures"
        assert overview["likes"] == 1
        assert overview["top_items"]
        detail = admin.get(f"/api/v1/admin/requests/{alice_feed['request_id']}")
        assert detail.status_code == 200
        assert detail.json()["request"]["user_id"] == alice_user["id"]
        assert {row["event_type"] for row in detail.json()["events"]} >= {"impression", "like"}
        debug = admin.get(f"/api/v1/admin/users/{alice_user['id']}/debug").json()
        assert debug["profile"]["version"] == profile_before["version"] + 1


def test_dashboard_feed_breakdown_aggregates_each_behavior(api_env) -> None:
    app, _, _ = api_env
    with TestClient(app) as alice, TestClient(app) as admin:
        _login(alice, "alice")
        _login(admin, "admin", "admin-pass")

        for feed_type in ("personalized", "popular", "explore"):
            feed = alice.get(f"/api/v1/feeds/{feed_type}?limit=2").json()
            first, second = feed["items"]
            events = [
                {
                    "event_id": f"imp:{feed['request_id']}:{first['item_id']}:{first['position']}",
                    "event_type": "impression",
                    "request_id": feed["request_id"],
                    "item_id": first["item_id"],
                    "position": first["position"],
                    "client_timestamp": datetime.now(UTC).isoformat(),
                },
                {
                    "event_id": f"{feed_type}-click",
                    "event_type": "click",
                    "request_id": feed["request_id"],
                    "item_id": first["item_id"],
                    "position": first["position"],
                    "client_timestamp": datetime.now(UTC).isoformat(),
                },
                {
                    "event_id": f"{feed_type}-like",
                    "event_type": "like",
                    "request_id": feed["request_id"],
                    "item_id": second["item_id"],
                    "position": second["position"],
                    "client_timestamp": datetime.now(UTC).isoformat(),
                },
                {
                    "event_id": f"{feed_type}-not-interested",
                    "event_type": "not_interested",
                    "request_id": feed["request_id"],
                    "item_id": first["item_id"],
                    "position": first["position"],
                    "client_timestamp": datetime.now(UTC).isoformat(),
                },
            ]
            response = alice.post("/api/v1/events/batch", json={"events": events})
            assert response.status_code == 200, response.text
            assert response.json()["accepted"] == 4

        overview = _dashboard(admin)
        assert overview["requests"] == 3
        assert overview["exposures"] == 6
        breakdown = {row["feed_type"]: row for row in overview["feed_breakdown"]}
        assert set(breakdown) == {"personalized", "popular", "explore"}
        for feed_type in breakdown:
            assert breakdown[feed_type] == {
                "feed_type": feed_type,
                "requests": 1,
                "exposures": 2,
                "served_exposures": 2,
                "impressions": 1,
                "viewable_impressions": 1,
                "clicks": 1,
                "likes": 1,
                "not_interested": 1,
                "shares": 0,
                "revisits": 0,
                "revisit_users": 0,
                "average_dwell_ms": 0.0,
                "ctr": 0.5,
                "ctr_denominator": "served_exposures",
                "served_ctr": 0.5,
                "viewable_ctr": 1.0,
                "share": pytest.approx(1 / 3),
            }


def test_viewable_impressions_are_client_reported_idempotent_and_user_bound(api_env) -> None:
    app, database, _ = api_env
    with TestClient(app) as alice, TestClient(app) as bob, TestClient(app) as admin:
        _login(alice, "alice")
        _login(bob, "bob")
        _login(admin, "admin", "admin-pass")
        feed = alice.get("/api/v1/feeds/popular?limit=12").json()
        assert len(feed["items"]) == 12
        with database.connect() as conn:
            assert conn.execute(
                "SELECT COUNT(*) FROM exposures WHERE request_id = ?",
                (feed["request_id"],),
            ).fetchone()[0] == 12
            assert conn.execute(
                "SELECT COUNT(*) FROM events WHERE request_id = ? AND event_type = 'impression'",
                (feed["request_id"],),
            ).fetchone()[0] == 0

        events = [
            {
                "event_id": f"imp:{feed['request_id']}:{item['item_id']}:{item['position']}",
                "event_type": "impression",
                "request_id": feed["request_id"],
                "item_id": item["item_id"],
                "position": item["position"],
                "client_timestamp": datetime.now(UTC).isoformat(),
            }
            for item in feed["items"][:3]
        ]
        events.extend(
            [
                {
                    "event_id": "click-with-viewable-impression",
                    "event_type": "click",
                    "request_id": feed["request_id"],
                    "item_id": feed["items"][1]["item_id"],
                    "position": feed["items"][1]["position"],
                    "client_timestamp": datetime.now(UTC).isoformat(),
                },
                {
                    "event_id": "click-without-viewable-impression",
                    "event_type": "click",
                    "request_id": feed["request_id"],
                    "item_id": feed["items"][3]["item_id"],
                    "position": feed["items"][3]["position"],
                    "client_timestamp": datetime.now(UTC).isoformat(),
                },
            ]
        )
        created = alice.post("/api/v1/events/batch", json={"events": events})
        assert created.status_code == 200
        assert created.json()["accepted"] == 5
        duplicate = alice.post("/api/v1/events/batch", json={"events": events})
        assert duplicate.json()["accepted"] == 0
        assert duplicate.json()["duplicates"] == 5
        forged = bob.post(
            "/api/v1/events/batch",
            json={"events": [{**events[0], "event_id": "imp:forged"}]},
        )
        assert forged.status_code == 422
        assert forged.json()["error"]["code"] == "exposure_mismatch"

        next_feed = alice.get("/api/v1/feeds/popular?limit=12").json()
        viewed = {item["item_id"] for item in feed["items"][:3]}
        unviewed = {item["item_id"] for item in feed["items"][3:]}
        assert viewed.isdisjoint({item["item_id"] for item in next_feed["items"]})
        assert unviewed & {item["item_id"] for item in next_feed["items"]}

        overview = _dashboard(admin)
        assert overview["served_exposures"] == 24
        assert overview["viewable_impressions"] == 3
        assert overview["clicks"] == 2
        assert overview["viewable_ctr"] == pytest.approx(1 / 3)
        popular = next(
            row for row in overview["feed_breakdown"] if row["feed_type"] == "popular"
        )
        assert popular["clicks"] == 2
        assert popular["viewable_impressions"] == 3
        assert popular["viewable_ctr"] == pytest.approx(1 / 3)


def test_dashboard_top_items_respect_event_time_window(api_env) -> None:
    app, database, _ = api_env
    with TestClient(app) as alice, TestClient(app) as admin:
        _login(alice, "alice")
        _login(admin, "admin", "admin-pass")
        feed = alice.get("/api/v1/feeds/popular?limit=1").json()
        item = feed["items"][0]
        events = [
            {
                "event_id": f"window-{event_type}",
                "event_type": event_type,
                "request_id": feed["request_id"],
                "item_id": item["item_id"],
                "position": item["position"],
                "client_timestamp": datetime.now(UTC).isoformat(),
            }
            for event_type in ("impression", "click", "like")
        ]
        assert alice.post("/api/v1/events/batch", json={"events": events}).status_code == 200

        start = "2026-06-01T00:00:00.000Z"
        end = "2026-06-01T01:00:00.000Z"
        with database.transaction(immediate=True) as conn:
            conn.execute(
                "UPDATE recommendation_requests SET created_at = ? WHERE request_id = ?",
                ("2026-06-01T00:15:00.000Z", feed["request_id"]),
            )
            conn.execute(
                "UPDATE exposures SET created_at = ? WHERE request_id = ?",
                ("2026-06-01T00:15:00.000Z", feed["request_id"]),
            )
            conn.execute(
                """
                UPDATE app_metadata SET value = ?, updated_at = ?
                WHERE key = 'viewable_impression_semantics_started_at'
                """,
                ("2026-05-31T00:00:00.000Z", "2026-05-31T00:00:00.000Z"),
            )

        def dashboard() -> dict:
            response = admin.get(
                "/api/v1/admin/dashboard/overview",
                params={"from": start, "to": end},
            )
            assert response.status_code == 200, response.text
            return response.json()

        def metrics(overview: dict) -> tuple[dict, dict]:
            feed_row = next(
                row for row in overview["feed_breakdown"] if row["feed_type"] == "popular"
            )
            item_row = next(
                row for row in overview["top_items"] if row["item_id"] == item["item_id"]
            )
            return feed_row, item_row

        for event_time, expected in (
            ("2026-06-01T01:00:00.001Z", 0),
            (start, 1),
            (end, 0),
        ):
            with database.transaction(immediate=True) as conn:
                conn.execute(
                    "UPDATE events SET received_at = ? WHERE request_id = ?",
                    (event_time, feed["request_id"]),
                )
            overview = dashboard()
            feed_row, item_row = metrics(overview)
            assert overview["clicks"] == expected
            assert overview["likes"] == expected
            assert overview["viewable_impressions"] == expected
            assert feed_row["clicks"] == expected
            assert feed_row["likes"] == expected
            assert feed_row["viewable_impressions"] == expected
            assert item_row["clicks"] == expected
            assert item_row["likes"] == expected
            assert item_row["viewable_impressions"] == expected
            assert item_row["viewable_ctr"] == float(expected)


def test_cursor_uses_stable_snapshot_across_profile_model_and_ops_changes(api_env) -> None:
    app, database, pointer = api_env
    with TestClient(app) as alice, TestClient(app) as bob, TestClient(app) as admin:
        _login(alice, "alice")
        _login(bob, "bob")
        _login(admin, "admin", "admin-pass")
        first = alice.get("/api/v1/feeds/popular?limit=3").json()
        assert first["snapshot_id"]
        assert first["next_cursor"]
        first_ids = [item["item_id"] for item in first["items"]]
        assert [item["snapshot_position"] for item in first["items"]] == [0, 1, 2]

        like = _client_event(first, "like", "snapshot-like")
        assert alice.post("/api/v1/events/batch", json={"events": [like]}).status_code == 200
        model_v2 = pointer.parent / "model-v2"
        shutil.copytree(pointer.parent / "model-v1", model_v2)
        manifest_path = model_v2 / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["model_version"] = "model-v2"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        pointer.write_text(
            json.dumps({"manifest": "model-v2/manifest.json", "nonce": "published"}),
            encoding="utf-8",
        )
        now = datetime.now(UTC)
        assert admin.post(
            "/api/v1/admin/boosts",
            json={
                "item_id": "20",
                "audience": "all",
                "user_ids": [],
                "feed_types": ["popular"],
                "position": 0,
                "priority": 100,
                "starts_at": (now - timedelta(minutes=1)).isoformat(),
                "ends_at": (now + timedelta(days=1)).isoformat(),
                "reason": "new snapshots only",
            },
        ).status_code == 200
        second = alice.get(
            "/api/v1/feeds/popular",
            params={"limit": 3, "cursor": first["next_cursor"]},
        )
        replay = alice.get(
            "/api/v1/feeds/popular",
            params={"limit": 3, "cursor": first["next_cursor"]},
        )
        assert second.status_code == replay.status_code == 200
        second_payload = second.json()
        replay_payload = replay.json()
        assert second_payload["snapshot_id"] == first["snapshot_id"]
        assert replay_payload["snapshot_id"] == first["snapshot_id"]
        assert second_payload["profile_version"] == first["profile_version"]
        assert second_payload["model_version"] == "model-v1"
        assert [item["item_id"] for item in second_payload["items"]] == [
            item["item_id"] for item in replay_payload["items"]
        ]
        assert [
            (
                item["source"],
                item["raw_score"],
                item["normalized_score"],
                item["rank_in_source"],
                item["model_version"],
            )
            for item in second_payload["items"]
        ] == [
            (
                item["source"],
                item["raw_score"],
                item["normalized_score"],
                item["rank_in_source"],
                item["model_version"],
            )
            for item in replay_payload["items"]
        ]
        assert all(item["raw_score"] is not None for item in second_payload["items"])
        assert all(item["normalized_score"] is not None for item in second_payload["items"])
        assert all(item["rank_in_source"] is not None for item in second_payload["items"])
        assert [item["snapshot_position"] for item in second_payload["items"]] == [3, 4, 5]
        assert set(first_ids).isdisjoint({item["item_id"] for item in second_payload["items"]})

        cross_user = bob.get(
            "/api/v1/feeds/popular",
            params={"limit": 3, "cursor": first["next_cursor"]},
        )
        assert cross_user.status_code == 422
        cross_feed = alice.get(
            "/api/v1/feeds/explore",
            params={"limit": 3, "cursor": first["next_cursor"]},
        )
        assert cross_feed.status_code == 422
        cursor = first["next_cursor"]
        changed = ("A" if cursor[len(cursor) // 2] != "A" else "B")
        tampered = cursor[: len(cursor) // 2] + changed + cursor[len(cursor) // 2 + 1 :]
        assert alice.get(
            "/api/v1/feeds/popular",
            params={"limit": 3, "cursor": tampered},
        ).status_code == 422

        with database.connect() as conn:
            snapshot_count = conn.execute(
                "SELECT COUNT(*) FROM feed_snapshot_items WHERE snapshot_id = ?",
                (first["snapshot_id"],),
            ).fetchone()[0]
            assert 6 < snapshot_count <= app.state.settings.feed_snapshot_max_items
            traced = conn.execute(
                """
                SELECT COUNT(*) FROM recommendation_requests
                WHERE snapshot_id = ? AND snapshot_offset = 3
                """,
                (first["snapshot_id"],),
            ).fetchone()[0]
            assert traced == 2
        refreshed = alice.get("/api/v1/feeds/popular?limit=3").json()
        assert refreshed["snapshot_id"] != first["snapshot_id"]
        assert refreshed["model_version"] == "model-v2"
        assert refreshed["items"][0]["item_id"] == "20"
        assert refreshed["items"][0]["source"] == "forced"


def test_offline_and_expiry_override_old_snapshot_without_restore_reactivation(api_env) -> None:
    app, database, _ = api_env
    with TestClient(app) as alice, TestClient(app) as admin:
        _login(alice, "alice")
        _login(admin, "admin", "admin-pass")
        first = alice.get("/api/v1/feeds/popular?limit=3").json()
        with database.connect() as conn:
            target = conn.execute(
                """
                SELECT item_id FROM feed_snapshot_items
                WHERE snapshot_id = ? AND snapshot_position = 3
                """,
                (first["snapshot_id"],),
            ).fetchone()["item_id"]

        offline = admin.patch(
            f"/api/v1/admin/items/{target}/status",
            json={"status": "offline", "reason": "snapshot invalidation test"},
        )
        assert offline.status_code == 200
        old_page = alice.get(
            "/api/v1/feeds/popular",
            params={"limit": 3, "cursor": first["next_cursor"]},
        ).json()
        assert target not in {item["item_id"] for item in old_page["items"]}
        assert len(old_page["items"]) == 2

        assert admin.patch(
            f"/api/v1/admin/items/{target}/status",
            json={"status": "online", "reason": "restore for new snapshots only"},
        ).status_code == 200
        replay = alice.get(
            "/api/v1/feeds/popular",
            params={"limit": 3, "cursor": first["next_cursor"]},
        ).json()
        assert target not in {item["item_id"] for item in replay["items"]}

        expired_at = "2000-01-01T00:00:00Z"
        with database.transaction(immediate=True) as conn:
            conn.execute(
                """
                UPDATE feed_snapshots
                SET expires_at = ?
                WHERE snapshot_id = ?
                """,
                (expired_at, first["snapshot_id"]),
            )
        expired_payload = decode_cursor(first["next_cursor"], app.state.settings.app_secret)
        expired_payload["expires_at"] = expired_at
        expired_cursor = encode_cursor(expired_payload, app.state.settings.app_secret)
        expired = alice.get(
            "/api/v1/feeds/popular",
            params={"limit": 3, "cursor": expired_cursor},
        )
        assert expired.status_code == 410
        assert expired.json()["error"]["code"] == "cursor_expired"

        fresh = alice.get("/api/v1/feeds/popular?limit=3").json()
        with database.transaction(immediate=True) as conn:
            conn.execute(
                """
                UPDATE feed_snapshots
                SET status = 'expired'
                WHERE snapshot_id = ?
                """,
                (fresh["snapshot_id"],),
            )
        expired_status = alice.get(
            "/api/v1/feeds/popular",
            params={"limit": 3, "cursor": fresh["next_cursor"]},
        )
        assert expired_status.status_code == 410
        assert expired_status.json()["error"]["code"] == "cursor_expired"

        with database.transaction(immediate=True) as conn:
            cleanup_time = datetime.now(UTC).isoformat()
            before_unexpired = conn.execute(
                "SELECT COUNT(*) FROM feed_snapshots WHERE expires_at > ?",
                (cleanup_time,),
            ).fetchone()[0]
            result = cleanup_snapshots(conn, cleanup_time)
            after_unexpired = conn.execute(
                "SELECT COUNT(*) FROM feed_snapshots WHERE expires_at > ?",
                (cleanup_time,),
            ).fetchone()[0]
        assert result["deleted_expired_snapshots"] >= 1
        assert result["retained_active_snapshots"] == before_unexpired
        assert after_unexpired == before_unexpired
        after_cleanup = alice.get(
            "/api/v1/feeds/popular",
            params={"limit": 3, "cursor": expired_cursor},
        )
        assert after_cleanup.status_code == 410
        assert after_cleanup.json()["error"]["code"] == "cursor_expired"


def test_boost_offline_priority_restore_direct_api_and_audit(api_env) -> None:
    app, _, _ = api_env
    with TestClient(app) as alice, TestClient(app) as admin:
        _login(alice, "alice")
        _login(admin, "admin", "admin-pass")
        item_id = "8"
        now = datetime.now(UTC)
        boost = admin.post(
            "/api/v1/admin/boosts",
            json={
                "item_id": item_id,
                "audience": "all",
                "user_ids": [],
                "feed_types": ["popular"],
                "position": 0,
                "priority": 100,
                "starts_at": (now - timedelta(minutes=1)).isoformat(),
                "ends_at": (now + timedelta(days=1)).isoformat(),
                "reason": "integration test",
            },
        )
        assert boost.status_code == 200, boost.text
        boosted_feed = alice.get("/api/v1/feeds/popular?limit=3").json()
        assert boosted_feed["items"][0]["item_id"] == item_id
        assert boosted_feed["items"][0]["is_forced"] is True

        offline = admin.patch(
            f"/api/v1/admin/items/{item_id}/status",
            json={"status": "offline", "reason": "policy test"},
        )
        assert offline.status_code == 200
        offline_feed = alice.get("/api/v1/feeds/popular?limit=8").json()
        assert item_id not in {item["item_id"] for item in offline_feed["items"]}
        assert alice.get(f"/api/v1/items/{item_id}").status_code == 404
        assert alice.get(f"/api/v1/items/{item_id}/cover").status_code == 404

        restored = admin.patch(
            f"/api/v1/admin/items/{item_id}/status",
            json={"status": "online", "reason": "policy cleared"},
        )
        assert restored.status_code == 200
        restored_feed = alice.get("/api/v1/feeds/popular?limit=3").json()
        assert restored_feed["items"][0]["item_id"] == item_id
        assert alice.get(f"/api/v1/items/{item_id}").status_code == 200
        operations = admin.get("/api/v1/admin/operations").json()["operations"]
        actions = {row["action"] for row in operations}
        assert {"boost_create", "item_offline", "item_restore"} <= actions


def test_atomic_idempotent_batch_offline_restore_and_audit(api_env) -> None:
    app, database, _ = api_env
    with TestClient(app) as alice, TestClient(app) as admin:
        _login(alice, "alice")
        _login(admin, "admin", "admin-pass")
        first = alice.get("/api/v1/feeds/popular?limit=2").json()
        with database.connect() as conn:
            target_ids = [
                str(row["item_id"])
                for row in conn.execute(
                    """
                    SELECT item_id FROM feed_snapshot_items
                    WHERE snapshot_id = ? AND snapshot_position IN (2, 3)
                    ORDER BY snapshot_position
                    """,
                    (first["snapshot_id"],),
                )
            ]
        assert len(target_ids) == 2
        body = {
            "item_ids": [target_ids[0], target_ids[0], target_ids[1]],
            "status": "offline",
            "reason": "batch moderation test",
            "idempotency_key": "batch-offline-test-key",
        }
        assert alice.patch("/api/v1/admin/items/batch/status", json=body).status_code == 403
        response = admin.patch("/api/v1/admin/items/batch/status", json=body)
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["success_count"] == 2
        assert result["changed_count"] == 2
        assert result["failure_count"] == 0
        replay = admin.patch("/api/v1/admin/items/batch/status", json=body).json()
        assert replay["batch_id"] == result["batch_id"]
        assert replay["idempotent_replay"] is True
        conflict = admin.patch(
            "/api/v1/admin/items/batch/status",
            json={**body, "status": "online"},
        )
        assert conflict.status_code == 409
        for item_id in target_ids:
            assert alice.get(f"/api/v1/items/{item_id}").status_code == 404
        old_page = alice.get(f"/api/v1/feeds/popular?limit=2&cursor={first['next_cursor']}").json()
        assert not (set(target_ids) & {item["item_id"] for item in old_page["items"]})
        for feed_type in ("personalized", "popular", "explore"):
            feed = alice.get(f"/api/v1/feeds/{feed_type}?limit=20").json()
            assert not (set(target_ids) & {item["item_id"] for item in feed["items"]})

        untouched_id = next(str(value) for value in range(1, 21) if str(value) not in target_ids)
        mixed = admin.patch(
            "/api/v1/admin/items/batch/status",
            json={
                "item_ids": [untouched_id, "missing-item"],
                "status": "offline",
                "reason": "must roll back",
                "idempotency_key": "batch-mixed-test-key",
            },
        )
        assert mixed.status_code == 404
        assert alice.get(f"/api/v1/items/{untouched_id}").status_code == 200
        assert admin.patch(
            "/api/v1/admin/items/batch/status",
            json={
                "item_ids": [str(value) for value in range(101)],
                "status": "offline",
                "reason": "over limit",
                "idempotency_key": "batch-over-limit-key",
            },
        ).status_code == 422

        restored = admin.patch(
            "/api/v1/admin/items/batch/status",
            json={
                "item_ids": target_ids,
                "status": "online",
                "reason": "batch restore test",
                "idempotency_key": "batch-restore-test-key",
            },
        )
        assert restored.status_code == 200
        assert restored.json()["changed_count"] == 2
        for item_id in target_ids:
            assert alice.get(f"/api/v1/items/{item_id}").status_code == 200
        with database.connect() as conn:
            assert conn.execute(
                "SELECT COUNT(*) FROM operations WHERE batch_id = ?",
                (result["batch_id"],),
            ).fetchone()[0] == 2
            assert conn.execute(
                "SELECT COUNT(*) FROM operation_batches WHERE batch_id = ?",
                (result["batch_id"],),
            ).fetchone()[0] == 1


def test_personalized_falls_back_to_popular_when_deep_unavailable(api_env) -> None:
    app, _, _ = api_env
    with TestClient(app) as alice:
        _login(alice, "alice")
        feed = alice.get("/api/v1/feeds/personalized?limit=3").json()
        assert feed["fallback_reason"] == "unified_model_unavailable"
        assert feed["diversity"]["source_mix"]["strategy"] == "fallback"
        assert feed["diversity"]["source_mix"]["selected"]["popular"] >= 3
        assert [item["source"] for item in feed["items"]] == ["popular"] * len(feed["items"])


def test_corrupt_model_falls_back_without_fabricating_model_result(api_env) -> None:
    app, _, pointer = api_env
    manifest_path = pointer.parent / "model-v1" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["item_factors.npy"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    pointer.write_text(
        json.dumps({"manifest": "model-v1/manifest.json", "nonce": "corrupt"}),
        encoding="utf-8",
    )
    with TestClient(app) as alice, TestClient(app) as admin:
        _login(alice, "alice")
        _login(admin, "admin", "admin-pass")
        response = alice.get("/api/v1/feeds/personalized?limit=3")
        assert response.status_code == 200
        payload = response.json()
        assert payload["fallback_reason"] == "unified_model_unavailable"
        assert payload["model_version"] == "fallback-popularity-v1"
        assert payload["items"]
        model_state = admin.get("/api/v1/admin/models").json()
        assert model_state["current_model_version"] is None
        assert "hash mismatch" in model_state["load_error"]
