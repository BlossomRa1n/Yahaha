from __future__ import annotations

import csv
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.cli import load_items, seed_accounts
from app.config import Settings
from app.db import Database
from app.main import create_app


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
        for value in range(1, 9):
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


def test_personalized_feed_events_profile_dashboard_and_isolation(api_env) -> None:
    app, database, _ = api_env
    with TestClient(app) as alice, TestClient(app) as bob, TestClient(app) as admin:
        alice_user = _login(alice, "alice")
        _login(bob, "bob")
        _login(admin, "admin", "admin-pass")
        alice_feed = alice.get("/api/v1/feeds/personalized?limit=2").json()
        bob_feed = bob.get("/api/v1/feeds/personalized?limit=2").json()
        assert [item["item_id"] for item in alice_feed["items"]] != [
            item["item_id"] for item in bob_feed["items"]
        ]
        assert {
            "request_id",
            "feed_type",
            "model_version",
            "profile_version",
            "fallback_reason",
            "next_cursor",
            "has_more",
            "items",
        } == set(alice_feed)
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
            ).fetchone()[0] == 2

        profile_before = alice.get("/api/v1/me/profile").json()
        event = _client_event(alice_feed, "favorite", "alice-like-1")
        accepted = alice.post("/api/v1/events/batch", json={"events": [event]})
        assert accepted.status_code == 200
        assert accepted.json() == {
            "accepted": 1,
            "duplicates": 0,
            "profile_version": profile_before["version"] + 1,
        }
        duplicate = alice.post("/api/v1/events/batch", json={"events": [event]}).json()
        assert duplicate["accepted"] == 0
        assert duplicate["duplicates"] == 1

        cross_user = bob.post(
            "/api/v1/events/batch",
            json={"events": [{**event, "event_id": "bob-forgery"}]},
        )
        assert cross_user.status_code == 422
        assert cross_user.json()["error"]["code"] == "exposure_mismatch"
        assert bob.get("/api/v1/me/events").json()["events"]
        assert all(
            row["request_id"] != alice_feed["request_id"]
            for row in bob.get("/api/v1/me/events").json()["events"]
        )

        next_feed = alice.get("/api/v1/feeds/personalized?limit=2").json()
        assert next_feed["profile_version"] == profile_before["version"] + 1
        assert not ({item["item_id"] for item in alice_feed["items"]} & {
            item["item_id"] for item in next_feed["items"]
        })
        assert any("feedback" in item["explanation"].lower() for item in next_feed["items"])

        overview = admin.get("/api/v1/admin/dashboard/overview").json()
        assert overview["requests"] >= 3
        assert overview["exposures"] >= 6
        assert overview["likes"] == 1
        assert overview["top_items"]
        detail = admin.get(f"/api/v1/admin/requests/{alice_feed['request_id']}")
        assert detail.status_code == 200
        assert detail.json()["request"]["user_id"] == alice_user["id"]
        assert {row["event_type"] for row in detail.json()["events"]} >= {"impression", "like"}
        debug = admin.get(f"/api/v1/admin/users/{alice_user['id']}/debug").json()
        assert debug["profile"]["version"] == profile_before["version"] + 1


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
        assert payload["fallback_reason"] == "model_unavailable"
        assert payload["model_version"] == "fallback-popularity-v1"
        assert payload["items"]
        model_state = admin.get("/api/v1/admin/models").json()
        assert model_state["current_model_version"] is None
        assert "hash mismatch" in model_state["load_error"]
