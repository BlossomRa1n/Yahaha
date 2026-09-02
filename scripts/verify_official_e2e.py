from __future__ import annotations

import argparse
import json
import tempfile
import uuid
from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app.artifacts import ArtifactStore
from app.cli import load_items, seed_accounts
from app.config import Settings
from app.db import Database
from app.main import create_app


def _expect(response: Any, status: int = 200) -> dict[str, Any]:
    assert response.status_code == status, response.text
    return response.json()


def _login(client: TestClient, username: str, password: str = "demo-pass") -> dict[str, Any]:
    return _expect(
        client.post(
            "/api/v1/auth/login",
            json={"username": username, "password": password},
        )
    )["user"]


def _event(feed: dict[str, Any], index: int, event_type: str) -> dict[str, Any]:
    item = feed["items"][index]
    return {
        "event_id": str(uuid.uuid4()),
        "event_type": event_type,
        "request_id": feed["request_id"],
        "item_id": item["item_id"],
        "position": item["position"],
        "client_timestamp": datetime.now(UTC).isoformat(),
    }


def verify(items_path: Path, model_pointer: Path) -> dict[str, Any]:
    artifact_store = ArtifactStore(model_pointer)
    artifact = artifact_store.get()
    assert artifact is not None, artifact_store.last_error or "model artifact is unavailable"
    assert len(artifact.user_ids) >= 2, "model must contain at least two mapped users"

    with tempfile.TemporaryDirectory(prefix="microlens-official-e2e-") as temp_name:
        settings = Settings(
            app_env="test",
            app_secret="official-e2e-secret-with-enough-entropy",
            database_path=Path(temp_name) / "app.db",
            model_pointer=model_pointer,
            session_hours=12,
            session_cookie="official_e2e_session",
        )
        database = Database(settings.database_path)
        database.reset()
        with database.transaction(immediate=True) as conn:
            item_count = load_items(conn, items_path)
            seed_accounts(conn, [str(value) for value in artifact.user_ids[:2]])

        app = create_app(settings)
        with ExitStack() as stack:
            alice = stack.enter_context(TestClient(app))
            bob = stack.enter_context(TestClient(app))
            carol = stack.enter_context(TestClient(app))
            admin = stack.enter_context(TestClient(app))
            alice_user = _login(alice, "alice")
            _login(bob, "bob")
            _login(carol, "carol")
            _login(admin, "admin", "admin-pass")

            assert alice.get("/api/v1/admin/dashboard/overview").status_code == 403
            before = _expect(admin.get("/api/v1/admin/dashboard/overview"))

            alice_feed = _expect(alice.get("/api/v1/feeds/personalized?limit=8"))
            bob_feed = _expect(bob.get("/api/v1/feeds/personalized?limit=8"))
            carol_feed = _expect(carol.get("/api/v1/feeds/personalized?limit=5"))
            assert len(alice_feed["items"]) >= 3
            assert [item["item_id"] for item in alice_feed["items"]] != [
                item["item_id"] for item in bob_feed["items"]
            ]
            assert carol_feed["fallback_reason"] == "cold_start"
            assert alice_feed["model_version"] == artifact.model_version
            assert all(
                {"position", "source", "score", "explanation", "model_version"} <= item.keys()
                for item in alice_feed["items"]
            )

            page_two = _expect(
                alice.get(
                    "/api/v1/feeds/personalized",
                    params={"limit": 8, "cursor": alice_feed["next_cursor"]},
                )
            )
            assert not (
                {item["item_id"] for item in alice_feed["items"]}
                & {item["item_id"] for item in page_two["items"]}
            )

            profile_before = _expect(alice.get("/api/v1/me/profile"))
            events = [
                _event(alice_feed, 0, "click"),
                _event(alice_feed, 1, "like"),
                _event(alice_feed, 2, "not_interested"),
            ]
            accepted = _expect(alice.post("/api/v1/events/batch", json={"events": events}))
            assert accepted["accepted"] == 3
            forged = {**events[0], "event_id": str(uuid.uuid4())}
            assert bob.post("/api/v1/events/batch", json={"events": [forged]}).status_code == 422

            profile_after = _expect(alice.get("/api/v1/me/profile"))
            assert profile_after["version"] == profile_before["version"] + 1
            next_feed = _expect(alice.get("/api/v1/feeds/personalized?limit=8"))
            rejected_item = events[2]["item_id"]
            assert rejected_item not in {item["item_id"] for item in next_feed["items"]}
            assert next_feed["profile_version"] == profile_after["version"]

            after = _expect(admin.get("/api/v1/admin/dashboard/overview"))
            assert after["requests"] >= before["requests"] + 5
            assert after["exposures"] >= before["exposures"] + 37
            assert after["clicks"] == before["clicks"] + 1
            assert after["likes"] == before["likes"] + 1
            request_detail = _expect(
                admin.get(f"/api/v1/admin/requests/{alice_feed['request_id']}")
            )
            assert {event["event_type"] for event in request_detail["events"]} >= {
                "impression",
                "click",
                "like",
                "not_interested",
            }

            candidate = _expect(admin.get("/api/v1/admin/items?status=online&limit=1"))["items"][0]
            item_id = candidate["item_id"]
            now = datetime.now(UTC)
            _expect(
                admin.post(
                    "/api/v1/admin/boosts",
                    json={
                        "item_id": item_id,
                        "audience": "users",
                        "user_ids": [alice_user["id"]],
                        "feed_types": ["personalized", "popular", "explore"],
                        "position": 0,
                        "priority": 100,
                        "starts_at": (now - timedelta(minutes=1)).isoformat(),
                        "ends_at": (now + timedelta(days=1)).isoformat(),
                        "reason": "official e2e verification",
                    },
                )
            )
            for feed_type in ("personalized", "popular", "explore"):
                boosted = _expect(alice.get(f"/api/v1/feeds/{feed_type}?limit=5"))
                assert boosted["items"][0]["item_id"] == item_id
                assert boosted["items"][0]["is_forced"] is True

            _expect(
                admin.patch(
                    f"/api/v1/admin/items/{item_id}/status",
                    json={"status": "offline", "reason": "official e2e policy check"},
                )
            )
            for feed_type in ("personalized", "popular", "explore"):
                offline_feed = _expect(alice.get(f"/api/v1/feeds/{feed_type}?limit=20"))
                assert item_id not in {item["item_id"] for item in offline_feed["items"]}
            assert alice.get(f"/api/v1/items/{item_id}").status_code == 404
            assert alice.get(f"/api/v1/items/{item_id}/cover").status_code == 404

            _expect(
                admin.patch(
                    f"/api/v1/admin/items/{item_id}/status",
                    json={"status": "online", "reason": "official e2e policy cleared"},
                )
            )
            restored = _expect(alice.get("/api/v1/feeds/popular?limit=5"))
            assert restored["items"][0]["item_id"] == item_id
            operations = _expect(admin.get("/api/v1/admin/operations"))["operations"]
            assert {"boost_create", "item_offline", "item_restore"} <= {
                row["action"] for row in operations
            }

            _expect(alice.post("/api/v1/auth/logout"))
            assert alice.get("/api/v1/auth/me").status_code == 401

            with database.connect() as conn:
                counts = {
                    table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                    for table in (
                        "users",
                        "items",
                        "recommendation_requests",
                        "exposures",
                        "events",
                        "operations",
                    )
                }

    return {
        "status": "passed",
        "model_version": artifact.model_version,
        "data_version": artifact.data_version,
        "items_imported": item_count,
        "alice_request_id": alice_feed["request_id"],
        "alice_bob_different": True,
        "carol_fallback": carol_feed["fallback_reason"],
        "dashboard_delta": {
            key: after[key] - before[key]
            for key in ("requests", "exposures", "clicks", "likes")
        },
        "database_counts": counts,
        "offline_precedence_verified": True,
        "logout_invalidated_session": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the official-data recommendation E2E")
    parser.add_argument("--items", type=Path, default=Path("data/processed/items.csv"))
    parser.add_argument("--model-pointer", type=Path, default=Path("artifacts/current.json"))
    args = parser.parse_args()
    result = verify(args.items.resolve(), args.model_pointer.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
