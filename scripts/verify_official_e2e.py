from __future__ import annotations

import argparse
from collections import Counter
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
from recsys.mixing import DYNAMIC_POLICY_VERSION, SAFE_POLICY_VERSION


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
    event_id = (
        f"imp:{feed['request_id']}:{item['item_id']}:{item['position']}"
        if event_type == "impression"
        else str(uuid.uuid4())
    )
    payload = {
        "event_id": event_id,
        "event_type": event_type,
        "request_id": feed["request_id"],
        "item_id": item["item_id"],
        "position": item["position"],
        "client_timestamp": datetime.now(UTC).isoformat(),
    }
    if event_type == "dwell":
        payload["dwell_ms"] = 30_000
    return payload


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
                {
                    "position",
                    "source",
                    "score",
                    "raw_score",
                    "normalized_score",
                    "rank_in_source",
                    "explanation",
                    "model_version",
                }
                <= item.keys()
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
            replayed_page_two = _expect(
                alice.get(
                    "/api/v1/feeds/personalized",
                    params={"limit": 8, "cursor": alice_feed["next_cursor"]},
                )
            )
            assert [item["item_id"] for item in replayed_page_two["items"]] == [
                item["item_id"] for item in page_two["items"]
            ]
            assert [
                (
                    item["source"],
                    item["raw_score"],
                    item["normalized_score"],
                    item["rank_in_source"],
                )
                for item in replayed_page_two["items"]
            ] == [
                (
                    item["source"],
                    item["raw_score"],
                    item["normalized_score"],
                    item["rank_in_source"],
                )
                for item in page_two["items"]
            ]
            initial_source_counts = Counter(
                item["source"] for item in alice_feed["items"] + page_two["items"]
            )
            source_mix = alice_feed["diversity"]["source_mix"]
            mix_policy_version = source_mix["mix_policy_version"]
            assert mix_policy_version in {
                SAFE_POLICY_VERSION,
                DYNAMIC_POLICY_VERSION,
            }
            assert initial_source_counts["content_profile"] > 0
            assert len(
                {
                    item["item_id"]
                    for item in alice_feed["items"] + page_two["items"]
                }
            ) == len(alice_feed["items"] + page_two["items"])
            if mix_policy_version == SAFE_POLICY_VERSION:
                assert set(initial_source_counts) <= {"model", "content_profile"}
                assert initial_source_counts["model"] > 0
            else:
                assert len(initial_source_counts) >= 2

            profile_before = _expect(alice.get("/api/v1/me/profile"))
            events = [
                _event(alice_feed, 0, "impression"),
                _event(alice_feed, 1, "impression"),
                _event(alice_feed, 2, "impression"),
                _event(alice_feed, 0, "click"),
                _event(alice_feed, 1, "like"),
                _event(alice_feed, 2, "not_interested"),
                _event(alice_feed, 0, "dwell"),
                _event(alice_feed, 1, "share"),
            ]
            accepted = _expect(alice.post("/api/v1/events/batch", json={"events": events}))
            assert accepted["accepted"] == 8
            duplicate = _expect(alice.post("/api/v1/events/batch", json={"events": events[:3]}))
            assert duplicate["accepted"] == 0
            assert duplicate["duplicates"] == 3
            forged = {**events[3], "event_id": str(uuid.uuid4())}
            assert bob.post("/api/v1/events/batch", json={"events": [forged]}).status_code == 422

            profile_after = _expect(alice.get("/api/v1/me/profile"))
            assert profile_after["version"] == profile_before["version"] + 1
            next_feed = _expect(alice.get("/api/v1/feeds/personalized?limit=8"))
            rejected_item = events[-1]["item_id"]
            assert rejected_item not in {item["item_id"] for item in next_feed["items"]}
            assert next_feed["profile_version"] == profile_after["version"]

            after = _expect(admin.get("/api/v1/admin/dashboard/overview"))
            assert after["requests"] >= before["requests"] + 5
            assert after["exposures"] >= before["exposures"] + 37
            assert after["viewable_impressions"] == before["viewable_impressions"] + 3
            assert after["clicks"] == before["clicks"] + 1
            assert after["likes"] == before["likes"] + 1
            assert after["shares"] == before["shares"] + 1
            assert after["dwell"]["count"] == before["dwell"]["count"] + 1
            dashboard_source_counts = {
                row["source"]: row["served_exposures"] for row in after["candidate_sources"]
            }
            assert dashboard_source_counts["content_profile"] > 0
            assert dashboard_source_counts["model"] > 0
            assert sum(dashboard_source_counts.values()) == after["served_exposures"]
            request_detail = _expect(
                admin.get(f"/api/v1/admin/requests/{alice_feed['request_id']}")
            )
            assert {event["event_type"] for event in request_detail["events"]} >= {
                "impression",
                "click",
                "like",
                "not_interested",
                "dwell",
                "share",
            }

            item_id = alice_feed["items"][0]["item_id"]
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
            revisit_feed = None
            for feed_type in ("personalized", "popular", "explore"):
                boosted = _expect(alice.get(f"/api/v1/feeds/{feed_type}?limit=5"))
                assert boosted["items"][0]["item_id"] == item_id
                assert boosted["items"][0]["is_forced"] is True
                if feed_type == "personalized":
                    revisit_feed = boosted
            assert revisit_feed is not None
            revisit_result = _expect(
                alice.post(
                    "/api/v1/events/batch",
                    json={"events": [_event(revisit_feed, 0, "impression")]},
                )
            )
            assert revisit_result["accepted"] == 1
            revisit_dashboard = _expect(admin.get("/api/v1/admin/dashboard/overview"))
            assert revisit_dashboard["revisits"] >= 1

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
                personalized_latencies = [
                    float(row["latency_ms"])
                    for row in conn.execute(
                        """
                        SELECT latency_ms FROM recommendation_requests
                        WHERE feed_type = 'personalized'
                        ORDER BY created_at
                        """
                    )
                ]

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
            for key in (
                "requests",
                "served_exposures",
                "viewable_impressions",
                "clicks",
                "likes",
                "shares",
            )
        },
        "engagement_verified": {
            "dwell_ms": 30000,
            "shares": 1,
            "revisits": revisit_dashboard["revisits"],
        },
        "snapshot_replay_stable": True,
        "initial_personalized_source_counts": dict(initial_source_counts),
        "mix_policy_version": mix_policy_version,
        "dashboard_source_counts": dashboard_source_counts,
        "personalized_latency_ms": {
            "min": min(personalized_latencies),
            "max": max(personalized_latencies),
            "mean": sum(personalized_latencies) / len(personalized_latencies),
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
