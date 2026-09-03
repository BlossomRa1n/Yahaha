from __future__ import annotations

import csv
import json
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from app.artifacts import ArtifactStore
from app.cli import load_items, seed_accounts
from app.config import Settings
from app.db import Database
from app.main import create_app
from recsys.data import prepare_data
from recsys.model import train_model
from recsys.mixing import DYNAMIC_POLICY_VERSION, SAFE_POLICY_VERSION


def _write_synthetic_raw(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True)
    item_ids = list(range(1, 151))
    with (raw_dir / "MicroLens-50k_pairs.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["user", "item", "timestamp"])
        for timestamp in range(1, 1501):
            writer.writerow([((timestamp - 1) % 10) + 1, ((timestamp - 1) % 150) + 1, timestamp])
    with (raw_dir / "MicroLens-50k_titles.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["item", "title"])
        for item_id in item_ids:
            writer.writerow([item_id, f"CI item {item_id}"])
    with (raw_dir / "MicroLens-50k_likes_and_views.txt").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        for item_id in item_ids:
            handle.write(f"{item_id}\t{item_id}\t{item_id * 10}\n")


def run_smoke(root: Path) -> dict[str, object]:
    raw_dir = root / "raw"
    processed_dir = root / "processed"
    artifacts_dir = root / "artifacts"
    _write_synthetic_raw(raw_dir)
    prepared = prepare_data(raw_dir, processed_dir, seed=20260902)
    trained = train_model(
        processed_dir,
        artifacts_dir,
        mode="smoke",
        max_users=10,
        max_eval_users=10,
        rank=4,
        seed=20260902,
    )
    settings = Settings(
        app_env="test",
        app_secret="ci-only-secret-with-at-least-32-characters",
        database_path=root / "app.db",
        model_pointer=artifacts_dir / "current.json",
        session_hours=1,
        session_cookie="ci_session",
    )
    database = Database(settings.database_path)
    database.reset()
    artifact = ArtifactStore(settings.model_pointer).get()
    if artifact is None:
        raise RuntimeError("CI smoke model could not be loaded")
    if artifact.content_item_vectors is None or artifact.item_cf_neighbors is None:
        raise RuntimeError("CI smoke hybrid retrieval artifacts could not be loaded")
    selected_policy = (artifact.mix_policy or {}).get("selected_policy_version")
    if selected_policy not in {SAFE_POLICY_VERSION, DYNAMIC_POLICY_VERSION}:
        raise RuntimeError("CI smoke model has no compatible locked mix policy")
    policy_search = artifact.metrics.get("mix_policy_search") or {}
    if policy_search.get("search_split") != "validation_only" or not policy_search.get(
        "test_policy_locked"
    ):
        raise RuntimeError("CI smoke mix policy was not locked from validation")
    with database.transaction(immediate=True) as conn:
        item_count = load_items(conn, processed_dir / "items.csv")
        seed_accounts(conn, [str(value) for value in artifact.user_ids[:2]])
    with TestClient(create_app(settings)) as client:
        login = client.post(
            "/api/v1/auth/login",
            json={"username": "alice", "password": "demo-pass"},
        )
        login.raise_for_status()
        feed = client.get("/api/v1/feeds/personalized?limit=5")
        feed.raise_for_status()
        feed_payload = feed.json()
        item = feed_payload["items"][0]
        events = client.post(
            "/api/v1/events/batch",
            json={
                "events": [
                    {
                        "event_id": f"imp:{feed_payload['request_id']}:{item['item_id']}:0",
                        "event_type": "impression",
                        "request_id": feed_payload["request_id"],
                        "item_id": item["item_id"],
                        "position": item["position"],
                        "client_timestamp": datetime.now(UTC).isoformat(),
                    },
                    {
                        "event_id": "ci-dwell",
                        "event_type": "dwell",
                        "request_id": feed_payload["request_id"],
                        "item_id": item["item_id"],
                        "position": item["position"],
                        "client_timestamp": datetime.now(UTC).isoformat(),
                        "dwell_ms": 5_000,
                    },
                    {
                        "event_id": "ci-share",
                        "event_type": "share",
                        "request_id": feed_payload["request_id"],
                        "item_id": item["item_id"],
                        "position": item["position"],
                        "client_timestamp": datetime.now(UTC).isoformat(),
                    },
                ]
            },
        )
        events.raise_for_status()
        health = client.get("/api/v1/health")
        health.raise_for_status()
        payload = health.json()
    if payload.get("model_version") != trained["model_version"]:
        raise RuntimeError("health endpoint did not load the published smoke model")
    return {
        "status": "passed",
        "data_version": prepared["data_version"],
        "model_version": trained["model_version"],
        "mix_policy_version": selected_policy,
        "interactions": prepared["counts"]["interactions"],
        "items": item_count,
        "feed_sources": sorted({row["source"] for row in feed_payload["items"]}),
        "events_accepted": events.json()["accepted"],
        "database": payload["database"],
    }


def main() -> int:
    root = Path(".tmp") / f"ci-smoke-{uuid.uuid4().hex}"
    try:
        print(json.dumps(run_smoke(root), indent=2, sort_keys=True))
    finally:
        shutil.rmtree(root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
