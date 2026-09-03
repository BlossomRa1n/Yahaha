from __future__ import annotations

import csv
import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.cli import load_items, run_online_retraining, seed_accounts
from app.config import Settings
from app.db import Database
from app.main import create_app
from recsys.data import prepare_data
from recsys.model import ModelTrainingError, train_model
from recsys.online_retrain import (
    OnlineEvent,
    _apply_user_item_feedback_rules,
    _event_feedback,
)


def _write_raw(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True)
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
        for item_id in range(1, 151):
            writer.writerow([item_id, f"Topic {item_id % 8} item {item_id}"])
    with (raw_dir / "MicroLens-50k_likes_and_views.txt").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        for item_id in range(1, 151):
            handle.write(f"{item_id}\t{item_id}\t{item_id * 10}\n")


def _event(feed: dict, index: int, event_type: str) -> dict:
    item = feed["items"][index]
    return {
        "event_id": (
            f"imp:{feed['request_id']}:{item['item_id']}:{item['position']}"
            if event_type == "impression"
            else str(uuid.uuid4())
        ),
        "event_type": event_type,
        "request_id": feed["request_id"],
        "item_id": item["item_id"],
        "position": item["position"],
        "client_timestamp": datetime.now(UTC).isoformat(),
    }


def _online_event(
    event_id: str,
    item_id: int,
    event_type: str,
    weight: float,
    disposition: str,
    timestamp_ms: int,
) -> OnlineEvent:
    return OnlineEvent(
        timestamp_ms=timestamp_ms,
        event_id=event_id,
        user_id=1,
        item_id=item_id,
        event_type=event_type,
        weight=weight,
        disposition=disposition,
        dwell_ms=None,
        visit_index=None,
        client_timestamp=None,
        received_at=datetime.fromtimestamp(timestamp_ms / 1000, UTC).isoformat(),
    )


def test_engagement_feedback_buckets_conflicts_and_cap() -> None:
    assert _event_feedback("dwell", 749) is None
    assert _event_feedback("dwell", 750) == (0.25, "positive")
    assert _event_feedback("dwell", 5_000) == (0.75, "positive")
    assert _event_feedback("dwell", 30_000) == (1.5, "positive")
    assert _event_feedback("share", None) == (4.0, "positive")
    assert _event_feedback("revisit", None) == (1.5, "positive")

    ruled = _apply_user_item_feedback_rules(
        [
            _online_event("click", 10, "click", 1.0, "positive", 1_000),
            _online_event("share", 10, "share", 4.0, "positive", 2_000),
            _online_event("revisit", 10, "revisit", 1.5, "positive", 3_000),
            _online_event("blocked-share", 20, "share", 4.0, "positive", 4_000),
            _online_event(
                "negative", 20, "not_interested", -2.0, "negative", 5_000
            ),
        ]
    )
    weights = {event.event_id: event.weight for event in ruled}
    dispositions = {event.event_id: event.disposition for event in ruled}
    assert weights["click"] + weights["share"] + weights["revisit"] == 6.0
    assert weights["revisit"] == 1.0
    assert weights["blocked-share"] == 0.0
    assert dispositions["blocked-share"] == "ignored_negative_conflict"


def test_real_event_window_retrains_publishes_and_failure_keeps_current(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    artifacts = tmp_path / "artifacts"
    _write_raw(raw)
    prepare_data(raw, processed, seed=17)
    initial = train_model(
        processed,
        artifacts,
        mode="smoke",
        max_users=10,
        max_eval_users=10,
        rank=4,
        seed=17,
    )
    settings = Settings(
        app_env="test",
        app_secret="online-retrain-test-secret-with-32-chars",
        database_path=tmp_path / "app.db",
        model_pointer=artifacts / "current.json",
        session_hours=1,
        session_cookie="retrain_session",
    )
    database = Database(settings.database_path)
    database.reset()
    with database.transaction(immediate=True) as conn:
        load_items(conn, processed / "items.csv")
        seed_accounts(conn, ["1", "2"])

    window_start = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    with TestClient(create_app(settings)) as client:
        assert client.post(
            "/api/v1/auth/login", json={"username": "alice", "password": "demo-pass"}
        ).status_code == 200
        feed = client.get("/api/v1/feeds/personalized?limit=15").json()
        assert len(feed["items"]) == 15
        assert client.post("/api/v1/events/batch", json={"events": [_event(feed, 0, "impression")]}).status_code == 200
        assert client.post("/api/v1/events/batch", json={"events": [_event(feed, 1, "not_interested")]}).status_code == 200
        dwell = _event(feed, 13, "dwell")
        dwell["dwell_ms"] = 30_000
        assert client.post(
            "/api/v1/events/batch", json={"events": [dwell]}
        ).status_code == 200
        assert client.post(
            "/api/v1/events/batch", json={"events": [_event(feed, 14, "share")]}
        ).status_code == 200
        for index in range(12):
            event_type = "like" if index % 4 == 0 else "click"
            response = client.post(
                "/api/v1/events/batch", json={"events": [_event(feed, index, event_type)]}
            )
            assert response.status_code == 200, response.text
    window_end = (datetime.now(UTC) + timedelta(seconds=1)).isoformat()

    result = run_online_retraining(
        settings=settings,
        start_time=window_start,
        end_time=window_end,
        base_processed_dir=processed,
        output_root=tmp_path / "retraining",
        artifacts_dir=artifacts,
        mode="smoke",
        max_users=10,
        max_eval_users=10,
        rank=4,
        seed=18,
    )
    assert result["model_version"] != initial["model_version"]
    assert result["data_version"].startswith("microlens50k-online-")
    assert result["event_count"] == 16
    # The click on the same item as not_interested is excluded by negative dominance.
    assert result["sample_count"] == 13
    summary = json.loads(
        (Path(result["processed_dir"]) / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["online_retraining"]["event_counts"] == {
        "click": 9,
        "dwell": 1,
        "impression": 1,
        "like": 3,
        "not_interested": 1,
        "share": 1,
    }
    assert summary["online_retraining"]["feedback_mapping"]["not_interested"]["weight"] == -2.0
    assert summary["online_retraining"]["feedback_mapping"]["share"]["weight"] == 4.0
    assert summary["online_retraining"]["feedback_mapping"]["dwell"]["weight"] == "bucketed"
    assert summary["cutoffs"]["train_cutoff_ms"] < summary["cutoffs"]["validation_cutoff_ms"]
    with database.connect() as conn:
        models = conn.execute("SELECT * FROM model_versions ORDER BY created_at").fetchall()
        run = conn.execute("SELECT * FROM training_runs WHERE run_id = ?", (result["run_id"],)).fetchone()
        assert len(models) == 2
        assert sum(row["status"] == "active" for row in models) == 1
        assert run["training_status"] == "succeeded"
        assert run["publish_status"] == "published"

    with TestClient(create_app(settings)) as client:
        client.post(
            "/api/v1/auth/login", json={"username": "alice", "password": "demo-pass"}
        )
        assert client.get("/api/v1/feeds/personalized?limit=3").json()["model_version"] == result["model_version"]

    with TestClient(create_app(settings)) as admin:
        assert admin.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "admin-pass"},
        ).status_code == 200
        comparison = admin.get(
            "/api/v1/admin/models/compare",
            params=[
                ("versions", initial["model_version"]),
                ("versions", result["model_version"]),
            ],
        )
        assert comparison.status_code == 200, comparison.text
        compared = comparison.json()
        assert compared["protocol_compatible"] is True
        assert compared["models"][1]["is_current"] is True
        assert compared["models"][1]["data_version"] == result["data_version"]
        assert "recall@10" in compared["models"][1]["metrics"]
        assert "ndcg@10" in compared["models"][1]["metrics"]

    pointer_before_failure = settings.model_pointer.read_bytes()
    with pytest.raises(ModelTrainingError, match="rank must be positive"):
        run_online_retraining(
            settings=settings,
            start_time=window_start,
            end_time=window_end,
            base_processed_dir=processed,
            output_root=tmp_path / "retraining",
            artifacts_dir=artifacts,
            mode="smoke",
            max_users=10,
            max_eval_users=10,
            rank=0,
            seed=19,
        )
    assert settings.model_pointer.read_bytes() == pointer_before_failure
    with database.connect() as conn:
        failed = conn.execute(
            "SELECT * FROM training_runs WHERE training_status = 'failed' ORDER BY created_at DESC"
        ).fetchone()
        assert failed is not None
        assert failed["publish_status"] == "failed"
