from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from app.artifacts import ArtifactStore
from app.cli import load_items
from app.db import Database
from app.recommendation import RecommendationService
from recsys.data import DataValidationError, prepare_data, sha256_file


def _write_raw(raw_dir: Path, *, missing_title: bool = False) -> None:
    raw_dir.mkdir(parents=True)
    interactions: list[tuple[int, int, int]] = []
    for timestamp in range(1, 31):
        user_id = ((timestamp - 1) % 5) + 1
        interactions.append((user_id, timestamp, timestamp // 3 + 1))
    with (raw_dir / "MicroLens-50k_pairs.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["user", "item", "timestamp"])
        writer.writerows(interactions)
    with (raw_dir / "MicroLens-50k_titles.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["item", "title"])
        for item_id in range(1, 31 - int(missing_title)):
            writer.writerow([item_id, f'Title, "{item_id}"'])
    with (raw_dir / "MicroLens-50k_likes_and_views.txt").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        for item_id in range(1, 31):
            handle.write(f"{item_id}\t{item_id}\t{item_id * 10}\n")


def _read_timestamps(path: Path) -> list[int]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [int(row["timestamp_ms"]) for row in csv.DictReader(handle)]


def test_prepare_is_deterministic_and_has_strict_time_boundaries(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_raw(raw_dir)
    first = tmp_path / "first"
    second = tmp_path / "second"

    summary = prepare_data(raw_dir, first, seed=7)
    repeated = prepare_data(raw_dir, second, seed=7)

    assert summary == repeated
    assert summary["counts"]["interactions"] == 30
    train = _read_timestamps(first / "train.csv")
    validation = _read_timestamps(first / "validation.csv")
    test = _read_timestamps(first / "test.csv")
    assert max(train) < min(validation)
    assert max(validation) < min(test)
    for name in (
        "train.csv",
        "validation.csv",
        "test.csv",
        "items.csv",
        "user_history.jsonl",
        "stats_snapshot.json",
        "summary.json",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()

    histories = [json.loads(line) for line in (first / "user_history.jsonl").read_text().splitlines()]
    assert len(histories) == 5
    assert all(max(history["timestamps_ms"]) <= max(train) for history in histories)
    snapshot = json.loads((first / "stats_snapshot.json").read_text(encoding="utf-8"))
    assert snapshot["available_at"] is None
    assert snapshot["historical_use_policy"] == "disabled_for_historical_training_and_evaluation"
    assert snapshot["source_file_hash"] == sha256_file(
        raw_dir / "MicroLens-50k_likes_and_views.txt"
    )
    assert snapshot["source_file_name"] == "MicroLens-50k_likes_and_views.txt"
    assert "source_file_mtime" in snapshot
    assert "ingest_time" not in snapshot
    assert snapshot["row_count"] == 30
    assert summary["likes_views_snapshot"] == snapshot

    available = prepare_data(
        raw_dir,
        tmp_path / "available",
        seed=7,
        stats_available_at="2026-09-02T12:00:00+08:00",
    )
    assert available["likes_views_snapshot"]["available_at"] == "2026-09-02T04:00:00Z"
    with (tmp_path / "available" / "items.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        row = next(csv.DictReader(handle))
    available_snapshot = available["likes_views_snapshot"]
    assert available_snapshot["snapshot_version"] != snapshot["snapshot_version"]
    assert row["stats_snapshot_version"] == available_snapshot["snapshot_version"]
    assert row["stats_available_at"] == "2026-09-02T04:00:00Z"

    database = Database(tmp_path / "stats.db")
    database.initialize()
    with database.transaction(immediate=True) as conn:
        assert load_items(conn, tmp_path / "available" / "items.csv") == 30
        stored = conn.execute(
            "SELECT * FROM item_stats_snapshots WHERE snapshot_version = ?",
            (available_snapshot["snapshot_version"],),
        ).fetchone()
        assert stored["source_file_hash"] == available_snapshot["source_file_hash"]
        assert stored["available_at"] == "2026-09-02T04:00:00Z"
        assert stored["ingest_time"] != available_snapshot["source_file_mtime"]
        recommender = RecommendationService(ArtifactStore(tmp_path / "missing.json"))
        before_available = recommender._popular(
            conn,
            artifact=None,
            pool_size=3,
            now="2026-09-02T03:59:59Z",
        )
        after_available = recommender._popular(
            conn,
            artifact=None,
            pool_size=3,
            now="2026-09-02T04:00:00Z",
        )
    assert all("excluded: unavailable at cutoff" in row.explanation for row in before_available)
    assert all("current likes/views snapshot" in row.explanation for row in after_available)

    with database.transaction(immediate=True) as conn:
        conn.execute(
            "DELETE FROM item_stats_snapshots WHERE snapshot_version = ?",
            (available_snapshot["snapshot_version"],),
        )
        without_db_provenance = recommender._popular(
            conn,
            artifact=None,
            pool_size=3,
            now="2026-09-02T04:00:01Z",
        )
    assert all(
        "excluded: provenance unavailable" in row.explanation
        for row in without_db_provenance
    )

    orphan_dir = tmp_path / "orphan"
    orphan_dir.mkdir()
    orphan_items = orphan_dir / "items.csv"
    orphan_items.write_bytes((tmp_path / "available" / "items.csv").read_bytes())
    orphan_database = Database(tmp_path / "orphan.db")
    orphan_database.initialize()
    with orphan_database.transaction(immediate=True) as conn:
        assert load_items(conn, orphan_items) == 30
        item = conn.execute("SELECT * FROM items ORDER BY item_id LIMIT 1").fetchone()
        assert item["stats_snapshot_version"] is None
        assert item["stats_available_at"] is None


def test_prepare_rejects_item_metadata_mismatch(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_raw(raw_dir, missing_title=True)
    with pytest.raises(DataValidationError, match="item metadata does not match"):
        prepare_data(raw_dir, tmp_path / "processed")


def test_load_items_rejects_snapshot_identity_mismatch(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_raw(raw_dir)
    processed = tmp_path / "processed"
    prepare_data(
        raw_dir,
        processed,
        seed=7,
        stats_available_at="2026-09-02T04:00:00Z",
    )
    snapshot_path = processed / "stats_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["available_at"] = "2026-09-03T04:00:00Z"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")

    database = Database(tmp_path / "tampered.db")
    database.initialize()
    with database.transaction(immediate=True) as conn:
        with pytest.raises(ValueError, match="snapshot version does not match provenance"):
            load_items(conn, processed / "items.csv")


def test_online_popularity_excludes_legacy_impressions(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_raw(raw_dir)
    processed = tmp_path / "processed"
    prepare_data(raw_dir, processed, seed=7)
    database = Database(tmp_path / "legacy.db")
    database.initialize()
    recommender = RecommendationService(ArtifactStore(tmp_path / "missing.json"))

    with database.transaction(immediate=True) as conn:
        load_items(conn, processed / "items.csv")
        conn.execute(
            """
            UPDATE app_metadata SET value = ?, updated_at = ?
            WHERE key = 'viewable_impression_semantics_started_at'
            """,
            ("2026-01-02T00:00:00Z", "2026-01-02T00:00:00Z"),
        )
        conn.execute(
            """
            INSERT INTO users(id, username, password_hash, role, is_active, created_at)
            VALUES ('u1', 'user', 'x', 'user', 1, '2026-01-01T00:00:00Z')
            """
        )
        for request_id, item_id, received_at in (
            ("legacy-request", "1", "2026-01-01T23:59:59Z"),
            ("current-request", "2", "2026-01-02T00:00:00Z"),
        ):
            conn.execute(
                """
                INSERT INTO recommendation_requests(
                    request_id, user_id, feed_type, profile_version, returned_count,
                    latency_ms, created_at
                ) VALUES (?, 'u1', 'popular', 0, 1, 0, ?)
                """,
                (request_id, received_at),
            )
            conn.execute(
                """
                INSERT INTO exposures(
                    request_id, user_id, item_id, position, source, score,
                    explanation, created_at
                ) VALUES (?, 'u1', ?, 0, 'popular', 1, 'test', ?)
                """,
                (request_id, item_id, received_at),
            )
            conn.execute(
                """
                INSERT INTO events(
                    event_id, event_type, request_id, user_id, item_id, position,
                    received_at
                ) VALUES (?, 'impression', ?, 'u1', ?, 0, ?)
                """,
                (f"event-{item_id}", request_id, item_id, received_at),
            )
        candidates = recommender._popular(
            conn,
            artifact=None,
            pool_size=30,
            now="2026-01-03T00:00:00Z",
        )

    scores = {candidate.item_id: candidate.score for candidate in candidates}
    assert scores["1"] == 0
    assert scores["2"] > 0
