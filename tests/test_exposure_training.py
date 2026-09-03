from __future__ import annotations

import json
import sqlite3

from recsys.exposure import build_exposure_training_rows
from recsys.two_stage import UNIFIED_FEATURE_SCHEMA_VERSION, UNIFIED_SOURCE_ORDER


def _connection() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE users(id TEXT PRIMARY KEY, dataset_user_id TEXT);
        CREATE TABLE feed_snapshots(
            snapshot_id TEXT PRIMARY KEY, user_id TEXT, feed_type TEXT, created_at TEXT
        );
        CREATE TABLE candidate_manifests(
            snapshot_id TEXT, item_id TEXT, primary_source TEXT,
            source_scores_json TEXT, source_calibrated_scores_json TEXT,
            source_mask_json TEXT, feature_schema_version TEXT
        );
        CREATE TABLE exposures(
            request_id TEXT, user_id TEXT, item_id TEXT, position INTEGER, snapshot_id TEXT
        );
        CREATE TABLE events(
            request_id TEXT, user_id TEXT, item_id TEXT, position INTEGER,
            event_type TEXT, dwell_ms INTEGER
        );
        """
    )
    return conn


def test_exposure_labels_require_viewability_and_a_mature_window() -> None:
    conn = _connection()
    zeros = json.dumps([0.0] * len(UNIFIED_SOURCE_ORDER))
    ones = json.dumps([1.0] * len(UNIFIED_SOURCE_ORDER))
    conn.execute("INSERT INTO users VALUES ('u1', '7')")
    conn.execute(
        "INSERT INTO feed_snapshots VALUES ('s1', 'u1', 'personalized', '2026-01-01T00:00:00Z')"
    )
    for item_id in ("10", "11", "12"):
        conn.execute(
            "INSERT INTO candidate_manifests VALUES (?, ?, 'dssm', ?, ?, ?, ?)",
            ("s1", item_id, ones, ones, zeros, UNIFIED_FEATURE_SCHEMA_VERSION),
        )
    for position, item_id in enumerate(("10", "11", "12")):
        conn.execute(
            "INSERT INTO exposures VALUES ('r1', 'u1', ?, ?, 's1')",
            (item_id, position),
        )
    conn.execute("INSERT INTO events VALUES ('r1', 'u1', '10', 0, 'impression', NULL)")
    conn.execute("INSERT INTO events VALUES ('r1', 'u1', '10', 0, 'click', NULL)")
    conn.execute("INSERT INTO events VALUES ('r1', 'u1', '11', 1, 'impression', NULL)")

    rows, audit = build_exposure_training_rows(
        conn, observation_end="2026-01-03T00:00:00Z"
    )

    assert [(row.item_id, row.label) for row in rows] == [(10, 0.8), (11, 0.0)]
    assert audit["status"] == "usable"
    assert audit["unlabeled_rows"] == 1

    immature, _ = build_exposure_training_rows(
        conn, observation_end="2026-01-01T12:00:00Z"
    )
    assert immature == []
