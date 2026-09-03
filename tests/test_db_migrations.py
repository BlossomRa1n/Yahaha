from __future__ import annotations

import sqlite3
from pathlib import Path

from app.db import Database


def test_existing_operations_table_migrates_batch_contract(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE operations (
                operation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                admin_user_id TEXT NOT NULL,
                action TEXT NOT NULL,
                item_id TEXT,
                target_id TEXT,
                reason TEXT NOT NULL,
                before_json TEXT NOT NULL,
                after_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )

    database = Database(path)
    database.initialize()

    with database.connect() as conn:
        columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(operations)")
        }
        assert "batch_id" in columns
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'operation_batches'"
        ).fetchone()
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = 'idx_operations_batch'"
        ).fetchone()
        candidate_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(candidate_manifests)")
        }
        assert {
            "source_scores_json",
            "source_calibrated_scores_json",
            "source_mask_json",
            "source_evidence_json",
            "ranker_score",
            "feature_schema_version",
        } <= candidate_columns
