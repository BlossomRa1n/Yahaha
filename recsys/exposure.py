from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExposureTrainingRow:
    snapshot_id: str
    dataset_user_id: int
    item_id: int
    label: float
    sample_weight: float
    position: int
    source_scores: tuple[float, ...]
    source_calibrated_scores: tuple[float, ...]
    source_mask: tuple[float, ...]
    primary_source: str
    feature_schema_version: str


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def build_exposure_training_rows(
    conn: sqlite3.Connection,
    *,
    observation_end: str,
    observation_hours: int = 24,
) -> tuple[list[ExposureTrainingRow], dict[str, Any]]:
    """Build mature labels without treating unviewed candidates as negatives."""
    if observation_hours < 1:
        raise ValueError("observation_hours must be positive")
    table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'candidate_manifests'"
    ).fetchone()
    if table is None:
        return [], {
            "status": "candidate_manifest_unavailable",
            "labeled_rows": 0,
            "positive_rows": 0,
            "negative_rows": 0,
        }
    mature_before = _parse_timestamp(observation_end) - timedelta(hours=observation_hours)
    rows = conn.execute(
        """
        SELECT
            cm.snapshot_id,
            u.dataset_user_id,
            cm.item_id,
            cm.primary_source,
            cm.source_scores_json,
            cm.source_calibrated_scores_json,
            cm.source_mask_json,
            cm.feature_schema_version,
            MIN(x.position) AS position,
            MAX(CASE WHEN ev.event_type = 'impression' THEN 1 ELSE 0 END) AS viewed,
            MAX(CASE WHEN ev.event_type = 'click' THEN 1 ELSE 0 END) AS clicked,
            MAX(CASE WHEN ev.event_type = 'like' THEN 1 ELSE 0 END) AS liked,
            MAX(CASE WHEN ev.event_type = 'share' THEN 1 ELSE 0 END) AS shared,
            MAX(CASE WHEN ev.event_type = 'revisit' THEN 1 ELSE 0 END) AS revisited,
            MAX(CASE WHEN ev.event_type = 'dwell' THEN ev.dwell_ms ELSE 0 END) AS dwell_ms
        FROM candidate_manifests cm
        JOIN feed_snapshots fs ON fs.snapshot_id = cm.snapshot_id
        JOIN users u ON u.id = fs.user_id
        LEFT JOIN exposures x
          ON x.snapshot_id = cm.snapshot_id AND x.item_id = cm.item_id
        LEFT JOIN events ev
          ON ev.request_id = x.request_id
         AND ev.user_id = x.user_id
         AND ev.item_id = x.item_id
         AND ev.position = x.position
        WHERE fs.feed_type = 'personalized'
          AND fs.created_at <= ?
          AND u.dataset_user_id IS NOT NULL
        GROUP BY
            cm.snapshot_id, u.dataset_user_id, cm.item_id, cm.primary_source,
            cm.source_scores_json, cm.source_calibrated_scores_json,
            cm.source_mask_json, cm.feature_schema_version
        """,
        (mature_before.isoformat().replace("+00:00", "Z"),),
    ).fetchall()
    samples: list[ExposureTrainingRow] = []
    unlabeled = 0
    for row in rows:
        if row["position"] is None or not bool(row["viewed"]):
            unlabeled += 1
            continue
        positive_strength = max(
            1.0 if row["liked"] or row["shared"] else 0.0,
            0.9 if row["revisited"] else 0.0,
            0.8 if row["clicked"] else 0.0,
            min(0.8, float(row["dwell_ms"] or 0) / 30_000.0),
        )
        label = positive_strength if positive_strength > 0 else 0.0
        position = int(row["position"])
        sample_weight = min(3.0, math.sqrt(position + 1.0))
        scores = tuple(float(value) for value in json.loads(row["source_scores_json"]))
        calibrated = tuple(
            float(value) for value in json.loads(row["source_calibrated_scores_json"])
        )
        mask = tuple(float(value) for value in json.loads(row["source_mask_json"]))
        samples.append(
            ExposureTrainingRow(
                snapshot_id=str(row["snapshot_id"]),
                dataset_user_id=int(row["dataset_user_id"]),
                item_id=int(row["item_id"]),
                label=label,
                sample_weight=sample_weight,
                position=position,
                source_scores=scores,
                source_calibrated_scores=calibrated,
                source_mask=mask,
                primary_source=str(row["primary_source"]),
                feature_schema_version=str(row["feature_schema_version"]),
            )
        )
    positive = sum(sample.label > 0 for sample in samples)
    return samples, {
        "status": "usable" if positive and positive < len(samples) else "insufficient_classes",
        "candidate_rows": len(rows),
        "labeled_rows": len(samples),
        "positive_rows": positive,
        "negative_rows": len(samples) - positive,
        "unlabeled_rows": unlabeled,
        "observation_hours": observation_hours,
        "mature_before": mature_before.isoformat().replace("+00:00", "Z"),
    }


def load_exposure_training_rows(
    database_path: Path,
    *,
    observation_end: str,
    observation_hours: int = 24,
) -> tuple[list[ExposureTrainingRow], dict[str, Any]]:
    if not database_path.is_file():
        return [], {"status": "database_unavailable", "labeled_rows": 0}
    conn = sqlite3.connect(database_path)
    conn.row_factory = sqlite3.Row
    try:
        return build_exposure_training_rows(
            conn,
            observation_end=observation_end,
            observation_hours=observation_hours,
        )
    finally:
        conn.close()
