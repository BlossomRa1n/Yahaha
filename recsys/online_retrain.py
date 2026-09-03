from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import sqlite3
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .data import canonical_optional_timestamp, sha256_file
from .popularity import timestamp_to_ms


EVENT_FEEDBACK = {
    "impression": {"weight": 0.0, "disposition": "context_only"},
    "click": {"weight": 1.0, "disposition": "positive"},
    "like": {"weight": 3.0, "disposition": "positive"},
    "dwell": {
        "weight": "bucketed",
        "disposition": "positive",
        "buckets_ms": {"750-4999": 0.25, "5000-29999": 0.75, "30000-600000": 1.5},
    },
    "share": {"weight": 4.0, "disposition": "positive"},
    "revisit": {"weight": 1.5, "disposition": "positive"},
    "not_interested": {"weight": -2.0, "disposition": "negative"},
}
POSITIVE_USER_ITEM_WEIGHT_CAP = 6.0


class OnlineRetrainingError(ValueError):
    pass


@dataclass(frozen=True, order=True)
class OnlineEvent:
    timestamp_ms: int
    event_id: str
    user_id: int
    item_id: int
    event_type: str
    weight: float
    disposition: str
    dwell_ms: int | None
    visit_index: int | None
    client_timestamp: str | None
    received_at: str


def _canonical_required(value: str, name: str) -> str:
    try:
        result = canonical_optional_timestamp(value)
    except ValueError as exc:
        raise OnlineRetrainingError(f"{name} must be ISO-8601") from exc
    if result is None:
        raise OnlineRetrainingError(f"{name} is required")
    return result


def _read_rows(path: Path) -> list[tuple[int, int, int]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["user_id", "item_id", "timestamp_ms"]:
            raise OnlineRetrainingError(f"invalid processed split schema: {path}")
        return [
            (int(row["user_id"]), int(row["item_id"]), int(row["timestamp_ms"]))
            for row in reader
        ]


def _write_rows(path: Path, rows: list[tuple[int, int, int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["user_id", "item_id", "timestamp_ms"])
        writer.writerows(rows)


def _event_feedback(event_type: str, dwell_ms: int | None) -> tuple[float, str] | None:
    mapping = EVENT_FEEDBACK.get(event_type)
    if mapping is None:
        return None
    if event_type != "dwell":
        return float(mapping["weight"]), str(mapping["disposition"])
    if dwell_ms is None or dwell_ms < 750 or dwell_ms > 600_000:
        return None
    if dwell_ms < 5_000:
        return 0.25, "positive"
    if dwell_ms < 30_000:
        return 0.75, "positive"
    return 1.5, "positive"


def _apply_user_item_feedback_rules(events: list[OnlineEvent]) -> list[OnlineEvent]:
    negative_pairs = {
        (event.user_id, event.item_id)
        for event in events
        if event.disposition == "negative"
    }
    consumed: defaultdict[tuple[int, int], float] = defaultdict(float)
    result: list[OnlineEvent] = []
    for event in sorted(events):
        pair = (event.user_id, event.item_id)
        if event.disposition != "positive":
            result.append(event)
            continue
        if pair in negative_pairs:
            result.append(replace(event, weight=0.0, disposition="ignored_negative_conflict"))
            continue
        remaining = max(0.0, POSITIVE_USER_ITEM_WEIGHT_CAP - consumed[pair])
        applied = min(event.weight, remaining)
        consumed[pair] += applied
        result.append(
            replace(
                event,
                weight=applied,
                disposition="positive" if applied > 0 else "ignored_positive_cap",
            )
        )
    return result


def _item_ids(items_path: Path) -> set[int]:
    with items_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "item_id" not in reader.fieldnames:
            raise OnlineRetrainingError("processed items.csv has no item_id")
        return {int(row["item_id"]) for row in reader}


def _split_positive(events: list[OnlineEvent]) -> tuple[dict[str, list[OnlineEvent]], int, int]:
    positives = sorted(event for event in events if event.disposition == "positive")
    timestamps = sorted({event.timestamp_ms for event in positives})
    if len(timestamps) < 3:
        raise OnlineRetrainingError(
            "online window needs positive feedback in at least three distinct received_at groups"
        )
    train_index = min(math.ceil(len(timestamps) * 0.8) - 1, len(timestamps) - 3)
    validation_index = min(
        max(train_index + 1, math.ceil(len(timestamps) * 0.9) - 1),
        len(timestamps) - 2,
    )
    train_cutoff = timestamps[train_index]
    validation_cutoff = timestamps[validation_index]
    if not timestamps[0] <= train_cutoff < validation_cutoff < timestamps[-1]:
        raise OnlineRetrainingError("online feedback cannot form strict chronological splits")
    splits = {"train": [], "validation": [], "test": []}
    for event in positives:
        if event.timestamp_ms <= train_cutoff:
            splits["train"].append(event)
        elif event.timestamp_ms <= validation_cutoff:
            splits["validation"].append(event)
        else:
            splits["test"].append(event)
    if not all(splits.values()):
        raise OnlineRetrainingError("online train/validation/test feedback splits must be non-empty")
    return splits, train_cutoff, validation_cutoff


def _read_online_events(
    conn: sqlite3.Connection,
    *,
    start: str,
    end: str,
    known_users: set[int],
    known_items: set[int],
) -> tuple[list[OnlineEvent], dict[str, int]]:
    rows = conn.execute(
        """
        SELECT e.event_id, e.event_type, e.item_id, e.client_timestamp,
               e.dwell_ms, e.visit_index, e.received_at,
               u.dataset_user_id
        FROM events e
        JOIN users u ON u.id = e.user_id
        WHERE e.received_at >= ? AND e.received_at < ?
        ORDER BY e.received_at, e.event_id
        """,
        (start, end),
    ).fetchall()
    events: list[OnlineEvent] = []
    event_ids: set[str] = set()
    skipped = Counter()
    for row in rows:
        event_id = str(row["event_id"])
        if event_id in event_ids:
            skipped["duplicate_event_id"] += 1
            continue
        event_ids.add(event_id)
        feedback = _event_feedback(str(row["event_type"]), row["dwell_ms"])
        if feedback is None:
            skipped["unsupported_event_type"] += 1
            continue
        try:
            user_id = int(row["dataset_user_id"])
        except (TypeError, ValueError):
            skipped["unmapped_user"] += 1
            continue
        try:
            item_id = int(row["item_id"])
        except (TypeError, ValueError):
            skipped["invalid_item_id"] += 1
            continue
        if user_id not in known_users:
            skipped["unknown_dataset_user"] += 1
            continue
        if item_id not in known_items:
            skipped["missing_item_metadata"] += 1
            continue
        try:
            timestamp_ms = timestamp_to_ms(str(row["received_at"]))
        except ValueError:
            skipped["invalid_received_at"] += 1
            continue
        events.append(
            OnlineEvent(
                timestamp_ms=timestamp_ms,
                event_id=event_id,
                user_id=user_id,
                item_id=item_id,
                event_type=str(row["event_type"]),
                weight=feedback[0],
                disposition=feedback[1],
                dwell_ms=(int(row["dwell_ms"]) if row["dwell_ms"] is not None else None),
                visit_index=(
                    int(row["visit_index"]) if row["visit_index"] is not None else None
                ),
                client_timestamp=row["client_timestamp"],
                received_at=str(row["received_at"]),
            )
        )
    return _apply_user_item_feedback_rules(events), dict(skipped)


def _write_feedback(path: Path, events: list[OnlineEvent]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "event_id",
                "user_id",
                "item_id",
                "timestamp_ms",
                "event_type",
                "weight",
                "disposition",
                "received_at",
            ]
        )
        for event in events:
            writer.writerow(
                [
                    event.event_id,
                    event.user_id,
                    event.item_id,
                    event.timestamp_ms,
                    event.event_type,
                    event.weight,
                    event.disposition,
                    event.received_at,
                ]
            )


def _write_history(path: Path, rows: list[tuple[int, int, int]]) -> None:
    histories: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for user_id, item_id, timestamp_ms in sorted(rows, key=lambda row: (row[2], row[0], row[1])):
        histories[user_id].append((item_id, timestamp_ms))
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for user_id in sorted(histories):
            values = histories[user_id]
            handle.write(
                json.dumps(
                    {
                        "user_id": user_id,
                        "item_ids": [item_id for item_id, _ in values],
                        "timestamps_ms": [timestamp for _, timestamp in values],
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def build_online_retraining_dataset(
    conn: sqlite3.Connection,
    *,
    base_processed_dir: Path,
    output_root: Path,
    start_time: str,
    end_time: str,
    seed: int,
) -> dict[str, Any]:
    start = _canonical_required(start_time, "start_time")
    end = _canonical_required(end_time, "end_time")
    if timestamp_to_ms(start) >= timestamp_to_ms(end):
        raise OnlineRetrainingError("start_time must be earlier than end_time")
    base_processed_dir = Path(base_processed_dir).resolve()
    required = ["train.csv", "validation.csv", "test.csv", "items.csv", "summary.json"]
    missing = [name for name in required if not (base_processed_dir / name).is_file()]
    if missing:
        raise OnlineRetrainingError(f"base processed data is missing: {missing}")
    base_summary = json.loads((base_processed_dir / "summary.json").read_text(encoding="utf-8"))
    base_rows = []
    for split_name in ("train", "validation", "test"):
        base_rows.extend(_read_rows(base_processed_dir / f"{split_name}.csv"))
    base_rows.sort(key=lambda row: (row[2], row[0], row[1]))
    base_max = max(row[2] for row in base_rows)
    if timestamp_to_ms(start) <= base_max:
        raise OnlineRetrainingError("online window must start after the base dataset time range")
    known_users = {row[0] for row in base_rows}
    known_items = _item_ids(base_processed_dir / "items.csv")
    events, skipped = _read_online_events(
        conn,
        start=start,
        end=end,
        known_users=known_users,
        known_items=known_items,
    )
    if not events:
        raise OnlineRetrainingError("online window contains no valid mapped events")
    positive_splits, train_cutoff, validation_cutoff = _split_positive(events)
    event_digest = hashlib.sha256()
    for event in events:
        event_digest.update(
            f"{event.event_id}|{event.user_id}|{event.item_id}|{event.timestamp_ms}|"
            f"{event.event_type}|{event.weight}\n".encode()
        )
    version_payload = {
        "base_data_version": base_summary.get("data_version"),
        "end_time": end,
        "event_sha256": event_digest.hexdigest(),
        "feedback_mapping": EVENT_FEEDBACK,
        "seed": seed,
        "start_time": start,
    }
    version_hash = hashlib.sha256(
        json.dumps(version_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    data_version = f"microlens50k-online-{version_hash}"
    output_root = Path(output_root).resolve()
    destination = output_root / data_version
    if destination.is_dir():
        return json.loads((destination / "summary.json").read_text(encoding="utf-8"))
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".staging-{data_version}-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        shutil.copy2(base_processed_dir / "items.csv", staging / "items.csv")
        if (base_processed_dir / "stats_snapshot.json").is_file():
            shutil.copy2(base_processed_dir / "stats_snapshot.json", staging / "stats_snapshot.json")
        _write_rows(staging / "train.csv", base_rows)
        validation_rows = [
            (event.user_id, event.item_id, event.timestamp_ms)
            for event in positive_splits["validation"]
        ]
        test_rows = [
            (event.user_id, event.item_id, event.timestamp_ms) for event in positive_splits["test"]
        ]
        _write_rows(staging / "validation.csv", validation_rows)
        _write_rows(staging / "test.csv", test_rows)
        training_events = [event for event in events if event.timestamp_ms <= train_cutoff]
        _write_feedback(staging / "online_feedback_train.csv", training_events)
        _write_feedback(staging / "online_events.csv", events)
        positive_train_rows = [
            (event.user_id, event.item_id, event.timestamp_ms)
            for event in positive_splits["train"]
        ]
        _write_history(staging / "user_history.jsonl", [*base_rows, *positive_train_rows])
        generated_names = [
            "train.csv",
            "validation.csv",
            "test.csv",
            "items.csv",
            "user_history.jsonl",
            "online_feedback_train.csv",
            "online_events.csv",
        ]
        if (staging / "stats_snapshot.json").is_file():
            generated_names.append("stats_snapshot.json")
        event_counts = Counter(event.event_type for event in events)
        disposition_counts = Counter(event.disposition for event in events)
        summary = {
            **base_summary,
            "schema_version": 3,
            "data_version": data_version,
            "base_data_version": base_summary.get("data_version"),
            "generated_at": datetime.now(UTC).isoformat(),
            "seed": seed,
            "cutoffs": {
                "train_cutoff_ms": train_cutoff,
                "train_cutoff_utc": datetime.fromtimestamp(train_cutoff / 1000, UTC).isoformat(),
                "validation_cutoff_ms": validation_cutoff,
                "validation_cutoff_utc": datetime.fromtimestamp(
                    validation_cutoff / 1000, UTC
                ).isoformat(),
            },
            "counts": {
                **dict(base_summary.get("counts") or {}),
                "historical_interactions": len(base_rows),
                "online_events": len(events),
                "online_positive_samples": sum(len(values) for values in positive_splits.values()),
                "train": len(base_rows) + len(positive_splits["train"]),
                "validation": len(validation_rows),
                "test": len(test_rows),
            },
            "split_policy": {
                "name": "all_known_history_plus_chronological_online_window",
                "historical_rule": "all base train/validation/test interactions are known before start_time",
                "online_train_rule": "received_at <= train_cutoff",
                "online_validation_rule": "train_cutoff < received_at <= validation_cutoff",
                "online_test_rule": "received_at > validation_cutoff and received_at < end_time",
            },
            "online_retraining": {
                "window": {"start": start, "end": end, "end_exclusive": True},
                "event_count": len(events),
                "event_counts": dict(sorted(event_counts.items())),
                "disposition_counts": dict(sorted(disposition_counts.items())),
                "positive_split_counts": {
                    name: len(values) for name, values in positive_splits.items()
                },
                "skipped": skipped,
                "feedback_mapping": EVENT_FEEDBACK,
                "event_sha256": event_digest.hexdigest(),
                "timestamp_source": "server_received_at",
                "leakage_rule": "training feedback timestamp_ms <= train_cutoff_ms",
            },
        }
        summary["generated_files"] = {
            name: {
                "bytes": (staging / name).stat().st_size,
                "sha256": sha256_file(staging / name),
            }
            for name in generated_names
        }
        (staging / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, destination)
        return summary
    finally:
        if staging.exists():
            shutil.rmtree(staging)
