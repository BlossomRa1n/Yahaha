from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


PAIRS_FILE = "MicroLens-50k_pairs.csv"
TITLES_FILE = "MicroLens-50k_titles.csv"
STATS_FILE = "MicroLens-50k_likes_and_views.txt"
SPLIT_POLICY = {
    "name": "global_timestamp_quantiles",
    "train_ratio": 0.8,
    "validation_ratio": 0.1,
    "test_ratio": 0.1,
    "train_rule": "timestamp_ms <= train_cutoff_ms",
    "validation_rule": "train_cutoff_ms < timestamp_ms <= validation_cutoff_ms",
    "test_rule": "timestamp_ms > validation_cutoff_ms",
}


class DataValidationError(ValueError):
    pass


@dataclass(frozen=True, order=True)
class Interaction:
    timestamp_ms: int
    user_id: int
    item_id: int


@dataclass(frozen=True)
class ItemMetadata:
    item_id: int
    title: str
    likes: int
    views: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_int(value: str, *, field: str, line: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"line {line}: {field} must be an integer") from exc
    if parsed <= 0:
        raise DataValidationError(f"line {line}: {field} must be positive")
    return parsed


def read_interactions(path: Path) -> tuple[list[Interaction], dict[str, int]]:
    interactions: list[Interaction] = []
    exact_rows: set[tuple[int, int, int]] = set()
    user_items: set[tuple[int, int]] = set()
    duplicate_rows = 0
    duplicate_user_items = 0

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["user", "item", "timestamp"]:
            raise DataValidationError(
                f"{path.name}: expected header user,item,timestamp; got {reader.fieldnames}"
            )
        for line_number, row in enumerate(reader, start=2):
            user_id = _positive_int(row["user"], field="user", line=line_number)
            item_id = _positive_int(row["item"], field="item", line=line_number)
            timestamp_ms = _positive_int(
                row["timestamp"], field="timestamp", line=line_number
            )
            exact_key = (user_id, item_id, timestamp_ms)
            user_item_key = (user_id, item_id)
            if exact_key in exact_rows:
                duplicate_rows += 1
            if user_item_key in user_items:
                duplicate_user_items += 1
            exact_rows.add(exact_key)
            user_items.add(user_item_key)
            interactions.append(Interaction(timestamp_ms, user_id, item_id))

    if not interactions:
        raise DataValidationError(f"{path.name}: no interactions")
    interactions.sort()
    return interactions, {
        "duplicate_rows": duplicate_rows,
        "duplicate_user_items": duplicate_user_items,
    }


def read_titles(path: Path) -> dict[int, str]:
    titles: dict[int, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["item", "title"]:
            raise DataValidationError(
                f"{path.name}: expected header item,title; got {reader.fieldnames}"
            )
        for line_number, row in enumerate(reader, start=2):
            item_id = _positive_int(row["item"], field="item", line=line_number)
            title = row["title"].strip()
            if not title:
                raise DataValidationError(f"line {line_number}: title must not be blank")
            if item_id in titles:
                raise DataValidationError(f"line {line_number}: duplicate title for item {item_id}")
            titles[item_id] = title
    if not titles:
        raise DataValidationError(f"{path.name}: no titles")
    return titles


def read_likes_and_views(path: Path) -> dict[int, tuple[int, int]]:
    stats: dict[int, tuple[int, int]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) != 3:
                raise DataValidationError(
                    f"line {line_number}: expected tab-separated item,likes,views"
                )
            item_id = _positive_int(parts[0], field="item", line=line_number)
            try:
                likes, views = int(parts[1]), int(parts[2])
            except ValueError as exc:
                raise DataValidationError(
                    f"line {line_number}: likes and views must be integers"
                ) from exc
            if likes < 0 or views < 0:
                raise DataValidationError(
                    f"line {line_number}: likes and views must be non-negative"
                )
            if item_id in stats:
                raise DataValidationError(f"line {line_number}: duplicate stats for item {item_id}")
            stats[item_id] = (likes, views)
    if not stats:
        raise DataValidationError(f"{path.name}: no likes/views rows")
    return stats


def _validate_item_sets(
    interactions: Iterable[Interaction],
    titles: dict[int, str],
    stats: dict[int, tuple[int, int]],
) -> set[int]:
    interaction_items = {row.item_id for row in interactions}
    title_items = set(titles)
    stats_items = set(stats)
    if interaction_items != title_items or interaction_items != stats_items:
        def sample(values: set[int]) -> list[int]:
            return sorted(values)[:10]

        details = {
            "missing_titles": sample(interaction_items - title_items),
            "extra_titles": sample(title_items - interaction_items),
            "missing_stats": sample(interaction_items - stats_items),
            "extra_stats": sample(stats_items - interaction_items),
        }
        raise DataValidationError(f"item metadata does not match interactions: {details}")
    return interaction_items


def _cutoffs(interactions: list[Interaction]) -> tuple[int, int]:
    timestamps = [row.timestamp_ms for row in interactions]
    train_index = math.ceil(len(timestamps) * 0.8) - 1
    validation_index = math.ceil(len(timestamps) * 0.9) - 1
    train_cutoff = timestamps[train_index]
    validation_cutoff = timestamps[validation_index]
    if not timestamps[0] <= train_cutoff < validation_cutoff < timestamps[-1]:
        raise DataValidationError(
            "at least three distinct timestamp regions are required for the split"
        )
    return train_cutoff, validation_cutoff


def _split_interactions(
    interactions: list[Interaction], train_cutoff: int, validation_cutoff: int
) -> dict[str, list[Interaction]]:
    splits = {"train": [], "validation": [], "test": []}
    for row in interactions:
        if row.timestamp_ms <= train_cutoff:
            splits["train"].append(row)
        elif row.timestamp_ms <= validation_cutoff:
            splits["validation"].append(row)
        else:
            splits["test"].append(row)
    if any(not rows for rows in splits.values()):
        raise DataValidationError("global time split produced an empty partition")
    if not (
        max(row.timestamp_ms for row in splits["train"])
        < min(row.timestamp_ms for row in splits["validation"])
        <= max(row.timestamp_ms for row in splits["validation"])
        < min(row.timestamp_ms for row in splits["test"])
    ):
        raise DataValidationError("time split boundary assertion failed")
    return splits


def _write_interactions(path: Path, rows: Iterable[Interaction]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["user_id", "item_id", "timestamp_ms"])
        for row in rows:
            writer.writerow([row.user_id, row.item_id, row.timestamp_ms])


def _write_items(
    path: Path, titles: dict[int, str], stats: dict[int, tuple[int, int]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["item_id", "title", "likes", "views"])
        for item_id in sorted(titles):
            likes, views = stats[item_id]
            writer.writerow([item_id, titles[item_id], likes, views])


def _write_user_history(
    path: Path, all_users: Iterable[int], train_rows: Iterable[Interaction]
) -> None:
    histories: dict[int, list[Interaction]] = defaultdict(list)
    for row in train_rows:
        histories[row.user_id].append(row)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for user_id in sorted(all_users):
            rows = histories.get(user_id, [])
            payload = {
                "item_ids": [row.item_id for row in rows],
                "timestamps_ms": [row.timestamp_ms for row in rows],
                "user_id": user_id,
            }
            handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True))
            handle.write("\n")


def _iso_timestamp(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat()


def _distribution(values: Iterable[int]) -> dict[str, int]:
    ordered = sorted(values)
    if not ordered:
        return {"min": 0, "p50": 0, "p95": 0, "max": 0}
    return {
        "min": ordered[0],
        "p50": ordered[math.floor((len(ordered) - 1) * 0.50)],
        "p95": ordered[math.floor((len(ordered) - 1) * 0.95)],
        "max": ordered[-1],
    }


def _data_version(raw_files: dict[str, dict[str, object]], seed: int) -> str:
    material = {
        "raw_sha256": {name: details["sha256"] for name, details in raw_files.items()},
        "schema_version": 1,
        "seed": seed,
        "split_policy": SPLIT_POLICY,
    }
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    return f"microlens50k-{hashlib.sha256(encoded).hexdigest()[:16]}"


def prepare_data(raw_dir: Path, out_dir: Path, seed: int = 20260901) -> dict[str, object]:
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    paths = {
        PAIRS_FILE: raw_dir / PAIRS_FILE,
        TITLES_FILE: raw_dir / TITLES_FILE,
        STATS_FILE: raw_dir / STATS_FILE,
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise DataValidationError(f"missing official MicroLens files: {missing}")

    interactions, duplicate_counts = read_interactions(paths[PAIRS_FILE])
    titles = read_titles(paths[TITLES_FILE])
    stats = read_likes_and_views(paths[STATS_FILE])
    interaction_items = _validate_item_sets(interactions, titles, stats)
    all_users = {row.user_id for row in interactions}
    train_cutoff, validation_cutoff = _cutoffs(interactions)
    splits = _split_interactions(interactions, train_cutoff, validation_cutoff)

    raw_files = {
        name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for name, path in paths.items()
    }
    per_user_counts: dict[int, int] = defaultdict(int)
    for row in interactions:
        per_user_counts[row.user_id] += 1

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=out_dir.parent) as temp_name:
        staging = Path(temp_name)
        for split_name, rows in splits.items():
            _write_interactions(staging / f"{split_name}.csv", rows)
        _write_items(staging / "items.csv", titles, stats)
        _write_user_history(staging / "user_history.jsonl", all_users, splits["train"])

        generated_names = [
            "train.csv",
            "validation.csv",
            "test.csv",
            "items.csv",
            "user_history.jsonl",
        ]
        summary: dict[str, object] = {
            "schema_version": 1,
            "data_version": _data_version(raw_files, seed),
            "dataset": "MicroLens-50K",
            "seed": seed,
            "raw_files": raw_files,
            "source_schema": {
                PAIRS_FILE: ["user", "item", "timestamp"],
                TITLES_FILE: ["item", "title"],
                STATS_FILE: ["item", "likes", "views"],
                "likes_and_views_schema_status": "inferred_from_official_filename_and_values",
            },
            "split_policy": SPLIT_POLICY,
            "cutoffs": {
                "train_cutoff_ms": train_cutoff,
                "train_cutoff_utc": _iso_timestamp(train_cutoff),
                "validation_cutoff_ms": validation_cutoff,
                "validation_cutoff_utc": _iso_timestamp(validation_cutoff),
            },
            "counts": {
                "interactions": len(interactions),
                "users": len(all_users),
                "items": len(interaction_items),
                "titles": len(titles),
                "likes_and_views": len(stats),
                "train": len(splits["train"]),
                "validation": len(splits["validation"]),
                "test": len(splits["test"]),
            },
            "split_counts": {
                split_name: {
                    "interactions": len(rows),
                    "users": len({row.user_id for row in rows}),
                    "items": len({row.item_id for row in rows}),
                }
                for split_name, rows in splits.items()
            },
            "quality": {
                **duplicate_counts,
                "missing_titles": 0,
                "missing_stats": 0,
                "blank_titles": 0,
                "negative_likes_or_views": 0,
                "interactions_per_user": _distribution(per_user_counts.values()),
            },
            "leakage_checks": {
                "same_timestamp_never_crosses_split": True,
                "train_max_lt_validation_min": True,
                "validation_max_lt_test_min": True,
                "untimed_likes_views_allowed_as_offline_feature": False,
                "user_history_scope": "train_only",
            },
            "time_range": {
                "min_timestamp_ms": interactions[0].timestamp_ms,
                "min_utc": _iso_timestamp(interactions[0].timestamp_ms),
                "max_timestamp_ms": interactions[-1].timestamp_ms,
                "max_utc": _iso_timestamp(interactions[-1].timestamp_ms),
            },
            "generated_files": {
                name: {"bytes": (staging / name).stat().st_size, "sha256": sha256_file(staging / name)}
                for name in generated_names
            },
        }
        summary_path = staging / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for name in [*generated_names, "summary.json"]:
            os.replace(staging / name, out_dir / name)
    return summary

