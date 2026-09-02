from __future__ import annotations

import argparse
import csv
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import ArtifactStore
from .config import Settings
from .db import Database
from .security import hash_password, isoformat


def _first(row: dict[str, str], names: tuple[str, ...], default: str = "") -> str:
    for name in names:
        value = row.get(name)
        if value is not None and value.strip() != "":
            return value.strip()
    return default


def _nonnegative_int(value: str) -> int:
    try:
        return max(0, int(float(value or 0)))
    except ValueError:
        return 0


def load_items(conn: Any, path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"Processed item metadata not found: {path}")
    now = isoformat()
    count = 0
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("items.csv has no header")
        for row in reader:
            item_id = _first(row, ("item_id", "video_id", "videoID", "item"))
            if not item_id or item_id in seen:
                continue
            seen.add(item_id)
            title = _first(row, ("title", "item_title", "video_title"), f"MicroLens item {item_id}")
            likes = _nonnegative_int(_first(row, ("likes", "like_count", "num_likes")))
            views = _nonnegative_int(_first(row, ("views", "view_count", "num_views")))
            raw_popularity = _first(row, ("popularity_score", "popularity", "hot_score"))
            try:
                popularity = float(raw_popularity) if raw_popularity else math.log1p(views) + 2 * math.log1p(likes)
            except ValueError:
                popularity = math.log1p(views) + 2 * math.log1p(likes)
            if not math.isfinite(popularity):
                popularity = 0.0
            cover_url = _first(row, ("cover_url", "cover_path")) or None
            conn.execute(
                """
                INSERT INTO items(
                    item_id, title, likes, views, popularity_score, cover_url,
                    status, status_version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'online', 1, ?, ?)
                ON CONFLICT(item_id) DO UPDATE SET
                    title = excluded.title,
                    likes = excluded.likes,
                    views = excluded.views,
                    popularity_score = excluded.popularity_score,
                    cover_url = excluded.cover_url,
                    updated_at = excluded.updated_at
                """,
                (item_id, title, likes, views, popularity, cover_url, now, now),
            )
            count += 1
    if count == 0:
        raise ValueError("items.csv did not contain any usable item rows")
    return count


def seed_accounts(conn: Any, dataset_user_ids: list[str]) -> None:
    now = isoformat()
    accounts = [
        ("alice", "demo-pass", "user", dataset_user_ids[0] if len(dataset_user_ids) > 0 else None),
        ("bob", "demo-pass", "user", dataset_user_ids[1] if len(dataset_user_ids) > 1 else None),
        ("carol", "demo-pass", "user", None),
        ("admin", "admin-pass", "admin", None),
    ]
    conn.execute(
        "UPDATE users SET dataset_user_id = NULL WHERE username IN ('alice', 'bob')"
    )
    for username, password, role, dataset_user_id in accounts:
        existing = conn.execute(
            "SELECT id FROM users WHERE username = ? COLLATE NOCASE",
            (username,),
        ).fetchone()
        user_id = existing["id"] if existing else str(uuid.uuid4())
        conn.execute(
            """
            INSERT INTO users(
                id, username, dataset_user_id, password_hash, role, is_active, created_at
            ) VALUES (?, ?, ?, ?, ?, 1, ?)
            ON CONFLICT(username) DO UPDATE SET
                dataset_user_id = excluded.dataset_user_id,
                password_hash = excluded.password_hash,
                role = excluded.role,
                is_active = 1
            """,
            (user_id, username, dataset_user_id, hash_password(password), role, now),
        )
        conn.execute(
            """
            INSERT INTO profiles(user_id, version, updated_at) VALUES (?, 0, ?)
            ON CONFLICT(user_id) DO NOTHING
            """,
            (user_id, now),
        )


def init_database(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    database = Database(settings.database_path)
    if args.reset:
        database.reset()
    else:
        database.initialize()
    artifact_store = ArtifactStore(settings.model_pointer)
    artifact = artifact_store.get()
    dataset_user_ids = [str(value) for value in artifact.user_ids[:2]] if artifact else []
    with database.transaction(immediate=True) as conn:
        item_count = load_items(conn, Path(args.items).resolve())
        seed_accounts(conn, dataset_user_ids)
    result = {
        "database": str(settings.database_path),
        "items": item_count,
        "accounts": ["alice", "bob", "carol", "admin"],
        "mapped_dataset_users": dataset_user_ids,
        "model_version": artifact.model_version if artifact else None,
        "model_warning": artifact_store.last_error,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


# This is staging metadata, not a claim that the current binary SVD consumes the weight.
EXPORT_EVENT_WEIGHTS = {"click": 1, "like": 3}


def _iso_to_epoch_ms(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(round(parsed.timestamp() * 1000))


def export_events(conn: Any, out_path: Path) -> dict[str, Any]:
    """Export real users' click/like events as a future-training staging snapshot.

    Each source event remains one row. ``weight`` preserves the intended signal strength,
    but the current train-only benchmark intentionally does not consume this file. A future
    retraining job must establish a new chronological cutoff before merging it.
    """
    rows = conn.execute(
        """
        SELECT e.event_type, u.dataset_user_id, e.item_id, e.received_at
        FROM events e
        JOIN users u ON u.id = e.user_id
        WHERE u.dataset_user_id IS NOT NULL
          AND e.event_type IN ('click', 'like')
        ORDER BY e.received_at, e.event_id
        """
    ).fetchall()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    events_seen = 0
    rows_written = 0
    skipped = 0
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["user", "item", "timestamp", "event_type", "weight"])
        for row in rows:
            try:
                user_id = int(row["dataset_user_id"])
                item_id = int(row["item_id"])
                timestamp_ms = _iso_to_epoch_ms(row["received_at"])
            except (TypeError, ValueError):
                skipped += 1
                continue
            writer.writerow(
                [
                    user_id,
                    item_id,
                    timestamp_ms,
                    row["event_type"],
                    EXPORT_EVENT_WEIGHTS[row["event_type"]],
                ]
            )
            rows_written += 1
            events_seen += 1
    return {
        "output": str(out_path),
        "events": events_seen,
        "rows": rows_written,
        "skipped": skipped,
        "weights": EXPORT_EVENT_WEIGHTS,
        "consumed_by_training": False,
        "usage": "staging_only_requires_new_chronological_split",
    }


def export_events_command(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    database = Database(settings.database_path)
    out_path = Path(args.out).resolve()
    with database.connect() as conn:
        result = export_events(conn, out_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MicroLens service administration")
    subparsers = parser.add_subparsers(dest="command", required=True)
    init_parser = subparsers.add_parser("init-db", help="Initialize items and demo accounts")
    init_parser.add_argument(
        "--items",
        default="data/processed/items.csv",
        help="Processed item metadata CSV",
    )
    init_parser.add_argument(
        "--reset",
        action="store_true",
        help="Explicitly delete and recreate the configured SQLite database",
    )
    init_parser.set_defaults(handler=init_database)
    export_parser = subparsers.add_parser(
        "export-events",
        help="Export real users' click/like events as a future-training staging snapshot",
    )
    export_parser.add_argument(
        "--out",
        default="data/staging/online_events.csv",
        help="Output CSV path (default: data/staging/online_events.csv)",
    )
    export_parser.set_defaults(handler=export_events_command)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
