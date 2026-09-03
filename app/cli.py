from __future__ import annotations

import argparse
import csv
import json
import math
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import ArtifactStore
from .config import Settings
from .db import Database
from .security import hash_password, isoformat
from recsys.data import canonical_optional_timestamp, stats_snapshot_version
from recsys.model import train_model
from recsys.online_retrain import build_online_retraining_dataset


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
    snapshot_path = path.with_name("stats_snapshot.json")
    snapshot: dict[str, Any] | None = None
    if snapshot_path.is_file():
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        required = {
            "snapshot_version",
            "available_at",
            "source_file_hash",
            "source_file_name",
            "source_file_mtime",
            "row_count",
        }
        if not isinstance(snapshot, dict) or not required <= set(snapshot):
            raise ValueError("stats_snapshot.json is missing required provenance fields")
        canonical_available_at = canonical_optional_timestamp(snapshot["available_at"])
        if snapshot["available_at"] != canonical_available_at:
            raise ValueError("stats_snapshot.json available_at must be canonical UTC")
        expected_version = stats_snapshot_version(
            str(snapshot["source_file_hash"]), canonical_available_at
        )
        if snapshot["snapshot_version"] != expected_version:
            raise ValueError("stats_snapshot.json snapshot version does not match provenance")
        quality = dict(snapshot.get("quality") or {})
        quality_json = json.dumps(quality, sort_keys=True)
        existing = conn.execute(
            "SELECT * FROM item_stats_snapshots WHERE snapshot_version = ?",
            (str(snapshot["snapshot_version"]),),
        ).fetchone()
        if existing is not None and (
            existing["available_at"],
            existing["source_file_hash"],
            existing["source_file_name"],
            int(existing["row_count"]),
            existing["quality_json"],
        ) != (
            snapshot.get("available_at"),
            str(snapshot["source_file_hash"]),
            str(snapshot["source_file_name"]),
            int(snapshot["row_count"]),
            quality_json,
        ):
            raise ValueError("snapshot version already exists with different provenance")
        conn.execute(
            """
            INSERT INTO item_stats_snapshots(
                snapshot_version, available_at, source_file_hash, source_file_name,
                ingest_time, row_count, quality_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(snapshot_version) DO NOTHING
            """,
            (
                str(snapshot["snapshot_version"]),
                snapshot.get("available_at"),
                str(snapshot["source_file_hash"]),
                str(snapshot["source_file_name"]),
                now,
                int(snapshot["row_count"]),
                quality_json,
                now,
            ),
        )
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
            snapshot_version = _first(row, ("stats_snapshot_version",)) or None
            stats_available_at = _first(row, ("stats_available_at",)) or None
            if snapshot is not None:
                if snapshot_version != str(snapshot["snapshot_version"]):
                    raise ValueError("items.csv snapshot version does not match stats_snapshot.json")
                if stats_available_at != snapshot.get("available_at"):
                    raise ValueError("items.csv snapshot available_at does not match stats_snapshot.json")
            else:
                # Legacy item files remain loadable, but unverified cumulative stats cannot rank.
                snapshot_version = None
                stats_available_at = None
            conn.execute(
                """
                INSERT INTO items(
                    item_id, title, likes, views, popularity_score, cover_url,
                    stats_snapshot_version, stats_available_at,
                    status, status_version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'online', 1, ?, ?)
                ON CONFLICT(item_id) DO UPDATE SET
                    title = excluded.title,
                    likes = excluded.likes,
                    views = excluded.views,
                    popularity_score = excluded.popularity_score,
                    cover_url = excluded.cover_url,
                    stats_snapshot_version = excluded.stats_snapshot_version,
                    stats_available_at = excluded.stats_available_at,
                    updated_at = excluded.updated_at
                """,
                (
                    item_id,
                    title,
                    likes,
                    views,
                    popularity,
                    cover_url,
                    snapshot_version,
                    stats_available_at,
                    now,
                    now,
                ),
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
        ("analyst", "analyst-pass", "analyst", None),
        ("operator", "operator-pass", "operator", None),
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


def cleanup_snapshots(conn: Any, now: str) -> dict[str, int]:
    active = int(
        conn.execute(
            "SELECT COUNT(*) FROM feed_snapshots WHERE expires_at > ?",
            (now,),
        ).fetchone()[0]
    )
    expired = int(
        conn.execute(
            "SELECT COUNT(*) FROM feed_snapshots WHERE expires_at <= ?",
            (now,),
        ).fetchone()[0]
    )
    conn.execute("DELETE FROM feed_snapshots WHERE expires_at <= ?", (now,))
    return {
        "deleted_expired_snapshots": expired,
        "retained_active_snapshots": active,
    }


def cleanup_snapshots_command(_: argparse.Namespace) -> int:
    settings = Settings.from_env()
    database = Database(settings.database_path)
    database.initialize()
    with database.transaction(immediate=True) as conn:
        result = cleanup_snapshots(conn, isoformat())
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _register_artifact(
    conn: Any,
    artifact: Any,
    *,
    status: str,
    window: dict[str, str] | None = None,
    event_count: int | None = None,
    sample_count: int | None = None,
) -> None:
    now = isoformat()
    protocol = dict(artifact.metrics.get("evaluation_protocol") or {})
    conn.execute(
        """
        INSERT INTO model_versions(
            model_version, data_version, algorithm, artifact_path, metrics_json,
            training_window_start, training_window_end, sample_count, event_count,
            training_status, publish_status, evaluation_protocol_json,
            status, created_at, activated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'succeeded', 'published', ?, ?, ?, ?)
        ON CONFLICT(model_version) DO UPDATE SET
            data_version = excluded.data_version,
            algorithm = excluded.algorithm,
            artifact_path = excluded.artifact_path,
            metrics_json = excluded.metrics_json,
            training_window_start = COALESCE(excluded.training_window_start, model_versions.training_window_start),
            training_window_end = COALESCE(excluded.training_window_end, model_versions.training_window_end),
            sample_count = COALESCE(excluded.sample_count, model_versions.sample_count),
            event_count = COALESCE(excluded.event_count, model_versions.event_count),
            training_status = 'succeeded',
            publish_status = 'published',
            evaluation_protocol_json = excluded.evaluation_protocol_json,
            status = excluded.status,
            activated_at = CASE WHEN excluded.status = 'active' THEN excluded.activated_at
                                ELSE model_versions.activated_at END
        """,
        (
            artifact.model_version,
            artifact.data_version,
            artifact.algorithm,
            str(artifact.manifest_path),
            json.dumps(artifact.metrics, sort_keys=True),
            window.get("start") if window else None,
            window.get("end") if window else None,
            sample_count,
            event_count,
            json.dumps(protocol, sort_keys=True),
            status,
            now,
            now if status == "active" else None,
        ),
    )


def _restore_pointer(path: Path, previous: bytes | None) -> None:
    if previous is None:
        if path.exists():
            path.unlink()
        return
    temporary = path.with_name(f".{path.name}.rollback-{uuid.uuid4().hex}")
    temporary.write_bytes(previous)
    os.replace(temporary, path)


def run_online_retraining(
    *,
    settings: Settings,
    start_time: str,
    end_time: str,
    base_processed_dir: Path,
    output_root: Path,
    artifacts_dir: Path,
    mode: str,
    max_users: int | None,
    max_eval_users: int | None,
    rank: int,
    seed: int,
) -> dict[str, Any]:
    database = Database(settings.database_path)
    database.initialize()
    artifacts_dir = Path(artifacts_dir).resolve()
    pointer_path = artifacts_dir / "current.json"
    if pointer_path != settings.model_pointer.resolve():
        raise ValueError("artifacts_dir/current.json must match configured MODEL_POINTER")
    run_id = str(uuid.uuid4())
    created_at = isoformat()
    config = {
        "mode": mode,
        "max_users": max_users,
        "max_eval_users": max_eval_users,
        "rank": rank,
        "seed": seed,
        "base_processed_dir": str(Path(base_processed_dir).resolve()),
    }
    previous_pointer = pointer_path.read_bytes() if pointer_path.is_file() else None
    prior_artifact = ArtifactStore(pointer_path).get()
    with database.transaction(immediate=True) as conn:
        if prior_artifact is not None:
            conn.execute("UPDATE model_versions SET status = 'inactive' WHERE status = 'active'")
            _register_artifact(conn, prior_artifact, status="active")
        conn.execute(
            """
            INSERT INTO training_runs(
                run_id, window_start, window_end, config_json,
                training_status, publish_status, created_at
            ) VALUES (?, ?, ?, ?, 'running', 'not_published', ?)
            """,
            (run_id, start_time, end_time, json.dumps(config, sort_keys=True), created_at),
        )
    try:
        with database.connect() as conn:
            summary = build_online_retraining_dataset(
                conn,
                base_processed_dir=base_processed_dir,
                output_root=output_root,
                start_time=start_time,
                end_time=end_time,
                seed=seed,
            )
        data_version = str(summary["data_version"])
        processed_dir = Path(output_root).resolve() / data_version
        online = dict(summary["online_retraining"])
        sample_count = int(summary["counts"]["online_positive_samples"])
        event_count = int(online["event_count"])
        with database.transaction(immediate=True) as conn:
            conn.execute(
                """
                UPDATE training_runs
                SET data_version = ?, event_count = ?, sample_count = ?
                WHERE run_id = ?
                """,
                (data_version, event_count, sample_count, run_id),
            )
        trained = train_model(
            processed_dir,
            artifacts_dir,
            mode=mode,
            max_users=max_users,
            max_eval_users=max_eval_users,
            rank=rank,
            seed=seed,
        )
        loaded = ArtifactStore(pointer_path).get()
        if loaded is None or loaded.model_version != trained["model_version"]:
            raise RuntimeError("published model failed online artifact validation")
        completed_at = isoformat()
        with database.transaction(immediate=True) as conn:
            conn.execute(
                "UPDATE model_versions SET status = 'inactive' WHERE status = 'active'"
            )
            _register_artifact(
                conn,
                loaded,
                status="active",
                window=dict(online["window"]),
                event_count=event_count,
                sample_count=sample_count,
            )
            conn.execute(
                """
                UPDATE training_runs
                SET model_version = ?, metrics_json = ?, artifact_path = ?,
                    training_status = 'succeeded', publish_status = 'published',
                    completed_at = ?
                WHERE run_id = ?
                """,
                (
                    loaded.model_version,
                    json.dumps(loaded.metrics, sort_keys=True),
                    str(loaded.manifest_path.parent),
                    completed_at,
                    run_id,
                ),
            )
        return {
            "run_id": run_id,
            "data_version": data_version,
            "model_version": loaded.model_version,
            "processed_dir": str(processed_dir),
            "artifact_dir": str(loaded.manifest_path.parent),
            "event_count": event_count,
            "sample_count": sample_count,
            "window": online["window"],
            "metrics": loaded.metrics,
            "training_status": "succeeded",
            "publish_status": "published",
            "previous_model_version": prior_artifact.model_version if prior_artifact else None,
        }
    except Exception as exc:
        current_pointer = pointer_path.read_bytes() if pointer_path.is_file() else None
        if current_pointer != previous_pointer:
            _restore_pointer(pointer_path, previous_pointer)
        with database.transaction(immediate=True) as conn:
            conn.execute(
                """
                UPDATE training_runs
                SET training_status = 'failed', publish_status = 'failed', error = ?, completed_at = ?
                WHERE run_id = ?
                """,
                (str(exc)[:2000], isoformat(), run_id),
            )
        raise


def retrain_events_command(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    artifacts_dir = (
        Path(args.artifacts_dir).resolve()
        if args.artifacts_dir
        else settings.model_pointer.resolve().parent
    )
    result = run_online_retraining(
        settings=settings,
        start_time=args.start_time,
        end_time=args.end_time,
        base_processed_dir=Path(args.base_processed_dir),
        output_root=Path(args.output_root),
        artifacts_dir=artifacts_dir,
        mode=args.mode,
        max_users=args.max_users,
        max_eval_users=args.max_eval_users,
        rank=args.rank,
        seed=args.seed,
    )
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
    cleanup_parser = subparsers.add_parser(
        "cleanup-snapshots",
        help="Delete expired persistent Feed snapshots and their candidate rows",
    )
    cleanup_parser.set_defaults(handler=cleanup_snapshots_command)
    retrain_parser = subparsers.add_parser(
        "retrain-events",
        help="Build a chronological online-event window, evaluate and atomically publish a model",
    )
    retrain_parser.add_argument("--start-time", required=True, help="Inclusive ISO-8601 window start")
    retrain_parser.add_argument("--end-time", required=True, help="Exclusive ISO-8601 window end")
    retrain_parser.add_argument("--base-processed-dir", default="data/processed")
    retrain_parser.add_argument("--output-root", default="data/retraining")
    retrain_parser.add_argument("--artifacts-dir")
    retrain_parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    retrain_parser.add_argument("--max-users", type=int)
    retrain_parser.add_argument("--max-eval-users", type=int)
    retrain_parser.add_argument("--rank", type=int, default=32)
    retrain_parser.add_argument("--seed", type=int, default=20260901)
    retrain_parser.set_defaults(handler=retrain_events_command)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
