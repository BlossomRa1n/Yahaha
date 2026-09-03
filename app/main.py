from __future__ import annotations

import base64
import csv
import io
import json
import logging
import secrets
import sqlite3
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from recsys.two_stage import UNIFIED_FEATURE_SCHEMA_VERSION

from .alerts import ALERT_METRICS, evaluate_alerts
from .artifacts import ArtifactStore, ModelArtifact
from .cache import build_cache
from .cli import run_online_retraining
from .config import Settings
from .db import Database
from .deep_artifacts import DeepArtifactStore
from .logging import configure_logging
from .multimodal_artifacts import MultimodalArtifactStore
from .recommendation import RecommendationService
from .schemas import (
    AlertRuleBody,
    AlertRuleUpdateBody,
    BatchItemStatusBody,
    BoostBody,
    BoostStatusBody,
    EventBatch,
    ItemStatusBody,
    LoginBody,
    RegisterBody,
    TrainingJobBody,
)
from .security import (
    decode_cursor,
    encode_cursor,
    hash_password,
    isoformat,
    issue_session_token,
    session_digest,
    utc_now,
    verify_password,
)

access_logger = logging.getLogger("app.access")


class APIError(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: Any = None,
    ) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details
        super().__init__(message)


def _error_payload(request: Request, error: APIError) -> dict[str, Any]:
    return {
        "error": {
            "code": error.code,
            "message": error.message,
            "request_id": getattr(request.state, "api_request_id", None),
            "details": error.details,
        }
    }


def _user_payload(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "username": row["username"],
        "role": row["role"],
        "dataset_user_id": row["dataset_user_id"],
    }


def _item_payload(row: sqlite3.Row, *, public: bool = False) -> dict[str, Any]:
    result = {
        "item_id": str(row["item_id"]),
        "title": row["title"],
        "likes": int(row["likes"]),
        "views": int(row["views"]),
        "popularity_score": float(row["popularity_score"]),
        "status": row["status"],
        "status_version": int(row["status_version"]),
        "updated_at": row["updated_at"],
    }
    if public:
        result.pop("status")
        result.pop("status_version")
        result["cover_url"] = f"/api/v1/items/{row['item_id']}/cover"
    return result


def _alert_rule_payload(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "metric": row["metric"],
        "operator": row["operator"],
        "threshold": float(row["threshold"]),
        "severity": row["severity"],
        "enabled": bool(row["enabled"]),
        "created_by": row["created_by"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _alert_event_payload(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "rule_id": row["rule_id"],
        "metric": row["metric"],
        "operator": row["operator"],
        "threshold": float(row["threshold"]),
        "severity": row["severity"],
        "observed_value": (
            float(row["observed_value"]) if row["observed_value"] is not None else None
        ),
        "status": row["status"],
        "triggered_at": row["triggered_at"],
        "resolved_at": row["resolved_at"],
        "acknowledged_by": row["acknowledged_by"],
        "acknowledged_at": row["acknowledged_at"],
        "last_evaluated_at": row["last_evaluated_at"],
    }


def _parse_datetime(value: str | None, default: datetime) -> datetime:
    if value is None:
        return default
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise APIError(422, "invalid_time_range", "Timestamps must be ISO-8601") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _canonical_datetime(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return isoformat(value.astimezone(UTC))


def _sortable_datetime(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _resolve_dashboard_range(
    from_value: str | None,
    to_value: str | None,
    *,
    now: datetime | None = None,
) -> tuple[datetime, datetime, str, str]:
    end = _parse_datetime(to_value, now or utc_now())
    start = _parse_datetime(from_value, end - timedelta(hours=24))
    if start >= end:
        raise APIError(422, "invalid_time_range", "from must be earlier than to")
    if end - start > timedelta(days=366):
        raise APIError(422, "invalid_time_range", "time range must not exceed 366 days")
    return start, end, _sortable_datetime(start), _sortable_datetime(end)


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    ordered = sorted(values)
    n = len(ordered)

    def quantile(q: float) -> float:
        position = q * (n - 1)
        lower = int(position)
        upper = min(lower + 1, n - 1)
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    return {
        "min": float(ordered[0]),
        "p50": quantile(0.50),
        "p95": quantile(0.95),
        "p99": quantile(0.99),
        "max": float(ordered[-1]),
    }


def _dwell_affinity(dwell_ms: int | None) -> float:
    if dwell_ms is None or dwell_ms < 750:
        return 0.0
    if dwell_ms < 5_000:
        return 0.25
    if dwell_ms < 30_000:
        return 0.75
    return 1.5


def _iso_to_epoch_seconds(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.timestamp())


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    database = Database(settings.database_path)
    artifacts = ArtifactStore(settings.model_pointer)
    deep_artifacts = (
        DeepArtifactStore(settings.experiment_model_pointer)
        if settings.experiment_model_pointer is not None
        else None
    )
    multimodal_artifacts = (
        MultimodalArtifactStore(settings.multimodal_model_pointer)
        if settings.multimodal_model_pointer is not None
        else None
    )
    recommender = RecommendationService(artifacts, deep_artifacts, multimodal_artifacts)
    cache = build_cache(settings.redis_url)
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        database.initialize()
        artifacts.get()
        if deep_artifacts is not None:
            # Validate and warm the unified ranker before accepting traffic.
            # A failed model remains disabled and the stable store still serves.
            deep_artifacts.get()
        if multimodal_artifacts is not None:
            multimodal_artifacts.get()
        yield

    app = FastAPI(
        title="MicroLens Recommendation MVP",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.database = database
    app.state.artifacts = artifacts
    app.state.deep_artifacts = deep_artifacts
    app.state.multimodal_artifacts = multimodal_artifacts
    app.state.recommender = recommender
    app.state.cache = cache

    configure_logging()

    @app.middleware("http")
    async def request_context(request: Request, call_next: Any) -> Response:
        request.state.api_request_id = str(uuid.uuid4())
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.api_request_id
        access_logger.info(
            "request completed",
            extra={
                "request_id": request.state.api_request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
            },
        )
        return response

    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, error: APIError) -> JSONResponse:
        return JSONResponse(
            status_code=error.status_code,
            content=_error_payload(request, error),
            headers={"X-Request-ID": getattr(request.state, "api_request_id", "")},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,
        error: RequestValidationError,
    ) -> JSONResponse:
        wrapped = APIError(
            422,
            "validation_error",
            "Request validation failed",
            jsonable_encoder(error.errors()),
        )
        return JSONResponse(status_code=422, content=_error_payload(request, wrapped))

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(
        request: Request,
        error: StarletteHTTPException,
    ) -> JSONResponse:
        code = "not_found" if error.status_code == 404 else "method_not_allowed" if error.status_code == 405 else "http_error"
        wrapped = APIError(error.status_code, code, str(error.detail))
        return JSONResponse(status_code=error.status_code, content=_error_payload(request, wrapped))

    @app.exception_handler(sqlite3.Error)
    async def database_error_handler(request: Request, _: sqlite3.Error) -> JSONResponse:
        wrapped = APIError(503, "database_unavailable", "Database operation failed")
        return JSONResponse(status_code=503, content=_error_payload(request, wrapped))

    def current_user(request: Request) -> sqlite3.Row:
        token = request.cookies.get(settings.session_cookie)
        if not token:
            raise APIError(401, "unauthorized", "Authentication required")
        with database.connect() as conn:
            user = conn.execute(
                """
                SELECT u.*
                FROM auth_sessions s
                JOIN users u ON u.id = s.user_id
                WHERE s.token_hash = ?
                  AND s.revoked_at IS NULL
                  AND s.expires_at > ?
                  AND u.is_active = 1
                """,
                (session_digest(token), isoformat()),
            ).fetchone()
        if user is None:
            raise APIError(401, "unauthorized", "Session is invalid or expired")
        return user

    ROLE_LEVELS = {"user": 0, "analyst": 1, "operator": 2, "admin": 3}

    def require_role(request: Request, minimum: str) -> sqlite3.Row:
        user = current_user(request)
        if ROLE_LEVELS.get(user["role"], 0) < ROLE_LEVELS[minimum]:
            raise APIError(
                403, "forbidden", f"{minimum.title()} role or higher required"
            )
        return user

    def admin_user(request: Request) -> sqlite3.Row:
        return require_role(request, "admin")

    def ensure_model(conn: sqlite3.Connection, artifact: ModelArtifact | None) -> None:
        if artifact is None:
            return
        now = isoformat()
        conn.execute(
            "UPDATE model_versions SET status = 'inactive' WHERE status = 'active' AND model_version != ?",
            (artifact.model_version,),
        )
        conn.execute(
            """
            INSERT INTO model_versions(
                model_version, data_version, algorithm, artifact_path,
                metrics_json, status, created_at, activated_at
            ) VALUES (?, ?, ?, ?, ?, 'active', ?, ?)
            ON CONFLICT(model_version) DO UPDATE SET
                data_version = excluded.data_version,
                algorithm = excluded.algorithm,
                artifact_path = excluded.artifact_path,
                metrics_json = excluded.metrics_json,
                status = 'active',
                activated_at = COALESCE(model_versions.activated_at, excluded.activated_at)
            """,
            (
                artifact.model_version,
                artifact.data_version,
                artifact.algorithm,
                str(artifact.manifest_path),
                json.dumps(artifact.metrics, sort_keys=True),
                now,
                now,
            ),
        )

    def profile_payload(conn: sqlite3.Connection, user_id: str) -> dict[str, Any]:
        profile = conn.execute(
            "SELECT version, updated_at FROM profiles WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if profile is None:
            raise APIError(404, "user_not_found", "User profile does not exist")
        summary = conn.execute(
            """
            SELECT
                COALESCE(SUM(exposure_count), 0) AS impressions,
                COALESCE(SUM(click_count), 0) AS clicks,
                COALESCE(SUM(like_count), 0) AS likes,
                COALESCE(SUM(dwell_ms_total), 0) AS dwell_ms,
                COALESCE(SUM(dwell_event_count), 0) AS dwell_events,
                COALESCE(SUM(share_count), 0) AS shares,
                COALESCE(SUM(revisit_count), 0) AS revisits,
                COALESCE(SUM(not_interested), 0) AS not_interested
            FROM user_item_state WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()
        positives = conn.execute(
            """
            SELECT item_id, affinity AS weight
            FROM user_item_state
            WHERE user_id = ? AND affinity > 0 AND not_interested = 0
            ORDER BY affinity DESC, last_event_at DESC LIMIT 20
            """,
            (user_id,),
        ).fetchall()
        negatives = conn.execute(
            """
            SELECT item_id, affinity AS weight
            FROM user_item_state
            WHERE user_id = ? AND (affinity < 0 OR not_interested = 1)
            ORDER BY affinity ASC, last_event_at DESC LIMIT 20
            """,
            (user_id,),
        ).fetchall()
        return {
            "user_id": user_id,
            "version": int(profile["version"]),
            "updated_at": profile["updated_at"],
            "summary": {key: int(summary[key]) for key in summary.keys()},
            "positive_items": [
                {"item_id": str(row["item_id"]), "weight": float(row["weight"])}
                for row in positives
            ],
            "negative_items": [
                {"item_id": str(row["item_id"]), "weight": float(row["weight"])}
                for row in negatives
            ],
        }

    def recent_events(conn: sqlite3.Connection, user_id: str, limit: int) -> list[dict[str, Any]]:
        rows = conn.execute(
            """
            SELECT ev.event_id, ev.event_type, ev.request_id, ev.item_id,
                   ev.position, ev.client_timestamp, ev.dwell_ms,
                   ev.visit_index, ev.received_at,
                   r.feed_type, e.source
            FROM events ev
            JOIN exposures e ON e.request_id = ev.request_id
                            AND e.user_id = ev.user_id
                            AND e.item_id = ev.item_id
                            AND e.position = ev.position
            JOIN recommendation_requests r ON r.request_id = ev.request_id
            WHERE ev.user_id = ?
            ORDER BY ev.received_at DESC, ev.rowid DESC LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
        return [
            {
                "event_id": row["event_id"],
                "event_type": row["event_type"],
                "request_id": row["request_id"],
                "item_id": str(row["item_id"]),
                "position": int(row["position"]),
                "feed_type": row["feed_type"],
                "source": row["source"],
                "client_timestamp": row["client_timestamp"],
                "dwell_ms": row["dwell_ms"],
                "visit_index": row["visit_index"],
                "received_at": row["received_at"],
            }
            for row in rows
        ]

    @app.get("/api/v1/health")
    def health() -> dict[str, Any]:
        with database.connect() as conn:
            conn.execute("SELECT 1").fetchone()
        artifact = artifacts.get()
        return {
            "status": "ok",
            "database": "ok",
            "model_version": artifact.model_version if artifact else None,
            "model_error": artifacts.last_error,
        }

    @app.post("/api/v1/auth/login")
    def login(body: LoginBody) -> JSONResponse:
        with database.connect() as conn:
            user = conn.execute(
                "SELECT * FROM users WHERE username = ? COLLATE NOCASE AND is_active = 1",
                (body.username,),
            ).fetchone()
        if user is None or not verify_password(body.password, user["password_hash"]):
            raise APIError(401, "invalid_credentials", "Invalid username or password")
        token, digest = issue_session_token()
        now = utc_now()
        expires_at = now + timedelta(hours=settings.session_hours)
        with database.transaction(immediate=True) as conn:
            conn.execute(
                "DELETE FROM auth_sessions WHERE expires_at <= ? OR revoked_at IS NOT NULL",
                (isoformat(now),),
            )
            conn.execute(
                """
                INSERT INTO auth_sessions(id, token_hash, user_id, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (str(uuid.uuid4()), digest, user["id"], isoformat(now), isoformat(expires_at)),
            )
        response = JSONResponse({"user": _user_payload(user)})
        response.set_cookie(
            settings.session_cookie,
            token,
            max_age=settings.session_hours * 3600,
            httponly=True,
            secure=settings.secure_cookie,
            samesite="strict",
            path="/",
        )
        return response

    @app.post("/api/v1/auth/register", status_code=201)
    def register(body: RegisterBody) -> JSONResponse:
        token, digest = issue_session_token()
        now = utc_now()
        now_text = isoformat(now)
        expires_at = now + timedelta(hours=settings.session_hours)
        user_id = str(uuid.uuid4())
        password_hash = hash_password(body.password)
        try:
            with database.transaction(immediate=True) as conn:
                conn.execute(
                    """
                    INSERT INTO users(
                        id, username, dataset_user_id, password_hash,
                        role, is_active, created_at
                    ) VALUES (?, ?, NULL, ?, 'user', 1, ?)
                    """,
                    (user_id, body.username, password_hash, now_text),
                )
                conn.execute(
                    "INSERT INTO profiles(user_id, version, updated_at) VALUES (?, 0, ?)",
                    (user_id, now_text),
                )
                conn.execute(
                    "DELETE FROM auth_sessions WHERE expires_at <= ? OR revoked_at IS NOT NULL",
                    (now_text,),
                )
                conn.execute(
                    """
                    INSERT INTO auth_sessions(id, token_hash, user_id, created_at, expires_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (str(uuid.uuid4()), digest, user_id, now_text, isoformat(expires_at)),
                )
                user = conn.execute(
                    "SELECT * FROM users WHERE id = ?",
                    (user_id,),
                ).fetchone()
        except sqlite3.IntegrityError as exc:
            if "users.username" in str(exc).lower():
                raise APIError(
                    409,
                    "username_taken",
                    "Username is already registered",
                ) from exc
            raise
        if user is None:
            raise APIError(
                503,
                "registration_failed",
                "Registered user could not be loaded",
            )
        response = JSONResponse({"user": _user_payload(user)}, status_code=201)
        response.set_cookie(
            settings.session_cookie,
            token,
            max_age=settings.session_hours * 3600,
            httponly=True,
            secure=settings.secure_cookie,
            samesite="strict",
            path="/",
        )
        return response

    @app.get("/api/v1/auth/me")
    def auth_me(request: Request) -> dict[str, Any]:
        return {"user": _user_payload(current_user(request))}

    @app.post("/api/v1/auth/logout")
    def logout(request: Request) -> JSONResponse:
        token = request.cookies.get(settings.session_cookie)
        if token:
            with database.transaction(immediate=True) as conn:
                conn.execute(
                    "UPDATE auth_sessions SET revoked_at = ? WHERE token_hash = ? AND revoked_at IS NULL",
                    (isoformat(), session_digest(token)),
                )
        response = JSONResponse({"ok": True})
        response.delete_cookie(settings.session_cookie, path="/")
        return response

    @app.get("/api/v1/feeds/{feed_type}")
    def feed(
        feed_type: str,
        request: Request,
        limit: int = Query(default=12, ge=1, le=50),
        cursor: str | None = Query(default=None, max_length=2048),
    ) -> dict[str, Any]:
        if feed_type not in {"personalized", "popular", "explore"}:
            raise APIError(404, "feed_not_found", "Unknown feed type")
        user = current_user(request)
        snapshot_id: str | None = None
        offset = 0
        cursor_expires_at: str | None = None
        if cursor:
            try:
                cursor_payload = decode_cursor(cursor, settings.app_secret)
                snapshot_id = str(cursor_payload["snapshot_id"])
                offset = int(cursor_payload["offset"])
                cursor_expires_at = str(cursor_payload["expires_at"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise APIError(422, "invalid_cursor", "Cursor is invalid") from exc
            if (
                cursor_payload.get("user_id") != user["id"]
                or cursor_payload.get("feed_type") != feed_type
                or offset < 0
            ):
                raise APIError(422, "invalid_cursor", "Cursor does not belong to this user and feed")
            try:
                if _parse_datetime(cursor_expires_at, utc_now()) <= utc_now():
                    raise APIError(
                        410,
                        "cursor_expired",
                        "Feed cursor has expired; refresh the Feed",
                    )
            except (TypeError, ValueError) as exc:
                raise APIError(422, "invalid_cursor", "Cursor expiry is invalid") from exc

        started = time.perf_counter()
        feed_request_id = str(uuid.uuid4())
        created_at = isoformat()
        with database.transaction(immediate=True) as conn:
            fresh_user = conn.execute("SELECT * FROM users WHERE id = ?", (user["id"],)).fetchone()
            profile = conn.execute(
                "SELECT version FROM profiles WHERE user_id = ?",
                (user["id"],),
            ).fetchone()
            if fresh_user is None or profile is None:
                raise APIError(401, "unauthorized", "User is no longer active")

            if snapshot_id is None:
                seed = secrets.randbelow(2**31)
                artifact = artifacts.get()
                ensure_model(conn, artifact)
                result = recommender.recommend(
                    conn,
                    user=fresh_user,
                    feed_type=feed_type,
                    limit=settings.feed_snapshot_max_items,
                    seed=seed,
                    include_boosts=True,
                    now=created_at,
                )
                serving_version = result.model_version or "fallback-popularity-v1"
                snapshot_id = str(uuid.uuid4())
                expires_at = isoformat(
                    utc_now() + timedelta(minutes=settings.feed_snapshot_ttl_minutes)
                )
                ops_revision = int(
                    conn.execute("SELECT COALESCE(MAX(operation_id), 0) FROM operations").fetchone()[0]
                )
                conn.execute(
                    """
                    INSERT INTO feed_snapshots(
                        snapshot_id, user_id, feed_type, model_version, profile_version,
                        ops_revision, seed, fallback_reason, diversity_json,
                        status, created_at, expires_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
                    """,
                    (
                        snapshot_id,
                        user["id"],
                        feed_type,
                        serving_version,
                        int(profile["version"]),
                        ops_revision,
                        seed,
                        result.fallback_reason,
                        json.dumps(result.diversity_metrics, sort_keys=True),
                        created_at,
                        expires_at,
                    ),
                )
                for snapshot_position, candidate in enumerate(result.candidates):
                    conn.execute(
                        """
                        INSERT INTO feed_snapshot_items(
                            snapshot_id, snapshot_position, item_id, source, score,
                            raw_score, normalized_score, rank_in_source,
                            explanation, model_version, is_forced
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            snapshot_id,
                            snapshot_position,
                            candidate.item_id,
                            candidate.source,
                            float(candidate.score),
                            candidate.raw_score,
                            candidate.normalized_score,
                            candidate.rank_in_source,
                            candidate.explanation,
                            candidate.model_version or serving_version,
                            int(candidate.is_forced),
                        ),
                    )
                for candidate in result.candidate_manifest:
                    conn.execute(
                        """
                        INSERT INTO candidate_manifests(
                            snapshot_id, item_id, primary_source, source_scores_json,
                            source_calibrated_scores_json, source_mask_json,
                            source_raw_scores_json, source_evidence_json,
                            ranker_score, ranker_rank,
                            model_version, feature_schema_version, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            snapshot_id,
                            candidate.item_id,
                            candidate.primary_source,
                            json.dumps(candidate.source_scores),
                            json.dumps(candidate.source_calibrated_scores),
                            json.dumps(candidate.source_mask),
                            json.dumps(candidate.source_raw_scores),
                            json.dumps(candidate.source_evidence(), sort_keys=True),
                            candidate.ranker_score,
                            candidate.ranker_rank,
                            serving_version,
                            UNIFIED_FEATURE_SCHEMA_VERSION,
                            created_at,
                        ),
                    )
                snapshot = conn.execute(
                    "SELECT * FROM feed_snapshots WHERE snapshot_id = ?",
                    (snapshot_id,),
                ).fetchone()
            else:
                snapshot = conn.execute(
                    "SELECT * FROM feed_snapshots WHERE snapshot_id = ?",
                    (snapshot_id,),
                ).fetchone()
                if (
                    snapshot is None
                    or snapshot["user_id"] != user["id"]
                    or snapshot["feed_type"] != feed_type
                ):
                    raise APIError(422, "invalid_cursor", "Cursor snapshot is invalid")
                if cursor_expires_at != snapshot["expires_at"]:
                    raise APIError(422, "invalid_cursor", "Cursor expiry does not match snapshot")
                if (
                    snapshot["status"] != "active"
                    or _parse_datetime(snapshot["expires_at"], utc_now()) <= utc_now()
                ):
                    raise APIError(410, "cursor_expired", "Feed cursor has expired; refresh the Feed")

            serving_version = snapshot["model_version"]
            snapshot_profile_version = int(snapshot["profile_version"])
            fallback_reason = snapshot["fallback_reason"]
            diversity_metrics = json.loads(snapshot["diversity_json"])
            page_end = offset + limit
            snapshot_rows = conn.execute(
                """
                SELECT si.*, i.title
                FROM feed_snapshot_items si
                JOIN items i ON i.item_id = si.item_id
                WHERE si.snapshot_id = ?
                  AND si.snapshot_position >= ?
                  AND si.snapshot_position < ?
                  AND si.invalidated_at IS NULL
                  AND i.status = 'online'
                ORDER BY si.snapshot_position
                """,
                (snapshot_id, offset, page_end),
            ).fetchall()
            has_more = conn.execute(
                """
                SELECT 1
                FROM feed_snapshot_items si
                JOIN items i ON i.item_id = si.item_id
                WHERE si.snapshot_id = ?
                  AND si.snapshot_position >= ?
                  AND si.invalidated_at IS NULL
                  AND i.status = 'online'
                LIMIT 1
                """,
                (snapshot_id, page_end),
            ).fetchone() is not None
            conn.execute(
                """
                INSERT INTO recommendation_requests(
                    request_id, user_id, feed_type, model_version, profile_version,
                    cursor, fallback_reason, snapshot_id, snapshot_offset,
                    returned_count, latency_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                """,
                (
                    feed_request_id,
                    user["id"],
                    feed_type,
                    serving_version,
                    snapshot_profile_version,
                    cursor,
                    fallback_reason,
                    snapshot_id,
                    offset,
                    len(snapshot_rows),
                    created_at,
                ),
            )
            response_items = []
            for position, item in enumerate(snapshot_rows):
                item_model_version = item["model_version"] or serving_version
                conn.execute(
                    """
                    INSERT INTO exposures(
                        request_id, user_id, item_id, position, source, score,
                        raw_score, normalized_score, rank_in_source,
                        explanation, model_version, is_forced, snapshot_id,
                        snapshot_position, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        feed_request_id,
                        user["id"],
                        item["item_id"],
                        position,
                        item["source"],
                        float(item["score"]),
                        item["raw_score"],
                        item["normalized_score"],
                        item["rank_in_source"],
                        item["explanation"],
                        item_model_version,
                        int(item["is_forced"]),
                        snapshot_id,
                        int(item["snapshot_position"]),
                        created_at,
                    ),
                )
                response_items.append(
                    {
                        "item_id": str(item["item_id"]),
                        "title": item["title"],
                        "cover_url": f"/api/v1/items/{item['item_id']}/cover",
                        "position": position,
                        "snapshot_position": int(item["snapshot_position"]),
                        "source": item["source"],
                        "score": float(item["score"]),
                        "raw_score": (
                            float(item["raw_score"]) if item["raw_score"] is not None else None
                        ),
                        "normalized_score": (
                            float(item["normalized_score"])
                            if item["normalized_score"] is not None
                            else None
                        ),
                        "rank_in_source": item["rank_in_source"],
                        "explanation": item["explanation"],
                        "model_version": item_model_version,
                        "is_forced": bool(item["is_forced"]),
                    }
                )
            latency_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
            conn.execute(
                "UPDATE recommendation_requests SET latency_ms = ? WHERE request_id = ?",
                (latency_ms, feed_request_id),
            )

        next_cursor = None
        if has_more:
            next_cursor = encode_cursor(
                {
                    "snapshot_id": snapshot_id,
                    "offset": offset + limit,
                    "user_id": user["id"],
                    "feed_type": feed_type,
                    "expires_at": snapshot["expires_at"],
                },
                settings.app_secret,
            )
        return {
            "request_id": feed_request_id,
            "snapshot_id": snapshot_id,
            "feed_type": feed_type,
            "model_version": serving_version,
            "profile_version": snapshot_profile_version,
            "fallback_reason": fallback_reason,
            "diversity": diversity_metrics,
            "next_cursor": next_cursor,
            "has_more": bool(next_cursor),
            "items": response_items,
        }

    @app.post("/api/v1/events/batch")
    def post_events(body: EventBatch, request: Request) -> dict[str, Any]:
        user = current_user(request)
        accepted = 0
        duplicates = 0
        profile_changed = False
        received_at = isoformat()
        with database.transaction(immediate=True) as conn:
            profile = conn.execute(
                "SELECT version FROM profiles WHERE user_id = ?",
                (user["id"],),
            ).fetchone()
            if profile is None:
                raise APIError(404, "profile_not_found", "Profile does not exist")
            for event in body.events:
                normalized_type = "like" if event.event_type == "favorite" else event.event_type
                existing = conn.execute(
                    "SELECT * FROM events WHERE event_id = ?",
                    (event.event_id,),
                ).fetchone()
                if existing is not None:
                    if (
                        existing["event_type"] == normalized_type
                        and existing["request_id"] == event.request_id
                        and existing["user_id"] == user["id"]
                        and str(existing["item_id"]) == event.item_id
                        and int(existing["position"]) == event.position
                        and existing["dwell_ms"] == event.dwell_ms
                    ):
                        duplicates += 1
                        continue
                    raise APIError(409, "event_id_conflict", "event_id was already used for another event")

                exposure = conn.execute(
                    """
                    SELECT 1 FROM exposures
                    WHERE request_id = ? AND user_id = ? AND item_id = ? AND position = ?
                    """,
                    (event.request_id, user["id"], event.item_id, event.position),
                ).fetchone()
                if exposure is None:
                    raise APIError(
                        422,
                        "exposure_mismatch",
                        "Event does not match this user's persisted exposure",
                        {"event_id": event.event_id},
                    )
                same_type = conn.execute(
                    """
                    SELECT * FROM events
                    WHERE request_id = ? AND user_id = ? AND item_id = ?
                      AND position = ? AND event_type = ?
                    """,
                    (
                        event.request_id,
                        user["id"],
                        event.item_id,
                        event.position,
                        normalized_type,
                    ),
                ).fetchone()
                if same_type is not None:
                    if normalized_type == "dwell" and event.dwell_ms is not None:
                        previous_dwell = int(same_type["dwell_ms"] or 0)
                        if event.dwell_ms > previous_dwell:
                            conn.execute(
                                "UPDATE events SET dwell_ms = ? WHERE event_id = ?",
                                (event.dwell_ms, same_type["event_id"]),
                            )
                            affinity_delta = _dwell_affinity(event.dwell_ms) - _dwell_affinity(
                                previous_dwell
                            )
                            conn.execute(
                                """
                                UPDATE user_item_state
                                SET dwell_ms_total = dwell_ms_total + ?,
                                    affinity = CASE
                                        WHEN not_interested = 1 THEN MIN(affinity, -4.0)
                                        ELSE MIN(8.0, affinity + ?)
                                    END,
                                    last_event_at = ?
                                WHERE user_id = ? AND item_id = ?
                                """,
                                (
                                    event.dwell_ms - previous_dwell,
                                    affinity_delta,
                                    received_at,
                                    user["id"],
                                    event.item_id,
                                ),
                            )
                            profile_changed = True
                    duplicates += 1
                    continue

                client_timestamp = _canonical_datetime(event.client_timestamp)
                visit_index = None
                if normalized_type == "impression":
                    prior_visits = int(
                        conn.execute(
                            """
                            SELECT COUNT(DISTINCT request_id) FROM events
                            WHERE user_id = ? AND item_id = ?
                              AND event_type = 'impression' AND request_id != ?
                            """,
                            (user["id"], event.item_id, event.request_id),
                        ).fetchone()[0]
                    )
                    visit_index = prior_visits + 1
                conn.execute(
                    """
                    INSERT INTO events(
                        event_id, event_type, request_id, user_id, item_id,
                        position, client_timestamp, dwell_ms, visit_index, received_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.event_id,
                        normalized_type,
                        event.request_id,
                        user["id"],
                        event.item_id,
                        event.position,
                        client_timestamp,
                        event.dwell_ms,
                        visit_index,
                        received_at,
                    ),
                )
                revisit_delta = 0
                if normalized_type == "impression" and visit_index and visit_index > 1:
                    revisit = conn.execute(
                        """
                        INSERT OR IGNORE INTO events(
                            event_id, event_type, request_id, user_id, item_id,
                            position, client_timestamp, visit_index, received_at
                        ) VALUES (?, 'revisit', ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            f"revisit:{event.request_id}:{event.item_id}:{event.position}",
                            event.request_id,
                            user["id"],
                            event.item_id,
                            event.position,
                            client_timestamp,
                            visit_index,
                            received_at,
                        ),
                    )
                    revisit_delta = int(revisit.rowcount > 0)
                impression_delta = 1 if normalized_type == "impression" else 0
                click_delta = 1 if normalized_type == "click" else 0
                like_delta = 1 if normalized_type == "like" else 0
                dwell_delta = event.dwell_ms if normalized_type == "dwell" else 0
                dwell_event_delta = 1 if normalized_type == "dwell" else 0
                share_delta = 1 if normalized_type == "share" else 0
                negative = 1 if normalized_type == "not_interested" else 0
                affinity_delta = {
                    "impression": 0.0,
                    "click": 1.0,
                    "like": 3.0,
                    "not_interested": -4.0,
                    "dwell": _dwell_affinity(event.dwell_ms),
                    "share": 4.0,
                }[normalized_type]
                affinity_delta += 1.5 * revisit_delta
                conn.execute(
                    """
                    INSERT INTO user_item_state(
                        user_id, item_id, exposure_count, click_count, like_count,
                        dwell_ms_total, dwell_event_count, share_count, revisit_count,
                        not_interested, affinity, last_event_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id, item_id) DO UPDATE SET
                        exposure_count = exposure_count + excluded.exposure_count,
                        click_count = click_count + excluded.click_count,
                        like_count = like_count + excluded.like_count,
                        dwell_ms_total = dwell_ms_total + excluded.dwell_ms_total,
                        dwell_event_count = dwell_event_count + excluded.dwell_event_count,
                        share_count = share_count + excluded.share_count,
                        revisit_count = revisit_count + excluded.revisit_count,
                        not_interested = MAX(not_interested, excluded.not_interested),
                        affinity = CASE
                            WHEN excluded.not_interested = 1
                              OR user_item_state.not_interested = 1
                            THEN MIN(user_item_state.affinity, -4.0)
                            ELSE MIN(8.0, user_item_state.affinity + excluded.affinity)
                        END,
                        last_event_at = excluded.last_event_at
                    """,
                    (
                        user["id"],
                        event.item_id,
                        impression_delta,
                        click_delta,
                        like_delta,
                        dwell_delta,
                        dwell_event_delta,
                        share_delta,
                        revisit_delta,
                        negative,
                        affinity_delta,
                        received_at,
                    ),
                )
                accepted += 1
                profile_changed = True
            if profile_changed:
                conn.execute(
                    "UPDATE profiles SET version = version + 1, updated_at = ? WHERE user_id = ?",
                    (received_at, user["id"]),
                )
            resulting = conn.execute(
                "SELECT version FROM profiles WHERE user_id = ?",
                (user["id"],),
            ).fetchone()
        return {
            "accepted": accepted,
            "duplicates": duplicates,
            "profile_version": int(resulting["version"]),
        }

    @app.get("/api/v1/me/profile")
    def me_profile(request: Request) -> dict[str, Any]:
        user = current_user(request)
        with database.connect() as conn:
            return profile_payload(conn, user["id"])

    @app.get("/api/v1/me/events")
    def me_events(
        request: Request,
        limit: int = Query(default=50, ge=1, le=200),
    ) -> dict[str, Any]:
        user = current_user(request)
        with database.connect() as conn:
            return {"events": recent_events(conn, user["id"], limit), "limit": limit}

    @app.get("/api/v1/items/{item_id}")
    def get_item(item_id: str, request: Request) -> dict[str, Any]:
        current_user(request)
        cache_key = f"item:public:{item_id}"
        cached = cache.get_json(cache_key)
        if cached is not None:
            return {"item": cached}
        with database.connect() as conn:
            item = conn.execute(
                "SELECT * FROM items WHERE item_id = ? AND status = 'online'",
                (item_id,),
            ).fetchone()
        if item is None:
            raise APIError(404, "item_not_found", "Item does not exist or is offline")
        payload = _item_payload(item, public=True)
        cache.set_json(cache_key, payload, ttl_seconds=60)
        return {"item": payload}

    @app.get("/api/v1/items/{item_id}/cover")
    def get_cover(item_id: str, request: Request) -> Response:
        current_user(request)
        with database.connect() as conn:
            exists = conn.execute(
                "SELECT 1 FROM items WHERE item_id = ? AND status = 'online'",
                (item_id,),
            ).fetchone()
        if exists is None:
            raise APIError(404, "item_not_found", "Item does not exist or is offline")
        # Deterministic 1x1 PNG placeholder; the MVP does not redistribute covers.
        png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
        )
        return Response(content=png, media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})

    @app.get("/api/v1/admin/dashboard/overview")
    def dashboard_overview(
        request: Request,
        from_value: str | None = Query(default=None, alias="from"),
        to_value: str | None = Query(default=None, alias="to"),
    ) -> dict[str, Any]:
        require_role(request, "analyst")
        start, end, start_text, end_text = _resolve_dashboard_range(from_value, to_value)
        with database.connect() as conn:
            semantics_row = conn.execute(
                """
                SELECT value FROM app_metadata
                WHERE key = 'viewable_impression_semantics_started_at'
                """
            ).fetchone()
            semantics_started_at = semantics_row["value"] if semantics_row else end_text
            viewable_start = max(start, _parse_datetime(semantics_started_at, end))
            viewable_start_text = _sortable_datetime(viewable_start)
            users = int(conn.execute("SELECT COUNT(*) FROM users WHERE role = 'user'").fetchone()[0])
            active_users = int(
                conn.execute(
                    """
                    SELECT COUNT(*) FROM (
                        SELECT user_id FROM recommendation_requests
                        WHERE created_at >= ? AND created_at < ?
                        UNION
                        SELECT user_id FROM events
                        WHERE event_type != 'impression' AND received_at >= ? AND received_at < ?
                    )
                    """,
                    (start_text, end_text, start_text, end_text),
                ).fetchone()[0]
            )
            requests_count = int(
                conn.execute(
                    "SELECT COUNT(*) FROM recommendation_requests WHERE created_at >= ? AND created_at < ?",
                    (start_text, end_text),
                ).fetchone()[0]
            )
            exposures_count = int(
                conn.execute(
                    "SELECT COUNT(*) FROM exposures WHERE created_at >= ? AND created_at < ?",
                    (start_text, end_text),
                ).fetchone()[0]
            )
            impressions_count = int(
                conn.execute(
                    """
                    SELECT COUNT(*) FROM events
                    WHERE event_type = 'impression'
                      AND received_at >= ? AND received_at < ?
                    """,
                    (viewable_start_text, end_text),
                ).fetchone()[0]
            )
            clicks = int(
                conn.execute(
                    "SELECT COUNT(*) FROM events WHERE event_type = 'click' AND received_at >= ? AND received_at < ?",
                    (start_text, end_text),
                ).fetchone()[0]
            )
            viewable_clicks = int(
                conn.execute(
                    """
                    SELECT COUNT(*) FROM events click_event
                    WHERE click_event.event_type = 'click'
                      AND click_event.received_at >= ? AND click_event.received_at < ?
                      AND EXISTS (
                          SELECT 1 FROM events impression_event
                          WHERE impression_event.event_type = 'impression'
                            AND impression_event.user_id = click_event.user_id
                            AND impression_event.request_id = click_event.request_id
                            AND impression_event.item_id = click_event.item_id
                            AND impression_event.position = click_event.position
                            AND impression_event.received_at >= ?
                            AND impression_event.received_at < ?
                      )
                    """,
                    (
                        viewable_start_text,
                        end_text,
                        viewable_start_text,
                        end_text,
                    ),
                ).fetchone()[0]
            )
            likes = int(
                conn.execute(
                    "SELECT COUNT(*) FROM events WHERE event_type = 'like' AND received_at >= ? AND received_at < ?",
                    (start_text, end_text),
                ).fetchone()[0]
            )
            shares = int(
                conn.execute(
                    "SELECT COUNT(*) FROM events WHERE event_type = 'share' AND received_at >= ? AND received_at < ?",
                    (start_text, end_text),
                ).fetchone()[0]
            )
            revisits = int(
                conn.execute(
                    "SELECT COUNT(*) FROM events WHERE event_type = 'revisit' AND received_at >= ? AND received_at < ?",
                    (start_text, end_text),
                ).fetchone()[0]
            )
            revisit_users = int(
                conn.execute(
                    """
                    SELECT COUNT(DISTINCT user_id) FROM events
                    WHERE event_type = 'revisit' AND received_at >= ? AND received_at < ?
                    """,
                    (start_text, end_text),
                ).fetchone()[0]
            )
            dwell_values = [
                float(row[0])
                for row in conn.execute(
                    """
                    SELECT dwell_ms FROM events
                    WHERE event_type = 'dwell' AND received_at >= ? AND received_at < ?
                    ORDER BY dwell_ms
                    """,
                    (start_text, end_text),
                )
            ]
            dwell = _percentiles(dwell_values)
            dwell["average"] = (
                sum(dwell_values) / len(dwell_values) if dwell_values else 0.0
            )
            dwell["count"] = len(dwell_values)
            latency_rows = conn.execute(
                "SELECT latency_ms FROM recommendation_requests WHERE created_at >= ? AND created_at < ?",
                (start_text, end_text),
            ).fetchall()
            latency = _percentiles([float(row[0]) for row in latency_rows])
            offline_items = int(
                conn.execute("SELECT COUNT(*) FROM items WHERE status = 'offline'").fetchone()[0]
            )
            now = isoformat()
            active_boosts = int(
                conn.execute(
                    """
                    SELECT COUNT(*) FROM boost_campaigns
                    WHERE active = 1 AND starts_at <= ? AND ends_at > ?
                    """,
                    (now, now),
                ).fetchone()[0]
            )
            feed_rows = conn.execute(
                """
                WITH feed_metrics AS (
                    SELECT feed_type, COUNT(*) AS requests, 0 AS exposures,
                           0 AS impressions, 0 AS clicks, 0 AS viewable_clicks,
                           0 AS likes, 0 AS not_interested
                    FROM recommendation_requests
                    WHERE created_at >= ? AND created_at < ?
                    GROUP BY feed_type
                    UNION ALL
                    SELECT r.feed_type, 0, COUNT(*), 0, 0, 0, 0, 0
                    FROM exposures e
                    JOIN recommendation_requests r ON r.request_id = e.request_id
                    WHERE e.created_at >= ? AND e.created_at < ?
                    GROUP BY r.feed_type
                    UNION ALL
                    SELECT r.feed_type, 0, 0,
                           COUNT(DISTINCT CASE
                               WHEN ev.event_type = 'impression' AND ev.received_at >= ?
                               THEN ev.event_id END),
                           COUNT(DISTINCT CASE WHEN ev.event_type = 'click' THEN ev.event_id END),
                           COUNT(DISTINCT CASE
                               WHEN ev.event_type = 'click' AND ev.received_at >= ?
                                AND EXISTS (
                                    SELECT 1 FROM events impression_event
                                    WHERE impression_event.event_type = 'impression'
                                      AND impression_event.user_id = ev.user_id
                                      AND impression_event.request_id = ev.request_id
                                      AND impression_event.item_id = ev.item_id
                                      AND impression_event.position = ev.position
                                      AND impression_event.received_at >= ?
                                      AND impression_event.received_at < ?
                                )
                               THEN ev.event_id END),
                           COUNT(DISTINCT CASE WHEN ev.event_type = 'like' THEN ev.event_id END),
                           COUNT(DISTINCT CASE WHEN ev.event_type = 'not_interested' THEN ev.event_id END)
                    FROM events ev
                    JOIN exposures e ON e.request_id = ev.request_id
                                    AND e.item_id = ev.item_id
                                    AND e.position = ev.position
                    JOIN recommendation_requests r ON r.request_id = ev.request_id
                    WHERE ev.received_at >= ? AND ev.received_at < ?
                    GROUP BY r.feed_type
                )
                SELECT feed_type,
                       SUM(requests) AS requests,
                       SUM(exposures) AS exposures,
                       SUM(impressions) AS impressions,
                       SUM(clicks) AS clicks,
                       SUM(viewable_clicks) AS viewable_clicks,
                       SUM(likes) AS likes,
                       SUM(not_interested) AS not_interested
                FROM feed_metrics
                GROUP BY feed_type
                """,
                (
                    start_text,
                    end_text,
                    start_text,
                    end_text,
                    viewable_start_text,
                    viewable_start_text,
                    viewable_start_text,
                    end_text,
                    start_text,
                    end_text,
                ),
            ).fetchall()
            feed_values = {
                row["feed_type"]: {
                    "feed_type": row["feed_type"],
                    "requests": int(row["requests"]),
                    "exposures": int(row["exposures"]),
                    "served_exposures": int(row["exposures"]),
                    "impressions": int(row["impressions"]),
                    "viewable_impressions": int(row["impressions"]),
                    "clicks": int(row["clicks"]),
                    "likes": int(row["likes"]),
                    "not_interested": int(row["not_interested"]),
                    "ctr": int(row["clicks"]) / int(row["exposures"])
                    if row["exposures"]
                    else 0.0,
                    "ctr_denominator": "served_exposures",
                    "served_ctr": int(row["clicks"]) / int(row["exposures"])
                    if row["exposures"]
                    else 0.0,
                    "viewable_ctr": int(row["viewable_clicks"]) / int(row["impressions"])
                    if row["impressions"]
                    else 0.0,
                    "share": int(row["exposures"]) / exposures_count
                    if exposures_count
                    else 0.0,
                }
                for row in feed_rows
            }
            engagement_rows = conn.execute(
                """
                SELECT r.feed_type,
                       COUNT(DISTINCT CASE WHEN ev.event_type = 'share' THEN ev.event_id END)
                           AS shares,
                       COUNT(DISTINCT CASE WHEN ev.event_type = 'revisit' THEN ev.event_id END)
                           AS revisits,
                       COUNT(DISTINCT CASE WHEN ev.event_type = 'revisit' THEN ev.user_id END)
                           AS revisit_users,
                       AVG(CASE WHEN ev.event_type = 'dwell' THEN ev.dwell_ms END)
                           AS average_dwell_ms
                FROM events ev
                JOIN recommendation_requests r ON r.request_id = ev.request_id
                WHERE ev.received_at >= ? AND ev.received_at < ?
                GROUP BY r.feed_type
                """,
                (start_text, end_text),
            ).fetchall()
            engagement_values = {
                row["feed_type"]: {
                    "shares": int(row["shares"]),
                    "revisits": int(row["revisits"]),
                    "revisit_users": int(row["revisit_users"]),
                    "average_dwell_ms": float(row["average_dwell_ms"] or 0.0),
                }
                for row in engagement_rows
            }
            top_rows = conn.execute(
                """
                SELECT i.item_id, i.title,
                       COUNT(DISTINCT e.id) AS exposures,
                       COUNT(DISTINCT CASE
                           WHEN ev.event_type = 'impression' AND ev.received_at >= ?
                           THEN ev.event_id END) AS impressions,
                       COUNT(DISTINCT CASE WHEN ev.event_type = 'click' THEN ev.event_id END) AS clicks,
                       COUNT(DISTINCT CASE
                           WHEN ev.event_type = 'click' AND ev.received_at >= ?
                           AND EXISTS (
                               SELECT 1 FROM events impression_event
                               WHERE impression_event.event_type = 'impression'
                                 AND impression_event.user_id = ev.user_id
                                 AND impression_event.request_id = ev.request_id
                                 AND impression_event.item_id = ev.item_id
                                 AND impression_event.position = ev.position
                                 AND impression_event.received_at >= ?
                                 AND impression_event.received_at < ?
                           )
                           THEN ev.event_id END) AS viewable_clicks,
                       COUNT(DISTINCT CASE WHEN ev.event_type = 'like' THEN ev.event_id END) AS likes
                FROM exposures e
                JOIN items i ON i.item_id = e.item_id
                LEFT JOIN events ev ON ev.request_id = e.request_id
                                   AND ev.item_id = e.item_id
                                   AND ev.position = e.position
                                   AND ev.received_at >= ? AND ev.received_at < ?
                WHERE e.created_at >= ? AND e.created_at < ?
                GROUP BY i.item_id, i.title
                ORDER BY exposures DESC, clicks DESC, i.item_id
                LIMIT 10
                """,
                (
                    viewable_start_text,
                    viewable_start_text,
                    viewable_start_text,
                    end_text,
                    start_text,
                    end_text,
                    start_text,
                    end_text,
                ),
            ).fetchall()
            source_rows = conn.execute(
                """
                SELECT source,
                       COUNT(*) AS served_exposures,
                       COUNT(DISTINCT request_id) AS requests
                FROM exposures
                WHERE created_at >= ? AND created_at < ?
                GROUP BY source
                ORDER BY served_exposures DESC, source
                """,
                (start_text, end_text),
            ).fetchall()
        artifact = artifacts.get()
        return {
            "range": {"from": start_text, "to": end_text},
            "users": users,
            "active_users": active_users,
            "requests": requests_count,
            "exposures": exposures_count,
            "served_exposures": exposures_count,
            "impressions": impressions_count,
            "viewable_impressions": impressions_count,
            "clicks": clicks,
            "likes": likes,
            "shares": shares,
            "revisits": revisits,
            "revisit_users": revisit_users,
            "dwell": dwell,
            "latency": latency,
            "ctr": clicks / exposures_count if exposures_count else 0.0,
            "ctr_denominator": "served_exposures",
            "served_ctr": clicks / exposures_count if exposures_count else 0.0,
            "viewable_ctr": viewable_clicks / impressions_count if impressions_count else 0.0,
            "viewable_impression_semantics_started_at": semantics_started_at,
            "viewable_metrics_from": viewable_start_text,
            "offline_items": offline_items,
            "active_boosts": active_boosts,
            "current_model_version": artifact.model_version if artifact else None,
            "feed_breakdown": [
                {
                    **feed_values.get(
                        feed_type,
                        {
                        "feed_type": feed_type,
                        "requests": 0,
                        "exposures": 0,
                        "served_exposures": 0,
                        "impressions": 0,
                        "viewable_impressions": 0,
                        "clicks": 0,
                        "likes": 0,
                        "not_interested": 0,
                        "ctr": 0.0,
                        "ctr_denominator": "served_exposures",
                        "served_ctr": 0.0,
                        "viewable_ctr": 0.0,
                        "share": 0.0,
                        },
                    ),
                    **engagement_values.get(
                        feed_type,
                        {
                            "shares": 0,
                            "revisits": 0,
                            "revisit_users": 0,
                            "average_dwell_ms": 0.0,
                        },
                    ),
                }
                for feed_type in ("personalized", "popular", "explore")
            ],
            "candidate_sources": [
                {
                    "source": row["source"],
                    "served_exposures": int(row["served_exposures"]),
                    "requests": int(row["requests"]),
                    "share": (
                        int(row["served_exposures"]) / exposures_count
                        if exposures_count
                        else 0.0
                    ),
                }
                for row in source_rows
            ],
            "top_items": [
                {
                    "item_id": str(row["item_id"]),
                    "title": row["title"],
                    "exposures": int(row["exposures"]),
                    "served_exposures": int(row["exposures"]),
                    "impressions": int(row["impressions"]),
                    "viewable_impressions": int(row["impressions"]),
                    "clicks": int(row["clicks"]),
                    "likes": int(row["likes"]),
                    "ctr": int(row["clicks"]) / int(row["exposures"]) if row["exposures"] else 0.0,
                    "ctr_denominator": "served_exposures",
                    "served_ctr": int(row["clicks"]) / int(row["exposures"])
                    if row["exposures"]
                    else 0.0,
                    "viewable_ctr": int(row["viewable_clicks"]) / int(row["impressions"])
                    if row["impressions"]
                    else 0.0,
                }
                for row in top_rows
            ],
        }

    @app.get("/api/v1/admin/dashboard/export.csv")
    def dashboard_export_csv(
        request: Request,
        from_value: str | None = Query(default=None, alias="from"),
        to_value: str | None = Query(default=None, alias="to"),
    ) -> Response:
        overview = dashboard_overview(request, from_value, to_value)
        fieldnames = [
            "record_type",
            "scope",
            "range_from",
            "range_to",
            "users",
            "active_users",
            "requests",
            "served_exposures",
            "viewable_impressions",
            "clicks",
            "likes",
            "not_interested",
            "shares",
            "revisits",
            "revisit_users",
            "average_dwell_ms",
            "p95_dwell_ms",
            "served_ctr",
            "viewable_ctr",
            "served_exposure_share",
            "item_id",
            "title",
            "offline_items",
            "active_boosts",
            "current_model_version",
        ]
        common = {
            "range_from": overview["range"]["from"],
            "range_to": overview["range"]["to"],
        }
        rows: list[dict[str, Any]] = [
            {
                **common,
                "record_type": "overview",
                "scope": "all",
                "users": overview["users"],
                "active_users": overview["active_users"],
                "requests": overview["requests"],
                "served_exposures": overview["served_exposures"],
                "viewable_impressions": overview["viewable_impressions"],
                "clicks": overview["clicks"],
                "likes": overview["likes"],
                "shares": overview["shares"],
                "revisits": overview["revisits"],
                "revisit_users": overview["revisit_users"],
                "average_dwell_ms": overview["dwell"]["average"],
                "p95_dwell_ms": overview["dwell"]["p95"],
                "served_ctr": overview["served_ctr"],
                "viewable_ctr": overview["viewable_ctr"],
                "offline_items": overview["offline_items"],
                "active_boosts": overview["active_boosts"],
                "current_model_version": overview["current_model_version"],
            }
        ]
        rows.extend(
            {
                **common,
                "record_type": "feed",
                "scope": feed["feed_type"],
                "requests": feed["requests"],
                "served_exposures": feed["served_exposures"],
                "viewable_impressions": feed["viewable_impressions"],
                "clicks": feed["clicks"],
                "likes": feed["likes"],
                "not_interested": feed["not_interested"],
                "shares": feed["shares"],
                "revisits": feed["revisits"],
                "revisit_users": feed["revisit_users"],
                "average_dwell_ms": feed["average_dwell_ms"],
                "served_ctr": feed["served_ctr"],
                "viewable_ctr": feed["viewable_ctr"],
                "served_exposure_share": feed["share"],
            }
            for feed in overview["feed_breakdown"]
        )
        rows.extend(
            {
                **common,
                "record_type": "candidate_source",
                "scope": source["source"],
                "requests": source["requests"],
                "served_exposures": source["served_exposures"],
                "served_exposure_share": source["share"],
            }
            for source in overview["candidate_sources"]
        )
        rows.extend(
            {
                **common,
                "record_type": "top_item",
                "scope": "item",
                "item_id": item["item_id"],
                "title": item["title"],
                "served_exposures": item["served_exposures"],
                "viewable_impressions": item["viewable_impressions"],
                "clicks": item["clicks"],
                "likes": item["likes"],
                "served_ctr": item["served_ctr"],
                "viewable_ctr": item["viewable_ctr"],
            }
            for item in overview["top_items"]
        )

        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(rows)
        start = _parse_datetime(overview["range"]["from"], utc_now())
        end = _parse_datetime(overview["range"]["to"], utc_now())
        filename = (
            f"dashboard_{start.strftime('%Y%m%dT%H%M%SZ')}_"
            f"{end.strftime('%Y%m%dT%H%M%SZ')}.csv"
        )
        return Response(
            content=("\ufeff" + output.getvalue()).encode("utf-8"),
            headers={
                "Content-Type": "text/csv; charset=utf-8",
                "Content-Disposition": f'attachment; filename="{filename}"',
            },
        )

    @app.get("/api/v1/admin/dashboard/timeseries")
    def dashboard_timeseries(
        request: Request,
        metric: str = Query(default="requests"),
        from_value: str | None = Query(default=None, alias="from"),
        to_value: str | None = Query(default=None, alias="to"),
    ) -> dict[str, Any]:
        require_role(request, "analyst")
        if metric not in (
            "requests",
            "exposures",
            "served_exposures",
            "impressions",
            "viewable_impressions",
            "clicks",
            "likes",
            "shares",
            "revisits",
            "dwell_events",
        ):
            raise APIError(
                422,
                "invalid_metric",
                "metric must be one of requests/served_exposures/viewable_impressions/clicks/likes",
            )
        start, end, start_text, end_text = _resolve_dashboard_range(from_value, to_value)

        span = end - start
        bucket_seconds = 3600 if span <= timedelta(hours=48) else 86400
        bucket = "hour" if bucket_seconds == 3600 else "day"

        sql = {
            "requests": "SELECT created_at FROM recommendation_requests WHERE created_at >= ? AND created_at < ?",
            "exposures": "SELECT created_at FROM exposures WHERE created_at >= ? AND created_at < ?",
            "served_exposures": "SELECT created_at FROM exposures WHERE created_at >= ? AND created_at < ?",
            "impressions": "SELECT received_at FROM events WHERE event_type = 'impression' AND received_at >= ? AND received_at < ?",
            "viewable_impressions": "SELECT received_at FROM events WHERE event_type = 'impression' AND received_at >= ? AND received_at < ?",
            "clicks": "SELECT received_at FROM events WHERE event_type = 'click' AND received_at >= ? AND received_at < ?",
            "likes": "SELECT received_at FROM events WHERE event_type = 'like' AND received_at >= ? AND received_at < ?",
            "shares": "SELECT received_at FROM events WHERE event_type = 'share' AND received_at >= ? AND received_at < ?",
            "revisits": "SELECT received_at FROM events WHERE event_type = 'revisit' AND received_at >= ? AND received_at < ?",
            "dwell_events": "SELECT received_at FROM events WHERE event_type = 'dwell' AND received_at >= ? AND received_at < ?",
        }[metric]

        query_start_text = start_text
        with database.connect() as conn:
            if metric in {"impressions", "viewable_impressions"}:
                semantics_row = conn.execute(
                    """
                    SELECT value FROM app_metadata
                    WHERE key = 'viewable_impression_semantics_started_at'
                    """
                ).fetchone()
                if semantics_row is not None:
                    query_start_text = _sortable_datetime(
                        max(start, _parse_datetime(semantics_row["value"], end))
                    )
            raw = conn.execute(sql, (query_start_text, end_text)).fetchall()

        buckets: dict[int, int] = {}
        for row in raw:
            slot = (_iso_to_epoch_seconds(row[0]) // bucket_seconds) * bucket_seconds
            buckets[slot] = buckets.get(slot, 0) + 1

        start_slot = (int(start.timestamp()) // bucket_seconds) * bucket_seconds
        # The query range is [start, end), so an exact bucket boundary belongs
        # to the preceding bucket rather than creating a trailing zero point.
        end_inclusive = end - timedelta(microseconds=1)
        end_slot = (int(end_inclusive.timestamp()) // bucket_seconds) * bucket_seconds
        points: list[dict[str, Any]] = []
        slot = start_slot
        while slot <= end_slot:
            points.append(
                {
                    "t": datetime.fromtimestamp(slot, tz=UTC).isoformat().replace("+00:00", "Z"),
                    "value": buckets.get(slot, 0),
                }
            )
            slot += bucket_seconds

        return {
            "metric": metric,
            "bucket": bucket,
            "range": {"from": start_text, "to": end_text},
            "metric_data_from": query_start_text,
            "points": points,
        }

    @app.get("/api/v1/admin/requests/{feed_request_id}")
    def admin_request_detail(feed_request_id: str, request: Request) -> dict[str, Any]:
        require_role(request, "analyst")
        with database.connect() as conn:
            row = conn.execute(
                """
                SELECT r.*, u.username
                FROM recommendation_requests r JOIN users u ON u.id = r.user_id
                WHERE r.request_id = ?
                """,
                (feed_request_id,),
            ).fetchone()
            if row is None:
                raise APIError(404, "request_not_found", "Recommendation request not found")
            exposures = conn.execute(
                """
                SELECT e.*, i.title
                FROM exposures e JOIN items i ON i.item_id = e.item_id
                WHERE e.request_id = ? ORDER BY e.position
                """,
                (feed_request_id,),
            ).fetchall()
            event_rows = conn.execute(
                """
                SELECT event_id, event_type, item_id, position,
                       client_timestamp, received_at
                FROM events WHERE request_id = ?
                ORDER BY position, received_at, event_id
                """,
                (feed_request_id,),
            ).fetchall()
        return {
            "request": {
                "request_id": row["request_id"],
                "snapshot_id": row["snapshot_id"],
                "snapshot_offset": row["snapshot_offset"],
                "user_id": row["user_id"],
                "username": row["username"],
                "feed_type": row["feed_type"],
                "model_version": row["model_version"],
                "profile_version": int(row["profile_version"]),
                "fallback_reason": row["fallback_reason"],
                "returned_count": int(row["returned_count"]),
                "created_at": row["created_at"],
                "latency_ms": float(row["latency_ms"]),
            },
            "items": [
                {
                    "item_id": str(item["item_id"]),
                    "title": item["title"],
                    "cover_url": f"/api/v1/items/{item['item_id']}/cover",
                    "position": int(item["position"]),
                    "snapshot_position": item["snapshot_position"],
                    "source": item["source"],
                    "score": float(item["score"]),
                    "explanation": item["explanation"],
                    "model_version": item["model_version"],
                    "is_forced": bool(item["is_forced"]),
                }
                for item in exposures
            ],
            "events": [
                {
                    "event_id": event["event_id"],
                    "event_type": event["event_type"],
                    "item_id": str(event["item_id"]),
                    "position": int(event["position"]),
                    "client_timestamp": event["client_timestamp"],
                    "received_at": event["received_at"],
                }
                for event in event_rows
            ],
        }

    @app.get("/api/v1/admin/users")
    def admin_users(request: Request) -> dict[str, Any]:
        admin_user(request)
        with database.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM users ORDER BY role, username"
            ).fetchall()
        return {"users": [_user_payload(row) for row in rows]}

    @app.get("/api/v1/admin/users/{user_id}/debug")
    def admin_user_debug(user_id: str, request: Request) -> dict[str, Any]:
        admin_user(request)
        with database.connect() as conn:
            user = conn.execute(
                "SELECT * FROM users WHERE id = ? OR username = ? COLLATE NOCASE",
                (user_id, user_id),
            ).fetchone()
            if user is None:
                raise APIError(404, "user_not_found", "User not found")
            resolved_user_id = user["id"]
            last_request = conn.execute(
                """
                SELECT request_id, feed_type, model_version, profile_version,
                       fallback_reason, returned_count, created_at, latency_ms
                FROM recommendation_requests WHERE user_id = ?
                ORDER BY created_at DESC LIMIT 1
                """,
                (resolved_user_id,),
            ).fetchone()
            profile = profile_payload(conn, resolved_user_id)
            events = recent_events(conn, resolved_user_id, 20)
        return {
            "user": _user_payload(user),
            "profile": profile,
            "recent_events": events,
            "last_request": dict(last_request) if last_request else None,
        }

    @app.get("/api/v1/admin/items")
    def admin_items(
        request: Request,
        q: str | None = Query(default=None, max_length=100),
        status: str | None = Query(default=None),
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        require_role(request, "analyst")
        if status not in {None, "online", "offline"}:
            raise APIError(422, "invalid_status", "status must be online or offline")
        where = []
        params: list[Any] = []
        if q:
            where.append("(item_id LIKE ? OR title LIKE ?)")
            params.extend([f"%{q}%", f"%{q}%"])
        if status:
            where.append("status = ?")
            params.append(status)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        with database.connect() as conn:
            total = int(conn.execute(f"SELECT COUNT(*) FROM items {clause}", params).fetchone()[0])
            rows = conn.execute(
                f"""
                SELECT * FROM items {clause}
                ORDER BY popularity_score DESC, item_id LIMIT ? OFFSET ?
                """,
                (*params, limit, offset),
            ).fetchall()
        return {
            "items": [_item_payload(row) for row in rows],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    def apply_item_status(
        conn: sqlite3.Connection,
        *,
        item: sqlite3.Row,
        status: str,
        now: str,
    ) -> tuple[dict[str, Any], dict[str, Any], str]:
        before = _item_payload(item)
        if item["status"] != status:
            conn.execute(
                """
                UPDATE items
                SET status = ?, status_version = status_version + 1, updated_at = ?
                WHERE item_id = ?
                """,
                (status, now, item["item_id"]),
            )
            cache.delete(f"item:public:{item['item_id']}")
            if status == "offline":
                conn.execute(
                    """
                    UPDATE feed_snapshot_items
                    SET invalidated_at = ?
                    WHERE item_id = ? AND invalidated_at IS NULL
                    """,
                    (now, item["item_id"]),
                )
        updated = conn.execute(
            "SELECT * FROM items WHERE item_id = ?", (item["item_id"],)
        ).fetchone()
        after = _item_payload(updated)
        action = (
            "item_offline"
            if status == "offline" and before["status"] != status
            else "item_restore"
            if status == "online" and before["status"] != status
            else "item_status_noop"
        )
        return before, after, action

    @app.patch("/api/v1/admin/items/batch/status")
    def set_item_status_batch(
        body: BatchItemStatusBody,
        request: Request,
    ) -> dict[str, Any]:
        admin = require_role(request, "operator")
        now = isoformat()
        item_ids = sorted(body.item_ids)
        request_payload = {
            "item_ids": item_ids,
            "status": body.status,
            "reason": body.reason,
        }
        request_json = json.dumps(request_payload, sort_keys=True, separators=(",", ":"))
        with database.transaction(immediate=True) as conn:
            existing = conn.execute(
                """
                SELECT * FROM operation_batches
                WHERE admin_user_id = ? AND idempotency_key = ?
                """,
                (admin["id"], body.idempotency_key),
            ).fetchone()
            if existing is not None:
                if existing["request_json"] != request_json:
                    raise APIError(
                        409,
                        "idempotency_conflict",
                        "Idempotency key was already used for a different batch request",
                    )
                result = json.loads(existing["result_json"])
                result["idempotent_replay"] = True
                return result

            placeholders = ",".join("?" for _ in item_ids)
            rows = conn.execute(
                f"SELECT * FROM items WHERE item_id IN ({placeholders})",
                item_ids,
            ).fetchall()
            by_id = {str(row["item_id"]): row for row in rows}
            missing = [item_id for item_id in item_ids if item_id not in by_id]
            if missing:
                raise APIError(
                    404,
                    "batch_items_not_found",
                    "Batch contains unknown items; no changes were applied",
                    missing,
                )

            batch_id = str(uuid.uuid4())
            results = []
            changed_count = 0
            for item_id in item_ids:
                before, after, action = apply_item_status(
                    conn,
                    item=by_id[item_id],
                    status=body.status,
                    now=now,
                )
                changed_count += int(before["status"] != after["status"])
                cursor = conn.execute(
                    """
                    INSERT INTO operations(
                        admin_user_id, action, item_id, target_id, batch_id, reason,
                        before_json, after_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        admin["id"],
                        action,
                        item_id,
                        item_id,
                        batch_id,
                        body.reason,
                        json.dumps(before, sort_keys=True),
                        json.dumps(after, sort_keys=True),
                        now,
                    ),
                )
                results.append(
                    {
                        "item_id": item_id,
                        "status": after["status"],
                        "changed": before["status"] != after["status"],
                        "operation_id": int(cursor.lastrowid),
                    }
                )
            result = {
                "batch_id": batch_id,
                "status": body.status,
                "success_count": len(results),
                "failure_count": 0,
                "changed_count": changed_count,
                "items": results,
                "idempotent_replay": False,
            }
            conn.execute(
                """
                INSERT INTO operation_batches(
                    batch_id, admin_user_id, idempotency_key,
                    request_json, result_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    batch_id,
                    admin["id"],
                    body.idempotency_key,
                    request_json,
                    json.dumps(result, sort_keys=True),
                    now,
                ),
            )
        return result

    @app.patch("/api/v1/admin/items/{item_id}/status")
    def set_item_status(item_id: str, body: ItemStatusBody, request: Request) -> dict[str, Any]:
        admin = require_role(request, "operator")
        now = isoformat()
        with database.transaction(immediate=True) as conn:
            item = conn.execute("SELECT * FROM items WHERE item_id = ?", (item_id,)).fetchone()
            if item is None:
                raise APIError(404, "item_not_found", "Item not found")
            before, after, action = apply_item_status(
                conn,
                item=item,
                status=body.status,
                now=now,
            )
            cursor = conn.execute(
                """
                INSERT INTO operations(
                    admin_user_id, action, item_id, target_id, reason,
                    before_json, after_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    admin["id"],
                    action,
                    item_id,
                    item_id,
                    body.reason,
                    json.dumps(before, sort_keys=True),
                    json.dumps(after, sort_keys=True),
                    now,
                ),
            )
            operation_id = int(cursor.lastrowid)
        return {"item": after, "operation_id": operation_id}

    def boost_payload(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "item_id": str(row["item_id"]),
            "audience": row["audience"],
            "user_ids": json.loads(row["user_ids_json"]),
            "feed_types": json.loads(row["feed_types_json"]),
            "position": int(row["position"]),
            "priority": int(row["priority"]),
            "starts_at": row["starts_at"],
            "ends_at": row["ends_at"],
            "reason": row["reason"],
            "active": bool(row["active"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    @app.post("/api/v1/admin/boosts")
    def create_boost(body: BoostBody, request: Request) -> dict[str, Any]:
        admin = require_role(request, "operator")
        now = isoformat()
        boost_id = str(uuid.uuid4())
        starts_at = _canonical_datetime(body.starts_at)
        ends_at = _canonical_datetime(body.ends_at)
        with database.transaction(immediate=True) as conn:
            item = conn.execute("SELECT * FROM items WHERE item_id = ?", (body.item_id,)).fetchone()
            if item is None:
                raise APIError(404, "item_not_found", "Item not found")
            if body.user_ids:
                placeholders = ",".join("?" for _ in body.user_ids)
                found = {
                    row["id"]
                    for row in conn.execute(
                        f"SELECT id FROM users WHERE is_active = 1 AND id IN ({placeholders})",
                        tuple(body.user_ids),
                    )
                }
                missing = sorted(set(body.user_ids) - found)
                if missing:
                    raise APIError(422, "unknown_boost_users", "Boost contains unknown users", missing)
            conn.execute(
                """
                INSERT INTO boost_campaigns(
                    id, item_id, audience, user_ids_json, feed_types_json,
                    position, priority, starts_at, ends_at, reason, active,
                    created_by, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
                """,
                (
                    boost_id,
                    body.item_id,
                    body.audience,
                    json.dumps(body.user_ids),
                    json.dumps(body.feed_types),
                    body.position,
                    body.priority,
                    starts_at,
                    ends_at,
                    body.reason,
                    admin["id"],
                    now,
                    now,
                ),
            )
            boost = conn.execute("SELECT * FROM boost_campaigns WHERE id = ?", (boost_id,)).fetchone()
            after = boost_payload(boost)
            cursor = conn.execute(
                """
                INSERT INTO operations(
                    admin_user_id, action, item_id, target_id, reason,
                    before_json, after_json, created_at
                ) VALUES (?, 'boost_create', ?, ?, ?, '{}', ?, ?)
                """,
                (
                    admin["id"],
                    body.item_id,
                    boost_id,
                    body.reason,
                    json.dumps(after, sort_keys=True),
                    now,
                ),
            )
            operation_id = int(cursor.lastrowid)
        return {"boost": after, "operation_id": operation_id}

    @app.get("/api/v1/admin/boosts")
    def list_boosts(request: Request) -> dict[str, Any]:
        require_role(request, "analyst")
        with database.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM boost_campaigns ORDER BY created_at DESC"
            ).fetchall()
        return {"boosts": [boost_payload(row) for row in rows]}

    @app.patch("/api/v1/admin/boosts/{boost_id}")
    def set_boost_status(
        boost_id: str,
        body: BoostStatusBody,
        request: Request,
    ) -> dict[str, Any]:
        admin = require_role(request, "operator")
        now = isoformat()
        with database.transaction(immediate=True) as conn:
            boost = conn.execute("SELECT * FROM boost_campaigns WHERE id = ?", (boost_id,)).fetchone()
            if boost is None:
                raise APIError(404, "boost_not_found", "Boost campaign not found")
            before = boost_payload(boost)
            conn.execute(
                "UPDATE boost_campaigns SET active = ?, updated_at = ? WHERE id = ?",
                (int(body.active), now, boost_id),
            )
            updated = conn.execute("SELECT * FROM boost_campaigns WHERE id = ?", (boost_id,)).fetchone()
            after = boost_payload(updated)
            cursor = conn.execute(
                """
                INSERT INTO operations(
                    admin_user_id, action, item_id, target_id, reason,
                    before_json, after_json, created_at
                ) VALUES (?, 'boost_status', ?, ?, ?, ?, ?, ?)
                """,
                (
                    admin["id"],
                    updated["item_id"],
                    boost_id,
                    body.reason,
                    json.dumps(before, sort_keys=True),
                    json.dumps(after, sort_keys=True),
                    now,
                ),
            )
            operation_id = int(cursor.lastrowid)
        return {"boost": after, "operation_id": operation_id}

    @app.get("/api/v1/admin/operations")
    def admin_operations(
        request: Request,
        limit: int = Query(default=100, ge=1, le=500),
    ) -> dict[str, Any]:
        require_role(request, "analyst")
        with database.connect() as conn:
            rows = conn.execute(
                """
                SELECT o.*, u.username AS admin_username
                FROM operations o JOIN users u ON u.id = o.admin_user_id
                ORDER BY o.operation_id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return {
            "operations": [
                {
                    "operation_id": int(row["operation_id"]),
                    "admin_user_id": row["admin_user_id"],
                    "admin_username": row["admin_username"],
                    "action": row["action"],
                    "item_id": str(row["item_id"]) if row["item_id"] is not None else None,
                    "target_id": row["target_id"],
                    "batch_id": row["batch_id"],
                    "reason": row["reason"],
                    "before": json.loads(row["before_json"]),
                    "after": json.loads(row["after_json"]),
                    "created_at": row["created_at"],
                }
                for row in rows
            ]
        }

    @app.get("/api/v1/admin/models")
    def admin_models(request: Request) -> dict[str, Any]:
        require_role(request, "analyst")
        artifact = artifacts.get()
        with database.transaction(immediate=True) as conn:
            ensure_model(conn, artifact)
            rows = conn.execute(
                "SELECT * FROM model_versions ORDER BY created_at DESC"
            ).fetchall()
            runs = conn.execute(
                "SELECT * FROM training_runs ORDER BY created_at DESC LIMIT 100"
            ).fetchall()

        def model_payload(row: sqlite3.Row) -> dict[str, Any]:
            return {
                "model_version": row["model_version"],
                "data_version": row["data_version"],
                "algorithm": row["algorithm"],
                "artifact_path": row["artifact_path"],
                "status": row["status"],
                "training_status": row["training_status"],
                "publish_status": row["publish_status"],
                "training_window": {
                    "start": row["training_window_start"],
                    "end": row["training_window_end"],
                },
                "sample_count": row["sample_count"],
                "event_count": row["event_count"],
                "metrics": json.loads(row["metrics_json"]),
                "evaluation_protocol": json.loads(row["evaluation_protocol_json"]),
                "created_at": row["created_at"],
                "activated_at": row["activated_at"],
                "is_current": artifact is not None
                and row["model_version"] == artifact.model_version,
            }

        return {
            "models": [model_payload(row) for row in rows],
            "training_runs": [
                {
                    "run_id": row["run_id"],
                    "data_version": row["data_version"],
                    "model_version": row["model_version"],
                    "window": {"start": row["window_start"], "end": row["window_end"]},
                    "event_count": int(row["event_count"]),
                    "sample_count": int(row["sample_count"]),
                    "training_status": row["training_status"],
                    "publish_status": row["publish_status"],
                    "error": row["error"],
                    "created_at": row["created_at"],
                    "completed_at": row["completed_at"],
                }
                for row in runs
            ],
            "current_model_version": artifact.model_version if artifact else None,
            "load_error": artifacts.last_error,
        }

    @app.get("/api/v1/admin/models/compare")
    def compare_models(
        request: Request,
        versions: list[str] = Query(default=[]),
    ) -> dict[str, Any]:
        require_role(request, "analyst")
        unique_versions = list(dict.fromkeys(versions))
        if len(unique_versions) < 2 or len(unique_versions) > 10:
            raise APIError(
                422,
                "invalid_model_selection",
                "Select between 2 and 10 unique model versions",
            )
        placeholders = ",".join("?" for _ in unique_versions)
        with database.connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM model_versions WHERE model_version IN ({placeholders})",
                unique_versions,
            ).fetchall()
        by_version = {row["model_version"]: row for row in rows}
        missing = [version for version in unique_versions if version not in by_version]
        if missing:
            raise APIError(
                404,
                "model_version_not_found",
                "One or more model versions do not exist",
                missing,
            )

        protocols = [
            json.loads(by_version[version]["evaluation_protocol_json"])
            for version in unique_versions
        ]
        protocol_method_fields = (
            "candidate_universe",
            "cohort_aggregation",
            "k",
            "negative_sampling",
            "negatives_per_query",
            "positive_labels",
            "popularity_scope",
            "popularity_feature_rule",
            "shared_queries_across_models",
            "untimed_likes_views_used",
        )
        protocol_keys = [
            json.dumps(
                {key: value.get(key) for key in protocol_method_fields},
                sort_keys=True,
                separators=(",", ":"),
            )
            for value in protocols
        ]
        compatible = len(set(protocol_keys)) == 1 and bool(protocols[0])
        baseline_version = unique_versions[0]

        def offline_metrics(row: sqlite3.Row) -> dict[str, float]:
            metrics = json.loads(row["metrics_json"])
            values = (((metrics.get("test") or {}).get("models") or {}).get("svd") or {})
            return {
                key: float(value)
                for key, value in values.items()
                if isinstance(value, (int, float))
            }

        baseline_metrics = offline_metrics(by_version[baseline_version])
        compared = []
        for version in unique_versions:
            row = by_version[version]
            metrics = offline_metrics(row)
            deltas = (
                {
                    key: metrics[key] - baseline_metrics[key]
                    for key in sorted(metrics.keys() & baseline_metrics.keys())
                }
                if compatible
                else None
            )
            compared.append(
                {
                    "model_version": version,
                    "data_version": row["data_version"],
                    "algorithm": row["algorithm"],
                    "training_window": {
                        "start": row["training_window_start"],
                        "end": row["training_window_end"],
                    },
                    "sample_count": row["sample_count"],
                    "event_count": row["event_count"],
                    "training_status": row["training_status"],
                    "publish_status": row["publish_status"],
                    "status": row["status"],
                    "is_current": row["status"] == "active",
                    "trained_at": row["created_at"],
                    "metrics": metrics,
                    "deltas_from_baseline": deltas,
                }
            )
        return {
            "baseline_version": baseline_version,
            "protocol_compatible": compatible,
            "compatibility_reason": (
                None
                if compatible
                else "Evaluation protocols are missing or differ; metric deltas are suppressed"
            ),
            "evaluation_protocol": protocols[0] if compatible else None,
            "models": compared,
        }

    @app.get("/api/v1/admin/alerts/metrics")
    def alert_metrics(request: Request) -> dict[str, Any]:
        require_role(request, "analyst")
        return {
            "metrics": [
                {"name": name, "label": spec["label"], "unit": spec["unit"]}
                for name, spec in ALERT_METRICS.items()
            ]
        }

    @app.get("/api/v1/admin/alerts/rules")
    def alert_rules(request: Request) -> dict[str, Any]:
        require_role(request, "analyst")
        with database.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM alert_rules ORDER BY created_at"
            ).fetchall()
        return {"rules": [_alert_rule_payload(row) for row in rows]}

    @app.post("/api/v1/admin/alerts/rules", status_code=201)
    def create_alert_rule(request: Request, body: AlertRuleBody) -> JSONResponse:
        admin = require_role(request, "operator")
        if body.metric not in ALERT_METRICS:
            raise APIError(
                422,
                "invalid_metric",
                f"Unknown alert metric: {body.metric}",
                {"available": list(ALERT_METRICS)},
            )
        now = isoformat()
        rule_id = str(uuid.uuid4())
        with database.connect() as conn:
            conn.execute(
                """
                INSERT INTO alert_rules(
                    id, name, metric, operator, threshold, severity,
                    enabled, created_by, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
                """,
                (
                    rule_id,
                    body.name,
                    body.metric,
                    body.operator,
                    body.threshold,
                    body.severity,
                    admin["id"],
                    now,
                    now,
                ),
            )
            rule = conn.execute(
                "SELECT * FROM alert_rules WHERE id = ?", (rule_id,)
            ).fetchone()
        return JSONResponse(status_code=201, content=_alert_rule_payload(rule))

    @app.patch("/api/v1/admin/alerts/rules/{rule_id}")
    def update_alert_rule(
        request: Request, rule_id: str, body: AlertRuleUpdateBody
    ) -> dict[str, Any]:
        require_role(request, "operator")
        updates: dict[str, Any] = {}
        if body.name is not None:
            updates["name"] = body.name
        if body.operator is not None:
            updates["operator"] = body.operator
        if body.threshold is not None:
            updates["threshold"] = body.threshold
        if body.severity is not None:
            updates["severity"] = body.severity
        if body.enabled is not None:
            updates["enabled"] = 1 if body.enabled else 0
        if not updates:
            raise APIError(422, "no_updates", "No fields to update")
        updates["updated_at"] = isoformat()
        assignments = ", ".join(f"{column} = ?" for column in updates)
        values = [*updates.values(), rule_id]
        with database.connect() as conn:
            cursor = conn.execute(
                f"UPDATE alert_rules SET {assignments} WHERE id = ?", values
            )
            if cursor.rowcount == 0:
                raise APIError(404, "alert_rule_not_found", "Alert rule does not exist")
            rule = conn.execute(
                "SELECT * FROM alert_rules WHERE id = ?", (rule_id,)
            ).fetchone()
        return _alert_rule_payload(rule)

    @app.delete("/api/v1/admin/alerts/rules/{rule_id}")
    def delete_alert_rule(request: Request, rule_id: str) -> dict[str, Any]:
        require_role(request, "operator")
        with database.connect() as conn:
            cursor = conn.execute("DELETE FROM alert_rules WHERE id = ?", (rule_id,))
            if cursor.rowcount == 0:
                raise APIError(404, "alert_rule_not_found", "Alert rule does not exist")
        return {"deleted": rule_id}

    @app.get("/api/v1/admin/alerts/events")
    def alert_events(
        request: Request,
        status: str | None = Query(default=None),
    ) -> dict[str, Any]:
        require_role(request, "analyst")
        with database.connect() as conn:
            summary = evaluate_alerts(
                conn, window_minutes=settings.alert_window_minutes
            )
            if status is not None:
                if status not in {"open", "resolved"}:
                    raise APIError(422, "invalid_status", "status must be open or resolved")
                rows = conn.execute(
                    "SELECT * FROM alert_events WHERE status = ? ORDER BY triggered_at DESC",
                    (status,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM alert_events ORDER BY triggered_at DESC"
                ).fetchall()
        return {
            "evaluation": summary,
            "events": [_alert_event_payload(row) for row in rows],
        }

    @app.post("/api/v1/admin/alerts/events/{event_id}/acknowledge")
    def acknowledge_alert_event(request: Request, event_id: str) -> dict[str, Any]:
        admin = require_role(request, "operator")
        with database.connect() as conn:
            cursor = conn.execute(
                """
                UPDATE alert_events
                SET acknowledged_by = ?, acknowledged_at = ?
                WHERE id = ? AND status = 'open'
                """,
                (admin["id"], isoformat(), event_id),
            )
            if cursor.rowcount == 0:
                raise APIError(
                    404, "alert_event_not_found", "Open alert event does not exist"
                )
            event = conn.execute(
                "SELECT * FROM alert_events WHERE id = ?", (event_id,)
            ).fetchone()
        return _alert_event_payload(event)

    @app.post("/api/v1/admin/alerts/evaluate")
    def evaluate_alerts_now(request: Request) -> dict[str, Any]:
        require_role(request, "operator")
        with database.connect() as conn:
            summary = evaluate_alerts(
                conn, window_minutes=settings.alert_window_minutes
            )
        return summary

    def _training_job_payload(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "job_id": row["job_id"],
            "run_id": row["run_id"],
            "status": row["status"],
            "window_start": row["window_start"],
            "window_end": row["window_end"],
            "config": json.loads(row["config_json"]),
            "error": row["error"],
            "created_by": row["created_by"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
        }

    def _execute_training_job(job_id: str, *, start_time: str, end_time: str, config: dict[str, Any]) -> None:
        with database.connect() as conn:
            conn.execute(
                "UPDATE training_jobs SET status = 'running', started_at = ? WHERE job_id = ?",
                (isoformat(), job_id),
            )
        try:
            result = run_online_retraining(
                settings=settings,
                start_time=start_time,
                end_time=end_time,
                base_processed_dir=Path(config["base_processed_dir"]),
                output_root=Path(config["output_root"]),
                artifacts_dir=settings.model_pointer.resolve().parent,
                mode=config["mode"],
                max_users=config["max_users"],
                max_eval_users=config["max_eval_users"],
                rank=config["rank"],
                seed=config["seed"],
            )
            with database.connect() as conn:
                conn.execute(
                    """
                    UPDATE training_jobs
                    SET status = 'succeeded', run_id = ?, completed_at = ?
                    WHERE job_id = ?
                    """,
                    (result["run_id"], isoformat(), job_id),
                )
        except Exception as exc:  # noqa: BLE001 - job failure is recorded, not raised
            with database.connect() as conn:
                conn.execute(
                    """
                    UPDATE training_jobs
                    SET status = 'failed', error = ?, completed_at = ?
                    WHERE job_id = ?
                    """,
                    (str(exc)[:2000], isoformat(), job_id),
                )

    @app.get("/api/v1/admin/training/jobs")
    def list_training_jobs(request: Request) -> dict[str, Any]:
        require_role(request, "analyst")
        with database.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM training_jobs ORDER BY created_at DESC"
            ).fetchall()
        return {"jobs": [_training_job_payload(row) for row in rows]}

    @app.get("/api/v1/admin/training/jobs/{job_id}")
    def get_training_job(request: Request, job_id: str) -> dict[str, Any]:
        require_role(request, "analyst")
        with database.connect() as conn:
            row = conn.execute(
                "SELECT * FROM training_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        if row is None:
            raise APIError(404, "training_job_not_found", "Training job does not exist")
        return _training_job_payload(row)

    @app.post("/api/v1/admin/training/jobs", status_code=202)
    def create_training_job(request: Request, body: TrainingJobBody) -> JSONResponse:
        admin = require_role(request, "operator")
        job_id = str(uuid.uuid4())
        now = isoformat()
        start_text = _canonical_datetime(body.start_time)
        end_text = _canonical_datetime(body.end_time)
        config = {
            "mode": body.mode,
            "max_users": body.max_users,
            "max_eval_users": body.max_eval_users,
            "rank": body.rank,
            "seed": body.seed,
            "base_processed_dir": body.base_processed_dir,
            "output_root": body.output_root,
        }
        with database.transaction(immediate=True) as conn:
            conn.execute(
                """
                INSERT INTO training_jobs(
                    job_id, status, window_start, window_end, config_json,
                    created_by, created_at
                ) VALUES (?, 'queued', ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    start_text,
                    end_text,
                    json.dumps(config, sort_keys=True),
                    admin["id"],
                    now,
                ),
            )
            job = conn.execute(
                "SELECT * FROM training_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        threading.Thread(
            target=_execute_training_job,
            args=(job_id,),
            kwargs={"start_time": start_text, "end_time": end_text, "config": config},
            daemon=True,
        ).start()
        return JSONResponse(status_code=202, content=_training_job_payload(job))

    web_root = Path(__file__).resolve().parent.parent / "web"
    index_path = web_root / "index.html"

    @app.get("/", include_in_schema=False)
    def web_index() -> Response:
        if not index_path.is_file():
            raise APIError(404, "web_not_built", "Web client is not available")
        return FileResponse(index_path)

    if web_root.is_dir():
        app.mount("/web", StaticFiles(directory=web_root), name="web")

    return app


app = create_app()
