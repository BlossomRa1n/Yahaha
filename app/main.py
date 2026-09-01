from __future__ import annotations

import base64
import json
import secrets
import sqlite3
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

from .artifacts import ArtifactStore, ModelArtifact
from .config import Settings
from .db import Database
from .recommendation import RecommendationService
from .schemas import BoostBody, BoostStatusBody, EventBatch, ItemStatusBody, LoginBody
from .security import (
    decode_cursor,
    encode_cursor,
    isoformat,
    issue_session_token,
    session_digest,
    utc_now,
    verify_password,
)


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


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    database = Database(settings.database_path)
    artifacts = ArtifactStore(settings.model_pointer)
    recommender = RecommendationService(artifacts)
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        database.initialize()
        yield

    app = FastAPI(
        title="MicroLens Recommendation MVP",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.database = database
    app.state.artifacts = artifacts
    app.state.recommender = recommender

    @app.middleware("http")
    async def request_context(request: Request, call_next: Any) -> Response:
        request.state.api_request_id = str(uuid.uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.api_request_id
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

    def admin_user(request: Request) -> sqlite3.Row:
        user = current_user(request)
        if user["role"] != "admin":
            raise APIError(403, "forbidden", "Administrator role required")
        return user

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
            SELECT event_id, event_type, request_id, item_id, position,
                   client_timestamp, received_at
            FROM events WHERE user_id = ?
            ORDER BY received_at DESC, rowid DESC LIMIT ?
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
                "client_timestamp": row["client_timestamp"],
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
        page = 0
        seed = secrets.randbelow(2**31)
        if cursor:
            try:
                cursor_payload = decode_cursor(cursor, settings.app_secret)
            except (ValueError, json.JSONDecodeError) as exc:
                raise APIError(422, "invalid_cursor", "Cursor is invalid") from exc
            if cursor_payload.get("user_id") != user["id"] or cursor_payload.get("feed_type") != feed_type:
                raise APIError(422, "invalid_cursor", "Cursor does not belong to this user and feed")
            page = int(cursor_payload.get("page", 0))
            seed = int(cursor_payload.get("seed", seed))

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
            artifact = artifacts.get()
            ensure_model(conn, artifact)
            result = recommender.recommend(
                conn,
                user=fresh_user,
                feed_type=feed_type,
                limit=limit,
                seed=seed,
                include_boosts=page == 0,
                now=created_at,
            )
            serving_version = result.model_version or "fallback-popularity-v1"
            candidate_ids = [candidate.item_id for candidate in result.candidates]
            item_rows: dict[str, sqlite3.Row] = {}
            if candidate_ids:
                placeholders = ",".join("?" for _ in candidate_ids)
                item_rows = {
                    str(row["item_id"]): row
                    for row in conn.execute(
                        f"SELECT * FROM items WHERE status = 'online' AND item_id IN ({placeholders})",
                        tuple(candidate_ids),
                    )
                }
            persisted = [candidate for candidate in result.candidates if candidate.item_id in item_rows]
            conn.execute(
                """
                INSERT INTO recommendation_requests(
                    request_id, user_id, feed_type, model_version, profile_version,
                    cursor, fallback_reason, returned_count, latency_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                """,
                (
                    feed_request_id,
                    user["id"],
                    feed_type,
                    serving_version,
                    int(profile["version"]),
                    cursor,
                    result.fallback_reason,
                    len(persisted),
                    created_at,
                ),
            )
            response_items = []
            for position, candidate in enumerate(persisted):
                item = item_rows[candidate.item_id]
                item_model_version = candidate.model_version or serving_version
                conn.execute(
                    """
                    INSERT INTO exposures(
                        request_id, user_id, item_id, position, source, score,
                        explanation, model_version, is_forced, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        feed_request_id,
                        user["id"],
                        candidate.item_id,
                        position,
                        candidate.source,
                        float(candidate.score),
                        candidate.explanation,
                        item_model_version,
                        int(candidate.is_forced),
                        created_at,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO events(
                        event_id, event_type, request_id, user_id, item_id,
                        position, client_timestamp, received_at
                    ) VALUES (?, 'impression', ?, ?, ?, ?, NULL, ?)
                    """,
                    (
                        f"imp:{feed_request_id}:{position}",
                        feed_request_id,
                        user["id"],
                        candidate.item_id,
                        position,
                        created_at,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO user_item_state(
                        user_id, item_id, exposure_count, last_event_at
                    ) VALUES (?, ?, 1, ?)
                    ON CONFLICT(user_id, item_id) DO UPDATE SET
                        exposure_count = exposure_count + 1,
                        last_event_at = excluded.last_event_at
                    """,
                    (user["id"], candidate.item_id, created_at),
                )
                response_items.append(
                    {
                        "item_id": candidate.item_id,
                        "title": item["title"],
                        "cover_url": f"/api/v1/items/{candidate.item_id}/cover",
                        "position": position,
                        "source": candidate.source,
                        "score": float(candidate.score),
                        "explanation": candidate.explanation,
                        "model_version": item_model_version,
                        "is_forced": candidate.is_forced,
                    }
                )
            latency_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
            conn.execute(
                "UPDATE recommendation_requests SET latency_ms = ? WHERE request_id = ?",
                (latency_ms, feed_request_id),
            )

        next_cursor = None
        if result.has_more and response_items:
            next_cursor = encode_cursor(
                {
                    "user_id": user["id"],
                    "feed_type": feed_type,
                    "page": page + 1,
                    "seed": seed,
                },
                settings.app_secret,
            )
        return {
            "request_id": feed_request_id,
            "feed_type": feed_type,
            "model_version": serving_version,
            "profile_version": int(profile["version"]),
            "fallback_reason": result.fallback_reason,
            "next_cursor": next_cursor,
            "has_more": bool(next_cursor),
            "items": response_items,
        }

    @app.post("/api/v1/events/batch")
    def post_events(body: EventBatch, request: Request) -> dict[str, Any]:
        user = current_user(request)
        accepted = 0
        duplicates = 0
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
                    SELECT event_id FROM events
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
                    duplicates += 1
                    continue

                client_timestamp = _canonical_datetime(event.client_timestamp)
                conn.execute(
                    """
                    INSERT INTO events(
                        event_id, event_type, request_id, user_id, item_id,
                        position, client_timestamp, received_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.event_id,
                        normalized_type,
                        event.request_id,
                        user["id"],
                        event.item_id,
                        event.position,
                        client_timestamp,
                        received_at,
                    ),
                )
                click_delta = 1 if normalized_type == "click" else 0
                like_delta = 1 if normalized_type == "like" else 0
                negative = 1 if normalized_type == "not_interested" else 0
                affinity_delta = {"click": 1.0, "like": 3.0, "not_interested": -4.0}[normalized_type]
                conn.execute(
                    """
                    INSERT INTO user_item_state(
                        user_id, item_id, click_count, like_count,
                        not_interested, affinity, last_event_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id, item_id) DO UPDATE SET
                        click_count = click_count + excluded.click_count,
                        like_count = like_count + excluded.like_count,
                        not_interested = MAX(not_interested, excluded.not_interested),
                        affinity = affinity + excluded.affinity,
                        last_event_at = excluded.last_event_at
                    """,
                    (
                        user["id"],
                        event.item_id,
                        click_delta,
                        like_delta,
                        negative,
                        affinity_delta,
                        received_at,
                    ),
                )
                accepted += 1
            if accepted:
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
        with database.connect() as conn:
            item = conn.execute(
                "SELECT * FROM items WHERE item_id = ? AND status = 'online'",
                (item_id,),
            ).fetchone()
        if item is None:
            raise APIError(404, "item_not_found", "Item does not exist or is offline")
        return {"item": _item_payload(item, public=True)}

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
        admin_user(request)
        end = _parse_datetime(to_value, utc_now())
        start = _parse_datetime(from_value, end - timedelta(hours=24))
        if start >= end:
            raise APIError(422, "invalid_time_range", "from must be earlier than to")
        start_text, end_text = isoformat(start), isoformat(end)
        with database.connect() as conn:
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
            clicks = int(
                conn.execute(
                    "SELECT COUNT(*) FROM events WHERE event_type = 'click' AND received_at >= ? AND received_at < ?",
                    (start_text, end_text),
                ).fetchone()[0]
            )
            likes = int(
                conn.execute(
                    "SELECT COUNT(*) FROM events WHERE event_type = 'like' AND received_at >= ? AND received_at < ?",
                    (start_text, end_text),
                ).fetchone()[0]
            )
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
                SELECT r.feed_type,
                       COUNT(DISTINCT r.request_id) AS requests,
                       COUNT(DISTINCT e.id) AS exposures,
                       COUNT(DISTINCT CASE WHEN ev.event_type = 'click' THEN ev.event_id END) AS clicks
                FROM recommendation_requests r
                LEFT JOIN exposures e ON e.request_id = r.request_id
                LEFT JOIN events ev ON ev.request_id = e.request_id
                                   AND ev.item_id = e.item_id
                                   AND ev.position = e.position
                WHERE r.created_at >= ? AND r.created_at < ?
                GROUP BY r.feed_type
                """,
                (start_text, end_text),
            ).fetchall()
            feed_values = {
                row["feed_type"]: {
                    "feed_type": row["feed_type"],
                    "requests": int(row["requests"]),
                    "exposures": int(row["exposures"]),
                    "clicks": int(row["clicks"]),
                }
                for row in feed_rows
            }
            top_rows = conn.execute(
                """
                SELECT i.item_id, i.title,
                       COUNT(DISTINCT e.id) AS exposures,
                       COUNT(DISTINCT CASE WHEN ev.event_type = 'click' THEN ev.event_id END) AS clicks,
                       COUNT(DISTINCT CASE WHEN ev.event_type = 'like' THEN ev.event_id END) AS likes
                FROM exposures e
                JOIN items i ON i.item_id = e.item_id
                LEFT JOIN events ev ON ev.request_id = e.request_id
                                   AND ev.item_id = e.item_id
                                   AND ev.position = e.position
                WHERE e.created_at >= ? AND e.created_at < ?
                GROUP BY i.item_id, i.title
                ORDER BY exposures DESC, clicks DESC, i.item_id
                LIMIT 10
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
            "clicks": clicks,
            "likes": likes,
            "ctr": clicks / exposures_count if exposures_count else 0.0,
            "offline_items": offline_items,
            "active_boosts": active_boosts,
            "current_model_version": artifact.model_version if artifact else None,
            "feed_breakdown": [
                feed_values.get(
                    feed_type,
                    {"feed_type": feed_type, "requests": 0, "exposures": 0, "clicks": 0},
                )
                for feed_type in ("personalized", "popular", "explore")
            ],
            "top_items": [
                {
                    "item_id": str(row["item_id"]),
                    "title": row["title"],
                    "exposures": int(row["exposures"]),
                    "clicks": int(row["clicks"]),
                    "likes": int(row["likes"]),
                    "ctr": int(row["clicks"]) / int(row["exposures"]) if row["exposures"] else 0.0,
                }
                for row in top_rows
            ],
        }

    @app.get("/api/v1/admin/requests/{feed_request_id}")
    def admin_request_detail(feed_request_id: str, request: Request) -> dict[str, Any]:
        admin_user(request)
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
        admin_user(request)
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

    @app.patch("/api/v1/admin/items/{item_id}/status")
    def set_item_status(item_id: str, body: ItemStatusBody, request: Request) -> dict[str, Any]:
        admin = admin_user(request)
        now = isoformat()
        with database.transaction(immediate=True) as conn:
            item = conn.execute("SELECT * FROM items WHERE item_id = ?", (item_id,)).fetchone()
            if item is None:
                raise APIError(404, "item_not_found", "Item not found")
            before = _item_payload(item)
            if item["status"] != body.status:
                conn.execute(
                    """
                    UPDATE items
                    SET status = ?, status_version = status_version + 1, updated_at = ?
                    WHERE item_id = ?
                    """,
                    (body.status, now, item_id),
                )
            updated = conn.execute("SELECT * FROM items WHERE item_id = ?", (item_id,)).fetchone()
            after = _item_payload(updated)
            action = (
                "item_offline"
                if body.status == "offline"
                else "item_restore"
                if before["status"] != body.status
                else "item_status_noop"
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
        admin = admin_user(request)
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
        admin_user(request)
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
        admin = admin_user(request)
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
        admin_user(request)
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
        admin_user(request)
        artifact = artifacts.get()
        with database.transaction(immediate=True) as conn:
            ensure_model(conn, artifact)
            rows = conn.execute(
                "SELECT * FROM model_versions ORDER BY created_at DESC"
            ).fetchall()
        return {
            "models": [
                {
                    "model_version": row["model_version"],
                    "data_version": row["data_version"],
                    "algorithm": row["algorithm"],
                    "artifact_path": row["artifact_path"],
                    "status": row["status"],
                    "metrics": json.loads(row["metrics_json"]),
                    "created_at": row["created_at"],
                    "activated_at": row["activated_at"],
                }
                for row in rows
            ],
            "current_model_version": artifact.model_version if artifact else None,
            "load_error": artifacts.last_error,
        }

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
