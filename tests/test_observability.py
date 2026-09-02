from __future__ import annotations

import io
import json
import logging
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from app.logging import JsonFormatter
from app.main import _percentiles
from tests.test_api import _client_event, _login, api_env


def _point_values(payload: dict) -> dict[str, int]:
    return {point["t"]: point["value"] for point in payload["points"]}


def test_dashboard_timeseries_requires_admin_and_validates_query(api_env) -> None:
    app, _, _ = api_env
    with TestClient(app) as anonymous, TestClient(app) as user, TestClient(app) as admin:
        assert anonymous.get("/api/v1/admin/dashboard/timeseries").status_code == 401
        _login(user, "alice")
        forbidden = user.get("/api/v1/admin/dashboard/timeseries")
        assert forbidden.status_code == 403
        assert forbidden.json()["error"]["code"] == "forbidden"

        _login(admin, "admin", "admin-pass")
        invalid_metric = admin.get(
            "/api/v1/admin/dashboard/timeseries",
            params={"metric": "ctr"},
        )
        assert invalid_metric.status_code == 422
        assert invalid_metric.json()["error"]["code"] == "invalid_metric"

        invalid_timestamp = admin.get(
            "/api/v1/admin/dashboard/timeseries",
            params={"from": "not-a-timestamp"},
        )
        assert invalid_timestamp.status_code == 422
        assert invalid_timestamp.json()["error"]["code"] == "invalid_time_range"

        reversed_range = admin.get(
            "/api/v1/admin/dashboard/timeseries",
            params={
                "from": "2026-01-02T00:00:00Z",
                "to": "2026-01-01T00:00:00Z",
            },
        )
        assert reversed_range.status_code == 422
        assert reversed_range.json()["error"]["code"] == "invalid_time_range"

        excessive_range = admin.get(
            "/api/v1/admin/dashboard/timeseries",
            params={
                "from": "2025-01-01T00:00:00Z",
                "to": "2026-01-03T00:00:00Z",
            },
        )
        assert excessive_range.status_code == 422
        assert excessive_range.json()["error"]["code"] == "invalid_time_range"


def test_dashboard_timeseries_aggregates_all_metrics_from_database(api_env) -> None:
    app, database, _ = api_env
    with TestClient(app) as user, TestClient(app) as admin:
        _login(user, "alice")
        _login(admin, "admin", "admin-pass")

        first_feed = user.get("/api/v1/feeds/popular?limit=2").json()
        second_feed = user.get("/api/v1/feeds/explore?limit=2").json()
        events = [
            _client_event(first_feed, "click", "observability-click-1"),
            _client_event(first_feed, "favorite", "observability-like-1"),
            _client_event(second_feed, "click", "observability-click-2"),
        ]
        for event in events:
            response = user.post("/api/v1/events/batch", json={"events": [event]})
            assert response.status_code == 200, response.text

        request_times = {
            first_feed["request_id"]: "2026-01-01T00:15:00.000Z",
            second_feed["request_id"]: "2026-01-01T02:15:00.000Z",
        }
        with database.transaction(immediate=True) as conn:
            for request_id, timestamp in request_times.items():
                conn.execute(
                    "UPDATE recommendation_requests SET created_at = ? WHERE request_id = ?",
                    (timestamp, request_id),
                )
                conn.execute(
                    "UPDATE exposures SET created_at = ? WHERE request_id = ?",
                    (timestamp, request_id),
                )
                conn.execute(
                    "UPDATE events SET received_at = ? WHERE request_id = ?",
                    (timestamp, request_id),
                )

        expected = {
            "requests": {"2026-01-01T00:00:00Z": 1, "2026-01-01T02:00:00Z": 1},
            "exposures": {"2026-01-01T00:00:00Z": 2, "2026-01-01T02:00:00Z": 2},
            "clicks": {"2026-01-01T00:00:00Z": 1, "2026-01-01T02:00:00Z": 1},
            "likes": {"2026-01-01T00:00:00Z": 1},
        }
        for metric, nonzero_points in expected.items():
            response = admin.get(
                "/api/v1/admin/dashboard/timeseries",
                params={
                    "metric": metric,
                    "from": "2026-01-01T00:00:00Z",
                    "to": "2026-01-01T03:30:00Z",
                },
            )
            assert response.status_code == 200, response.text
            payload = response.json()
            assert payload["metric"] == metric
            assert payload["bucket"] == "hour"
            assert sum(point["value"] for point in payload["points"]) == sum(
                nonzero_points.values()
            )
            values = _point_values(payload)
            for timestamp, value in nonzero_points.items():
                assert values[timestamp] == value

        daily = admin.get(
            "/api/v1/admin/dashboard/timeseries",
            params={
                "metric": "requests",
                "from": "2025-12-31T00:00:00Z",
                "to": "2026-01-03T12:00:00Z",
            },
        )
        assert daily.status_code == 200
        assert daily.json()["bucket"] == "day"
        assert _point_values(daily.json())["2026-01-01T00:00:00Z"] == 2

        exact_boundary = admin.get(
            "/api/v1/admin/dashboard/timeseries",
            params={
                "metric": "requests",
                "from": "2026-01-01T00:00:00Z",
                "to": "2026-01-01T03:00:00Z",
            },
        )
        assert exact_boundary.status_code == 200
        assert [point["t"] for point in exact_boundary.json()["points"]] == [
            "2026-01-01T00:00:00Z",
            "2026-01-01T01:00:00Z",
            "2026-01-01T02:00:00Z",
        ]


def test_dashboard_latency_percentiles_are_interpolated_from_request_rows(api_env) -> None:
    app, database, _ = api_env
    with TestClient(app) as user, TestClient(app) as admin:
        _login(user, "alice")
        _login(admin, "admin", "admin-pass")
        request_ids = [
            user.get(f"/api/v1/feeds/{feed_type}?limit=1").json()["request_id"]
            for feed_type in ("popular", "explore", "personalized", "popular", "explore")
        ]
        with database.transaction(immediate=True) as conn:
            for request_id, latency in zip(request_ids, (0.0, 10.0, 20.0, 30.0, 40.0), strict=True):
                conn.execute(
                    "UPDATE recommendation_requests SET latency_ms = ? WHERE request_id = ?",
                    (latency, request_id),
                )

        overview = admin.get("/api/v1/admin/dashboard/overview")
        assert overview.status_code == 200, overview.text
        assert overview.json()["latency"] == pytest.approx(
            {"min": 0.0, "p50": 20.0, "p95": 38.0, "p99": 39.6, "max": 40.0}
        )

    assert _percentiles([]) == {"min": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    assert _percentiles([7.5]) == {
        "min": 7.5,
        "p50": 7.5,
        "p95": 7.5,
        "p99": 7.5,
        "max": 7.5,
    }


def test_access_log_is_parseable_json_and_correlates_with_response(api_env) -> None:
    app, _, _ = api_env
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    access_logger = logging.getLogger("app.access")
    access_logger.addHandler(handler)
    try:
        with TestClient(app) as client:
            response = client.get("/api/v1/health")
    finally:
        access_logger.removeHandler(handler)

    assert response.status_code == 200
    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert len(records) == 1
    record = records[0]
    assert record["logger"] == "app.access"
    assert record["msg"] == "request completed"
    assert record["request_id"] == response.headers["X-Request-ID"]
    assert record["method"] == "GET"
    assert record["path"] == "/api/v1/health"
    assert record["status_code"] == 200
    assert isinstance(record["duration_ms"], float)
    assert record["duration_ms"] >= 0
    assert logging.getLogger("uvicorn.access").disabled is True
    datetime.fromisoformat(record["ts"].replace("Z", "+00:00")).astimezone(UTC)
