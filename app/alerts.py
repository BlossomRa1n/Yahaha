"""Threshold alerting over live dashboard metrics.

Rules are stored in ``alert_rules`` and evaluated against real, DB-aggregated
metrics (request/exposure/event counts, CTR, latency percentiles, offline-item
count). Evaluation is deterministic and side-effect safe: it only reads live
tables and writes to ``alert_events``. There is no fixed/fake data — a rule
fires only when the current metric value actually breaches its threshold.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Callable

from .security import isoformat, utc_now

ALERT_OPERATORS = (">", "<", ">=", "<=")
ALERT_SEVERITIES = ("info", "warn", "critical")


def _sortable(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _count(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...]) -> float:
    return float(conn.execute(sql, params).fetchone()[0])


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    position = q * (n - 1)
    lower = int(position)
    upper = min(lower + 1, n - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _ctr(conn: sqlite3.Connection, start: str, end: str) -> float:
    exposures = _count(
        conn,
        "SELECT COUNT(*) FROM exposures WHERE created_at >= ? AND created_at < ?",
        (start, end),
    )
    clicks = _count(
        conn,
        "SELECT COUNT(*) FROM events WHERE event_type = 'click' AND received_at >= ? AND received_at < ?",
        (start, end),
    )
    return clicks / exposures if exposures else 0.0


def _active_users(conn: sqlite3.Connection, start: str, end: str) -> float:
    return _count(
        conn,
        """
        SELECT COUNT(*) FROM (
            SELECT user_id FROM recommendation_requests
            WHERE created_at >= ? AND created_at < ?
            UNION
            SELECT user_id FROM events
            WHERE event_type != 'impression' AND received_at >= ? AND received_at < ?
        )
        """,
        (start, end, start, end),
    )


def _latency_p95(conn: sqlite3.Connection, start: str, end: str) -> float:
    rows = conn.execute(
        "SELECT latency_ms FROM recommendation_requests WHERE created_at >= ? AND created_at < ?",
        (start, end),
    ).fetchall()
    return _percentile([float(row[0]) for row in rows], 0.95)


# Metric name -> {label, unit, compute(conn, start, end) -> float}
ALERT_METRICS: dict[str, dict[str, Any]] = {
    "requests": {
        "label": "推荐请求数",
        "unit": "count",
        "compute": lambda conn, s, e: _count(
            conn,
            "SELECT COUNT(*) FROM recommendation_requests WHERE created_at >= ? AND created_at < ?",
            (s, e),
        ),
    },
    "exposures": {
        "label": "曝光数",
        "unit": "count",
        "compute": lambda conn, s, e: _count(
            conn,
            "SELECT COUNT(*) FROM exposures WHERE created_at >= ? AND created_at < ?",
            (s, e),
        ),
    },
    "impressions": {
        "label": "可见曝光数",
        "unit": "count",
        "compute": lambda conn, s, e: _count(
            conn,
            "SELECT COUNT(*) FROM events WHERE event_type = 'impression' AND received_at >= ? AND received_at < ?",
            (s, e),
        ),
    },
    "clicks": {
        "label": "点击数",
        "unit": "count",
        "compute": lambda conn, s, e: _count(
            conn,
            "SELECT COUNT(*) FROM events WHERE event_type = 'click' AND received_at >= ? AND received_at < ?",
            (s, e),
        ),
    },
    "likes": {
        "label": "点赞数",
        "unit": "count",
        "compute": lambda conn, s, e: _count(
            conn,
            "SELECT COUNT(*) FROM events WHERE event_type = 'like' AND received_at >= ? AND received_at < ?",
            (s, e),
        ),
    },
    "ctr": {
        "label": "点击率 (CTR)",
        "unit": "ratio",
        "compute": _ctr,
    },
    "active_users": {
        "label": "活跃用户数",
        "unit": "count",
        "compute": _active_users,
    },
    "latency_p95": {
        "label": "P95 延迟",
        "unit": "ms",
        "compute": _latency_p95,
    },
    "offline_items": {
        "label": "下线内容数",
        "unit": "count",
        "compute": lambda conn, s, e: _count(
            conn,
            "SELECT COUNT(*) FROM items WHERE status = 'offline'",
            (),
        ),
    },
}


def _compare(value: float, operator: str, threshold: float) -> bool:
    if operator == ">":
        return value > threshold
    if operator == ">=":
        return value >= threshold
    if operator == "<":
        return value < threshold
    if operator == "<=":
        return value <= threshold
    return False


def evaluate_alerts(
    conn: sqlite3.Connection,
    now: datetime | None = None,
    *,
    window_minutes: int = 60,
) -> dict[str, int]:
    """Evaluate all enabled rules and transition alert-event state.

    A rule with no open event and a currently-breaching value opens one; a rule
    whose value has recovered closes its open event. Returns trigger/resolve
    counts so callers can report the transition summary honestly.
    """
    now = now or utc_now()
    end_text = _sortable(now)
    start_text = _sortable(now - timedelta(minutes=max(1, window_minutes)))
    triggered = 0
    resolved = 0
    rules = conn.execute(
        "SELECT * FROM alert_rules WHERE enabled = 1 ORDER BY created_at"
    ).fetchall()
    for rule in rules:
        metric_name = rule["metric"]
        spec = ALERT_METRICS.get(metric_name)
        if spec is None:
            continue
        observed = float(spec["compute"](conn, start_text, end_text))
        firing = _compare(observed, rule["operator"], float(rule["threshold"]))
        open_event = conn.execute(
            """
            SELECT id FROM alert_events
            WHERE rule_id = ? AND status = 'open'
            ORDER BY triggered_at DESC LIMIT 1
            """,
            (rule["id"],),
        ).fetchone()
        if firing:
            if open_event is None:
                conn.execute(
                    """
                    INSERT INTO alert_events(
                        id, rule_id, metric, operator, threshold, severity,
                        observed_value, status, triggered_at, last_evaluated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'open', ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        rule["id"],
                        metric_name,
                        rule["operator"],
                        float(rule["threshold"]),
                        rule["severity"],
                        observed,
                        isoformat(now),
                        isoformat(now),
                    ),
                )
                triggered += 1
            else:
                conn.execute(
                    """
                    UPDATE alert_events
                    SET observed_value = ?, last_evaluated_at = ?
                    WHERE id = ?
                    """,
                    (observed, isoformat(now), open_event["id"]),
                )
        elif open_event is not None:
            conn.execute(
                """
                UPDATE alert_events
                SET status = 'resolved', resolved_at = ?,
                    observed_value = ?, last_evaluated_at = ?
                WHERE id = ?
                """,
                (isoformat(now), observed, isoformat(now), open_event["id"]),
            )
            resolved += 1
    return {"triggered": triggered, "resolved": resolved}
