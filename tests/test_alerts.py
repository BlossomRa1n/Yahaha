"""Threshold alerting (app/alerts.py + /api/v1/admin/alerts/*) integration tests.

These tests exercise the live DB-aggregated alert pipeline end to end: role
guards, rule CRUD, deterministic trigger/resolve/acknowledge state transitions,
and disabled-rule/validation behaviour. No fake metrics are injected — rules
fire only when a real metric value breaches its threshold.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.cli import load_items, seed_accounts
from app.config import Settings
from app.db import Database
from app.main import create_app


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_artifact(root: Path) -> Path:
    version_dir = root / "model-v1"
    version_dir.mkdir(parents=True)
    arrays = {
        "user_ids.npy": np.array(["dataset-alice", "dataset-bob"]),
        "item_ids.npy": np.array([str(value) for value in range(1, 9)]),
        "user_factors.npy": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        "item_factors.npy": np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.8, 0.2],
                [0.1, 0.9],
                [0.0, 1.0],
                [0.2, 0.8],
                [0.5, 0.5],
                [-0.2, 0.1],
            ],
            dtype=np.float64,
        ),
    }
    for name, value in arrays.items():
        np.save(version_dir / name, value, allow_pickle=False)
    (version_dir / "popularity.json").write_text(
        json.dumps(
            {"items": [{"item_id": str(value), "score": 9 - value} for value in range(1, 9)]}
        ),
        encoding="utf-8",
    )
    files = {name: {"sha256": _sha256(version_dir / name)} for name in (*arrays, "popularity.json")}
    manifest = {
        "schema_version": 1,
        "model_version": "model-v1",
        "data_version": "fixture-v1",
        "algorithm": "truncated_svd",
        "metrics": {"recall@5": 0.5, "ndcg@5": 0.4},
        "files": files,
    }
    (version_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    pointer = root / "current.json"
    pointer.write_text(json.dumps({"manifest": "model-v1/manifest.json"}), encoding="utf-8")
    return pointer


def _write_items(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id", "title", "likes", "views"])
        writer.writeheader()
        for value in range(1, 21):
            writer.writerow(
                {
                    "item_id": str(value),
                    "title": f"Item {value}",
                    "likes": (9 - value) * 10,
                    "views": (9 - value) * 100,
                }
            )


@pytest.fixture
def alert_env(tmp_path: Path):
    pointer = _write_artifact(tmp_path / "artifacts")
    items_path = tmp_path / "items.csv"
    _write_items(items_path)
    settings = Settings(
        app_env="test",
        app_secret="test-secret-with-enough-entropy",
        database_path=tmp_path / "app.db",
        model_pointer=pointer,
        session_hours=12,
        session_cookie="test_session",
    )
    database = Database(settings.database_path)
    database.reset()
    with database.transaction(immediate=True) as conn:
        load_items(conn, items_path)
        seed_accounts(conn, ["dataset-alice", "dataset-bob"])
    app = create_app(settings)
    return app, database


def _login(client: TestClient, username: str, password: str = "demo-pass") -> dict:
    response = client.post("/api/v1/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200, response.text
    return response.json()["user"]


def test_alert_rules_crud_and_role_guards(alert_env) -> None:
    app, _ = alert_env
    with TestClient(app) as user, TestClient(app) as analyst, TestClient(app) as operator:
        _login(user, "alice")
        _login(analyst, "analyst", "analyst-pass")
        operator_user = _login(operator, "operator", "operator-pass")

        # Reads require analyst; writes require operator.
        assert user.get("/api/v1/admin/alerts/metrics").status_code == 403
        assert user.get("/api/v1/admin/alerts/rules").status_code == 403

        metrics = analyst.get("/api/v1/admin/alerts/metrics")
        assert metrics.status_code == 200, metrics.text
        metric_names = {row["name"] for row in metrics.json()["metrics"]}
        assert metric_names == {
            "requests",
            "exposures",
            "impressions",
            "clicks",
            "likes",
            "ctr",
            "active_users",
            "latency_p95",
            "offline_items",
        }
        for row in metrics.json()["metrics"]:
            assert {"name", "label", "unit"} == set(row)

        assert analyst.get("/api/v1/admin/alerts/rules").json() == {"rules": []}
        assert analyst.post(
            "/api/v1/admin/alerts/rules",
            json={"name": "nope", "metric": "requests", "operator": ">", "threshold": 0},
        ).status_code == 403

        # Unknown metric is rejected with the available-metric catalogue.
        invalid = operator.post(
            "/api/v1/admin/alerts/rules",
            json={"name": "bad", "metric": "nope", "operator": ">", "threshold": 0},
        )
        assert invalid.status_code == 422
        assert invalid.json()["error"]["code"] == "invalid_metric"
        assert "requests" in invalid.json()["error"]["details"]["available"]

        created = operator.post(
            "/api/v1/admin/alerts/rules",
            json={
                "name": "request flood",
                "metric": "requests",
                "operator": ">",
                "threshold": 0,
                "severity": "critical",
            },
        )
        assert created.status_code == 201, created.text
        rule = created.json()
        assert rule["enabled"] is True
        assert rule["created_by"] == operator_user["id"]
        assert rule["metric"] == "requests"
        assert rule["severity"] == "critical"
        assert rule["threshold"] == 0

        # PATCH a real field.
        patched = operator.patch(
            f"/api/v1/admin/alerts/rules/{rule['id']}",
            json={"threshold": 10, "severity": "warn"},
        )
        assert patched.status_code == 200, patched.text
        assert patched.json()["threshold"] == 10
        assert patched.json()["severity"] == "warn"

        # Empty PATCH and missing rule.
        assert operator.patch(
            f"/api/v1/admin/alerts/rules/{rule['id']}", json={}
        ).status_code == 422
        assert operator.patch(
            "/api/v1/admin/alerts/rules/missing", json={"threshold": 1}
        ).status_code == 404

        listing = analyst.get("/api/v1/admin/alerts/rules").json()
        assert len(listing["rules"]) == 1
        assert listing["rules"][0]["threshold"] == 10

        assert operator.delete(f"/api/v1/admin/alerts/rules/{rule['id']}").json() == {
            "deleted": rule["id"]
        }
        assert operator.delete(f"/api/v1/admin/alerts/rules/{rule['id']}").status_code == 404


def test_alert_evaluation_trigger_resolve_and_acknowledge(alert_env) -> None:
    app, _ = alert_env
    with TestClient(app) as analyst, TestClient(app) as operator, TestClient(app) as admin:
        _login(analyst, "analyst", "analyst-pass")
        operator_user = _login(operator, "operator", "operator-pass")
        _login(admin, "admin", "admin-pass")

        rule = operator.post(
            "/api/v1/admin/alerts/rules",
            json={
                "name": "too many offline items",
                "metric": "offline_items",
                "operator": ">",
                "threshold": 0,
                "severity": "critical",
            },
        ).json()

        # No offline items yet -> no trigger.
        assert operator.post("/api/v1/admin/alerts/evaluate").json() == {
            "triggered": 0,
            "resolved": 0,
        }

        # Take an item offline -> breach -> open event.
        assert admin.patch(
            "/api/v1/admin/items/8/status",
            json={"status": "offline", "reason": "alert trigger test"},
        ).status_code == 200
        assert operator.post("/api/v1/admin/alerts/evaluate").json()["triggered"] == 1

        open_events = analyst.get("/api/v1/admin/alerts/events", params={"status": "open"}).json()
        assert len(open_events["events"]) == 1
        event = open_events["events"][0]
        assert event["rule_id"] == rule["id"]
        assert event["metric"] == "offline_items"
        assert event["severity"] == "critical"
        assert event["status"] == "open"
        assert event["observed_value"] >= 1
        assert event["resolved_at"] is None

        # Acknowledge the open event.
        ack = operator.post(
            f"/api/v1/admin/alerts/events/{event['id']}/acknowledge"
        )
        assert ack.status_code == 200, ack.text
        assert ack.json()["acknowledged_by"] == operator_user["id"]
        assert ack.json()["acknowledged_at"] is not None
        assert ack.json()["status"] == "open"

        # Restore the item -> value recovers -> resolve the open event.
        assert admin.patch(
            "/api/v1/admin/items/8/status",
            json={"status": "online", "reason": "alert resolve test"},
        ).status_code == 200
        assert operator.post("/api/v1/admin/alerts/evaluate").json()["resolved"] == 1

        assert analyst.get(
            "/api/v1/admin/alerts/events", params={"status": "open"}
        ).json()["events"] == []
        resolved = analyst.get(
            "/api/v1/admin/alerts/events", params={"status": "resolved"}
        ).json()["events"]
        assert len(resolved) == 1
        assert resolved[0]["resolved_at"] is not None

        # A resolved event cannot be acknowledged.
        assert operator.post(
            f"/api/v1/admin/alerts/events/{event['id']}/acknowledge"
        ).status_code == 404


def test_alert_disabled_rule_invalid_query_and_cascade_delete(alert_env) -> None:
    app, database = alert_env
    with TestClient(app) as alice, TestClient(app) as operator, TestClient(app) as analyst:
        _login(alice, "alice")
        _login(operator, "operator", "operator-pass")
        _login(analyst, "analyst", "analyst-pass")

        # Generate a real request so `requests` is non-zero.
        assert alice.get("/api/v1/feeds/popular?limit=1").status_code == 200

        rule = operator.post(
            "/api/v1/admin/alerts/rules",
            json={
                "name": "any requests",
                "metric": "requests",
                "operator": ">",
                "threshold": 0,
                "severity": "info",
            },
        ).json()

        # Disable -> evaluation ignores it.
        assert operator.patch(
            f"/api/v1/admin/alerts/rules/{rule['id']}", json={"enabled": False}
        ).json()["enabled"] is False
        assert operator.post("/api/v1/admin/alerts/evaluate").json() == {
            "triggered": 0,
            "resolved": 0,
        }
        assert analyst.get("/api/v1/admin/alerts/events").json()["events"] == []

        # Invalid status filter.
        assert analyst.get(
            "/api/v1/admin/alerts/events", params={"status": "bogus"}
        ).status_code == 422

        # Re-enable -> fires; deleting the rule cascades its event away.
        operator.patch(f"/api/v1/admin/alerts/rules/{rule['id']}", json={"enabled": True})
        assert operator.post("/api/v1/admin/alerts/evaluate").json()["triggered"] == 1
        with database.connect() as conn:
            assert conn.execute(
                "SELECT COUNT(*) FROM alert_events WHERE rule_id = ?", (rule["id"],)
            ).fetchone()[0] == 1
        operator.delete(f"/api/v1/admin/alerts/rules/{rule['id']}")
        with database.connect() as conn:
            assert conn.execute(
                "SELECT COUNT(*) FROM alert_events WHERE rule_id = ?", (rule["id"],)
            ).fetchone()[0] == 0
