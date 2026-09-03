"""Async training jobs (app/main.py /api/v1/admin/training/*).

The async endpoint queues a retraining job and runs ``run_online_retraining``
in a background thread, recording lifecycle transitions in ``training_jobs``.
These tests stub the actual training call so the job queue, role guards,
validation, and queued -> running -> succeeded/failed transitions are exercised
deterministically without touching the model pointer or heavy training.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from app.cli import seed_accounts
from app.config import Settings
from app.db import Database
from app.main import create_app


@pytest.fixture
def training_env(tmp_path, monkeypatch):
    settings = Settings(
        app_env="test",
        app_secret="test-secret-with-enough-entropy",
        database_path=tmp_path / "app.db",
        model_pointer=tmp_path / "artifacts" / "current.json",
        session_hours=12,
        session_cookie="test_session",
    )
    database = Database(settings.database_path)
    database.reset()
    with database.transaction(immediate=True) as conn:
        seed_accounts(conn, [])
    app = create_app(settings)
    return app, database


def _login(client: TestClient, username: str, password: str) -> dict:
    response = client.post("/api/v1/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200, response.text
    return response.json()["user"]


def _wait_for_terminal(client: TestClient, job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/api/v1/admin/training/jobs/{job_id}").json()
        if job["status"] in ("succeeded", "failed"):
            return job
        time.sleep(0.02)
    raise AssertionError(f"training job {job_id} did not reach a terminal state in time")


def test_async_training_job_lifecycle_and_role_guards(training_env, monkeypatch) -> None:
    app, _ = training_env
    monkeypatch.setattr(
        "app.main.run_online_retraining",
        lambda **kwargs: {"run_id": "run-fake-1"},
    )
    body = {
        "start_time": "2026-01-01T00:00:00Z",
        "end_time": "2026-01-02T00:00:00Z",
        "mode": "smoke",
        "rank": 16,
    }
    with TestClient(app) as user, TestClient(app) as analyst, TestClient(app) as operator:
        _login(user, "alice", "demo-pass")
        _login(analyst, "analyst", "analyst-pass")
        _login(operator, "operator", "operator-pass")

        # Role guards: reads need analyst, writes need operator.
        assert user.get("/api/v1/admin/training/jobs").status_code == 403
        assert user.post("/api/v1/admin/training/jobs", json=body).status_code == 403
        assert analyst.post("/api/v1/admin/training/jobs", json=body).status_code == 403
        assert analyst.get("/api/v1/admin/training/jobs").json() == {"jobs": []}

        # Invalid window is rejected.
        assert operator.post(
            "/api/v1/admin/training/jobs",
            json={**body, "start_time": body["end_time"], "end_time": body["start_time"]},
        ).status_code == 422

        # Submit a job: returns immediately as queued.
        response = operator.post("/api/v1/admin/training/jobs", json=body)
        assert response.status_code == 202, response.text
        queued = response.json()
        assert queued["status"] == "queued"
        assert queued["run_id"] is None
        job_id = queued["job_id"]

        final = _wait_for_terminal(analyst, job_id)
        assert final["status"] == "succeeded", final
        assert final["run_id"] == "run-fake-1"
        assert final["started_at"] is not None
        assert final["completed_at"] is not None
        assert final["config"]["mode"] == "smoke"
        assert final["config"]["rank"] == 16

        listing = analyst.get("/api/v1/admin/training/jobs").json()["jobs"]
        assert listing[0]["job_id"] == job_id
        assert listing[0]["status"] == "succeeded"

        assert analyst.get("/api/v1/admin/training/jobs/missing").status_code == 404


def test_async_training_job_records_failure(training_env, monkeypatch) -> None:
    app, _ = training_env

    def _boom(**kwargs):
        raise RuntimeError("training blew up")

    monkeypatch.setattr("app.main.run_online_retraining", _boom)
    body = {
        "start_time": "2026-01-01T00:00:00Z",
        "end_time": "2026-01-02T00:00:00Z",
        "mode": "smoke",
    }
    with TestClient(app) as operator, TestClient(app) as analyst:
        _login(operator, "operator", "operator-pass")
        _login(analyst, "analyst", "analyst-pass")
        response = operator.post("/api/v1/admin/training/jobs", json=body)
        assert response.status_code == 202, response.text
        job_id = response.json()["job_id"]

        final = _wait_for_terminal(analyst, job_id)
        assert final["status"] == "failed", final
        assert final["run_id"] is None
        assert "training blew up" in final["error"]
        assert final["completed_at"] is not None
