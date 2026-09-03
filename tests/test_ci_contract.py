from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_github_ci_covers_locked_tests_frontend_and_smoke() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "pull_request:" in workflow
    assert "branches:\n      - main" in workflow
    assert "uv sync --locked --group dev" in workflow
    assert "cache-dependency-glob: uv.lock" in workflow
    assert "uv run python -m pytest -p no:cacheprovider" in workflow
    assert "node --check web/app.js" in workflow
    assert "node --check web/api.js" in workflow
    assert "uv run python -m scripts.ci_smoke" in workflow
    assert "APP_DATABASE: .tmp/ci-app.db" in workflow
    assert "DATABASE_PATH:" not in workflow
    assert "continue-on-error" not in workflow
    assert "data/raw" not in workflow


def test_ci_smoke_is_synthetic_and_exercises_model_db_and_health() -> None:
    source = (ROOT / "scripts" / "ci_smoke.py").read_text(encoding="utf-8")
    assert "_write_synthetic_raw" in source
    assert "prepare_data(" in source
    assert "train_model(" in source
    assert "Database(settings.database_path)" in source
    assert 'client.get("/api/v1/health")' in source
    assert '"event_type": "dwell"' in source
    assert '"event_type": "share"' in source
    assert "artifact.item_cf_neighbors" in source
    assert "data/raw" not in source
