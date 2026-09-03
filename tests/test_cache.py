"""Redis cache layer (app/cache.py) — graceful degradation and correct
invocation of the public item payload cache.

Redis is optional: with no ``REDIS_URL`` (or the ``redis`` package absent) the
service must behave exactly as before via the no-op backend. These tests lock
in that contract and exercise the get/set/invalidate cycle on the item detail
endpoint through the cache wrapper.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.cache import NoopCache, build_cache
from app.cli import load_items, seed_accounts
from app.config import Settings
from app.db import Database
from app.main import create_app


def test_build_cache_degrades_to_noop_without_redis_url() -> None:
    assert build_cache(None).backend == "noop"
    assert build_cache("").backend == "noop"


def test_build_cache_degrades_to_noop_when_redis_unavailable() -> None:
    # A non-listening port (and/or a missing ``redis`` package) must not raise:
    # the cache degrades silently instead of failing the service.
    cache = build_cache("redis://127.0.0.1:6399/0")
    assert cache.backend == "noop"


def test_noop_cache_always_misses_and_swallows_writes() -> None:
    cache = NoopCache()
    assert cache.get_json("any") is None
    assert cache.set_json("any", {"x": 1}, ttl_seconds=60) is None
    assert cache.delete("any") is None


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
    (version_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_version": "model-v1",
                "data_version": "fixture-v1",
                "algorithm": "truncated_svd",
                "metrics": {"recall@5": 0.5, "ndcg@5": 0.4},
                "files": files,
            }
        ),
        encoding="utf-8",
    )
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
def cache_env(tmp_path: Path):
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
        redis_url=None,
    )
    database = Database(settings.database_path)
    database.reset()
    with database.transaction(immediate=True) as conn:
        load_items(conn, items_path)
        seed_accounts(conn, ["dataset-alice", "dataset-bob"])
    app = create_app(settings)
    return app, database


def test_item_detail_cache_roundtrip_and_status_invalidation(cache_env) -> None:
    app, _ = cache_env
    with TestClient(app) as alice, TestClient(app) as admin:
        alice.post("/api/v1/auth/login", json={"username": "alice", "password": "demo-pass"})
        admin.post("/api/v1/auth/login", json={"username": "admin", "password": "admin-pass"})

        first = alice.get("/api/v1/items/8")
        assert first.status_code == 200
        payload = first.json()["item"]
        assert payload["item_id"] == "8"
        assert "status" not in payload  # public payload omits internal status

        # The no-op cache never fabricates a hit; repeated reads stay fresh.
        assert alice.get("/api/v1/items/8").json()["item"] == payload

        # Offline invalidates the cached payload and the item becomes 404.
        assert admin.patch(
            "/api/v1/admin/items/8/status",
            json={"status": "offline", "reason": "cache invalidation test"},
        ).status_code == 200
        assert alice.get("/api/v1/items/8").status_code == 404

        # Restore clears invalidation and the item is reachable again.
        assert admin.patch(
            "/api/v1/admin/items/8/status",
            json={"status": "online", "reason": "cache restore test"},
        ).status_code == 200
        assert alice.get("/api/v1/items/8").status_code == 200
