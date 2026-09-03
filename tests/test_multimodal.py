from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.sparse import csr_matrix

from app.artifacts import ModelArtifact
from app.multimodal_artifacts import MultimodalArtifact, MultimodalArtifactStore
from recsys.vision import VisionError, audit_and_extract_covers


def _jpeg(color: tuple[int, int, int]) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(stream, format="JPEG")
    return stream.getvalue()


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cover_archive_is_mapped_verified_and_path_traversal_is_rejected(
    tmp_path: Path,
) -> None:
    items = tmp_path / "items.csv"
    items.write_text("item_id,title,likes,views\n1,a,0,0\n2,b,0,0\n", encoding="utf-8")
    archive = tmp_path / "covers.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("MicroLens-50k_covers/1.jpg", _jpeg((255, 0, 0)))
        handle.writestr("MicroLens-50k_covers/2.jpg", _jpeg((0, 255, 0)))
    report = audit_and_extract_covers(archive, tmp_path / "covers", items)
    assert report["mapping_success_rate"] == 1.0
    assert report["damaged_rate"] == 0.0
    assert (tmp_path / "covers" / "1.jpg").is_file()

    unsafe = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(unsafe, "w") as handle:
        handle.writestr("../outside.jpg", _jpeg((0, 0, 0)))
    with pytest.raises(VisionError, match="unsafe or unexpected"):
        audit_and_extract_covers(unsafe, tmp_path / "unsafe", items)
    assert not (tmp_path / "outside.jpg").exists()


def _stable_artifact(tmp_path: Path) -> ModelArtifact:
    return ModelArtifact(
        model_version="stable-v1",
        data_version="data-v1",
        algorithm="test",
        manifest_path=tmp_path / "stable" / "manifest.json",
        metrics={},
        user_ids=np.asarray([1]),
        item_ids=np.asarray([10, 11]),
        user_factors=np.ones((1, 2)),
        item_factors=np.ones((2, 2)),
        popularity=[],
        content_item_ids=np.asarray([10, 11]),
        content_user_ids=np.asarray([1]),
        content_item_vectors=csr_matrix(np.asarray([[1.0, 0.0], [0.0, 1.0]])),
        content_user_vectors=csr_matrix(np.asarray([[1.0, 0.0]])),
    )


def test_multimodal_rerank_uses_visual_signal_and_missing_image_falls_back(
    tmp_path: Path,
) -> None:
    artifact = MultimodalArtifact(
        model_version="multi-v1",
        base_model_version="stable-v1",
        data_version="data-v1",
        manifest_path=tmp_path / "manifest.json",
        metrics={},
        item_ids=np.asarray([10, 11]),
        item_embeddings=np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        available=np.asarray([True, False]),
        user_ids=np.asarray([1]),
        user_profiles=np.asarray([[0.0, 1.0]], dtype=np.float32),
        visual_weight=0.25,
    )
    ranked = artifact.rerank(
        dataset_user_id="1",
        candidates=[("11", 1.0), ("10", 1.0)],
        stable=_stable_artifact(tmp_path),
    )
    assert ranked[0].item_id == "10"
    assert ranked[0].visual_score == pytest.approx(1.0)
    missing = next(value for value in ranked if value.item_id == "11")
    assert missing.visual_score is None
    assert np.isfinite(missing.fusion_score)
    visual = artifact.retrieve_visual(dataset_user_id="1", limit=10)
    assert visual == [("10", pytest.approx(1.0))]
    assert artifact.retrieve_visual(dataset_user_id="missing", limit=10) == []


def _write_multimodal_artifact(root: Path) -> Path:
    artifact = root / "multi-v1"
    artifact.mkdir()
    arrays = {
        "visual_item_ids.npy": np.asarray([10, 11], dtype=np.int64),
        "visual_item_embeddings.npy": np.ones((2, 2), dtype=np.float32),
        "visual_available.npy": np.asarray([True, True]),
        "visual_user_ids.npy": np.asarray([1], dtype=np.int64),
        "visual_user_profiles.npy": np.ones((1, 2), dtype=np.float32),
    }
    for name, value in arrays.items():
        np.save(artifact / name, value, allow_pickle=False)
    (artifact / "metrics.json").write_text("{}", encoding="utf-8")
    (artifact / "extraction.json").write_text("{}", encoding="utf-8")
    files = {
        path.name: {"bytes": path.stat().st_size, "sha256": _hash(path)}
        for path in artifact.iterdir()
    }
    (artifact / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "mobilenet_text_fusion_experiment",
                "model_version": "multi-v1",
                "base_model_version": "stable-v1",
                "data_version": "data-v1",
                "files": files,
                "metrics": {},
                "serving": {
                    "selected_visual_weight": 0.25,
                    "selected_warm_visual_weight": 0.1,
                    "selected_cold_visual_weight": 0.2,
                },
            }
        ),
        encoding="utf-8",
    )
    pointer = root / "multimodal-current.json"
    pointer.write_text(json.dumps({"manifest": "multi-v1/manifest.json"}), encoding="utf-8")
    return pointer


def test_multimodal_store_validates_hash_and_keeps_previous_good(tmp_path: Path) -> None:
    pointer = _write_multimodal_artifact(tmp_path)
    store = MultimodalArtifactStore(pointer)
    loaded = store.get()
    assert loaded is not None
    assert loaded.visual_weight == 0.25
    assert loaded.warm_visual_weight == 0.1
    assert loaded.cold_visual_weight == 0.2
    pointer.write_text(json.dumps({"manifest": "missing/manifest.json"}) + "\n", encoding="utf-8")
    os.utime(pointer, None)
    assert store.get() is loaded
    assert store.last_error is not None


def test_multimodal_store_rejects_corrupt_file(tmp_path: Path) -> None:
    pointer = _write_multimodal_artifact(tmp_path)
    manifest_path = tmp_path / "multi-v1" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["visual_item_embeddings.npy"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    store = MultimodalArtifactStore(pointer)
    assert store.get() is None
    assert "invalid multimodal artifact" in str(store.last_error)
