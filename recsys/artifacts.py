from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import load_npz, save_npz, spmatrix


ARRAY_FILES = {
    "user_ids.npy": 1,
    "item_ids.npy": 1,
    "user_factors.npy": 2,
    "item_factors.npy": 2,
}
SUPPORT_FILES = ("popularity.json", "metrics.json", "evaluation.md")


class ArtifactValidationError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _array_metadata(path: Path) -> dict[str, Any]:
    array = np.load(path, allow_pickle=False)
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": sha256_file(path),
    }


def write_staged_artifact(
    artifacts_dir: Path,
    *,
    manifest_base: dict[str, Any],
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    popularity: dict[str, Any],
    metrics: dict[str, Any],
    evaluation_markdown: str,
    extra_artifacts: dict[str, Any] | None = None,
) -> Path:
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    staging = artifacts_dir / f".staging-{manifest_base['model_version']}-{uuid.uuid4().hex}"
    staging.mkdir()
    arrays = {
        "user_ids.npy": np.asarray(user_ids, dtype=np.int64),
        "item_ids.npy": np.asarray(item_ids, dtype=np.int64),
        "user_factors.npy": np.asarray(user_factors, dtype=np.float32),
        "item_factors.npy": np.asarray(item_factors, dtype=np.float32),
    }
    for name, array in arrays.items():
        with (staging / name).open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
    _write_json(staging / "popularity.json", popularity)
    _write_json(staging / "metrics.json", metrics)
    (staging / "evaluation.md").write_text(evaluation_markdown, encoding="utf-8")

    for name, value in sorted((extra_artifacts or {}).items()):
        if Path(name).name != name or name.startswith("."):
            raise ArtifactValidationError(f"unsafe extra artifact name: {name}")
        path = staging / name
        if name.endswith(".npy"):
            with path.open("wb") as handle:
                np.save(handle, np.asarray(value), allow_pickle=False)
        elif name.endswith(".npz") and isinstance(value, spmatrix):
            save_npz(path, value, compressed=True)
        elif name.endswith(".json") and isinstance(value, (dict, list)):
            _write_json(path, value)
        else:
            raise ArtifactValidationError(f"unsupported extra artifact: {name}")

    files: dict[str, dict[str, Any]] = {
        name: _array_metadata(staging / name) for name in ARRAY_FILES
    }
    for name in SUPPORT_FILES:
        files[name] = {"bytes": (staging / name).stat().st_size, "sha256": sha256_file(staging / name)}
    for name in sorted(extra_artifacts or {}):
        path = staging / name
        if name.endswith(".npy"):
            files[name] = _array_metadata(path)
        elif name.endswith(".npz"):
            matrix = load_npz(path)
            files[name] = {
                "dtype": str(matrix.dtype),
                "format": matrix.format,
                "shape": list(matrix.shape),
                "sha256": sha256_file(path),
            }
        else:
            files[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    manifest = {**manifest_base, "files": files}
    _write_json(staging / "manifest.json", manifest)
    validate_artifact_dir(staging)
    return staging


def validate_artifact_dir(artifact_dir: Path) -> dict[str, Any]:
    artifact_dir = Path(artifact_dir)
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ArtifactValidationError("manifest.json is missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError("manifest.json is invalid") from exc
    if manifest.get("schema_version") != 1 or not manifest.get("model_version"):
        raise ArtifactValidationError("manifest schema_version/model_version is invalid")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ArtifactValidationError("manifest files section is missing")

    loaded: dict[str, np.ndarray] = {}
    for name, expected_ndim in ARRAY_FILES.items():
        path = artifact_dir / name
        details = files.get(name)
        if not path.is_file() or not isinstance(details, dict):
            raise ArtifactValidationError(f"{name} is missing")
        if sha256_file(path) != details.get("sha256"):
            raise ArtifactValidationError(f"{name} checksum mismatch")
        try:
            array = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ArtifactValidationError(f"{name} cannot be loaded without pickle") from exc
        if array.ndim != expected_ndim:
            raise ArtifactValidationError(f"{name} expected {expected_ndim} dimensions")
        if list(array.shape) != details.get("shape") or str(array.dtype) != details.get("dtype"):
            raise ArtifactValidationError(f"{name} shape/dtype mismatch")
        if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise ArtifactValidationError(f"{name} contains non-finite or non-numeric values")
        loaded[name] = array

    user_ids = loaded["user_ids.npy"]
    item_ids = loaded["item_ids.npy"]
    user_factors = loaded["user_factors.npy"]
    item_factors = loaded["item_factors.npy"]
    if user_factors.shape[0] != user_ids.shape[0]:
        raise ArtifactValidationError("user factor/id row counts differ")
    if item_factors.shape[0] != item_ids.shape[0]:
        raise ArtifactValidationError("item factor/id row counts differ")
    if user_factors.shape[1] != item_factors.shape[1]:
        raise ArtifactValidationError("user/item factor dimensions differ")
    if len(np.unique(user_ids)) != len(user_ids) or len(np.unique(item_ids)) != len(item_ids):
        raise ArtifactValidationError("user_ids/item_ids must be unique")

    for name in SUPPORT_FILES:
        path = artifact_dir / name
        details = files.get(name)
        if not path.is_file() or not isinstance(details, dict):
            raise ArtifactValidationError(f"{name} is missing")
        if sha256_file(path) != details.get("sha256"):
            raise ArtifactValidationError(f"{name} checksum mismatch")
    known_files = {*ARRAY_FILES, *SUPPORT_FILES}
    for name, details in files.items():
        if name in known_files:
            continue
        path = artifact_dir / name
        if Path(name).name != name or not path.is_file() or not isinstance(details, dict):
            raise ArtifactValidationError(f"invalid extra artifact: {name}")
        if sha256_file(path) != details.get("sha256"):
            raise ArtifactValidationError(f"{name} checksum mismatch")
    try:
        json.loads((artifact_dir / "popularity.json").read_text(encoding="utf-8"))
        json.loads((artifact_dir / "metrics.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactValidationError("support JSON is invalid") from exc

    content = manifest.get("content_features")
    if content is not None:
        required_content = {
            "content_item_ids.npy",
            "content_user_ids.npy",
            "content_item_vectors.npz",
            "content_user_vectors.npz",
            "content_idf.npy",
            "content_vectorizer.json",
        }
        if content.get("schema_version") != 1 or not required_content.issubset(files):
            raise ArtifactValidationError("content feature manifest is incomplete")
        content_item_ids = np.load(artifact_dir / "content_item_ids.npy", allow_pickle=False)
        content_user_ids = np.load(artifact_dir / "content_user_ids.npy", allow_pickle=False)
        content_idf = np.load(artifact_dir / "content_idf.npy", allow_pickle=False)
        item_vectors = load_npz(artifact_dir / "content_item_vectors.npz")
        user_vectors = load_npz(artifact_dir / "content_user_vectors.npz")
        if content_item_ids.ndim != 1 or content_user_ids.ndim != 1 or content_idf.ndim != 1:
            raise ArtifactValidationError("content feature id/idf arrays must be one dimensional")
        if item_vectors.shape != (len(content_item_ids), len(content_idf)):
            raise ArtifactValidationError("content item vector shape mismatch")
        if user_vectors.shape != (len(content_user_ids), len(content_idf)):
            raise ArtifactValidationError("content user vector shape mismatch")
        if not np.isfinite(item_vectors.data).all() or not np.isfinite(user_vectors.data).all():
            raise ArtifactValidationError("content vectors contain non-finite values")
        try:
            vectorizer = json.loads(
                (artifact_dir / "content_vectorizer.json").read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as exc:
            raise ArtifactValidationError("content_vectorizer.json is invalid") from exc
        if (
            vectorizer.get("schema_version") != 1
            or len(vectorizer.get("vocabulary") or {}) != len(content_idf)
        ):
            raise ArtifactValidationError("content vectorizer state is incompatible")
    item_cf = manifest.get("item_cf")
    if item_cf is not None:
        required_cf = {
            "item_cf_neighbors.npz",
            "item_cf_user_history.npz",
            "item_cf_config.json",
        }
        if item_cf.get("schema_version") != 1 or not required_cf.issubset(files):
            raise ArtifactValidationError("item CF manifest is incomplete")
        neighbors = load_npz(artifact_dir / "item_cf_neighbors.npz")
        history = load_npz(artifact_dir / "item_cf_user_history.npz")
        if neighbors.shape != (len(item_ids), len(item_ids)):
            raise ArtifactValidationError("item CF neighbor shape mismatch")
        if history.shape != (len(user_ids), len(item_ids)):
            raise ArtifactValidationError("item CF history shape mismatch")
        if not np.isfinite(neighbors.data).all() or not np.isfinite(history.data).all():
            raise ArtifactValidationError("item CF artifacts contain non-finite values")
        if neighbors.diagonal().any():
            raise ArtifactValidationError("item CF must not retain self similarity")
        try:
            cf_config = json.loads(
                (artifact_dir / "item_cf_config.json").read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as exc:
            raise ArtifactValidationError("item_cf_config.json is invalid") from exc
        if cf_config.get("schema_version") != 1:
            raise ArtifactValidationError("item CF config is incompatible")
    return manifest


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def publish_artifact(staging_dir: Path, artifacts_dir: Path) -> Path:
    staging_dir = Path(staging_dir)
    artifacts_dir = Path(artifacts_dir)
    manifest = validate_artifact_dir(staging_dir)
    model_version = str(manifest["model_version"])
    if Path(model_version).name != model_version or model_version.startswith("."):
        raise ArtifactValidationError("model_version is not a safe directory name")
    destination = artifacts_dir / model_version
    if destination.exists():
        raise ArtifactValidationError(f"model version already exists: {model_version}")
    os.replace(staging_dir, destination)
    pointer = {
        "artifact_path": model_version,
        "manifest": f"{model_version}/manifest.json",
        "model_version": model_version,
        "published_at": manifest.get("created_at"),
        "schema_version": 1,
    }
    _atomic_write_json(artifacts_dir / "current.json", pointer)
    return destination
