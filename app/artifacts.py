from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, load_npz
from recsys.mixing import DYNAMIC_POLICY_VERSION, SAFE_POLICY_VERSION


class ArtifactError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelArtifact:
    model_version: str
    data_version: str | None
    algorithm: str
    manifest_path: Path
    metrics: dict[str, Any]
    user_ids: np.ndarray
    item_ids: np.ndarray
    user_factors: np.ndarray
    item_factors: np.ndarray
    popularity: list[tuple[str, float]]
    content_item_ids: np.ndarray | None = None
    content_user_ids: np.ndarray | None = None
    content_item_vectors: csr_matrix | None = None
    content_user_vectors: csr_matrix | None = None
    content_metadata: dict[str, Any] | None = None
    item_cf_neighbors: csr_matrix | None = None
    item_cf_user_history: csr_matrix | None = None
    item_cf_metadata: dict[str, Any] | None = None
    mix_policy: dict[str, Any] | None = None
    feature_errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "user_index", {str(v): i for i, v in enumerate(self.user_ids)})
        object.__setattr__(self, "item_index", {str(v): i for i, v in enumerate(self.item_ids)})
        object.__setattr__(
            self,
            "content_item_index",
            {
                str(value): index
                for index, value in enumerate(
                    self.content_item_ids if self.content_item_ids is not None else ()
                )
            },
        )
        object.__setattr__(
            self,
            "content_user_index",
            {
                str(value): index
                for index, value in enumerate(
                    self.content_user_ids if self.content_user_ids is not None else ()
                )
            },
        )


class ArtifactStore:
    def __init__(self, pointer_path: Path | str):
        self.pointer_path = Path(pointer_path).resolve()
        self._lock = threading.Lock()
        self._cache_key: tuple[int, int] | None = None
        self._artifact: ModelArtifact | None = None
        self.last_error: str | None = None

    def get(self) -> ModelArtifact | None:
        try:
            stat = self.pointer_path.stat()
            key = (stat.st_mtime_ns, stat.st_size)
        except OSError as exc:
            self.last_error = f"model pointer unavailable: {exc}"
            return self._artifact
        if self._cache_key == key:
            return self._artifact
        with self._lock:
            if self._cache_key == key:
                return self._artifact
            try:
                loaded = self._load()
            except (OSError, ValueError, KeyError, ArtifactError, json.JSONDecodeError) as exc:
                self.last_error = str(exc)
                self._cache_key = key
                return self._artifact
            self._artifact = loaded
            self._cache_key = key
            self.last_error = None
            return loaded

    def _load(self) -> ModelArtifact:
        pointer = self._read_json(self.pointer_path)
        manifest_path = self._resolve_manifest(pointer)
        manifest = self._read_json(manifest_path)
        files = manifest.get("files") or manifest.get("artifacts") or {}
        required = ("user_ids.npy", "item_ids.npy", "user_factors.npy", "item_factors.npy")
        for name in required:
            path = (manifest_path.parent / name).resolve()
            self._assert_child(path, manifest_path.parent)
            if not path.is_file():
                raise ArtifactError(f"missing model file: {name}")
            expected_hash = self._file_hash(files, name, manifest)
            if not expected_hash:
                raise ArtifactError(f"manifest does not hash {name}")
            if not self._matches_hash(path, expected_hash):
                raise ArtifactError(f"hash mismatch: {name}")

        user_ids = np.load(manifest_path.parent / "user_ids.npy", allow_pickle=False)
        item_ids = np.load(manifest_path.parent / "item_ids.npy", allow_pickle=False)
        user_factors = np.load(manifest_path.parent / "user_factors.npy", allow_pickle=False)
        item_factors = np.load(manifest_path.parent / "item_factors.npy", allow_pickle=False)
        self._validate_arrays(user_ids, item_ids, user_factors, item_factors)

        popularity_path = manifest_path.parent / "popularity.json"
        popularity: list[tuple[str, float]] = []
        if popularity_path.is_file():
            expected_hash = self._file_hash(files, "popularity.json", manifest)
            if not expected_hash or not self._matches_hash(popularity_path, expected_hash):
                raise ArtifactError("invalid popularity.json hash")
            popularity = self._parse_popularity(self._read_json(popularity_path))

        metrics = manifest.get("metrics") or manifest.get("evaluation", {}).get("metrics")
        metrics_path = manifest_path.parent / "metrics.json"
        if metrics is None and metrics_path.is_file():
            expected_hash = self._file_hash(files, "metrics.json", manifest)
            if not expected_hash or not self._matches_hash(metrics_path, expected_hash):
                raise ArtifactError("invalid metrics.json hash")
            metrics = self._read_json(metrics_path)

        content_item_ids: np.ndarray | None = None
        content_user_ids: np.ndarray | None = None
        content_item_vectors: csr_matrix | None = None
        content_user_vectors: csr_matrix | None = None
        content_metadata: dict[str, Any] | None = None
        feature_errors: list[str] = []
        if manifest.get("content_features") is not None:
            try:
                (
                    content_item_ids,
                    content_user_ids,
                    content_item_vectors,
                    content_user_vectors,
                    content_metadata,
                ) = self._load_content_features(manifest_path.parent, manifest, files)
            except (OSError, ValueError, KeyError, ArtifactError, json.JSONDecodeError) as exc:
                feature_errors.append(f"content_features: {exc}")
        item_cf_neighbors: csr_matrix | None = None
        item_cf_user_history: csr_matrix | None = None
        item_cf_metadata: dict[str, Any] | None = None
        if manifest.get("item_cf") is not None:
            try:
                (
                    item_cf_neighbors,
                    item_cf_user_history,
                    item_cf_metadata,
                ) = self._load_item_cf(manifest_path.parent, manifest, files, user_ids, item_ids)
            except (OSError, ValueError, KeyError, ArtifactError, json.JSONDecodeError) as exc:
                feature_errors.append(f"item_cf: {exc}")

        version = str(manifest.get("model_version") or manifest.get("version") or "")
        if not version:
            raise ArtifactError("manifest is missing model_version")
        mix_policy = dict(manifest.get("mix_policy") or {})
        selected_policy = mix_policy.get("selected_policy_version")
        if selected_policy not in {None, SAFE_POLICY_VERSION, DYNAMIC_POLICY_VERSION}:
            raise ArtifactError(f"unsupported mix policy: {selected_policy}")
        if mix_policy and mix_policy.get("schema_version") != 1:
            raise ArtifactError("unsupported mix policy schema")
        return ModelArtifact(
            model_version=version,
            data_version=(str(manifest["data_version"]) if manifest.get("data_version") else None),
            algorithm=str(manifest.get("algorithm") or "unknown"),
            manifest_path=manifest_path,
            metrics=dict(metrics or {}),
            user_ids=user_ids,
            item_ids=item_ids,
            user_factors=user_factors.astype(np.float64, copy=False),
            item_factors=item_factors.astype(np.float64, copy=False),
            popularity=popularity,
            content_item_ids=content_item_ids,
            content_user_ids=content_user_ids,
            content_item_vectors=content_item_vectors,
            content_user_vectors=content_user_vectors,
            content_metadata=content_metadata,
            item_cf_neighbors=item_cf_neighbors,
            item_cf_user_history=item_cf_user_history,
            item_cf_metadata=item_cf_metadata,
            mix_policy=mix_policy,
            feature_errors=tuple(feature_errors),
        )

    def _load_item_cf(
        self,
        artifact_dir: Path,
        manifest: dict[str, Any],
        files: dict[str, Any],
        user_ids: np.ndarray,
        item_ids: np.ndarray,
    ) -> tuple[csr_matrix, csr_matrix, dict[str, Any]]:
        section = manifest.get("item_cf")
        if not isinstance(section, dict) or section.get("schema_version") != 1:
            raise ArtifactError("unsupported item CF schema")
        names = (
            "item_cf_neighbors.npz",
            "item_cf_user_history.npz",
            "item_cf_config.json",
        )
        for name in names:
            path = (artifact_dir / name).resolve()
            self._assert_child(path, artifact_dir)
            expected_hash = self._file_hash(files, name, manifest)
            if not path.is_file() or not expected_hash or not self._matches_hash(path, expected_hash):
                raise ArtifactError(f"invalid item CF artifact: {name}")
        neighbors = load_npz(artifact_dir / "item_cf_neighbors.npz").tocsr()
        history = load_npz(artifact_dir / "item_cf_user_history.npz").tocsr()
        config = self._read_json(artifact_dir / "item_cf_config.json")
        if neighbors.shape != (len(item_ids), len(item_ids)):
            raise ArtifactError("item CF neighbor shape mismatch")
        if history.shape != (len(user_ids), len(item_ids)):
            raise ArtifactError("item CF history shape mismatch")
        if not np.isfinite(neighbors.data).all() or not np.isfinite(history.data).all():
            raise ArtifactError("item CF contains non-finite values")
        if neighbors.diagonal().any() or config.get("schema_version") != 1:
            raise ArtifactError("item CF artifact is incompatible")
        return neighbors, history, dict(section)

    def _load_content_features(
        self,
        artifact_dir: Path,
        manifest: dict[str, Any],
        files: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, csr_matrix, csr_matrix, dict[str, Any]]:
        section = manifest.get("content_features")
        if not isinstance(section, dict) or section.get("schema_version") != 1:
            raise ArtifactError("unsupported content feature schema")
        names = (
            "content_item_ids.npy",
            "content_user_ids.npy",
            "content_item_vectors.npz",
            "content_user_vectors.npz",
            "content_idf.npy",
            "content_vectorizer.json",
        )
        for name in names:
            path = (artifact_dir / name).resolve()
            self._assert_child(path, artifact_dir)
            expected_hash = self._file_hash(files, name, manifest)
            if not path.is_file() or not expected_hash or not self._matches_hash(path, expected_hash):
                raise ArtifactError(f"invalid content artifact: {name}")
        item_ids = np.load(artifact_dir / "content_item_ids.npy", allow_pickle=False)
        user_ids = np.load(artifact_dir / "content_user_ids.npy", allow_pickle=False)
        idf = np.load(artifact_dir / "content_idf.npy", allow_pickle=False)
        item_vectors = load_npz(artifact_dir / "content_item_vectors.npz").tocsr()
        user_vectors = load_npz(artifact_dir / "content_user_vectors.npz").tocsr()
        vectorizer = self._read_json(artifact_dir / "content_vectorizer.json")
        if item_ids.ndim != 1 or user_ids.ndim != 1 or idf.ndim != 1:
            raise ArtifactError("content id/idf arrays must be one dimensional")
        if item_vectors.shape != (len(item_ids), len(idf)):
            raise ArtifactError("content item vector shape mismatch")
        if user_vectors.shape != (len(user_ids), len(idf)):
            raise ArtifactError("content user vector shape mismatch")
        if (
            not np.isfinite(item_vectors.data).all()
            or not np.isfinite(user_vectors.data).all()
            or not np.isfinite(idf).all()
        ):
            raise ArtifactError("content feature contains non-finite values")
        if len(vectorizer.get("vocabulary") or {}) != len(idf):
            raise ArtifactError("content vocabulary/idf mismatch")
        metadata = dict(section)
        metadata["vectorizer"] = {
            key: value for key, value in vectorizer.items() if key != "vocabulary"
        }
        return item_ids, user_ids, item_vectors, user_vectors, metadata

    def _resolve_manifest(self, pointer: dict[str, Any]) -> Path:
        raw = pointer.get("manifest") or pointer.get("manifest_path")
        if raw:
            candidate = Path(str(raw))
            if not candidate.is_absolute():
                candidate = self.pointer_path.parent / candidate
        else:
            version = pointer.get("model_version") or pointer.get("version") or pointer.get("active_version")
            if not version:
                raise ArtifactError("model pointer has neither manifest nor version")
            candidate = self.pointer_path.parent / str(version) / "manifest.json"
        candidate = candidate.resolve()
        self._assert_child(candidate, self.pointer_path.parent)
        return candidate

    @staticmethod
    def _assert_child(path: Path, parent: Path) -> None:
        try:
            path.relative_to(parent.resolve())
        except ValueError as exc:
            raise ArtifactError("artifact path escapes artifact directory") from exc

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ArtifactError(f"expected JSON object: {path.name}")
        return value

    @staticmethod
    def _file_hash(files: dict[str, Any], name: str, manifest: dict[str, Any]) -> str | None:
        value = files.get(name)
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            return value.get("sha256") or value.get("hash")
        hashes = manifest.get("file_hashes") or {}
        value = hashes.get(name)
        return str(value) if value else None

    @staticmethod
    def _matches_hash(path: Path, expected: str) -> bool:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest().lower() == expected.removeprefix("sha256:").lower()

    @staticmethod
    def _validate_arrays(
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        user_factors: np.ndarray,
        item_factors: np.ndarray,
    ) -> None:
        if user_ids.ndim != 1 or item_ids.ndim != 1:
            raise ArtifactError("id arrays must be one dimensional")
        if user_factors.ndim != 2 or item_factors.ndim != 2:
            raise ArtifactError("factor arrays must be two dimensional")
        if user_factors.shape[0] != user_ids.shape[0]:
            raise ArtifactError("user id/factor shape mismatch")
        if item_factors.shape[0] != item_ids.shape[0]:
            raise ArtifactError("item id/factor shape mismatch")
        if user_factors.shape[1] != item_factors.shape[1]:
            raise ArtifactError("latent dimensions do not match")
        if not np.isfinite(user_factors).all() or not np.isfinite(item_factors).all():
            raise ArtifactError("model factors contain non-finite values")
        if len({str(v) for v in user_ids}) != len(user_ids):
            raise ArtifactError("duplicate user ids")
        if len({str(v) for v in item_ids}) != len(item_ids):
            raise ArtifactError("duplicate item ids")

    @staticmethod
    def _parse_popularity(payload: dict[str, Any]) -> list[tuple[str, float]]:
        values: Any = payload.get("items", payload)
        if isinstance(values, dict):
            return sorted(
                ((str(key), float(score)) for key, score in values.items()),
                key=lambda pair: (-pair[1], pair[0]),
            )
        if isinstance(values, list):
            parsed = []
            for index, row in enumerate(values):
                if isinstance(row, dict):
                    parsed.append((str(row["item_id"]), float(row.get("score", -index))))
                else:
                    parsed.append((str(row), float(-index)))
            return parsed
        raise ArtifactError("unsupported popularity.json shape")
