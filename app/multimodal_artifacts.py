from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .artifacts import ArtifactError, ModelArtifact


@dataclass(frozen=True)
class MultimodalRank:
    item_id: str
    base_score: float
    text_score: float
    visual_score: float | None
    fusion_score: float
    rank: int


@dataclass
class MultimodalArtifact:
    model_version: str
    base_model_version: str
    data_version: str
    manifest_path: Path
    metrics: dict[str, Any]
    item_ids: np.ndarray
    item_embeddings: np.ndarray
    available: np.ndarray
    user_ids: np.ndarray
    user_profiles: np.ndarray
    visual_weight: float | None = None
    warm_visual_weight: float | None = None
    cold_visual_weight: float | None = None

    def __post_init__(self) -> None:
        fallback_weight = float(self.visual_weight or 0.0)
        if self.warm_visual_weight is None:
            self.warm_visual_weight = fallback_weight
        if self.cold_visual_weight is None:
            self.cold_visual_weight = fallback_weight
        if self.visual_weight is None:
            self.visual_weight = float(self.cold_visual_weight)
        self.item_index = {str(value): index for index, value in enumerate(self.item_ids)}
        self.user_index = {str(value): index for index, value in enumerate(self.user_ids)}

    @staticmethod
    def _rank_normalize(values: np.ndarray, item_ids: Sequence[str]) -> np.ndarray:
        if len(values) <= 1:
            return np.ones(len(values), dtype=np.float64)
        order = np.lexsort((np.asarray(item_ids), -values))
        ranks = np.empty(len(values), dtype=np.float64)
        ranks[order] = np.arange(len(values), dtype=np.float64)
        return 1.0 - ranks / float(len(values) - 1)

    def rerank(
        self,
        *,
        dataset_user_id: str,
        candidates: Sequence[tuple[str, float]],
        stable: ModelArtifact,
    ) -> list[MultimodalRank]:
        if stable.model_version != self.base_model_version:
            raise ArtifactError("multimodal experiment base model does not match stable model")
        if not candidates:
            return []
        item_ids = [str(item_id) for item_id, _ in candidates]
        base_scores = np.asarray([float(score) for _, score in candidates], dtype=np.float64)
        base_rank = self._rank_normalize(base_scores, item_ids)
        visual_scores = np.full(len(item_ids), -1.0, dtype=np.float64)
        user_row = self.user_index.get(str(dataset_user_id))
        if user_row is not None:
            for index, item_id in enumerate(item_ids):
                item_row = self.item_index.get(item_id)
                if item_row is not None and self.available[item_row]:
                    visual_scores[index] = float(
                        self.item_embeddings[item_row] @ self.user_profiles[user_row]
                    )
        visual_available = visual_scores > -1.0
        visual_rank = self._rank_normalize(visual_scores, item_ids)

        text_scores = np.zeros(len(item_ids), dtype=np.float64)
        content_user_row = stable.content_user_index.get(str(dataset_user_id))
        if (
            content_user_row is not None
            and stable.content_item_vectors is not None
            and stable.content_user_vectors is not None
        ):
            valid = [
                (index, stable.content_item_index.get(item_id))
                for index, item_id in enumerate(item_ids)
            ]
            valid = [(index, row) for index, row in valid if row is not None]
            if valid:
                values = (
                    stable.content_item_vectors[
                        np.asarray([row for _, row in valid], dtype=np.int64)
                    ]
                    @ stable.content_user_vectors[content_user_row].T
                ).toarray().ravel()
                for (index, _), value in zip(valid, values):
                    text_scores[index] = float(value)
        text_rank = self._rank_normalize(text_scores, item_ids)
        visual_weights = np.asarray(
            [
                self.cold_visual_weight
                if item_id not in stable.item_index
                else self.warm_visual_weight
                for item_id in item_ids
            ],
            dtype=np.float64,
        )
        content_fusion = np.where(
            visual_available,
            (1.0 - visual_weights) * text_rank + visual_weights * visual_rank,
            text_rank,
        )
        # Keep the trained DeepFM order dominant while allowing validated content fusion
        # to affect the experimental result deterministically.
        serving_score = 0.75 * base_rank + 0.25 * content_fusion
        order = np.lexsort((np.asarray(item_ids), -serving_score))
        return [
            MultimodalRank(
                item_id=item_ids[position],
                base_score=float(base_scores[position]),
                text_score=float(text_scores[position]),
                visual_score=(
                    float(visual_scores[position]) if visual_available[position] else None
                ),
                fusion_score=float(serving_score[position]),
                rank=rank,
            )
            for rank, position in enumerate(order)
        ]

    def retrieve_visual(
        self,
        *,
        dataset_user_id: str,
        limit: int,
    ) -> list[tuple[str, float]]:
        """Return a bounded visual-recall list from the cached cutoff-safe profile."""
        user_row = self.user_index.get(str(dataset_user_id))
        if user_row is None or limit <= 0:
            return []
        scores = self.item_embeddings @ self.user_profiles[user_row]
        valid = np.flatnonzero(self.available & np.isfinite(scores))
        if valid.size == 0:
            return []
        order = valid[np.lexsort((self.item_ids[valid], -scores[valid]))]
        return [
            (str(self.item_ids[row]), float(scores[row]))
            for row in order[:limit]
        ]


class MultimodalArtifactStore:
    REQUIRED_FILES = (
        "metrics.json",
        "extraction.json",
        "visual_item_ids.npy",
        "visual_item_embeddings.npy",
        "visual_available.npy",
        "visual_user_ids.npy",
        "visual_user_profiles.npy",
    )

    def __init__(self, pointer_path: Path | str):
        self.pointer_path = Path(pointer_path).resolve()
        self._lock = threading.Lock()
        self._cache_key: tuple[int, int] | None = None
        self._artifact: MultimodalArtifact | None = None
        self.last_error: str | None = None

    def get(self) -> MultimodalArtifact | None:
        try:
            stat = self.pointer_path.stat()
            key = (stat.st_mtime_ns, stat.st_size)
        except OSError as exc:
            self.last_error = f"multimodal pointer unavailable: {exc}"
            return self._artifact
        if key == self._cache_key:
            return self._artifact
        with self._lock:
            if key == self._cache_key:
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

    def _load(self) -> MultimodalArtifact:
        pointer = self._read_json(self.pointer_path)
        manifest_path = (self.pointer_path.parent / str(pointer["manifest"])).resolve()
        self._assert_child(manifest_path, self.pointer_path.parent)
        manifest = self._read_json(manifest_path)
        if (
            manifest.get("schema_version") != 1
            or manifest.get("artifact_type") != "mobilenet_text_fusion_experiment"
        ):
            raise ArtifactError("unsupported multimodal artifact")
        files = manifest.get("files") or {}
        for name in self.REQUIRED_FILES:
            path = (manifest_path.parent / name).resolve()
            self._assert_child(path, manifest_path.parent)
            expected = (files.get(name) or {}).get("sha256")
            if not path.is_file() or not expected or self._sha256(path) != expected:
                raise ArtifactError(f"invalid multimodal artifact: {name}")
        item_ids = np.load(manifest_path.parent / "visual_item_ids.npy", allow_pickle=False)
        embeddings = np.load(
            manifest_path.parent / "visual_item_embeddings.npy", allow_pickle=False
        )
        available = np.load(manifest_path.parent / "visual_available.npy", allow_pickle=False)
        user_ids = np.load(manifest_path.parent / "visual_user_ids.npy", allow_pickle=False)
        profiles = np.load(
            manifest_path.parent / "visual_user_profiles.npy", allow_pickle=False
        )
        if item_ids.ndim != 1 or user_ids.ndim != 1 or available.ndim != 1:
            raise ArtifactError("multimodal id/availability arrays must be one dimensional")
        if embeddings.shape[0] != len(item_ids) or profiles.shape[0] != len(user_ids):
            raise ArtifactError("multimodal embedding/id shape mismatch")
        if available.shape[0] != len(item_ids) or embeddings.shape[1] != profiles.shape[1]:
            raise ArtifactError("multimodal feature shape mismatch")
        if not np.isfinite(embeddings).all() or not np.isfinite(profiles).all():
            raise ArtifactError("multimodal artifact contains non-finite values")
        serving = dict(manifest.get("serving") or {})
        fallback_weight = float(serving.get("selected_visual_weight", 0.0))
        warm_visual_weight = float(
            serving.get("selected_warm_visual_weight", fallback_weight)
        )
        cold_visual_weight = float(
            serving.get("selected_cold_visual_weight", fallback_weight)
        )
        if not all(
            0.0 <= value <= 1.0
            for value in (warm_visual_weight, cold_visual_weight)
        ):
            raise ArtifactError("multimodal visual weight is invalid")
        return MultimodalArtifact(
            model_version=str(manifest["model_version"]),
            base_model_version=str(manifest["base_model_version"]),
            data_version=str(manifest["data_version"]),
            manifest_path=manifest_path,
            metrics=dict(manifest.get("metrics") or {}),
            item_ids=item_ids,
            item_embeddings=embeddings.astype(np.float32, copy=False),
            available=available.astype(bool, copy=False),
            user_ids=user_ids,
            user_profiles=profiles.astype(np.float32, copy=False),
            visual_weight=fallback_weight,
            warm_visual_weight=warm_visual_weight,
            cold_visual_weight=cold_visual_weight,
        )

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ArtifactError(f"expected JSON object: {path.name}")
        return value

    @staticmethod
    def _assert_child(path: Path, root: Path) -> None:
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise ArtifactError("multimodal path escapes artifact root") from exc

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
