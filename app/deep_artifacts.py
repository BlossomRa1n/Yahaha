from __future__ import annotations

import hashlib
import json
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from safetensors.torch import load_file

from recsys.deep import DeepFM, DeepFMConfig, resolve_torch_device
from recsys.two_stage import UNIFIED_SOURCE_ORDER, UnifiedCandidate, safe_quality_prior

from .artifacts import ArtifactError, ModelArtifact


@dataclass(frozen=True)
class DeepRecommendation:
    item_id: str
    dssm_score: float
    deepfm_score: float
    rank: int


@dataclass
class DeepExperimentArtifact:
    model_version: str
    base_model_version: str
    data_version: str
    manifest_path: Path
    metrics: dict[str, Any]
    user_ids: np.ndarray
    item_ids: np.ndarray
    item_model_indices: np.ndarray
    user_embeddings: np.ndarray
    item_embeddings: np.ndarray
    user_profiles: np.ndarray
    item_content: np.ndarray
    item_popularity: np.ndarray
    item_pop_bucket: np.ndarray
    user_history_bucket: np.ndarray
    deepfm: DeepFM
    device: torch.device
    retrieval_top_n: int
    rank_strategy: str
    stable_rank_weight: float
    serving_mode: str = "legacy_protected_rerank"
    required_multimodal_version: str | None = None
    source_limits: dict[str, int] | None = None
    source_caps_at_10: dict[str, int] | None = None
    user_history_density: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.user_index = {str(value): index for index, value in enumerate(self.user_ids)}
        self.item_index = {str(value): index for index, value in enumerate(self.item_ids)}

    def recommend(
        self,
        *,
        dataset_user_id: str,
        feedback: Iterable[tuple[str, float]],
        stable: ModelArtifact,
        limit: int,
    ) -> list[DeepRecommendation]:
        if stable.model_version != self.base_model_version:
            raise ArtifactError("deep experiment base model does not match stable model")
        user_row = self.user_index.get(str(dataset_user_id))
        if user_row is None:
            return []
        user_embedding = self.user_embeddings[user_row].astype(np.float32, copy=True)
        profile = self.user_profiles[user_row].astype(np.float32, copy=True)
        feedback_rows: list[tuple[int, float]] = []
        for item_id, affinity in feedback:
            item_row = self.item_index.get(str(item_id))
            if item_row is not None and math.isfinite(float(affinity)) and affinity > 0:
                feedback_rows.append((item_row, min(4.0, float(affinity))))
        if feedback_rows:
            weights = np.asarray([value for _, value in feedback_rows], dtype=np.float32)
            rows = np.asarray([row for row, _ in feedback_rows], dtype=np.int64)
            user_embedding += 0.2 * np.average(
                self.item_embeddings[rows], axis=0, weights=weights
            )
            profile += np.average(self.item_content[rows], axis=0, weights=weights)
            user_embedding /= max(1e-8, float(np.linalg.norm(user_embedding)))
            profile /= max(1e-8, float(np.linalg.norm(profile)))

        retrieval_scores = self.item_embeddings @ user_embedding
        candidate_count = min(max(limit, self.retrieval_top_n), len(self.item_ids))
        order = np.lexsort((self.item_ids, -retrieval_scores))[:candidate_count]
        stable_user_row = stable.user_index.get(str(dataset_user_id))
        protected_item_rows = np.empty(0, dtype=np.int64)
        if self.rank_strategy == "protected_top10_rerank" and stable_user_row is not None:
            stable_values = np.full(len(self.item_ids), -1e12, dtype=np.float64)
            content_values = np.zeros(len(self.item_ids), dtype=np.float64)
            available = np.zeros(len(self.item_ids), dtype=bool)
            for stable_row, stable_item_id in enumerate(stable.item_ids):
                item_row = self.item_index.get(str(stable_item_id))
                if item_row is not None:
                    available[item_row] = True
                    stable_values[item_row] = float(
                        stable.user_factors[stable_user_row]
                        @ stable.item_factors[stable_row]
                    )
            content_user_row = stable.content_user_index.get(str(dataset_user_id))
            if (
                content_user_row is not None
                and stable.content_item_ids is not None
                and stable.content_item_vectors is not None
                and stable.content_user_vectors is not None
            ):
                raw_content = np.asarray(
                    (
                        stable.content_item_vectors
                        @ stable.content_user_vectors[content_user_row].T
                    )
                    .toarray()
                    .ravel(),
                    dtype=np.float64,
                )
                for content_row, content_item_id in enumerate(stable.content_item_ids):
                    item_row = self.item_index.get(str(content_item_id))
                    if item_row is not None:
                        content_values[item_row] = raw_content[content_row]
            cold_usable = (~available) & (content_values > 0)
            warm_order = [
                int(index)
                for index in np.lexsort((self.item_ids, -stable_values))
                if available[int(index)]
            ]
            cold_order = [
                int(index)
                for index in np.lexsort((self.item_ids, -content_values))
                if cold_usable[int(index)]
            ]
            protected_item_rows = np.asarray(
                [*warm_order[:7], *cold_order[:3]], dtype=np.int64
            )
            combined = [*map(int, protected_item_rows), *map(int, order)]
            order = np.asarray(list(dict.fromkeys(combined))[: candidate_count + 10])
            candidate_count = len(order)
        svd_scores = np.zeros(candidate_count, dtype=np.float32)
        if stable_user_row is not None:
            for index, item_row in enumerate(order):
                stable_item_row = stable.item_index.get(str(self.item_ids[item_row]))
                if stable_item_row is not None:
                    svd_scores[index] = float(
                        stable.user_factors[stable_user_row]
                        @ stable.item_factors[stable_item_row]
                    )
        content_scores = self.item_content[order] @ profile
        cf_scores = np.zeros(candidate_count, dtype=np.float32)
        if (
            stable_user_row is not None
            and stable.item_cf_neighbors is not None
            and stable.item_cf_user_history is not None
        ):
            history = stable.item_cf_user_history[stable_user_row]
            stable_candidate_rows = [
                stable.item_index.get(str(self.item_ids[item_row])) for item_row in order
            ]
            valid = [
                (index, stable_item_row)
                for index, stable_item_row in enumerate(stable_candidate_rows)
                if stable_item_row is not None
            ]
            if valid:
                values = (
                    history
                    @ stable.item_cf_neighbors[
                        :, np.asarray([row for _, row in valid], dtype=np.int64)
                    ]
                ).toarray().ravel()
                for (index, _), value in zip(valid, values):
                    cf_scores[index] = float(value)
        categorical = np.column_stack(
            (
                np.full(candidate_count, user_row + 1, dtype=np.int64),
                self.item_model_indices[order],
                self.item_pop_bucket[order],
                np.full(
                    candidate_count,
                    self.user_history_bucket[user_row],
                    dtype=np.int64,
                ),
                np.zeros(candidate_count, dtype=np.int64),
            )
        )
        continuous = np.column_stack(
            (
                retrieval_scores[order],
                svd_scores,
                content_scores,
                cf_scores,
                self.item_popularity[order],
            )
        ).astype(np.float32)
        self.deepfm.eval()
        with torch.inference_mode():
            ranked_scores = self.deepfm(
                torch.from_numpy(categorical).to(device=self.device, dtype=torch.long),
                torch.from_numpy(continuous).to(device=self.device, dtype=torch.float32),
            ).cpu().numpy()
        ranked = np.lexsort((self.item_ids[order], -ranked_scores))
        if protected_item_rows.size:
            order_position = {int(item_row): index for index, item_row in enumerate(order)}
            protected = np.asarray(
                [
                    order_position[int(item_row)]
                    for item_row in protected_item_rows
                    if int(item_row) in order_position
                ],
                dtype=np.int64,
            )
            if protected.size:
                stable_ranks = np.arange(len(protected), dtype=np.float64)
                stable_rank_scores = 1.0 - stable_ranks / max(1, len(protected) - 1)
                deep_order = np.lexsort(
                    (self.item_ids[order[protected]], -ranked_scores[protected])
                )
                deep_rank_scores = np.empty(len(protected), dtype=np.float64)
                deep_rank_scores[deep_order] = 1.0 - np.arange(
                    len(protected), dtype=np.float64
                ) / max(1, len(protected) - 1)
                protected_scores = (
                    self.stable_rank_weight * stable_rank_scores
                    + (1.0 - self.stable_rank_weight) * deep_rank_scores
                )
                protected_ranked = protected[
                    np.lexsort(
                        (self.item_ids[order[protected]], -protected_scores)
                    )
                ]
                protected_set = set(map(int, protected))
                ranked = np.asarray(
                    [
                        *map(int, protected_ranked),
                        *(int(index) for index in ranked if int(index) not in protected_set),
                    ],
                    dtype=np.int64,
                )
        return [
            DeepRecommendation(
                item_id=str(self.item_ids[order[position]]),
                dssm_score=float(retrieval_scores[order[position]]),
                deepfm_score=float(ranked_scores[position]),
                rank=rank,
            )
            for rank, position in enumerate(ranked[:limit])
        ]

    def retrieve_dssm(
        self,
        *,
        dataset_user_id: str,
        feedback: Iterable[tuple[str, float]],
        limit: int,
    ) -> list[tuple[str, float]]:
        user_row = self.user_index.get(str(dataset_user_id))
        if user_row is None or limit <= 0:
            return []
        user_embedding = self.user_embeddings[user_row].astype(np.float32, copy=True)
        feedback_rows: list[tuple[int, float]] = []
        for item_id, affinity in feedback:
            item_row = self.item_index.get(str(item_id))
            if item_row is not None and math.isfinite(float(affinity)) and affinity > 0:
                feedback_rows.append((item_row, min(4.0, float(affinity))))
        if feedback_rows:
            weights = np.asarray([value for _, value in feedback_rows], dtype=np.float32)
            rows = np.asarray([row for row, _ in feedback_rows], dtype=np.int64)
            user_embedding += 0.2 * np.average(
                self.item_embeddings[rows], axis=0, weights=weights
            )
            user_embedding /= max(1e-8, float(np.linalg.norm(user_embedding)))
        scores = self.item_embeddings @ user_embedding
        order = np.lexsort((self.item_ids, -scores))[: min(limit, len(self.item_ids))]
        return [(str(self.item_ids[row]), float(scores[row])) for row in order]

    def rank_unified(
        self,
        *,
        dataset_user_id: str,
        candidates: Iterable[UnifiedCandidate],
        limit: int,
    ) -> list[DeepRecommendation]:
        if self.serving_mode not in {"unified_multisource_v2", "unified_multisource_v3"}:
            raise ArtifactError("deep artifact does not support unified ranking")
        user_row = self.user_index.get(str(dataset_user_id))
        values = list(candidates)
        if user_row is None or not values or limit <= 0:
            return []
        item_rows = np.asarray(
            [self.item_index.get(value.item_id, -1) for value in values], dtype=np.int64
        )
        keep = item_rows >= 0
        values = [value for value, allowed in zip(values, keep) if bool(allowed)]
        item_rows = item_rows[keep]
        if not values:
            return []
        source_index = {name: index + 1 for index, name in enumerate(UNIFIED_SOURCE_ORDER)}
        categorical = np.column_stack(
            (
                self.item_pop_bucket[item_rows],
                np.full(len(values), self.user_history_bucket[user_row], dtype=np.int64),
                np.asarray(
                    [source_index[value.primary_source] for value in values], dtype=np.int64
                ),
            )
        )
        history_density = (
            float(self.user_history_density[user_row])
            if self.user_history_density is not None
            else float(self.user_history_bucket[user_row])
            / max(1.0, float(self.user_history_bucket.max()))
        )
        continuous = np.asarray(
            [
                [
                    *(
                        value.feature_values()
                        if self.serving_mode == "unified_multisource_v3"
                        else (*value.source_scores, *value.source_mask)
                    ),
                    history_density,
                    float(self.item_model_indices[item_row] == 0),
                    value.source_mask[UNIFIED_SOURCE_ORDER.index("visual")],
                ]
                for value, item_row in zip(values, item_rows)
            ],
            dtype=np.float32,
        )
        if continuous.shape[1] != self.deepfm.config.continuous_dim:
            raise ArtifactError("unified feature schema does not match DeepFM")
        self.deepfm.eval()
        with torch.inference_mode():
            scores = self.deepfm(
                torch.from_numpy(categorical).to(device=self.device, dtype=torch.long),
                torch.from_numpy(continuous).to(device=self.device, dtype=torch.float32),
            ).cpu().numpy()
        if self.serving_mode == "unified_multisource_v3":
            order = np.lexsort(
                (np.asarray([value.item_id for value in values]), -scores)
            )[: min(limit, len(values))]
            return [
                DeepRecommendation(
                    item_id=values[position].item_id,
                    dssm_score=float(
                        values[position].source_scores[
                            UNIFIED_SOURCE_ORDER.index("dssm")
                        ]
                    ),
                    deepfm_score=float(scores[position]),
                    rank=rank,
                )
                for rank, position in enumerate(order)
            ]
        deep_order = np.lexsort(
            (np.asarray([value.item_id for value in values]), -scores)
        )
        deep_normalized = np.empty(len(values), dtype=np.float64)
        deep_normalized[deep_order] = 1.0 - np.arange(len(values), dtype=np.float64) / max(
            1, len(values) - 1
        )
        cold_ids = {
            value.item_id
            for value, item_row in zip(values, item_rows)
            if self.item_model_indices[item_row] == 0
        }
        prior = safe_quality_prior(values, cold_item_ids=cold_ids)
        ranked_scores = (
            (1.0 - self.stable_rank_weight) * deep_normalized
            + self.stable_rank_weight
            * np.asarray([prior[value.item_id] for value in values], dtype=np.float64)
        )
        order = np.lexsort(
            (np.asarray([value.item_id for value in values]), -ranked_scores)
        )[: min(limit, len(values))]
        return [
            DeepRecommendation(
                item_id=values[position].item_id,
                dssm_score=float(
                    values[position].source_scores[UNIFIED_SOURCE_ORDER.index("dssm")]
                ),
                deepfm_score=float(ranked_scores[position]),
                rank=rank,
            )
            for rank, position in enumerate(order)
        ]


class DeepArtifactStore:
    REQUIRED_FILES = (
        "dssm.safetensors",
        "deepfm.safetensors",
        "dssm_config.json",
        "deepfm_config.json",
        "metrics.json",
        "training.json",
        "deep_user_ids.npy",
        "deep_item_ids.npy",
        "deep_item_model_indices.npy",
        "deep_user_embeddings.npy",
        "deep_item_embeddings.npy",
        "deep_user_profiles.npy",
        "deep_item_content.npy",
        "deep_item_popularity.npy",
        "deep_item_pop_bucket.npy",
        "deep_user_history_bucket.npy",
    )

    def __init__(self, pointer_path: Path | str, device: str = "auto"):
        self.pointer_path = Path(pointer_path).resolve()
        self.device = resolve_torch_device(device)
        self._lock = threading.Lock()
        self._cache_key: tuple[int, int] | None = None
        self._artifact: DeepExperimentArtifact | None = None
        self.last_error: str | None = None

    def get(self) -> DeepExperimentArtifact | None:
        try:
            stat = self.pointer_path.stat()
            key = (stat.st_mtime_ns, stat.st_size)
        except OSError as exc:
            self.last_error = f"experiment pointer unavailable: {exc}"
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

    def _load(self) -> DeepExperimentArtifact:
        pointer = self._read_json(self.pointer_path)
        manifest_path = (self.pointer_path.parent / str(pointer["manifest"])).resolve()
        self._assert_child(manifest_path, self.pointer_path.parent)
        manifest = self._read_json(manifest_path)
        if manifest.get("schema_version") != 1:
            raise ArtifactError("unsupported deep experiment schema")
        if manifest.get("artifact_type") not in {
            "dssm_deepfm_experiment",
            "dssm_deepfm_model",
        }:
            raise ArtifactError("unexpected deep artifact type")
        files = manifest.get("files") or {}
        for name in self.REQUIRED_FILES:
            path = (manifest_path.parent / name).resolve()
            self._assert_child(path, manifest_path.parent)
            expected = (files.get(name) or {}).get("sha256")
            if not path.is_file() or not expected or self._sha256(path) != expected:
                raise ArtifactError(f"invalid deep artifact: {name}")
        arrays = {
            name: np.load(manifest_path.parent / name, allow_pickle=False)
            for name in self.REQUIRED_FILES
            if name.endswith(".npy")
        }
        user_ids = arrays["deep_user_ids.npy"]
        item_ids = arrays["deep_item_ids.npy"]
        user_embeddings = arrays["deep_user_embeddings.npy"]
        item_embeddings = arrays["deep_item_embeddings.npy"]
        user_profiles = arrays["deep_user_profiles.npy"]
        item_content = arrays["deep_item_content.npy"]
        if user_ids.ndim != 1 or item_ids.ndim != 1:
            raise ArtifactError("deep id arrays must be one dimensional")
        if user_embeddings.shape[0] != len(user_ids) or item_embeddings.shape[0] != len(item_ids):
            raise ArtifactError("deep embedding/id shape mismatch")
        if user_profiles.shape[0] != len(user_ids) or item_content.shape[0] != len(item_ids):
            raise ArtifactError("deep profile/content shape mismatch")
        for value in arrays.values():
            if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
                raise ArtifactError("deep artifact contains non-finite values")
        config_payload = self._read_json(manifest_path.parent / "deepfm_config.json")
        config = DeepFMConfig(
            categorical_sizes=tuple(int(value) for value in config_payload["categorical_sizes"]),
            continuous_dim=int(config_payload["continuous_dim"]),
            embedding_dim=int(config_payload["embedding_dim"]),
            hidden_dims=tuple(int(value) for value in config_payload["hidden_dims"]),
            dropout=float(config_payload["dropout"]),
        )
        deepfm = DeepFM(config).to(self.device)
        deepfm.load_state_dict(load_file(manifest_path.parent / "deepfm.safetensors"))
        deepfm.eval()
        with torch.inference_mode():
            warmup = deepfm(
                torch.zeros(
                    (1, len(config.categorical_sizes)),
                    dtype=torch.long,
                    device=self.device,
                ),
                torch.zeros(
                    (1, config.continuous_dim),
                    dtype=torch.float32,
                    device=self.device,
                ),
            )
        if warmup.shape != (1,) or not torch.isfinite(warmup).all():
            raise ArtifactError("deep model warmup failed")
        serving = dict(manifest.get("serving") or {})
        serving_mode = str(serving.get("mode", "legacy_protected_rerank"))
        if serving_mode not in {
            "legacy_protected_rerank",
            "unified_multisource_v2",
            "unified_multisource_v3",
        }:
            raise ArtifactError("unsupported deep serving mode")
        rank_strategy = str(serving.get("rank_strategy", "rank_fusion"))
        if rank_strategy not in {
            "rank_fusion",
            "protected_top10_rerank",
            "unified_deepfm",
        }:
            raise ArtifactError("unsupported deep rank strategy")
        stable_rank_weight = float(serving.get("stable_rank_weight", 0.0))
        if not math.isfinite(stable_rank_weight) or not 0.0 <= stable_rank_weight <= 1.0:
            raise ArtifactError("invalid stable rank weight")
        history_density = None
        required_multimodal = None
        density_path = manifest_path.parent / "deep_user_history_density.npy"
        if serving_mode in {"unified_multisource_v2", "unified_multisource_v3"}:
            expected = (files.get("deep_user_history_density.npy") or {}).get("sha256")
            if (
                not density_path.is_file()
                or not expected
                or self._sha256(density_path) != expected
            ):
                raise ArtifactError("invalid deep artifact: deep_user_history_density.npy")
            history_density = np.load(density_path, allow_pickle=False).astype(
                np.float32, copy=False
            )
            if history_density.shape != (len(user_ids),) or not np.isfinite(
                history_density
            ).all():
                raise ArtifactError("invalid deep user history density")
            expected_width = len(UNIFIED_SOURCE_ORDER) * (
                3 if serving_mode == "unified_multisource_v3" else 2
            ) + 3
            if config.continuous_dim != expected_width:
                raise ArtifactError("unsupported unified DeepFM feature width")
            required_multimodal = str(serving.get("required_multimodal_version") or "")
            if not required_multimodal:
                raise ArtifactError("unified deep artifact does not pin multimodal version")
            if rank_strategy != "unified_deepfm":
                raise ArtifactError("unified deep artifact has an incompatible rank strategy")
            if serving_mode == "unified_multisource_v3" and stable_rank_weight != 0.0:
                raise ArtifactError("unified v3 must not blend a stable ranking prior")
            source_limits = dict(serving.get("source_limits") or {})
            source_caps = dict(serving.get("source_caps_at_10") or {})
            if set(source_limits) != set(UNIFIED_SOURCE_ORDER):
                raise ArtifactError("unified deep artifact has incomplete source limits")
            if set(source_caps) != set(UNIFIED_SOURCE_ORDER):
                raise ArtifactError("unified deep artifact has incomplete source caps")
        return DeepExperimentArtifact(
            model_version=str(manifest["model_version"]),
            base_model_version=str(manifest["base_model_version"]),
            data_version=str(manifest["data_version"]),
            manifest_path=manifest_path,
            metrics=dict(manifest.get("metrics") or {}),
            user_ids=user_ids,
            item_ids=item_ids,
            item_model_indices=arrays["deep_item_model_indices.npy"],
            user_embeddings=user_embeddings.astype(np.float32, copy=False),
            item_embeddings=item_embeddings.astype(np.float32, copy=False),
            user_profiles=user_profiles.astype(np.float32, copy=False),
            item_content=item_content.astype(np.float32, copy=False),
            item_popularity=arrays["deep_item_popularity.npy"].astype(np.float32, copy=False),
            item_pop_bucket=arrays["deep_item_pop_bucket.npy"].astype(np.int64, copy=False),
            user_history_bucket=arrays["deep_user_history_bucket.npy"].astype(np.int64, copy=False),
            deepfm=deepfm,
            device=self.device,
            retrieval_top_n=int(serving.get("retrieval_top_n", 200)),
            rank_strategy=rank_strategy,
            stable_rank_weight=stable_rank_weight,
            serving_mode=serving_mode,
            required_multimodal_version=required_multimodal,
            source_limits={
                str(key): int(value)
                for key, value in dict(serving.get("source_limits") or {}).items()
            },
            source_caps_at_10={
                str(key): int(value)
                for key, value in dict(serving.get("source_caps_at_10") or {}).items()
            },
            user_history_density=history_density,
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
            raise ArtifactError("deep artifact path escapes root") from exc

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
