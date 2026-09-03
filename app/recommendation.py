from __future__ import annotations

import json
import math
import random
import re
import sqlite3
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from .artifacts import ArtifactStore, ModelArtifact
from .deep_artifacts import DeepArtifactStore
from .multimodal_artifacts import MultimodalArtifactStore
from recsys.mixing import (
    DYNAMIC_POLICY_VERSION,
    MixCandidate as Candidate,
    MixContext,
    mix_candidates,
)
from recsys.two_stage import (
    UNIFIED_SOURCE_ORDER,
    apply_source_caps,
    merge_candidate_sources,
)
from recsys.popularity import (
    DEFAULT_HALF_LIFE_DAYS,
    build_popularity_features,
    snapshot_is_available,
    timestamp_to_ms,
)


@dataclass(frozen=True)
class RecommendationResult:
    candidates: list[Candidate]
    model_version: str | None
    fallback_reason: str | None
    has_more: bool
    diversity_metrics: dict[str, Any]
    candidate_manifest: tuple[Any, ...] = ()


class RecommendationService:
    def __init__(
        self,
        artifacts: ArtifactStore,
        deep_artifacts: DeepArtifactStore | None = None,
        multimodal_artifacts: MultimodalArtifactStore | None = None,
    ):
        self.artifacts = artifacts
        self.deep_artifacts = deep_artifacts
        self.multimodal_artifacts = multimodal_artifacts

    def recommend(
        self,
        conn: sqlite3.Connection,
        *,
        user: sqlite3.Row,
        feed_type: str,
        limit: int,
        seed: int,
        include_boosts: bool,
        now: str,
    ) -> RecommendationResult:
        artifact = self.artifacts.get()
        model_version = artifact.model_version if artifact else None
        fallback_reason: str | None = None
        mix_metrics: dict[str, Any] = {
            "strategy": "single_source",
            "selected": {},
            "relaxations": [],
        }
        pool_size = max(limit * 20, 200)

        if feed_type == "personalized":
            ordinary: list[Candidate] = []
            candidate_manifest: tuple[Any, ...] = ()
            deep_error: str | None = None
            deep_artifact = self.deep_artifacts.get() if self.deep_artifacts else None
            if deep_artifact is not None and artifact is not None and user["dataset_user_id"]:
                try:
                    ordinary, candidate_manifest = self._deep_personalized(
                        conn,
                        artifact=artifact,
                        dataset_user_id=str(user["dataset_user_id"]),
                        user_id=str(user["id"]),
                        pool_size=pool_size,
                        result_limit=limit,
                        now=now,
                        seed=seed,
                    )
                except Exception as exc:
                    deep_error = f"{type(exc).__name__}: {exc}"
                if ordinary:
                    model_version = ordinary[0].model_version or deep_artifact.model_version
                    if deep_artifact.serving_mode in {
                        "unified_multisource_v2",
                        "unified_multisource_v3",
                    }:
                        selected: dict[str, int] = {}
                        for candidate in ordinary:
                            selected[candidate.source] = selected.get(candidate.source, 0) + 1
                        mix_metrics = {
                            "strategy": "unified_two_stage_v2",
                            "mix_policy_version": "unified_two_stage_v2",
                            "selected": selected,
                            "relaxations": [],
                            "base_model_version": artifact.model_version,
                            "deep_model_version": deep_artifact.model_version,
                            "multimodal_model_version": deep_artifact.required_multimodal_version,
                            "source_limits": deep_artifact.source_limits,
                            "source_caps_at_10": deep_artifact.source_caps_at_10,
                        }
                    else:
                        source_name = ordinary[0].source
                        mix_metrics = {
                            "strategy": source_name,
                            "selected": {source_name: len(ordinary)},
                            "relaxations": [],
                            "base_model_version": artifact.model_version,
                        }
            if not ordinary:
                candidate_manifest = ()
                ordinary = self._normalize_source(
                    self._popular(conn, artifact=artifact, pool_size=pool_size, now=now)
                )
                if deep_error:
                    fallback_reason = "deep_experiment_failed"
                elif not user["dataset_user_id"]:
                    fallback_reason = "cold_start"
                else:
                    fallback_reason = "unified_model_unavailable"
                mix_metrics = {
                    "strategy": "fallback",
                    "selected": {"popular": len(ordinary)},
                    "relaxations": [fallback_reason],
                }
                if deep_error:
                    mix_metrics["deep_fallback_error"] = deep_error
        elif feed_type == "popular":
            candidate_manifest = ()
            ordinary = self._normalize_source(
                self._popular(conn, artifact=artifact, pool_size=pool_size, now=now)
            )
        else:
            candidate_manifest = ()
            ordinary = self._normalize_source(
                self._explore(conn, user_id=user["id"], seed=seed, pool_size=pool_size)
            )

        seen = {
            str(row["item_id"])
            for row in conn.execute(
                """
                SELECT DISTINCT item_id
                FROM events
                WHERE user_id = ?
                  AND event_type IN ('impression', 'click', 'like', 'not_interested')
                """,
                (user["id"],),
            )
        }
        negative = {
            str(row["item_id"])
            for row in conn.execute(
                "SELECT item_id FROM user_item_state WHERE user_id = ? AND not_interested = 1",
                (user["id"],),
            )
        }
        ordinary = self._deduplicate(
            candidate
            for candidate in ordinary
            if candidate.item_id not in seen and candidate.item_id not in negative
        )

        if len(ordinary) < limit:
            fill = self._explore(conn, user_id=user["id"], seed=seed, pool_size=pool_size)
            existing = {candidate.item_id for candidate in ordinary}
            ordinary.extend(
                candidate
                for candidate in fill
                if candidate.item_id not in existing
                and candidate.item_id not in seen
                and candidate.item_id not in negative
            )
            ordinary = self._deduplicate(ordinary)
            if feed_type == "personalized" and fallback_reason is None:
                fallback_reason = "candidate_shortfall"

        boosts = (
            self._boosts(conn, user_id=user["id"], feed_type=feed_type, now=now)
            if include_boosts
            else []
        )
        # Offline is authoritative before boosts and diversity. The final query below
        # repeats this guard immediately before persistence to close race-like gaps.
        validation_pool = self._deduplicate([*ordinary, *boosts])
        online: set[str] = set()
        if validation_pool:
            placeholders = ",".join("?" for _ in validation_pool)
            online = {
                str(row["item_id"])
                for row in conn.execute(
                    f"SELECT item_id FROM items WHERE status = 'online' AND item_id IN ({placeholders})",
                    tuple(candidate.item_id for candidate in validation_pool),
                )
            }
        ordinary = [candidate for candidate in ordinary if candidate.item_id in online]
        boosts = [candidate for candidate in boosts if candidate.item_id in online]
        ordinary = ordinary[:limit]
        diverse, diversity_metrics = self._rerank_diverse(conn, ordinary)
        diversity_metrics["source_mix"] = mix_metrics
        merged = self._place_boosts(diverse, boosts, limit)

        if merged:
            placeholders = ",".join("?" for _ in merged)
            online = {
                str(row["item_id"])
                for row in conn.execute(
                    f"SELECT item_id FROM items WHERE status = 'online' AND item_id IN ({placeholders})",
                    tuple(candidate.item_id for candidate in merged),
                )
            }
            merged = [candidate for candidate in merged if candidate.item_id in online]

        selected_ids = {candidate.item_id for candidate in merged}
        for candidate in diverse:
            if len(merged) >= limit:
                break
            if candidate.item_id in online and candidate.item_id not in selected_ids:
                merged.append(candidate)
                selected_ids.add(candidate.item_id)

        has_more = any(
            candidate.item_id in online and candidate.item_id not in selected_ids
            for candidate in diverse
        )
        return RecommendationResult(
            candidates=merged[:limit],
            model_version=model_version,
            fallback_reason=fallback_reason,
            has_more=has_more,
            diversity_metrics=diversity_metrics,
            candidate_manifest=candidate_manifest,
        )

    def _deep_personalized(
        self,
        conn: sqlite3.Connection,
        *,
        artifact: ModelArtifact,
        dataset_user_id: str,
        user_id: str,
        pool_size: int,
        result_limit: int,
        now: str,
        seed: int,
    ) -> tuple[list[Candidate], tuple[Any, ...]]:
        if self.deep_artifacts is None:
            return [], ()
        deep = self.deep_artifacts.get()
        if deep is None:
            return [], ()
        if deep.serving_mode in {"unified_multisource_v2", "unified_multisource_v3"}:
            return self._unified_deep_personalized(
                conn,
                artifact=artifact,
                deep=deep,
                dataset_user_id=dataset_user_id,
                user_id=user_id,
                result_limit=result_limit,
                now=now,
                seed=seed,
            )
        feedback = [
            (str(row["item_id"]), float(row["affinity"]))
            for row in conn.execute(
                """
                SELECT item_id, affinity
                FROM user_item_state
                WHERE user_id = ? AND affinity > 0 AND not_interested = 0
                ORDER BY last_event_at, item_id
                """,
                (user_id,),
            )
        ]
        ranked = deep.recommend(
            dataset_user_id=dataset_user_id,
            feedback=feedback,
            stable=artifact,
            limit=result_limit,
        )
        multimodal = self.multimodal_artifacts.get() if self.multimodal_artifacts else None
        multimodal_by_item = {}
        if multimodal is not None and ranked:
            fused = multimodal.rerank(
                dataset_user_id=dataset_user_id,
                candidates=[(value.item_id, value.deepfm_score) for value in ranked],
                stable=artifact,
            )
            ranked_by_item = {value.item_id: value for value in ranked}
            ranked = [ranked_by_item[value.item_id] for value in fused]
            multimodal_by_item = {value.item_id: value for value in fused}
        denominator = max(1, len(ranked) - 1)
        return [
            Candidate(
                item_id=value.item_id,
                source=(
                    "dssm_deepfm_multimodal" if multimodal_by_item else "dssm_deepfm"
                ),
                score=1.0 - rank / denominator,
                raw_score=value.deepfm_score,
                normalized_score=1.0 - rank / denominator,
                rank_in_source=rank,
                explanation=(
                    "DSSM personalized retrieval followed by DeepFM ranking; "
                    f"retrieval cosine={value.dssm_score:.4f}"
                    + (
                        "; MobileNet cover + title late fusion "
                        f"visual={multimodal_by_item[value.item_id].visual_score!s} "
                        f"fusion={multimodal_by_item[value.item_id].fusion_score:.4f}"
                        if multimodal_by_item
                        else ""
                    )
                ),
                model_version=(
                    f"{deep.model_version}+{multimodal.model_version}"
                    if multimodal is not None and multimodal_by_item
                    else deep.model_version
                ),
                eligible=math.isfinite(value.deepfm_score),
                confidence=1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, value.deepfm_score)))),
                support=max(1, len(feedback)),
                is_cold=str(value.item_id) not in artifact.item_index,
            )
            for rank, value in enumerate(ranked)
            if math.isfinite(value.deepfm_score)
        ], ()

    def _unified_deep_personalized(
        self,
        conn: sqlite3.Connection,
        *,
        artifact: ModelArtifact,
        deep: Any,
        dataset_user_id: str,
        user_id: str,
        result_limit: int,
        now: str,
        seed: int,
    ) -> tuple[list[Candidate], tuple[Any, ...]]:
        multimodal = self.multimodal_artifacts.get() if self.multimodal_artifacts else None
        if (
            multimodal is None
            or multimodal.model_version != deep.required_multimodal_version
            or multimodal.base_model_version != artifact.model_version
            or multimodal.data_version != deep.data_version
        ):
            raise RuntimeError("unified experiment multimodal version is unavailable or incompatible")
        feedback = [
            (str(row["item_id"]), float(row["affinity"]))
            for row in conn.execute(
                """
                SELECT item_id, affinity
                FROM user_item_state
                WHERE user_id = ? AND affinity > 0 AND not_interested = 0
                ORDER BY last_event_at, item_id
                """,
                (user_id,),
            )
        ]
        source_limits = dict(deep.source_limits or {})
        svd_candidates = self._svd_candidates(
            conn,
            artifact=artifact,
            dataset_user_id=dataset_user_id,
            user_id=user_id,
            pool_size=max(source_limits.get("svd", 150), 1),
        )
        dssm_candidates = [
            Candidate(
                item_id=item_id,
                source="dssm",
                score=score,
                explanation="DSSM full-catalog retrieval",
                model_version=deep.model_version,
                eligible=math.isfinite(score),
                confidence=(score + 1.0) / 2.0,
                support=max(1, len(feedback)),
                is_cold=item_id not in artifact.item_index,
            )
            for item_id, score in deep.retrieve_dssm(
                dataset_user_id=dataset_user_id,
                feedback=feedback,
                limit=max(source_limits.get("dssm", 200), 1),
            )
        ]
        visual_candidates = [
            Candidate(
                item_id=item_id,
                source="visual",
                score=score,
                explanation="MobileNet cover similarity to cutoff-safe visual profile",
                model_version=multimodal.model_version,
                eligible=math.isfinite(score),
                confidence=(score + 1.0) / 2.0,
                support=1,
                is_cold=item_id not in artifact.item_index,
            )
            for item_id, score in multimodal.retrieve_visual(
                dataset_user_id=dataset_user_id,
                limit=max(source_limits.get("visual", 150), 1),
            )
        ]
        sources = {
            "svd": svd_candidates,
            "dssm": dssm_candidates,
            "content": self._content_profile(
                conn,
                artifact=artifact,
                dataset_user_id=dataset_user_id,
                user_id=user_id,
                pool_size=max(source_limits.get("content", 150), 1),
            ),
            "visual": visual_candidates,
            "item_cf": self._item_cf(
                conn,
                artifact=artifact,
                dataset_user_id=dataset_user_id,
                user_id=user_id,
                pool_size=max(source_limits.get("item_cf", 100), 1),
            ),
            "popular": self._popular(
                conn,
                artifact=artifact,
                pool_size=max(source_limits.get("popular", 50), 1),
                now=now,
            ),
            "explore": self._explore(
                conn,
                user_id=user_id,
                seed=seed,
                pool_size=max(source_limits.get("explore", 50), 1),
            ),
        }
        excluded = {
            str(row["item_id"])
            for row in conn.execute(
                """
                SELECT DISTINCT item_id FROM events
                WHERE user_id = ?
                  AND event_type IN ('impression', 'click', 'like', 'not_interested')
                UNION
                SELECT item_id FROM user_item_state
                WHERE user_id = ? AND not_interested = 1
                """,
                (user_id, user_id),
            )
        }
        online = {
            str(row["item_id"])
            for row in conn.execute("SELECT item_id FROM items WHERE status = 'online'")
        }
        unified, union_diagnostics = merge_candidate_sources(
            sources,
            source_limits=source_limits,
            excluded_item_ids=excluded,
            eligible_item_ids=online,
        )
        ranked = deep.rank_unified(
            dataset_user_id=dataset_user_id,
            candidates=unified,
            limit=len(unified),
        )
        unified_by_item = {value.item_id: value for value in unified}
        ranked_by_item = {value.item_id: value for value in ranked}
        candidate_manifest = tuple(
            replace(
                value,
                ranker_score=(
                    ranked_by_item[value.item_id].deepfm_score
                    if value.item_id in ranked_by_item
                    else None
                ),
                ranker_rank=(
                    ranked_by_item[value.item_id].rank
                    if value.item_id in ranked_by_item
                    else None
                ),
            )
            for value in unified
        )
        denominator = max(1, len(ranked) - 1)
        values = []
        for rank, value in enumerate(ranked):
            unified_value = unified_by_item[value.item_id]
            active_sources = [
                name
                for name, present in zip(UNIFIED_SOURCE_ORDER, unified_value.source_mask)
                if present
            ]
            values.append(
                Candidate(
                    item_id=value.item_id,
                    source=unified_value.primary_source,
                    score=1.0 - rank / denominator,
                    raw_score=value.deepfm_score,
                    normalized_score=1.0 - rank / denominator,
                    rank_in_source=rank,
                    explanation=(
                        "Unified DeepFM rerank; sources="
                        + ",".join(active_sources)
                        + f"; dssm={value.dssm_score:.4f}"
                    ),
                    model_version=f"{deep.model_version}+{multimodal.model_version}",
                    eligible=True,
                    confidence=1.0
                    / (1.0 + math.exp(-max(-20.0, min(20.0, value.deepfm_score)))),
                    support=sum(int(flag) for flag in unified_value.source_mask),
                    is_cold=value.item_id not in artifact.item_index,
                )
            )
        capped, cap_diagnostics = apply_source_caps(
            values,
            limit=result_limit,
            caps_at_10=deep.source_caps_at_10,
        )
        candidate_details = (
            f"; union={union_diagnostics['union_size']}; "
            f"source_caps_relaxed={cap_diagnostics['relaxed']}"
        )
        return [
            replace(candidate, explanation=candidate.explanation + candidate_details)
            for candidate in capped
        ], candidate_manifest

    @staticmethod
    def _svd_candidates(
        conn: sqlite3.Connection,
        *,
        artifact: ModelArtifact,
        dataset_user_id: str,
        user_id: str,
        pool_size: int,
    ) -> list[Candidate]:
        user_index = artifact.user_index.get(str(dataset_user_id))
        if user_index is None:
            return []
        base_vector = artifact.user_factors[user_index]
        feedback_vector = np.zeros_like(base_vector, dtype=np.float64)
        feedback_rows = conn.execute(
            """
            SELECT item_id, affinity FROM user_item_state
            WHERE user_id = ? AND affinity != 0 AND not_interested = 0
            """,
            (user_id,),
        ).fetchall()
        feedback_count = 0
        for row in feedback_rows:
            item_index = artifact.item_index.get(str(row["item_id"]))
            if item_index is not None:
                feedback_vector += float(row["affinity"]) * artifact.item_factors[item_index]
                feedback_count += 1
        if feedback_count and np.linalg.norm(feedback_vector) > 0:
            feedback_vector /= np.linalg.norm(feedback_vector)
        scores = artifact.item_factors @ base_vector
        if feedback_count:
            norms = np.linalg.norm(artifact.item_factors, axis=1)
            scores = scores + 0.35 * (
                artifact.item_factors @ feedback_vector
            ) / np.maximum(norms, 1e-12)
        order = np.argsort(-scores, kind="stable")[:pool_size]
        return [
            Candidate(
                item_id=str(artifact.item_ids[index]),
                source="svd",
                score=float(scores[index]),
                explanation="SVD affinity plus cutoff-safe online profile",
                model_version=artifact.model_version,
                eligible=math.isfinite(float(scores[index])),
                confidence=0.5 + 0.5 * math.tanh(float(scores[index])),
                support=max(1, feedback_count),
            )
            for index in order
            if math.isfinite(float(scores[index]))
        ]

    @staticmethod
    def _title_tokens(title: str) -> frozenset[str]:
        return frozenset(
            token.lower()
            for token in re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", title)
            if len(token) > 1 or not token.isascii()
        )

    @staticmethod
    def _title_similarity(left: frozenset[str], right: frozenset[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)

    @classmethod
    def _diversity_metrics(
        cls,
        candidates: list[Candidate],
        tokens: dict[str, frozenset[str]],
    ) -> dict[str, float | int]:
        sample = candidates[:50]
        adjacent = [
            cls._title_similarity(tokens[left.item_id], tokens[right.item_id])
            for left, right in zip(sample, sample[1:])
        ]
        pairs = [
            cls._title_similarity(tokens[sample[left].item_id], tokens[sample[right].item_id])
            for left in range(len(sample))
            for right in range(left + 1, len(sample))
        ]
        return {
            "evaluated_items": len(sample),
            "duplicate_items": len(sample) - len({item.item_id for item in sample}),
            "unique_title_tokens": len(
                set().union(*(tokens[item.item_id] for item in sample)) if sample else set()
            ),
            "adjacent_title_similarity": sum(adjacent) / len(adjacent) if adjacent else 0.0,
            "intra_list_diversity": 1.0 - (sum(pairs) / len(pairs) if pairs else 0.0),
        }

    @classmethod
    def _rerank_diverse(
        cls,
        conn: sqlite3.Connection,
        candidates: list[Candidate],
    ) -> tuple[list[Candidate], dict[str, Any]]:
        if not candidates:
            empty_metrics = cls._diversity_metrics([], {})
            return [], {"strategy": "title_token_mmr", "before": empty_metrics, "after": empty_metrics}
        placeholders = ",".join("?" for _ in candidates)
        titles = {
            str(row["item_id"]): str(row["title"])
            for row in conn.execute(
                f"SELECT item_id, title FROM items WHERE item_id IN ({placeholders})",
                tuple(candidate.item_id for candidate in candidates),
            )
        }
        raw_tokens = {
            candidate.item_id: cls._title_tokens(titles.get(candidate.item_id, ""))
            for candidate in candidates
        }
        document_frequency: dict[str, int] = {}
        for values in raw_tokens.values():
            for token in values:
                document_frequency[token] = document_frequency.get(token, 0) + 1
        common = {
            token
            for token, count in document_frequency.items()
            if count / len(candidates) > 0.5
        }
        tokens = {
            item_id: frozenset(values - common)
            for item_id, values in raw_tokens.items()
        }
        before = cls._diversity_metrics(candidates, tokens)
        usable_ratio = sum(bool(values) for values in tokens.values()) / len(candidates)
        if usable_ratio < 0.6:
            return candidates, {
                "strategy": "title_token_mmr",
                "applied": False,
                "reason": "insufficient_title_metadata",
                "usable_title_ratio": usable_ratio,
                "before": before,
                "after": before,
            }
        remaining = list(enumerate(candidates))
        maximum_similarity = {index: 0.0 for index in range(len(candidates))}
        selected: list[Candidate] = []
        total = max(1, len(candidates) - 1)
        while remaining:
            def objective(entry: tuple[int, Candidate]) -> tuple[float, int, str]:
                index, candidate = entry
                relevance = 1.0 - index / total
                similarity = maximum_similarity[index]
                return (0.45 * relevance - 0.55 * similarity, -index, candidate.item_id)

            chosen_index, chosen = max(remaining, key=objective)
            penalty = maximum_similarity[chosen_index]
            selected.append(
                replace(
                    chosen,
                    explanation=(
                        f"{chosen.explanation}; diversity rerank title-overlap={penalty:.3f}"
                    ),
                )
            )
            remaining.remove((chosen_index, chosen))
            for index, candidate in remaining:
                maximum_similarity[index] = max(
                    maximum_similarity[index],
                    cls._title_similarity(tokens[candidate.item_id], tokens[chosen.item_id]),
                )
        after = cls._diversity_metrics(selected, tokens)
        return selected, {
            "strategy": "title_token_mmr",
            "applied": True,
            "lambda_relevance": 0.45,
            "before": before,
            "after": after,
        }

    @staticmethod
    def _normalize_source(candidates: list[Candidate]) -> list[Candidate]:
        denominator = max(1, len(candidates) - 1)
        return [
            replace(
                candidate,
                raw_score=(
                    float(candidate.raw_score)
                    if candidate.raw_score is not None
                    else float(candidate.score)
                ),
                normalized_score=1.0 - rank / denominator,
                rank_in_source=rank,
            )
            for rank, candidate in enumerate(candidates)
        ]

    @classmethod
    def _mix_sources(
        cls,
        sources: dict[str, list[Candidate]],
        limit: int,
        *,
        context: MixContext | None = None,
        policy_version: str = DYNAMIC_POLICY_VERSION,
    ) -> tuple[list[Candidate], dict[str, Any]]:
        return mix_candidates(policy_version, sources, limit, context)

    @staticmethod
    def _item_cf(
        conn: sqlite3.Connection,
        *,
        artifact: ModelArtifact,
        dataset_user_id: str | None,
        user_id: str,
        pool_size: int,
    ) -> list[Candidate]:
        if artifact.item_cf_neighbors is None or artifact.item_cf_user_history is None:
            return []
        weights: dict[int, float] = {}
        model_user_index = (
            artifact.user_index.get(str(dataset_user_id)) if dataset_user_id else None
        )
        if model_user_index is not None:
            history = artifact.item_cf_user_history[model_user_index]
            for item_index in history.indices:
                weights[int(item_index)] = 1.0
        rows = conn.execute(
            """
            SELECT item_id, affinity
            FROM user_item_state
            WHERE user_id = ? AND affinity > 0 AND not_interested = 0
            ORDER BY last_event_at, item_id
            """,
            (user_id,),
        ).fetchall()
        for row in rows:
            item_index = artifact.item_index.get(str(row["item_id"]))
            if item_index is None:
                continue
            weights[item_index] = min(
                4.0,
                weights.get(item_index, 0.0) + max(0.0, float(row["affinity"])),
            )
        if not weights:
            return []
        history_indices = np.asarray(sorted(weights), dtype=np.int64)
        history_weights = np.asarray(
            [weights[int(index)] for index in history_indices],
            dtype=np.float64,
        )
        scores = np.asarray(
            history_weights @ artifact.item_cf_neighbors[history_indices]
        ).ravel()
        finite = np.flatnonzero(np.isfinite(scores) & (scores > 0))
        if finite.size == 0:
            return []
        support_counts = np.asarray(
            artifact.item_cf_neighbors[history_indices][:, finite].getnnz(axis=0)
        ).ravel()
        support_by_index = {
            int(item_index): int(support)
            for item_index, support in zip(finite, support_counts)
        }
        order = finite[np.lexsort((artifact.item_ids[finite], -scores[finite]))]
        return [
            Candidate(
                item_id=str(artifact.item_ids[index]),
                source="item_cf",
                score=float(scores[index]),
                explanation="Similar to cutoff-safe positive-history items",
                model_version=artifact.model_version,
                eligible=bool(scores[index] > 0 and support_by_index.get(int(index), 0) > 0),
                confidence=min(1.0, float(scores[index])),
                support=support_by_index.get(int(index), 0),
            )
            for index in order[:pool_size]
        ]

    @staticmethod
    def _content_profile(
        conn: sqlite3.Connection,
        *,
        artifact: ModelArtifact,
        dataset_user_id: str | None,
        user_id: str,
        pool_size: int,
    ) -> list[Candidate]:
        if (
            artifact.content_item_ids is None
            or artifact.content_item_vectors is None
            or artifact.content_user_vectors is None
        ):
            return []
        dimension = artifact.content_item_vectors.shape[1]
        profile = csr_matrix((1, dimension), dtype=np.float64)
        content_user_index = (
            artifact.content_user_index.get(str(dataset_user_id)) if dataset_user_id else None
        )
        if content_user_index is not None:
            profile = artifact.content_user_vectors[content_user_index].astype(
                np.float64, copy=True
            )
        feedback_rows = conn.execute(
            """
            SELECT item_id, affinity
            FROM user_item_state
            WHERE user_id = ? AND affinity > 0 AND not_interested = 0
            ORDER BY last_event_at, item_id
            """,
            (user_id,),
        ).fetchall()
        feedback_vectors = []
        feedback_weights = []
        for row in feedback_rows:
            item_index = artifact.content_item_index.get(str(row["item_id"]))
            if item_index is None:
                continue
            feedback_vectors.append(artifact.content_item_vectors[item_index])
            feedback_weights.append(min(4.0, max(0.0, float(row["affinity"]))))
        if feedback_vectors:
            feedback = csr_matrix((1, dimension), dtype=np.float64)
            for vector, weight in zip(feedback_vectors, feedback_weights):
                feedback = feedback + weight * vector
            profile = profile + feedback
        if profile.nnz == 0:
            return []
        profile = normalize(profile, norm="l2", axis=1, copy=False)
        scores = np.asarray(
            (artifact.content_item_vectors @ profile.T).toarray().ravel(),
            dtype=np.float64,
        )
        finite = np.flatnonzero(np.isfinite(scores) & (scores > 0))
        if finite.size == 0:
            return []
        order = finite[np.lexsort((artifact.content_item_ids[finite], -scores[finite]))]
        return [
            Candidate(
                item_id=str(artifact.content_item_ids[index]),
                source="content_profile",
                score=float(scores[index]),
                explanation="Title similarity to cutoff-safe positive history",
                model_version=artifact.model_version,
                eligible=bool(scores[index] > 0),
                confidence=min(1.0, max(0.0, float(scores[index]))),
                support=max(1, len(feedback_vectors)),
                is_cold=str(artifact.content_item_ids[index]) not in artifact.item_index,
            )
            for index in order[:pool_size]
        ]

    def _popular(
        self,
        conn: sqlite3.Connection,
        *,
        artifact: ModelArtifact | None,
        pool_size: int,
        now: str,
    ) -> list[Candidate]:
        if artifact and artifact.popularity:
            return [
                Candidate(
                    item_id=item_id,
                    source="popular",
                    score=score,
                    explanation="Leakage-safe train interactions through the model cutoff",
                    model_version=artifact.model_version,
                    eligible=math.isfinite(float(score)),
                    confidence=1.0 if score > 0 else 0.1,
                    support=max(0, int(score)),
                )
                for item_id, score in artifact.popularity[:pool_size]
            ]
        cutoff_ms = timestamp_to_ms(now)
        semantics_row = conn.execute(
            """
            SELECT value FROM app_metadata
            WHERE key = 'viewable_impression_semantics_started_at'
            """
        ).fetchone()
        semantics_started_at = str(semantics_row["value"]) if semantics_row else now
        event_rows = conn.execute(
            """
            SELECT item_id, received_at
            FROM events
            WHERE event_type = 'impression'
              AND received_at >= ?
              AND received_at <= ?
            """,
            (semantics_started_at, now),
        ).fetchall()
        interaction_features = build_popularity_features(
            (
                (str(row["item_id"]), timestamp_to_ms(row["received_at"]), 1.0)
                for row in event_rows
            ),
            feature_cutoff_ms=cutoff_ms,
            half_life_days=DEFAULT_HALF_LIFE_DAYS,
        )
        rows = conn.execute(
            """
            SELECT
                i.item_id,
                i.popularity_score,
                i.stats_snapshot_version,
                i.stats_available_at,
                s.snapshot_version AS verified_snapshot_version
            FROM items i
            LEFT JOIN item_stats_snapshots s
              ON s.snapshot_version = i.stats_snapshot_version
             AND s.available_at = i.stats_available_at
            WHERE i.status = 'online'
            """,
        ).fetchall()
        candidates = []
        for row in rows:
            item_id = str(row["item_id"])
            features = interaction_features.get(item_id)
            cumulative = features.cumulative_interactions if features else 0.0
            decay = features.time_decay_score if features else 0.0
            raw_snapshot_allowed = snapshot_is_available(
                row["stats_available_at"],
                cutoff_ms,
            ) and row["verified_snapshot_version"] is not None
            snapshot_score = float(row["popularity_score"]) if raw_snapshot_allowed else 0.0
            score = decay + 1e-6 * cumulative + 1e-9 * snapshot_score
            sources = [
                "viewable impressions through current cutoff; "
                f"half_life_days={DEFAULT_HALF_LIFE_DAYS:g}"
            ]
            if raw_snapshot_allowed:
                sources.append(
                    f"current likes/views snapshot {row['stats_snapshot_version']} "
                    f"available at {row['stats_available_at']}"
                )
            elif row["stats_snapshot_version"] and row["verified_snapshot_version"] is None:
                sources.append("likes/views snapshot excluded: provenance unavailable")
            elif row["stats_snapshot_version"]:
                sources.append("likes/views snapshot excluded: unavailable at cutoff")
            candidates.append(
                Candidate(
                    item_id=item_id,
                    source="fallback" if artifact is None else "popular",
                    score=score,
                    explanation="; ".join(sources),
                    model_version=artifact.model_version if artifact else None,
                    eligible=math.isfinite(float(score)),
                    confidence=1.0 if score > 0 else 0.1,
                    support=max(0, int(cumulative)),
                )
            )
        return sorted(
            candidates,
            key=lambda candidate: (-candidate.score, candidate.item_id),
        )[:pool_size]

    def _explore(
        self,
        conn: sqlite3.Connection,
        *,
        user_id: str,
        seed: int,
        pool_size: int,
    ) -> list[Candidate]:
        rows = conn.execute(
            """
            SELECT i.item_id, COUNT(e.id) AS exposure_count
            FROM items i
            LEFT JOIN exposures e ON e.item_id = i.item_id
            WHERE i.status = 'online'
            GROUP BY i.item_id
            ORDER BY exposure_count ASC, i.item_id
            LIMIT ?
            """,
            (pool_size * 2,),
        ).fetchall()
        rng = random.Random(f"{seed}:{user_id}")
        candidates = [
            Candidate(
                item_id=str(row["item_id"]),
                source="explore",
                score=1.0 / (1.0 + int(row["exposure_count"])) + rng.random() * 0.01,
                explanation="Low-exposure discovery candidate",
                model_version=None,
                eligible=True,
                confidence=0.05,
                support=int(row["exposure_count"]),
            )
            for row in rows
        ]
        return sorted(candidates, key=lambda candidate: (-candidate.score, candidate.item_id))[:pool_size]

    @staticmethod
    def _boosts(
        conn: sqlite3.Connection,
        *,
        user_id: str,
        feed_type: str,
        now: str,
    ) -> list[Candidate]:
        rows = conn.execute(
            """
            SELECT * FROM boost_campaigns
            WHERE active = 1 AND starts_at <= ? AND ends_at > ?
            ORDER BY priority DESC, created_at ASC
            """,
            (now, now),
        ).fetchall()
        candidates = []
        for row in rows:
            user_ids = json.loads(row["user_ids_json"])
            feed_types = json.loads(row["feed_types_json"])
            if row["audience"] == "users" and user_id not in user_ids:
                continue
            if feed_types and feed_type not in feed_types:
                continue
            candidates.append(
                Candidate(
                    item_id=str(row["item_id"]),
                    source="forced",
                    score=1000.0 + float(row["priority"]),
                    explanation=f"Operational boost: {row['reason']}",
                    model_version=None,
                    is_forced=True,
                    desired_position=int(row["position"]),
                    priority=int(row["priority"]),
                )
            )
        return RecommendationService._deduplicate(candidates)

    @staticmethod
    def _place_boosts(
        ordinary: list[Candidate],
        boosts: list[Candidate],
        limit: int,
    ) -> list[Candidate]:
        boost_ids = {candidate.item_id for candidate in boosts}
        result = [candidate for candidate in ordinary if candidate.item_id not in boost_ids]
        for candidate in reversed(sorted(boosts, key=lambda value: (-value.priority, value.item_id))):
            position = min(candidate.desired_position, len(result), max(0, limit - 1))
            result.insert(position, candidate)
        return RecommendationService._deduplicate(result)[:limit]

    @staticmethod
    def _deduplicate(values: Any) -> list[Candidate]:
        result: list[Candidate] = []
        seen: set[str] = set()
        for value in values:
            if value.item_id in seen:
                continue
            seen.add(value.item_id)
            result.append(value)
        return result
