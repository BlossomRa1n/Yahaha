from __future__ import annotations

import json
import math
import random
import sqlite3
from dataclasses import dataclass
from typing import Any

import numpy as np

from .artifacts import ArtifactStore, ModelArtifact


@dataclass(frozen=True)
class Candidate:
    item_id: str
    source: str
    score: float
    explanation: str
    model_version: str | None
    is_forced: bool = False
    desired_position: int = 0
    priority: int = 0


@dataclass(frozen=True)
class RecommendationResult:
    candidates: list[Candidate]
    model_version: str | None
    fallback_reason: str | None
    has_more: bool


class RecommendationService:
    def __init__(self, artifacts: ArtifactStore):
        self.artifacts = artifacts

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
        pool_size = max(limit * 20, 200)

        if feed_type == "personalized":
            ordinary, fallback_reason = self._personalized(
                conn,
                artifact=artifact,
                dataset_user_id=user["dataset_user_id"],
                user_id=user["id"],
                pool_size=pool_size,
            )
        elif feed_type == "popular":
            ordinary = self._popular(conn, artifact=artifact, pool_size=pool_size)
        else:
            ordinary = self._explore(conn, user_id=user["id"], seed=seed, pool_size=pool_size)

        seen = {
            str(row["item_id"])
            for row in conn.execute(
                "SELECT DISTINCT item_id FROM exposures WHERE user_id = ?",
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
        merged = self._place_boosts(ordinary, boosts, limit)

        # The final authoritative filter runs after boost insertion and immediately
        # before the caller persists request facts in the same transaction.
        validation_pool = self._deduplicate([*merged, *ordinary])
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
            merged = [candidate for candidate in merged if candidate.item_id in online]

        selected_ids = {candidate.item_id for candidate in merged}
        for candidate in ordinary:
            if len(merged) >= limit:
                break
            if candidate.item_id in online and candidate.item_id not in selected_ids:
                merged.append(candidate)
                selected_ids.add(candidate.item_id)

        has_more = any(
            candidate.item_id in online and candidate.item_id not in selected_ids
            for candidate in ordinary
        )
        return RecommendationResult(
            candidates=merged[:limit],
            model_version=model_version,
            fallback_reason=fallback_reason,
            has_more=has_more,
        )

    def _personalized(
        self,
        conn: sqlite3.Connection,
        *,
        artifact: ModelArtifact | None,
        dataset_user_id: str | None,
        user_id: str,
        pool_size: int,
    ) -> tuple[list[Candidate], str | None]:
        if artifact is None:
            return self._popular(conn, artifact=None, pool_size=pool_size), "model_unavailable"
        user_index = artifact.user_index.get(str(dataset_user_id)) if dataset_user_id else None
        if user_index is None:
            return self._popular(conn, artifact=artifact, pool_size=pool_size), "cold_start"

        base_vector = artifact.user_factors[user_index]
        feedback_vector = np.zeros_like(base_vector, dtype=np.float64)
        feedback_rows = conn.execute(
            """
            SELECT item_id, affinity
            FROM user_item_state
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
            similarity = (artifact.item_factors @ feedback_vector) / np.maximum(norms, 1e-12)
            scores = scores + 0.35 * similarity
        order = np.argsort(-scores, kind="stable")[:pool_size]
        explanation = (
            "Latent affinity plus recent positive feedback"
            if feedback_count
            else "Learned latent affinity from chronological history"
        )
        return [
            Candidate(
                item_id=str(artifact.item_ids[index]),
                source="model",
                score=float(scores[index]),
                explanation=explanation,
                model_version=artifact.model_version,
            )
            for index in order
            if math.isfinite(float(scores[index]))
        ], None

    def _popular(
        self,
        conn: sqlite3.Connection,
        *,
        artifact: ModelArtifact | None,
        pool_size: int,
    ) -> list[Candidate]:
        if artifact and artifact.popularity:
            return [
                Candidate(
                    item_id=item_id,
                    source="popular",
                    score=score,
                    explanation="Train-only popularity baseline",
                    model_version=artifact.model_version,
                )
                for item_id, score in artifact.popularity[:pool_size]
            ]
        rows = conn.execute(
            """
            SELECT item_id, popularity_score
            FROM items
            WHERE status = 'online'
            ORDER BY popularity_score DESC, item_id
            LIMIT ?
            """,
            (pool_size,),
        ).fetchall()
        return [
            Candidate(
                item_id=str(row["item_id"]),
                source="fallback" if artifact is None else "popular",
                score=float(row["popularity_score"]),
                explanation="Database popularity fallback" if artifact is None else "Popularity baseline",
                model_version=artifact.model_version if artifact else None,
            )
            for row in rows
        ]

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
