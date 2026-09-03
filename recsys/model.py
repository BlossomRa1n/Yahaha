from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .artifacts import publish_artifact, write_staged_artifact
from .content import build_content_features
from .item_cf import build_item_cf
from .mixing import (
    DYNAMIC_POLICY_VERSION,
    SAFE_POLICY_VERSION,
    MixCandidate,
    MixContext,
    mix_candidates,
)
from .popularity import (
    DEFAULT_HALF_LIFE_DAYS,
    WINDOW_DAYS,
    build_popularity_features,
)


NEGATIVES_PER_QUERY = 100
METRIC_K = 10
SAMPLED_NEGATIVE_PROTOCOL = "deterministic_sampled_negatives_v1"
SAMPLED_ALL_ITEMS_KEY = "sampled_all_items"
LEGACY_SAMPLED_ALL_ITEMS_KEY = "full_catalog"


class ModelTrainingError(ValueError):
    pass


@dataclass(frozen=True)
class EvaluationQuery:
    user_id: int
    positive_item_ids: tuple[int, ...]
    negative_item_ids: tuple[int, ...]


@dataclass(frozen=True)
class CatalogEvaluationQuery:
    """A query evaluated against every eligible catalog item."""

    user_id: int
    positive_item_ids: tuple[int, ...]
    excluded_item_ids: tuple[int, ...]


def _read_processed_interactions(path: Path) -> list[tuple[int, int, int]]:
    rows: list[tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["user_id", "item_id", "timestamp_ms"]:
            raise ModelTrainingError(f"{path}: invalid processed interaction schema")
        for line_number, row in enumerate(reader, start=2):
            try:
                values = (int(row["user_id"]), int(row["item_id"]), int(row["timestamp_ms"]))
            except (TypeError, ValueError) as exc:
                raise ModelTrainingError(f"{path}: line {line_number} has non-integer values") from exc
            rows.append(values)
    return rows


def _read_online_feedback(path: Path) -> list[tuple[int, int, int, str, float]]:
    if not path.is_file():
        return []
    rows: list[tuple[int, int, int, str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = [
            "event_id",
            "user_id",
            "item_id",
            "timestamp_ms",
            "event_type",
            "weight",
            "disposition",
            "received_at",
        ]
        if reader.fieldnames != expected:
            raise ModelTrainingError(f"{path}: invalid online feedback schema")
        event_ids: set[str] = set()
        for line_number, row in enumerate(reader, start=2):
            event_id = str(row["event_id"])
            if event_id in event_ids:
                raise ModelTrainingError(f"{path}: duplicate event_id at line {line_number}")
            event_ids.add(event_id)
            try:
                values = (
                    int(row["user_id"]),
                    int(row["item_id"]),
                    int(row["timestamp_ms"]),
                    str(row["event_type"]),
                    float(row["weight"]),
                )
            except (TypeError, ValueError) as exc:
                raise ModelTrainingError(
                    f"{path}: line {line_number} has invalid feedback values"
                ) from exc
            if not math.isfinite(values[4]):
                raise ModelTrainingError(f"{path}: line {line_number} has non-finite weight")
            rows.append(values)
    return rows


def _stable_user_key(seed: int, user_id: int, purpose: str) -> bytes:
    return hashlib.sha256(f"{seed}:{purpose}:{user_id}".encode()).digest()


def _select_users(
    user_ids: Iterable[int], *, seed: int, limit: int | None, purpose: str
) -> list[int]:
    unique = sorted(set(user_ids))
    if limit is None or limit >= len(unique):
        return unique
    return sorted(
        unique,
        key=lambda user_id: (_stable_user_key(seed, user_id, purpose), user_id),
    )[:limit]


def _per_user_seed(seed: int, user_id: int, split_name: str) -> int:
    digest = hashlib.sha256(f"{seed}:{split_name}:{user_id}".encode()).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _sample_negatives(
    item_ids: np.ndarray,
    excluded: set[int],
    *,
    count: int,
    seed: int,
) -> tuple[int, ...]:
    if len(item_ids) - len(excluded.intersection(map(int, item_ids))) < count:
        return ()
    rng = np.random.default_rng(seed)
    selected: set[int] = set()
    while len(selected) < count:
        needed = count - len(selected)
        indices = rng.integers(0, len(item_ids), size=max(needed * 2, 32))
        for index in indices:
            item_id = int(item_ids[int(index)])
            if item_id not in excluded:
                selected.add(item_id)
                if len(selected) == count:
                    break
    return tuple(sorted(selected))


def _query_checksum(queries: Sequence[EvaluationQuery]) -> str:
    digest = hashlib.sha256()
    for query in queries:
        digest.update(f"u:{query.user_id}|p:".encode())
        digest.update(",".join(map(str, query.positive_item_ids)).encode())
        digest.update(b"|n:")
        digest.update(",".join(map(str, query.negative_item_ids)).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def build_evaluation_queries(
    *,
    split_name: str,
    target_rows: Sequence[tuple[int, int, int]],
    known_rows: Sequence[tuple[int, int, int]],
    model_user_ids: np.ndarray,
    model_item_ids: np.ndarray,
    seed: int,
    max_eval_users: int | None,
) -> tuple[list[EvaluationQuery], dict[str, int | float | str]]:
    target_by_user: dict[int, set[int]] = defaultdict(set)
    known_by_user: dict[int, set[int]] = defaultdict(set)
    model_users = set(map(int, model_user_ids))
    model_items = set(map(int, model_item_ids))
    for user_id, item_id, _ in target_rows:
        if user_id in model_users:
            target_by_user[user_id].add(item_id)
    for user_id, item_id, _ in known_rows:
        if user_id in model_users:
            known_by_user[user_id].add(item_id)

    total_positives = sum(len(values) for values in target_by_user.values())
    scorable_by_user = {
        user_id: sorted(item_id for item_id in values if item_id in model_items)
        for user_id, values in target_by_user.items()
    }
    scorable_by_user = {user_id: values for user_id, values in scorable_by_user.items() if values}
    scorable_positives = sum(len(values) for values in scorable_by_user.values())
    eligible_users = [
        user_id
        for user_id, positives in scorable_by_user.items()
        if len(model_items - known_by_user[user_id] - set(positives)) >= NEGATIVES_PER_QUERY
    ]
    selected_users = _select_users(
        eligible_users,
        seed=seed,
        limit=max_eval_users,
        purpose=f"evaluate-{split_name}",
    )
    queries: list[EvaluationQuery] = []
    for user_id in selected_users:
        positives = tuple(scorable_by_user[user_id])
        excluded = known_by_user[user_id] | set(positives)
        negatives = _sample_negatives(
            model_item_ids,
            excluded,
            count=NEGATIVES_PER_QUERY,
            seed=_per_user_seed(seed, user_id, split_name),
        )
        if len(negatives) == NEGATIVES_PER_QUERY:
            queries.append(EvaluationQuery(user_id, positives, negatives))
    coverage = scorable_positives / total_positives if total_positives else 0.0
    cohort: dict[str, int | float | str] = {
        "protocol": SAMPLED_NEGATIVE_PROTOCOL,
        "positive_scope": "warm_train_items",
        "negatives_per_query": NEGATIVES_PER_QUERY,
        "candidate_item_universe": len(model_item_ids),
        "eligible_users_before_limit": len(eligible_users),
        "evaluated_users": len(queries),
        "item_coverage": coverage,
        "query_set_sha256": _query_checksum(queries),
        "scorable_positive_interactions": scorable_positives,
        "target_positive_interactions": total_positives,
        "users_with_scorable_positives": len(scorable_by_user),
    }
    return queries, cohort


def build_sampled_all_items_queries(
    *,
    split_name: str,
    target_rows: Sequence[tuple[int, int, int]],
    known_rows: Sequence[tuple[int, int, int]],
    model_user_ids: np.ndarray,
    catalog_item_ids: np.ndarray,
    seed: int,
    max_eval_users: int | None,
) -> tuple[list[EvaluationQuery], dict[str, int | float | str]]:
    """Build an all-positive cohort with deterministic sampled catalog negatives."""
    model_users = set(map(int, model_user_ids))
    catalog_items = set(map(int, catalog_item_ids))
    target_by_user: dict[int, set[int]] = defaultdict(set)
    known_by_user: dict[int, set[int]] = defaultdict(set)
    for user_id, item_id, _ in target_rows:
        if user_id in model_users and item_id in catalog_items:
            target_by_user[user_id].add(item_id)
    for user_id, item_id, _ in known_rows:
        if user_id in model_users and item_id in catalog_items:
            known_by_user[user_id].add(item_id)
    eligible_users = [
        user_id
        for user_id, positives in target_by_user.items()
        if positives and len(catalog_items - known_by_user[user_id] - positives) >= NEGATIVES_PER_QUERY
    ]
    selected_users = _select_users(
        eligible_users,
        seed=seed,
        limit=max_eval_users,
        purpose=f"evaluate-full-{split_name}",
    )
    queries: list[EvaluationQuery] = []
    for user_id in selected_users:
        positives = tuple(sorted(target_by_user[user_id]))
        negatives = _sample_negatives(
            catalog_item_ids,
            known_by_user[user_id] | set(positives),
            count=NEGATIVES_PER_QUERY,
            seed=_per_user_seed(seed, user_id, f"full-{split_name}"),
        )
        if len(negatives) == NEGATIVES_PER_QUERY:
            queries.append(EvaluationQuery(user_id, positives, negatives))
    return queries, {
        "protocol": SAMPLED_NEGATIVE_PROTOCOL,
        "positive_scope": "all_catalog_items",
        "negatives_per_query": NEGATIVES_PER_QUERY,
        "candidate_item_universe": len(catalog_item_ids),
        "eligible_users_before_limit": len(eligible_users),
        "evaluated_users": len(queries),
        "evaluated_positive_interactions": sum(len(query.positive_item_ids) for query in queries),
        "query_set_sha256": _query_checksum(queries),
        "target_positive_interactions": sum(len(values) for values in target_by_user.values()),
    }


def build_full_evaluation_queries(
    **kwargs: Any,
) -> tuple[list[EvaluationQuery], dict[str, int | float | str]]:
    """Compatibility alias for the historically misnamed sampled protocol."""
    return build_sampled_all_items_queries(**kwargs)


def sampled_all_items_metrics(split_metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Read the unified sampled cohort from new or legacy stable artifacts."""
    value = split_metrics.get(SAMPLED_ALL_ITEMS_KEY)
    if value is None:
        value = split_metrics.get(LEGACY_SAMPLED_ALL_ITEMS_KEY)
    return dict(value or {})


def _catalog_query_checksum(queries: Sequence[CatalogEvaluationQuery]) -> str:
    digest = hashlib.sha256()
    for query in queries:
        digest.update(f"u:{query.user_id}|p:".encode())
        digest.update(",".join(map(str, query.positive_item_ids)).encode())
        digest.update(b"|x:")
        digest.update(",".join(map(str, query.excluded_item_ids)).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def build_catalog_evaluation_queries(
    *,
    split_name: str,
    target_rows: Sequence[tuple[int, int, int]],
    known_rows: Sequence[tuple[int, int, int]],
    model_user_ids: np.ndarray,
    catalog_item_ids: np.ndarray,
    seed: int,
    max_eval_users: int | None,
    allowed_positive_item_ids: np.ndarray | None = None,
) -> tuple[list[CatalogEvaluationQuery], dict[str, int | float | str]]:
    """Build a deterministic cohort for genuine all-catalog replay."""
    model_users = set(map(int, model_user_ids))
    catalog_items = set(map(int, catalog_item_ids))
    allowed_positives = (
        catalog_items
        if allowed_positive_item_ids is None
        else catalog_items.intersection(map(int, allowed_positive_item_ids))
    )
    target_by_user: dict[int, set[int]] = defaultdict(set)
    known_by_user: dict[int, set[int]] = defaultdict(set)
    all_target_count = 0
    for user_id, item_id, _ in target_rows:
        if user_id in model_users and item_id in catalog_items:
            all_target_count += 1
            if item_id in allowed_positives:
                target_by_user[user_id].add(item_id)
    for user_id, item_id, _ in known_rows:
        if user_id in model_users and item_id in catalog_items:
            known_by_user[user_id].add(item_id)
    eligible_users = [
        user_id
        for user_id, positives in target_by_user.items()
        if positives and bool(catalog_items - known_by_user[user_id])
    ]
    selected_users = _select_users(
        eligible_users,
        seed=seed,
        limit=max_eval_users,
        purpose=f"evaluate-catalog-{split_name}",
    )
    queries = [
        CatalogEvaluationQuery(
            user_id=user_id,
            positive_item_ids=tuple(sorted(target_by_user[user_id])),
            excluded_item_ids=tuple(
                sorted(known_by_user[user_id] - target_by_user[user_id])
            ),
        )
        for user_id in selected_users
    ]
    return queries, {
        "protocol": "complete_eligible_catalog_v1",
        "candidate_item_universe": len(catalog_items),
        "eligible_users_before_limit": len(eligible_users),
        "evaluated_users": len(queries),
        "evaluated_positive_interactions": sum(
            len(query.positive_item_ids) for query in queries
        ),
        "target_positive_interactions": all_target_count,
        "query_set_sha256": _catalog_query_checksum(queries),
    }


def _rank_normalize(scores: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.size <= 1:
        return np.ones(values.shape, dtype=np.float64)
    order = np.lexsort((item_ids, -values))
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return 1.0 - ranks / float(len(values) - 1)


def _queries_for_items(
    queries: Sequence[EvaluationQuery],
    allowed_items: set[int],
) -> list[EvaluationQuery]:
    return [
        EvaluationQuery(
            query.user_id,
            tuple(item_id for item_id in query.positive_item_ids if item_id in allowed_items),
            query.negative_item_ids,
        )
        for query in queries
        if any(item_id in allowed_items for item_id in query.positive_item_ids)
    ]


def evaluate_queries(
    queries: Sequence[EvaluationQuery],
    score_function: Callable[[int, np.ndarray], np.ndarray],
    *,
    k: int = METRIC_K,
) -> dict[str, float]:
    if not queries:
        return {f"recall@{k}": 0.0, f"ndcg@{k}": 0.0, f"hitrate@{k}": 0.0}
    recalls: list[float] = []
    ndcgs: list[float] = []
    hitrates: list[float] = []
    for query in queries:
        candidates = np.asarray(
            [*query.positive_item_ids, *query.negative_item_ids], dtype=np.int64
        )
        scores = np.asarray(score_function(query.user_id, candidates), dtype=np.float64)
        if scores.shape != candidates.shape or not np.isfinite(scores).all():
            raise ModelTrainingError("score function returned invalid scores")
        order = np.lexsort((candidates, -scores))[:k]
        ranked = candidates[order]
        positive_set = set(query.positive_item_ids)
        hits = [rank for rank, item_id in enumerate(ranked, start=1) if int(item_id) in positive_set]
        recalls.append(len(hits) / len(positive_set))
        hitrates.append(float(bool(hits)))
        dcg = sum(1.0 / math.log2(rank + 1) for rank in hits)
        ideal_count = min(len(positive_set), k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
        ndcgs.append(dcg / idcg if idcg else 0.0)
    return {
        f"recall@{k}": float(np.mean(recalls)),
        f"ndcg@{k}": float(np.mean(ndcgs)),
        f"hitrate@{k}": float(np.mean(hitrates)),
    }


def evaluate_catalog_queries(
    queries: Sequence[CatalogEvaluationQuery],
    catalog_item_ids: np.ndarray,
    score_function: Callable[[int, np.ndarray], np.ndarray],
    *,
    k: int = METRIC_K,
) -> dict[str, float]:
    """Evaluate against the complete eligible catalog, with no sampled negatives."""
    if not queries:
        return {f"recall@{k}": 0.0, f"ndcg@{k}": 0.0, f"hitrate@{k}": 0.0}
    catalog = np.asarray(catalog_item_ids, dtype=np.int64)
    recalls: list[float] = []
    ndcgs: list[float] = []
    hitrates: list[float] = []
    for query in queries:
        excluded = set(query.excluded_item_ids)
        candidates = np.asarray(
            [int(item_id) for item_id in catalog if int(item_id) not in excluded],
            dtype=np.int64,
        )
        scores = np.asarray(score_function(query.user_id, candidates), dtype=np.float64)
        if scores.shape != candidates.shape or not np.isfinite(scores).all():
            raise ModelTrainingError("score function returned invalid catalog scores")
        order = np.lexsort((candidates, -scores))[:k]
        ranked = candidates[order]
        positive_set = set(query.positive_item_ids)
        hits = [
            rank
            for rank, item_id in enumerate(ranked, start=1)
            if int(item_id) in positive_set
        ]
        recalls.append(len(hits) / len(positive_set))
        hitrates.append(float(bool(hits)))
        dcg = sum(1.0 / math.log2(rank + 1) for rank in hits)
        ideal_count = min(len(positive_set), k)
        idcg = sum(
            1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1)
        )
        ndcgs.append(dcg / idcg if idcg else 0.0)
    return {
        f"recall@{k}": float(np.mean(recalls)),
        f"ndcg@{k}": float(np.mean(ndcgs)),
        f"hitrate@{k}": float(np.mean(hitrates)),
    }


def _anonymous_id(prefix: str, value: int) -> str:
    digest = hashlib.sha256(f"microlens-evaluation:{prefix}:{value}".encode()).hexdigest()
    return f"{prefix}-{digest[:10]}"


def analyze_badcases(
    queries: Sequence[EvaluationQuery],
    score_function: Callable[[int, np.ndarray], np.ndarray],
    *,
    k: int = METRIC_K,
    limit: int = 5,
) -> list[dict[str, object]]:
    """Return deterministic, anonymized query-level misses for the model report."""
    badcases: list[dict[str, object]] = []
    for query in queries:
        candidates = np.asarray(
            [*query.positive_item_ids, *query.negative_item_ids], dtype=np.int64
        )
        scores = np.asarray(score_function(query.user_id, candidates), dtype=np.float64)
        if scores.shape != candidates.shape or not np.isfinite(scores).all():
            raise ModelTrainingError("score function returned invalid scores")
        order = np.lexsort((candidates, -scores))
        ranks = {int(candidates[index]): rank for rank, index in enumerate(order, start=1)}
        missed = [item_id for item_id in query.positive_item_ids if ranks[item_id] > k]
        if not missed:
            continue
        positive_ranks = [ranks[item_id] for item_id in query.positive_item_ids]
        badcases.append(
            {
                "candidate_count": len(candidates),
                "positive_count": len(query.positive_item_ids),
                "positive_item_ids": [
                    _anonymous_id("i", item_id) for item_id in query.positive_item_ids
                ],
                "positive_ranks": positive_ranks,
                "reason": "no_positive_in_top_k"
                if min(positive_ranks) > k
                else "some_positives_below_top_k",
                "top_k": k,
                "user_id": _anonymous_id("u", query.user_id),
            }
        )
        if len(badcases) == limit:
            break
    return badcases


def _random_scores(seed: int, user_id: int, item_ids: np.ndarray) -> np.ndarray:
    mask = (1 << 64) - 1
    values: list[float] = []
    for item_id in map(int, item_ids):
        value = (item_id + (user_id << 32) + seed) & mask
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        value ^= value >> 31
        values.append((value & ((1 << 53) - 1)) / float(1 << 53))
    return np.asarray(values, dtype=np.float64)


def _evaluation_markdown(metrics: dict[str, object]) -> str:
    lines = [
        "# Offline Evaluation",
        "",
        "All models use the same per-split users, positives and 100 deterministic negatives.",
        "The candidate universe and popularity counts come only from the training split.",
        "Untimed likes/views metadata is not an offline model feature.",
        "",
    ]
    for split_name in ("validation", "test"):
        split = metrics[split_name]
        lines.extend(
            [
                f"## {split_name.title()}",
                "",
                f"Evaluated users: {split['cohort']['evaluated_users']}",
                f"Warm-item coverage: {split['cohort']['item_coverage']:.6f}",
                "",
                "| Model | Recall@10 | NDCG@10 | HitRate@10 |",
                "|---|---:|---:|---:|",
            ]
        )
        for model_name in (
            "popular",
            "random",
            "svd",
            "content_only",
            "item_cf",
            "svd_content_fallback",
            "hybrid_all_sources",
        ):
            if model_name not in split["models"]:
                continue
            values = split["models"][model_name]
            lines.append(
                f"| {model_name} | {values['recall@10']:.6f} | "
                f"{values['ndcg@10']:.6f} | {values['hitrate@10']:.6f} |"
            )
        lines.append("")
        sampled_all_items = sampled_all_items_metrics(split)
        if sampled_all_items:
            lines.extend(
                [
                    "### Sampled all-item and cold-item evaluation",
                    "",
                    f"Evaluated users: {sampled_all_items['cohort']['evaluated_users']}",
                    f"Cold target interactions: {sampled_all_items['cohort']['cold_target_positive_interactions']}",
                    f"Content cold-item coverage: {sampled_all_items['cohort']['cold_item_coverage']:.6f}",
                    "Unscorable SVD positives remain misses in this cohort.",
                    "",
                    "| Model | Recall@10 | NDCG@10 | HitRate@10 |",
                    "|---|---:|---:|---:|",
                ]
            )
            for model_name in (
                "svd_only",
                "content_only",
                "item_cf",
                "svd_content_fallback",
                "hybrid_all_sources",
            ):
                values = sampled_all_items["models"][model_name]
                lines.append(
                    f"| {model_name} | {values['recall@10']:.6f} | "
                    f"{values['ndcg@10']:.6f} | {values['hitrate@10']:.6f} |"
                )
            lines.append("")
        lines.extend(
            [
                "### Reproducible SVD bad cases",
                "",
                "Identifiers below are stable one-way aliases. Ranks are measured against the same",
                "sampled candidate set used by the metrics, not against the full catalog.",
                "",
                "| User | Positive items | Positive ranks | Candidates | Reason |",
                "|---|---|---:|---:|---|",
            ]
        )
        badcases = split.get("badcases", {}).get("svd", [])
        if badcases:
            for case in badcases:
                lines.append(
                    f"| {case['user_id']} | {', '.join(case['positive_item_ids'])} | "
                    f"{', '.join(map(str, case['positive_ranks']))} | "
                    f"{case['candidate_count']} | {case['reason']} |"
                )
        else:
            lines.append("| none in evaluated cohort | - | - | - | - |")
        lines.extend(
            [
                "",
                f"Coverage limitation: {split['cohort']['target_positive_interactions'] - split['cohort']['scorable_positive_interactions']} "
                "target positives were not rankable because their items were absent from the train-only candidate universe.",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "The data is sparse, so popularity may remain strong. Cold users and items are reported as",
            "coverage rather than silently removed. Online likes/views are catalog/display signals only.",
            "",
        ]
    )
    return "\n".join(lines)


def train_model(
    processed_dir: Path,
    artifacts_dir: Path,
    *,
    mode: str = "smoke",
    max_users: int | None = None,
    max_eval_users: int | None = None,
    rank: int = 32,
    seed: int = 20260901,
) -> dict[str, object]:
    from scipy.sparse import csr_matrix
    from sklearn.decomposition import TruncatedSVD

    if mode not in {"smoke", "full"}:
        raise ModelTrainingError("mode must be smoke or full")
    if rank < 1:
        raise ModelTrainingError("rank must be positive")
    processed_dir = Path(processed_dir)
    artifacts_dir = Path(artifacts_dir)
    required = ["train.csv", "validation.csv", "test.csv", "items.csv", "summary.json"]
    missing = [name for name in required if not (processed_dir / name).is_file()]
    if missing:
        raise ModelTrainingError(f"missing processed files: {missing}")
    summary = json.loads((processed_dir / "summary.json").read_text(encoding="utf-8"))
    data_version = summary.get("data_version")
    if not data_version:
        raise ModelTrainingError("processed summary has no data_version")

    train_rows = _read_processed_interactions(processed_dir / "train.csv")
    validation_rows = _read_processed_interactions(processed_dir / "validation.csv")
    test_rows = _read_processed_interactions(processed_dir / "test.csv")
    online_feedback = _read_online_feedback(processed_dir / "online_feedback_train.csv")
    if not train_rows:
        raise ModelTrainingError("training split is empty")
    if mode == "smoke" and max_users is None:
        max_users = 2000
    selected_users = _select_users(
        (row[0] for row in train_rows), seed=seed, limit=max_users, purpose=f"train-{mode}"
    )
    feedback_users = sorted({row[0] for row in online_feedback} & {row[0] for row in train_rows})
    if max_users is not None and len(feedback_users) > max_users:
        raise ModelTrainingError("max_users is smaller than the mapped online feedback user cohort")
    if feedback_users:
        remaining = [user_id for user_id in selected_users if user_id not in set(feedback_users)]
        selected_users = sorted(
            [
                *feedback_users,
                *remaining[: None if max_users is None else max_users - len(feedback_users)],
            ]
        )
    selected_user_set = set(selected_users)
    sampled_train = [row for row in train_rows if row[0] in selected_user_set]
    item_ids = np.asarray(sorted({row[1] for row in sampled_train}), dtype=np.int64)
    user_ids = np.asarray(sorted(selected_users), dtype=np.int64)
    if len(user_ids) < 2 or len(item_ids) < 2:
        raise ModelTrainingError("at least two training users and items are required")

    user_index = {int(value): index for index, value in enumerate(user_ids)}
    item_index = {int(value): index for index, value in enumerate(item_ids)}
    matrix_rows = np.fromiter((user_index[row[0]] for row in sampled_train), dtype=np.int64)
    matrix_cols = np.fromiter((item_index[row[1]] for row in sampled_train), dtype=np.int64)
    matrix = csr_matrix(
        (np.ones(len(sampled_train), dtype=np.float32), (matrix_rows, matrix_cols)),
        shape=(len(user_ids), len(item_ids)),
        dtype=np.float32,
    )
    matrix.data[:] = 1.0
    feature_cutoff_ms = int(summary["cutoffs"]["train_cutoff_ms"])
    usable_feedback = [
        row
        for row in online_feedback
        if row[0] in user_index
        and row[1] in item_index
        and row[2] <= feature_cutoff_ms
        and row[4] != 0
    ]
    if usable_feedback:
        feedback_matrix = csr_matrix(
            (
                np.asarray([row[4] for row in usable_feedback], dtype=np.float32),
                (
                    np.asarray([user_index[row[0]] for row in usable_feedback], dtype=np.int64),
                    np.asarray([item_index[row[1]] for row in usable_feedback], dtype=np.int64),
                ),
            ),
            shape=(len(user_ids), len(item_ids)),
            dtype=np.float32,
        )
        matrix = matrix + feedback_matrix
    effective_rank = min(rank, len(user_ids) - 1, len(item_ids) - 1)
    svd = TruncatedSVD(
        n_components=effective_rank,
        algorithm="randomized",
        n_iter=7,
        random_state=seed,
    )
    user_factors = svd.fit_transform(matrix).astype(np.float32)
    item_factors = svd.components_.T.astype(np.float32)
    observed_scores = np.einsum(
        "ij,ij->i", user_factors[matrix_rows], item_factors[matrix_cols]
    )
    observed_positive_mse = float(np.mean(np.square(1.0 - observed_scores)))

    popularity_events = [
        (item_id, timestamp_ms, 1.0)
        for _, item_id, timestamp_ms in sampled_train
    ] + [
        (item_id, timestamp_ms, weight)
        for _, item_id, timestamp_ms, _, weight in usable_feedback
        if weight > 0
    ]
    popularity_half_life_candidates = (1.0, 7.0, 30.0)
    popularity_features_by_half_life = {
        half_life: build_popularity_features(
            popularity_events,
            feature_cutoff_ms=feature_cutoff_ms,
            half_life_days=half_life,
        )
        for half_life in popularity_half_life_candidates
    }
    selected_popularity_half_life = DEFAULT_HALF_LIFE_DAYS
    popularity_features = popularity_features_by_half_life[selected_popularity_half_life]
    popularity_counts = {
        int(item_id): int(features.cumulative_interactions)
        for item_id, features in popularity_features.items()
    }
    popularity_scores = np.asarray(
        [popularity_counts[int(item_id)] for item_id in item_ids], dtype=np.float64
    )
    item_position = {int(value): index for index, value in enumerate(item_ids)}
    content = build_content_features(
        items_path=processed_dir / "items.csv",
        train_rows=sampled_train,
        user_ids=user_ids,
        feature_cutoff_ms=feature_cutoff_ms,
        online_feedback=usable_feedback,
    )
    content_item_position = {
        int(value): index for index, value in enumerate(content.item_ids)
    }
    content_user_position = {
        int(value): index for index, value in enumerate(content.user_ids)
    }
    content_nonzero = np.diff(content.item_vectors.indptr) > 0
    item_cf = build_item_cf(
        train_rows=sampled_train,
        user_ids=user_ids,
        item_ids=item_ids,
        feature_cutoff_ms=feature_cutoff_ms,
        top_n=50,
        min_support=2,
    )

    if max_eval_users is None:
        max_eval_users = 500 if mode == "smoke" else 5000
    split_queries: dict[str, tuple[list[EvaluationQuery], dict[str, int | float | str]]] = {}
    feedback_positive_rows = [
        (user_id, item_id, timestamp_ms)
        for user_id, item_id, timestamp_ms, _, weight in usable_feedback
        if weight > 0
    ]
    split_queries["validation"] = build_evaluation_queries(
        split_name="validation",
        target_rows=validation_rows,
        known_rows=[*sampled_train, *feedback_positive_rows, *validation_rows],
        model_user_ids=user_ids,
        model_item_ids=item_ids,
        seed=seed,
        max_eval_users=max_eval_users,
    )
    split_queries["test"] = build_evaluation_queries(
        split_name="test",
        target_rows=test_rows,
        known_rows=[*sampled_train, *feedback_positive_rows, *validation_rows, *test_rows],
        model_user_ids=user_ids,
        model_item_ids=item_ids,
        seed=seed,
        max_eval_users=max_eval_users,
    )
    full_split_queries = {
        "validation": build_sampled_all_items_queries(
            split_name="validation",
            target_rows=validation_rows,
            known_rows=[*sampled_train, *feedback_positive_rows, *validation_rows],
            model_user_ids=user_ids,
            catalog_item_ids=content.item_ids,
            seed=seed,
            max_eval_users=max_eval_users,
        ),
        "test": build_sampled_all_items_queries(
            split_name="test",
            target_rows=test_rows,
            known_rows=[*sampled_train, *feedback_positive_rows, *validation_rows, *test_rows],
            model_user_ids=user_ids,
            catalog_item_ids=content.item_ids,
            seed=seed,
            max_eval_users=max_eval_users,
        ),
    }

    def popularity_candidate_score(
        features: dict[str, Any], candidates: np.ndarray
    ) -> np.ndarray:
        return np.asarray(
            [
                (
                    features[str(int(item_id))].time_decay_score
                    + 1e-6 * features[str(int(item_id))].cumulative_interactions
                )
                if str(int(item_id)) in features
                else 0.0
                for item_id in candidates
            ],
            dtype=np.float64,
        )

    popularity_validation_metrics = {
        str(half_life): evaluate_queries(
            full_split_queries["validation"][0],
            lambda _user_id, candidates, candidate_features=features: (
                popularity_candidate_score(candidate_features, candidates)
            ),
        )
        for half_life, features in popularity_features_by_half_life.items()
    }
    selected_popularity_half_life = max(
        popularity_half_life_candidates,
        key=lambda half_life: (
            popularity_validation_metrics[str(half_life)]["ndcg@10"],
            popularity_validation_metrics[str(half_life)]["recall@10"],
            -half_life,
        ),
    )
    popularity_features = popularity_features_by_half_life[selected_popularity_half_life]
    popularity_counts = {
        int(item_id): int(features.cumulative_interactions)
        for item_id, features in popularity_features.items()
    }
    popularity_score_by_item = {
        int(item_id): (
            features.time_decay_score + 1e-6 * features.cumulative_interactions
        )
        for item_id, features in popularity_features.items()
    }
    popularity_scores = np.asarray(
        [popularity_score_by_item[int(item_id)] for item_id in item_ids],
        dtype=np.float64,
    )

    def popular_score(_: int, candidates: np.ndarray) -> np.ndarray:
        return np.asarray(
            [popularity_scores[item_position[int(item_id)]] for item_id in candidates],
            dtype=np.float64,
        )

    def random_score(user_id: int, candidates: np.ndarray) -> np.ndarray:
        return _random_scores(seed, user_id, candidates)

    def svd_score(user_id: int, candidates: np.ndarray) -> np.ndarray:
        positions = np.asarray([item_position[int(item_id)] for item_id in candidates])
        return item_factors[positions] @ user_factors[user_index[user_id]]

    def content_score(user_id: int, candidates: np.ndarray) -> np.ndarray:
        user_position = content_user_position[user_id]
        positions = np.asarray(
            [content_item_position[int(item_id)] for item_id in candidates],
            dtype=np.int64,
        )
        return np.asarray(
            (content.item_vectors[positions] @ content.user_vectors[user_position].T)
            .toarray()
            .ravel(),
            dtype=np.float64,
        )

    def full_svd_score(user_id: int, candidates: np.ndarray) -> np.ndarray:
        values = np.full(len(candidates), -1e12, dtype=np.float64)
        available = [
            (index, item_position[int(item_id)])
            for index, item_id in enumerate(candidates)
            if int(item_id) in item_position
        ]
        if available:
            target_indices = np.asarray([row[0] for row in available], dtype=np.int64)
            factor_indices = np.asarray([row[1] for row in available], dtype=np.int64)
            values[target_indices] = item_factors[factor_indices] @ user_factors[user_index[user_id]]
        return values

    def full_popular_score(_: int, candidates: np.ndarray) -> np.ndarray:
        return np.asarray(
            [popularity_score_by_item.get(int(item_id), 0.0) for item_id in candidates],
            dtype=np.float64,
        )

    def item_cf_score(user_id: int, candidates: np.ndarray) -> np.ndarray:
        candidate_positions = [
            item_position.get(int(item_id)) for item_id in candidates
        ]
        values = np.zeros(len(candidates), dtype=np.float64)
        available = [
            (index, position)
            for index, position in enumerate(candidate_positions)
            if position is not None
        ]
        if not available:
            return values
        history = item_cf.user_history[user_index[user_id]]
        history_positions = history.indices
        if history_positions.size == 0:
            return values
        target_positions = np.asarray([row[1] for row in available], dtype=np.int64)
        scores = np.asarray(
            item_cf.neighbors[history_positions][:, target_positions].sum(axis=0)
        ).ravel()
        values[np.asarray([row[0] for row in available], dtype=np.int64)] = scores
        return values

    def hybrid_score(user_id: int, candidates: np.ndarray) -> np.ndarray:
        svd_values = full_svd_score(user_id, candidates)
        content_values = content_score(user_id, candidates)
        popular_values = full_popular_score(user_id, candidates)
        available = np.asarray(
            [int(item_id) in item_position for item_id in candidates],
            dtype=bool,
        )
        cold_usable = (~available) & (content_values > 0)
        if not cold_usable.any():
            return svd_values
        blended = (
            0.55 * _rank_normalize(svd_values, candidates)
            + 0.35 * _rank_normalize(content_values, candidates)
            + 0.10 * _rank_normalize(popular_values, candidates)
        )
        warm_order = [
            int(index)
            for index in np.lexsort((candidates, -svd_values))
            if available[int(index)]
        ]
        cold_order = [
            int(index)
            for index in np.lexsort((candidates, -content_values))
            if cold_usable[int(index)]
        ]
        selected = [*warm_order[:7], *cold_order[:3]]
        selected_set = set(selected)
        remainder = [
            int(index)
            for index in np.lexsort((candidates, -blended))
            if int(index) not in selected_set
        ]
        final_order = [*selected, *remainder]
        scores = np.empty(len(candidates), dtype=np.float64)
        for rank_index, candidate_index in enumerate(final_order):
            scores[candidate_index] = float(len(candidates) - rank_index)
        return scores

    def mixed_ranking(
        user_id: int,
        candidates: np.ndarray,
        policy_version: str,
    ) -> tuple[list[MixCandidate], dict[str, object], dict[str, list[MixCandidate]]]:
        model_values = full_svd_score(user_id, candidates)
        content_values = content_score(user_id, candidates)
        cf_values = item_cf_score(user_id, candidates)
        popular_values = full_popular_score(user_id, candidates)
        explore_values = _random_scores(seed, user_id, candidates)
        history = item_cf.user_history[user_index[user_id]]
        history_positions = history.indices
        cf_support_values = np.zeros(len(candidates), dtype=np.int64)
        candidate_model_positions = [
            item_position.get(int(item_id)) for item_id in candidates
        ]
        cf_targets = [
            (index, position)
            for index, position in enumerate(candidate_model_positions)
            if position is not None
        ]
        if history_positions.size and cf_targets:
            support = np.asarray(
                item_cf.neighbors[history_positions][
                    :, np.asarray([row[1] for row in cf_targets], dtype=np.int64)
                ].getnnz(axis=0)
            ).ravel()
            cf_support_values[
                np.asarray([row[0] for row in cf_targets], dtype=np.int64)
            ] = support

        sources: dict[str, list[MixCandidate]] = {
            "model": [],
            "content_profile": [],
            "item_cf": [],
            "popular": [],
            "explore": [],
        }
        for index, item_id_value in enumerate(candidates):
            item_id = int(item_id_value)
            if model_values[index] > -1e11 and math.isfinite(float(model_values[index])):
                sources["model"].append(
                    MixCandidate(
                        item_id=str(item_id),
                        source="model",
                        score=float(model_values[index]),
                        explanation="cutoff-safe SVD score",
                        model_version=None,
                        eligible=True,
                        confidence=0.5 + 0.5 * math.tanh(float(model_values[index])),
                        support=int(history.getnnz()),
                    )
                )
            if content_values[index] > 0 and math.isfinite(float(content_values[index])):
                sources["content_profile"].append(
                    MixCandidate(
                        item_id=str(item_id),
                        source="content_profile",
                        score=float(content_values[index]),
                        explanation="cutoff-safe title-profile score",
                        model_version=None,
                        eligible=True,
                        confidence=min(1.0, float(content_values[index])),
                        support=max(1, int(history.getnnz())),
                        is_cold=item_id not in item_position,
                    )
                )
            if cf_values[index] > 0 and math.isfinite(float(cf_values[index])):
                sources["item_cf"].append(
                    MixCandidate(
                        item_id=str(item_id),
                        source="item_cf",
                        score=float(cf_values[index]),
                        explanation="cutoff-safe item co-occurrence score",
                        model_version=None,
                        eligible=bool(cf_support_values[index] > 0),
                        confidence=min(1.0, float(cf_values[index])),
                        support=int(cf_support_values[index]),
                    )
                )
            sources["popular"].append(
                MixCandidate(
                    item_id=str(item_id),
                    source="popular",
                    score=float(popular_values[index]),
                    explanation="cutoff-safe popularity score",
                    model_version=None,
                    eligible=math.isfinite(float(popular_values[index])),
                    confidence=1.0 if popular_values[index] > 0 else 0.1,
                    support=max(0, int(popular_values[index])),
                    is_cold=item_id not in item_position,
                )
            )
            sources["explore"].append(
                MixCandidate(
                    item_id=str(item_id),
                    source="explore",
                    score=float(explore_values[index]),
                    explanation="deterministic exploration score",
                    model_version=None,
                    eligible=True,
                    confidence=0.05,
                    support=0,
                    is_cold=item_id not in item_position,
                )
            )
        context = MixContext(
            warm_user=True,
            history_count=int(history.getnnz()),
            cf_support=max(
                (candidate.support for candidate in sources["item_cf"]), default=0
            ),
            content_confidence=max(
                (candidate.confidence for candidate in sources["content_profile"]),
                default=0.0,
            ),
        )
        mixed, diagnostics = mix_candidates(
            policy_version, sources, len(candidates), context
        )
        return mixed, diagnostics, sources

    def policy_score(policy_version: str) -> Callable[[int, np.ndarray], np.ndarray]:
        def score(user_id: int, candidates: np.ndarray) -> np.ndarray:
            mixed, _, _ = mixed_ranking(user_id, candidates, policy_version)
            positions = {candidate.item_id: rank for rank, candidate in enumerate(mixed)}
            return np.asarray(
                [
                    float(len(candidates) - positions[str(int(item_id))])
                    if str(int(item_id)) in positions
                    else -1e12
                    for item_id in candidates
                ],
                dtype=np.float64,
            )

        return score

    safe_policy_score = policy_score(SAFE_POLICY_VERSION)
    dynamic_policy_score = policy_score(DYNAMIC_POLICY_VERSION)
    validation_queries = split_queries["validation"][0]
    validation_full_queries = full_split_queries["validation"][0]
    validation_policy_metrics = {
        SAFE_POLICY_VERSION: {
            "warm": evaluate_queries(validation_queries, safe_policy_score),
            "full": evaluate_queries(validation_full_queries, safe_policy_score),
        },
        DYNAMIC_POLICY_VERSION: {
            "warm": evaluate_queries(validation_queries, dynamic_policy_score),
            "full": evaluate_queries(validation_full_queries, dynamic_policy_score),
        },
    }
    safe_validation = validation_policy_metrics[SAFE_POLICY_VERSION]
    dynamic_validation = validation_policy_metrics[DYNAMIC_POLICY_VERSION]
    gate_checks = {
        "full_recall_within_1pct": dynamic_validation["full"]["recall@10"]
        >= 0.99 * safe_validation["full"]["recall@10"],
        "full_ndcg_within_1pct": dynamic_validation["full"]["ndcg@10"]
        >= 0.99 * safe_validation["full"]["ndcg@10"],
        "warm_recall_within_1pct": dynamic_validation["warm"]["recall@10"]
        >= 0.99 * safe_validation["warm"]["recall@10"],
        "warm_ndcg_within_1pct": dynamic_validation["warm"]["ndcg@10"]
        >= 0.99 * safe_validation["warm"]["ndcg@10"],
        "validation_composite_not_worse": (
            dynamic_validation["full"]["recall@10"]
            + dynamic_validation["full"]["ndcg@10"]
            + dynamic_validation["warm"]["recall@10"]
            + dynamic_validation["warm"]["ndcg@10"]
        )
        >= (
            safe_validation["full"]["recall@10"]
            + safe_validation["full"]["ndcg@10"]
            + safe_validation["warm"]["recall@10"]
            + safe_validation["warm"]["ndcg@10"]
        ),
    }
    selected_policy_version = (
        DYNAMIC_POLICY_VERSION if all(gate_checks.values()) else SAFE_POLICY_VERSION
    )
    selected_policy_score = policy_score(selected_policy_version)

    def analyze_policy(
        queries: Sequence[EvaluationQuery], policy_version: str
    ) -> dict[str, object]:
        selected_counts = defaultdict(int)
        independent_hits = defaultdict(int)
        jaccard_totals = defaultdict(float)
        relaxation_count = 0
        fallback_queries = 0
        for query in queries:
            candidates = np.asarray(
                [*query.positive_item_ids, *query.negative_item_ids], dtype=np.int64
            )
            mixed, diagnostics, sources = mixed_ranking(
                query.user_id, candidates, policy_version
            )
            for candidate in mixed[:METRIC_K]:
                selected_counts[candidate.source] += 1
            query_relaxations = sum(
                int(int(relaxation["slot"]) < METRIC_K)
                for relaxation in list(diagnostics["relaxations"])
            )
            relaxation_count += query_relaxations
            fallback_queries += int(query_relaxations > 0)
            positives = {str(item_id) for item_id in query.positive_item_ids}
            source_top_items: dict[str, set[str]] = {}
            for source, values in sources.items():
                ranked = sorted(
                    (candidate for candidate in values if candidate.eligible),
                    key=lambda candidate: (-candidate.score, candidate.item_id),
                )[:METRIC_K]
                source_top_items[source] = {candidate.item_id for candidate in ranked}
                independent_hits[source] += int(
                    bool(positives & source_top_items[source])
                )
            source_names = sorted(source_top_items)
            for left_index, left in enumerate(source_names):
                for right in source_names[left_index + 1 :]:
                    union = source_top_items[left] | source_top_items[right]
                    jaccard_totals[f"{left}:{right}"] += (
                        len(source_top_items[left] & source_top_items[right]) / len(union)
                        if union
                        else 0.0
                    )
            assert len({candidate.item_id for candidate in mixed}) == len(mixed)
        query_count = len(queries)
        return {
            "queries": query_count,
            "average_selected_quota": {
                source: count / query_count if query_count else 0.0
                for source, count in sorted(selected_counts.items())
            },
            "independent_source_hitrate@10": {
                source: count / query_count if query_count else 0.0
                for source, count in sorted(independent_hits.items())
            },
            "average_source_jaccard": {
                pair: value / query_count if query_count else 0.0
                for pair, value in sorted(jaccard_totals.items())
            },
            "quota_relaxations": relaxation_count,
            "fallback_queries": fallback_queries,
            "duplicate_results": 0,
        }

    selected_validation_diagnostics = analyze_policy(
        validation_full_queries, selected_policy_version
    )

    metrics: dict[str, object] = {
        "schema_version": 1,
        "mix_policy_search": {
            "search_split": "validation_only",
            "test_policy_locked": True,
            "candidate_metrics": validation_policy_metrics,
            "quality_gate": gate_checks,
            "selected_policy_version": selected_policy_version,
            "selected_validation_diagnostics": selected_validation_diagnostics,
            "post_mix_stage": (
                "online title-token MMR records before/after diversity metrics per request"
            ),
        },
        "evaluation_protocol": {
            "candidate_universe": "items observed in train for selected training users",
            "cohort_aggregation": "macro over users",
            "k": METRIC_K,
            "negative_sampling": "100 unique seeded train-item negatives per user",
            "negatives_per_query": NEGATIVES_PER_QUERY,
            "popularity_scope": "train_only",
            "popularity_feature_cutoff_ms": feature_cutoff_ms,
            "popularity_feature_rule": "event_timestamp_ms <= feature_cutoff_ms",
            "popularity_windows_days": list(WINDOW_DAYS),
            "popularity_time_decay_half_life_days": DEFAULT_HALF_LIFE_DAYS,
            "selected_popularity_time_decay_half_life_days": selected_popularity_half_life,
            "popularity_half_life_selection": "validation_only_ndcg_then_recall",
            "popularity_source": "interaction_events_only",
            "positive_labels": "all warm positives in the named time split",
            "shared_queries_across_models": True,
            "untimed_likes_views_used": False,
            "content_vocabulary_fit_scope": "train-visible item titles only",
            "content_transform_scope": "full catalog with frozen vocabulary",
            "protocol": SAMPLED_NEGATIVE_PROTOCOL,
            "all_item_positive_scope": "all catalog positives, including cold items",
            "unscorable_positive_rule": "unscorable SVD positives remain misses",
        },
    }
    for split_name, (queries, cohort) in split_queries.items():
        full_queries, full_cohort = full_split_queries[split_name]
        target_rows = validation_rows if split_name == "validation" else test_rows
        target_items = [
            int(item_id)
            for user_id, item_id, _ in target_rows
            if int(user_id) in user_index and int(item_id) in content_item_position
        ]
        cold_target_items = [item_id for item_id in target_items if item_id not in item_position]
        content_covered_cold = sum(
            bool(content_nonzero[content_item_position[item_id]]) for item_id in cold_target_items
        )
        buckets = {
            "cold": {item_id for item_id in content_item_position if item_id not in item_position},
            "tail_1": {item_id for item_id, count in popularity_counts.items() if count == 1},
            "tail_2_4": {
                item_id for item_id, count in popularity_counts.items() if 2 <= count <= 4
            },
            "head_5_plus": {
                item_id for item_id, count in popularity_counts.items() if count >= 5
            },
        }
        metrics[split_name] = {
            "cohort": cohort,
            "models": {
                "popular": evaluate_queries(queries, popular_score),
                "random": evaluate_queries(queries, random_score),
                "svd": evaluate_queries(queries, svd_score),
                "content_only": evaluate_queries(queries, content_score),
                "item_cf": evaluate_queries(queries, item_cf_score),
                "svd_content_fallback": evaluate_queries(queries, hybrid_score),
                "hybrid_all_sources": evaluate_queries(queries, selected_policy_score),
            },
            SAMPLED_ALL_ITEMS_KEY: {
                "cohort": {
                    **full_cohort,
                    "cold_target_positive_interactions": len(cold_target_items),
                    "content_covered_cold_interactions": content_covered_cold,
                    "cold_item_coverage": (
                        content_covered_cold / len(cold_target_items)
                        if cold_target_items
                        else 0.0
                    ),
                },
                "models": {
                    "svd_only": evaluate_queries(full_queries, full_svd_score),
                    "content_only": evaluate_queries(full_queries, content_score),
                    "item_cf": evaluate_queries(full_queries, item_cf_score),
                    "svd_content_fallback": evaluate_queries(full_queries, hybrid_score),
                    "hybrid_all_sources": evaluate_queries(
                        full_queries, selected_policy_score
                    ),
                },
                "popularity_buckets": {
                    name: {
                        "evaluated_positive_interactions": sum(
                            len(query.positive_item_ids)
                            for query in _queries_for_items(full_queries, allowed)
                        ),
                        "content_only": evaluate_queries(
                            _queries_for_items(full_queries, allowed), content_score
                        ),
                        "svd_content_fallback": evaluate_queries(
                            _queries_for_items(full_queries, allowed), hybrid_score
                        ),
                    }
                    for name, allowed in buckets.items()
                },
            },
            "badcases": {"svd": analyze_badcases(queries, svd_score)},
        }
        if int(cohort["evaluated_users"]) <= 0:
            raise ModelTrainingError(f"{split_name} evaluation has no eligible users")
        for model_metrics in metrics[split_name]["models"].values():
            if any(not math.isfinite(float(value)) for value in model_metrics.values()):
                raise ModelTrainingError(f"{split_name} evaluation produced non-finite metrics")

    config = {
        "algorithm": "sklearn.decomposition.TruncatedSVD",
        "effective_rank": effective_rank,
        "max_eval_users": max_eval_users,
        "max_users": max_users,
        "mode": mode,
        "n_iter": 7,
        "requested_rank": rank,
        "seed": seed,
        "training_scope": "train_only",
        "popularity_feature_cutoff_ms": feature_cutoff_ms,
        "popularity_windows_days": list(WINDOW_DAYS),
        "popularity_time_decay_half_life_days": selected_popularity_half_life,
        "popularity_half_life_candidates_days": list(popularity_half_life_candidates),
        "popularity_half_life_validation_metrics": popularity_validation_metrics,
        "online_feedback_enabled": bool(online_feedback),
        "online_feedback_mapping": (summary.get("online_retraining") or {}).get(
            "feedback_mapping", {}
        ),
        "content_features": {
            "analyzer": "char",
            "max_features": 8192,
            "min_df": 2,
            "ngram_range": [2, 4],
            "profile_source": "cutoff-safe positive training history",
        },
        "item_cf": {
            "algorithm": "train_only_sparse_item_cosine",
            "min_support": 2,
            "top_n": 50,
        },
        "mix_policy": {
            "candidate_policy_versions": [
                SAFE_POLICY_VERSION,
                DYNAMIC_POLICY_VERSION,
            ],
            "search_split": "validation_only",
            "selected_policy_version": selected_policy_version,
            "test_policy_locked": True,
        },
    }
    config_digest = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:8]
    now = datetime.now(timezone.utc)
    model_version = f"svd-{now.strftime('%Y%m%dT%H%M%S%fZ')}-{config_digest}"
    popularity = {
        "items": [
            {
                "count": count,
                "cumulative_interactions": count,
                "interactions_1d": popularity_features[str(item_id)].interactions_1d,
                "interactions_7d": popularity_features[str(item_id)].interactions_7d,
                "interactions_30d": popularity_features[str(item_id)].interactions_30d,
                "time_decay_score": popularity_features[str(item_id)].time_decay_score,
                "recent_growth": popularity_features[str(item_id)].recent_growth,
                "item_id": item_id,
                "score": float(popularity_score_by_item[item_id]),
            }
            for item_id, count in sorted(
                popularity_counts.items(), key=lambda pair: (-pair[1], pair[0])
            )
        ],
        "feature_cutoff_ms": feature_cutoff_ms,
        "feature_source": "train_interactions",
        "feature_rule": "event_timestamp_ms <= feature_cutoff_ms",
        "time_decay_half_life_days": selected_popularity_half_life,
        "score_formula": "time_decay_score + 1e-6 * cumulative_interactions",
        "selection_split": "validation_only",
        "candidate_half_life_days": list(popularity_half_life_candidates),
        "validation_metrics": popularity_validation_metrics,
        "windows_days": list(WINDOW_DAYS),
        "model_version": model_version,
        "schema_version": 1,
        "scope": "selected_users_train_split_only",
    }
    training_summary = {
        "explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
        "observed_positive_mse": observed_positive_mse,
        "train_interactions": len(sampled_train),
        "train_items": len(item_ids),
        "train_users": len(user_ids),
        "online_feedback_events_available": len(online_feedback),
        "online_feedback_events_used": len(usable_feedback),
        "online_positive_feedback_used": sum(1 for row in usable_feedback if row[4] > 0),
        "online_negative_feedback_used": sum(1 for row in usable_feedback if row[4] < 0),
        "content_catalog_items": len(content.item_ids),
        "content_train_visible_items": content.metadata["train_visible_items"],
        "content_vocabulary_size": content.metadata["vocabulary_size"],
        "content_nonzero_item_vectors": content.metadata["nonzero_item_vectors"],
        "content_nonzero_user_vectors": content.metadata["nonzero_user_vectors"],
        "item_cf_neighbor_edges": item_cf.metadata["neighbor_edges"],
        "item_cf_items_with_neighbors": item_cf.metadata["items_with_neighbors"],
        "mix_policy_version": selected_policy_version,
    }
    manifest_base = {
        "algorithm": "truncated_svd_implicit_feedback",
        "created_at": now.isoformat(),
        "data_version": data_version,
        "metrics": metrics,
        "model_version": model_version,
        "schema_version": 1,
        "training_config": config,
        "training_summary": training_summary,
        "untimed_likes_views_used_as_feature": False,
        "likes_views_snapshot": {
            **dict(summary.get("likes_views_snapshot") or {}),
            "used_for_training_or_evaluation": False,
            "reason": "cumulative snapshot is not historical interaction evidence",
        },
        "popularity_features": {
            "feature_cutoff_ms": feature_cutoff_ms,
            "feature_source": "train_interactions",
            "feature_rule": "event_timestamp_ms <= feature_cutoff_ms",
            "time_decay_half_life_days": selected_popularity_half_life,
            "score_formula": "time_decay_score + 1e-6 * cumulative_interactions",
            "selection_split": "validation_only",
            "candidate_half_life_days": list(popularity_half_life_candidates),
            "validation_metrics": popularity_validation_metrics,
            "windows_days": list(WINDOW_DAYS),
        },
        "online_retraining": dict(summary.get("online_retraining") or {}),
        "content_features": {
            **content.metadata,
            "schema_version": 1,
            "required_files": [
                "content_item_ids.npy",
                "content_user_ids.npy",
                "content_item_vectors.npz",
                "content_user_vectors.npz",
                "content_idf.npy",
                "content_vectorizer.json",
            ],
        },
        "item_cf": {
            **item_cf.metadata,
            "schema_version": 1,
            "required_files": [
                "item_cf_neighbors.npz",
                "item_cf_user_history.npz",
                "item_cf_config.json",
            ],
        },
        "mix_policy": {
            "schema_version": 1,
            "selected_policy_version": selected_policy_version,
            "search_split": "validation_only",
            "test_policy_locked": True,
            "quality_gate": gate_checks,
            "validation_metrics": validation_policy_metrics,
        },
    }
    staging: Path | None = None
    try:
        staging = write_staged_artifact(
            artifacts_dir,
            manifest_base=manifest_base,
            user_ids=user_ids,
            item_ids=item_ids,
            user_factors=user_factors,
            item_factors=item_factors,
            popularity=popularity,
            metrics=metrics,
            evaluation_markdown=_evaluation_markdown(metrics),
            extra_artifacts={
                "content_item_ids.npy": content.item_ids,
                "content_user_ids.npy": content.user_ids,
                "content_item_vectors.npz": content.item_vectors,
                "content_user_vectors.npz": content.user_vectors,
                "content_idf.npy": content.idf,
                "content_vectorizer.json": content.vectorizer_state,
                "item_cf_neighbors.npz": item_cf.neighbors,
                "item_cf_user_history.npz": item_cf.user_history,
                "item_cf_config.json": item_cf.metadata,
            },
        )
        destination = publish_artifact(staging, artifacts_dir)
        staging = None
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
    return {
        "artifact_dir": str(destination),
        "data_version": data_version,
        "metrics": metrics,
        "model_version": model_version,
        "training_summary": training_summary,
    }
