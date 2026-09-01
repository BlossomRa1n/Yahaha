from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from .artifacts import publish_artifact, write_staged_artifact


NEGATIVES_PER_QUERY = 100
METRIC_K = 10


class ModelTrainingError(ValueError):
    pass


@dataclass(frozen=True)
class EvaluationQuery:
    user_id: int
    positive_item_ids: tuple[int, ...]
    negative_item_ids: tuple[int, ...]


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
        for model_name in ("popular", "random", "svd"):
            values = split["models"][model_name]
            lines.append(
                f"| {model_name} | {values['recall@10']:.6f} | "
                f"{values['ndcg@10']:.6f} | {values['hitrate@10']:.6f} |"
            )
        lines.append("")
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
    required = ["train.csv", "validation.csv", "test.csv", "summary.json"]
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
    if not train_rows:
        raise ModelTrainingError("training split is empty")
    if mode == "smoke" and max_users is None:
        max_users = 2000
    selected_users = _select_users(
        (row[0] for row in train_rows), seed=seed, limit=max_users, purpose=f"train-{mode}"
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

    popularity_counts = Counter(row[1] for row in sampled_train)
    popularity_scores = np.asarray(
        [popularity_counts[int(item_id)] for item_id in item_ids], dtype=np.float64
    )
    item_position = {int(value): index for index, value in enumerate(item_ids)}

    if max_eval_users is None:
        max_eval_users = 500 if mode == "smoke" else 5000
    split_queries: dict[str, tuple[list[EvaluationQuery], dict[str, int | float | str]]] = {}
    split_queries["validation"] = build_evaluation_queries(
        split_name="validation",
        target_rows=validation_rows,
        known_rows=[*sampled_train, *validation_rows],
        model_user_ids=user_ids,
        model_item_ids=item_ids,
        seed=seed,
        max_eval_users=max_eval_users,
    )
    split_queries["test"] = build_evaluation_queries(
        split_name="test",
        target_rows=test_rows,
        known_rows=[*sampled_train, *validation_rows, *test_rows],
        model_user_ids=user_ids,
        model_item_ids=item_ids,
        seed=seed,
        max_eval_users=max_eval_users,
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

    metrics: dict[str, object] = {
        "schema_version": 1,
        "evaluation_protocol": {
            "candidate_universe": "items observed in train for selected training users",
            "cohort_aggregation": "macro over users",
            "k": METRIC_K,
            "negative_sampling": "100 unique seeded train-item negatives per user",
            "negatives_per_query": NEGATIVES_PER_QUERY,
            "popularity_scope": "train_only",
            "positive_labels": "all warm positives in the named time split",
            "shared_queries_across_models": True,
            "untimed_likes_views_used": False,
        },
    }
    for split_name, (queries, cohort) in split_queries.items():
        metrics[split_name] = {
            "cohort": cohort,
            "models": {
                "popular": evaluate_queries(queries, popular_score),
                "random": evaluate_queries(queries, random_score),
                "svd": evaluate_queries(queries, svd_score),
            },
        }

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
    }
    config_digest = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:8]
    now = datetime.now(timezone.utc)
    model_version = f"svd-{now.strftime('%Y%m%dT%H%M%S%fZ')}-{config_digest}"
    popularity = {
        "items": [
            {"count": count, "item_id": item_id, "score": float(count)}
            for item_id, count in sorted(
                popularity_counts.items(), key=lambda pair: (-pair[1], pair[0])
            )
        ],
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
