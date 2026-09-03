from __future__ import annotations

import numpy as np

from recsys.deep import _build_replay_training_features
from recsys.model import (
    SAMPLED_NEGATIVE_PROTOCOL,
    build_catalog_evaluation_queries,
    build_full_evaluation_queries,
    build_sampled_all_items_queries,
    evaluate_catalog_queries,
    sampled_all_items_metrics,
)
from recsys.two_stage import UNIFIED_SOURCE_ORDER


def test_catalog_evaluation_scores_every_non_history_item() -> None:
    catalog = np.asarray([10, 11, 12, 13], dtype=np.int64)
    queries, cohort = build_catalog_evaluation_queries(
        split_name="validation",
        target_rows=[(1, 13, 20)],
        known_rows=[(1, 10, 10)],
        model_user_ids=np.asarray([1]),
        catalog_item_ids=catalog,
        seed=7,
        max_eval_users=None,
    )
    seen_candidates: list[int] = []

    def score(_: int, candidates: np.ndarray) -> np.ndarray:
        seen_candidates.extend(map(int, candidates))
        return candidates.astype(np.float64)

    metrics = evaluate_catalog_queries(queries, catalog, score)

    assert cohort["protocol"] == "complete_eligible_catalog_v1"
    assert seen_candidates == [11, 12, 13]
    assert metrics == {"recall@10": 1.0, "ndcg@10": 1.0, "hitrate@10": 1.0}


def test_sampled_all_items_protocol_is_deterministic_and_legacy_compatible() -> None:
    kwargs = {
        "split_name": "validation",
        "target_rows": [(1, 13, 20)],
        "known_rows": [(1, 10, 10)],
        "model_user_ids": np.asarray([1]),
        "catalog_item_ids": np.arange(10, 112, dtype=np.int64),
        "seed": 7,
        "max_eval_users": None,
    }
    queries, cohort = build_sampled_all_items_queries(**kwargs)
    legacy_queries, legacy_cohort = build_full_evaluation_queries(**kwargs)

    assert queries == legacy_queries
    assert cohort["query_set_sha256"] == legacy_cohort["query_set_sha256"]
    assert cohort["protocol"] == SAMPLED_NEGATIVE_PROTOCOL
    assert cohort["negatives_per_query"] == 100
    assert len(queries[0].negative_item_ids) == 100

    current = {"sampled_all_items": {"cohort": cohort}}
    legacy = {"full_catalog": {"cohort": legacy_cohort}}
    assert sampled_all_items_metrics(current)["cohort"] == cohort
    assert sampled_all_items_metrics(legacy)["cohort"] == legacy_cohort


def test_ranker_training_replays_full_catalog_before_selecting_hard_negatives() -> None:
    catalog = np.asarray([10, 11, 12, 13, 14], dtype=np.int64)
    calls: list[list[int]] = []

    def query_features(_: int, candidates: np.ndarray):
        calls.append(list(map(int, candidates)))
        cats = np.zeros((len(candidates), 3), dtype=np.int64)
        cats[:, 2] = 1
        continuous = np.zeros(
            (len(candidates), len(UNIFIED_SOURCE_ORDER) * 3 + 3), dtype=np.float32
        )
        continuous[:, 0] = np.linspace(1.0, 0.2, len(candidates))
        included = np.ones(len(candidates), dtype=bool)
        return cats, continuous, included

    cats, continuous, labels, weights, audit = _build_replay_training_features(
        np.asarray([0]),
        np.asarray([[0, 1, 2]], dtype=np.int64),
        catalog_ids=catalog,
        user_ids=np.asarray([1]),
        known_by_user={1: {0}},
        query_features=query_features,
        negatives_per_positive=2,
        seed=7,
    )

    assert calls == [[10, 11, 12, 13, 14]]
    assert cats.shape == (3, 3)
    assert continuous.shape[1] == len(UNIFIED_SOURCE_ORDER) * 3 + 3
    assert labels.tolist() == [1.0, 0.0, 0.0]
    assert weights.tolist() == [1.0, 1.0, 1.0]
    assert audit["retrieved_positive_groups"] == 1
