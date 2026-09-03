from __future__ import annotations

import numpy as np

from app.recommendation import Candidate, RecommendationService
from recsys.item_cf import build_item_cf
from recsys.mixing import (
    DYNAMIC_POLICY_VERSION,
    SAFE_POLICY_VERSION,
    MixContext,
    mix_candidates,
)


def _candidate(source: str, item_id: int, score: float = 1.0) -> Candidate:
    return Candidate(
        item_id=f"{source}-{item_id}",
        source=source,
        score=score,
        explanation=source,
        model_version="hybrid-v1",
    )


def test_item_cf_is_cutoff_safe_bounded_and_has_no_self_similarity() -> None:
    rows = [
        (1, 10, 100), (1, 20, 100),
        (2, 10, 200), (2, 20, 200),
        (3, 10, 300), (3, 30, 300),
        (1, 30, 2_000),
    ]
    bundle = build_item_cf(
        train_rows=rows,
        user_ids=np.asarray([1, 2, 3]),
        item_ids=np.asarray([10, 20, 30]),
        feature_cutoff_ms=1_000,
        top_n=1,
        min_support=2,
    )

    assert not bundle.neighbors.diagonal().any()
    assert bundle.neighbors[0, 1] > 0
    assert bundle.neighbors[0, 2] == 0
    assert bundle.metadata["neighbor_edges"] == 2
    assert bundle.metadata["top_n"] == 1
    assert bundle.metadata["feature_cutoff_ms"] == 1_000


def test_multisource_mixer_enforces_quota_normalization_and_determinism() -> None:
    sources = {
        source: [_candidate(source, index, 20.0 - index) for index in range(12)]
        for source in ("model", "content_profile", "item_cf", "popular", "explore")
    }

    first, diagnostics = RecommendationService._mix_sources(sources, 10)
    second, repeated = RecommendationService._mix_sources(sources, 10)

    assert [candidate.item_id for candidate in first] == [candidate.item_id for candidate in second]
    assert diagnostics == repeated
    assert diagnostics["selected"] == {
        "model": 7,
        "content_profile": 2,
        "popular": 1,
    }
    assert diagnostics["mix_policy_version"] == DYNAMIC_POLICY_VERSION
    assert not diagnostics["relaxations"]
    assert len({candidate.item_id for candidate in first}) == 10
    assert all(candidate.raw_score is not None for candidate in first)
    assert all(candidate.normalized_score is not None for candidate in first)
    assert all(candidate.rank_in_source is not None for candidate in first)


def test_multisource_mixer_records_exhaustion_without_duplicates() -> None:
    mixed, diagnostics = RecommendationService._mix_sources(
        {
            "model": [_candidate("model", index) for index in range(6)],
            "content_profile": [],
            "item_cf": [],
            "popular": [_candidate("popular", index) for index in range(6)],
            "explore": [],
        },
        8,
    )

    assert len(mixed) == 8
    assert len({candidate.item_id for candidate in mixed}) == 8
    assert diagnostics["relaxations"]


def test_mixer_excludes_ineligible_and_unsupported_cf_before_normalization() -> None:
    sources = {
        "model": [_candidate("model", index) for index in range(10)],
        "content_profile": [_candidate("content_profile", index) for index in range(10)],
        "item_cf": [
            Candidate(
                item_id="cf-zero",
                source="item_cf",
                score=0.0,
                explanation="unsupported",
                model_version="hybrid-v1",
                eligible=True,
                support=0,
            ),
            Candidate(
                item_id="cf-valid",
                source="item_cf",
                score=0.8,
                explanation="supported",
                model_version="hybrid-v1",
                eligible=True,
                confidence=0.8,
                support=3,
            ),
        ],
    }

    mixed, diagnostics = mix_candidates(
        DYNAMIC_POLICY_VERSION,
        sources,
        10,
        MixContext(history_count=8, cf_support=3, content_confidence=0.8),
    )

    assert "cf-zero" not in {candidate.item_id for candidate in mixed}
    assert "cf-valid" in {candidate.item_id for candidate in mixed}
    assert diagnostics["available"]["item_cf"] == 1


def test_dynamic_policy_changes_with_user_and_source_confidence() -> None:
    sources = {
        source: [_candidate(source, index) for index in range(12)]
        for source in ("model", "content_profile", "item_cf", "popular", "explore")
    }
    dense, dense_diagnostics = mix_candidates(
        DYNAMIC_POLICY_VERSION,
        sources,
        10,
        MixContext(history_count=8, cf_support=3, content_confidence=0.8),
    )
    cold, cold_diagnostics = mix_candidates(
        DYNAMIC_POLICY_VERSION,
        sources,
        10,
        MixContext(warm_user=False, content_confidence=0.8),
    )

    assert dense_diagnostics["selected"]["item_cf"] == 1
    assert cold_diagnostics["selected"] == {
        "content_profile": 7,
        "popular": 2,
        "explore": 1,
    }
    assert [candidate.source for candidate in dense] != [candidate.source for candidate in cold]


def test_safe_policy_uses_only_model_and_cold_content_for_warm_users() -> None:
    warm_content = [
        Candidate(
            item_id="shared-0",
            source="content_profile",
            score=1.0,
            explanation="warm",
            model_version="hybrid-v1",
        )
    ]
    cold_content = [
        Candidate(
            item_id=f"cold-{index}",
            source="content_profile",
            score=0.9 - index * 0.01,
            explanation="cold",
            model_version="hybrid-v1",
            is_cold=True,
        )
        for index in range(3)
    ]
    mixed, diagnostics = mix_candidates(
        SAFE_POLICY_VERSION,
        {
            "model": [
                Candidate(
                    item_id=f"shared-{index}",
                    source="model",
                    score=10.0 - index,
                    explanation="model",
                    model_version="hybrid-v1",
                )
                for index in range(10)
            ],
            "content_profile": [*warm_content, *cold_content],
            "item_cf": [_candidate("item_cf", 1)],
            "popular": [_candidate("popular", 1)],
            "explore": [_candidate("explore", 1)],
        },
        10,
        MixContext(warm_user=True, history_count=5),
    )

    assert diagnostics["mix_policy_version"] == SAFE_POLICY_VERSION
    assert diagnostics["selected"] == {"model": 7, "content_profile": 3}
    assert len({candidate.item_id for candidate in mixed}) == 10
    assert not {"item_cf", "popular", "explore"} & {
        candidate.source for candidate in mixed
    }
