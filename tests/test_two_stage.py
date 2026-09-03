from __future__ import annotations

from dataclasses import dataclass

from recsys.two_stage import (
    UNIFIED_SOURCE_ORDER,
    apply_source_caps,
    calibrate_source_score,
    merge_candidate_sources,
    safe_quality_prior,
)


@dataclass(frozen=True)
class SourceCandidate:
    item_id: str
    source: str
    raw_score: float
    eligible: bool = True
    support: int = 1


def test_union_deduplicates_and_preserves_all_seven_source_features() -> None:
    sources = {
        source: [SourceCandidate("shared", source, 10.0 - index)]
        for index, source in enumerate(UNIFIED_SOURCE_ORDER)
    }
    sources["svd"].append(SourceCandidate("svd-only", "svd", 0.5))

    merged, diagnostics = merge_candidate_sources(sources)

    assert diagnostics["union_size"] == 2
    shared = next(candidate for candidate in merged if candidate.item_id == "shared")
    assert shared.source_mask == (1.0,) * len(UNIFIED_SOURCE_ORDER)
    assert len(shared.source_scores) == len(UNIFIED_SOURCE_ORDER)
    assert shared.primary_source == "svd"


def test_union_filters_before_ranking_and_rejects_unsupported_cf() -> None:
    sources = {
        "svd": [
            SourceCandidate("seen", "svd", 9.0),
            SourceCandidate("offline", "svd", 8.0),
            SourceCandidate("ok", "svd", 7.0),
        ],
        "item_cf": [
            SourceCandidate("zero", "item_cf", 0.0, support=3),
            SourceCandidate("unsupported", "item_cf", 2.0, support=0),
            SourceCandidate("cf-ok", "item_cf", 1.0, support=2),
        ],
    }

    merged, diagnostics = merge_candidate_sources(
        sources,
        excluded_item_ids={"seen"},
        eligible_item_ids={"ok", "cf-ok"},
    )

    assert {candidate.item_id for candidate in merged} == {"ok", "cf-ok"}
    assert diagnostics["admitted"]["svd"] == 1
    assert diagnostics["admitted"]["item_cf"] == 1


def test_source_limits_are_upper_bounds_and_never_force_missing_sources() -> None:
    sources = {
        "svd": [SourceCandidate(str(index), "svd", 10.0 - index) for index in range(4)]
    }

    merged, diagnostics = merge_candidate_sources(sources, source_limits={"svd": 2})

    assert [candidate.item_id for candidate in merged] == ["0", "1"]
    assert diagnostics["admitted"]["svd"] == 2
    assert all(diagnostics["admitted"][source] == 0 for source in UNIFIED_SOURCE_ORDER[1:])


def test_calibrated_scores_are_bounded_and_part_of_unified_features() -> None:
    sources = {
        "svd": [SourceCandidate("shared", "svd", -0.25)],
        "popular": [SourceCandidate("shared", "popular", 1000.0)],
    }

    merged, _ = merge_candidate_sources(sources)

    assert len(merged) == 1
    assert all(0.0 <= value <= 1.0 for value in merged[0].source_calibrated_scores)
    assert len(merged[0].feature_values()) == len(UNIFIED_SOURCE_ORDER) * 3
    assert calibrate_source_score("popular", 1000.0) < 1.0
    evidence = merged[0].source_evidence()
    assert evidence["svd"]["raw_score"] == -0.25
    assert evidence["svd"]["rank_in_source"] == 0
    assert evidence["svd"]["eligible"] is True
    assert evidence["svd"]["feature_version"] == "svd-candidate-v1"
    assert "content" not in evidence


def test_source_caps_apply_to_each_top10_window_and_relax_only_to_fill() -> None:
    ranked = [
        SourceCandidate(f"s{index:02d}", "svd", 100.0 - index) for index in range(14)
    ] + [
        SourceCandidate(f"d{index:02d}", "dssm", 50.0 - index) for index in range(10)
    ]

    selected, diagnostics = apply_source_caps(
        ranked,
        limit=20,
        caps_at_10={"svd": 6, "dssm": 6},
    )

    assert len(selected) == 20
    for start in (0, 10):
        window = selected[start : start + 10]
        assert sum(candidate.source == "svd" for candidate in window) <= 6
        assert sum(candidate.source == "dssm" for candidate in window) <= 6
    assert diagnostics["window_size"] == 10

    only_svd, relaxed = apply_source_caps(ranked[:10], limit=10, caps_at_10={"svd": 6})
    assert len(only_svd) == 10
    assert relaxed["relaxed"] == 4


def test_quality_prior_protects_warm_and_cold_without_closing_union() -> None:
    sources = {
        "svd": [SourceCandidate(f"warm-{index}", "svd", 20 - index) for index in range(9)],
        "content": [
            SourceCandidate(f"cold-{index}", "content", 20 - index) for index in range(5)
        ],
        "visual": [SourceCandidate("visual-only", "visual", 1.0)],
    }
    merged, _ = merge_candidate_sources(sources)

    scores = safe_quality_prior(
        merged, cold_item_ids={f"cold-{index}" for index in range(5)} | {"visual-only"}
    )
    ordered = sorted(scores, key=lambda item_id: (-scores[item_id], item_id))

    assert ordered[:7] == [f"warm-{index}" for index in range(7)]
    assert ordered[7:10] == [f"cold-{index}" for index in range(3)]
    assert "visual-only" in scores
