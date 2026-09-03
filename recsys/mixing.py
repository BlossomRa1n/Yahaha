from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence


SOURCE_ORDER = ("model", "content_profile", "item_cf", "popular", "explore")
SAFE_POLICY_VERSION = "safe_svd_content_v2"
DYNAMIC_POLICY_VERSION = "dynamic_confidence_v2"


@dataclass(frozen=True)
class MixCandidate:
    item_id: str
    source: str
    score: float
    explanation: str
    model_version: str | None
    feature_version: str | None = None
    raw_score: float | None = None
    normalized_score: float | None = None
    rank_in_source: int | None = None
    eligible: bool = True
    confidence: float = 1.0
    support: int = 1
    is_cold: bool = False
    is_forced: bool = False
    desired_position: int = 0
    priority: int = 0


@dataclass(frozen=True)
class MixContext:
    warm_user: bool = True
    history_count: int = 0
    cf_support: int = 0
    content_confidence: float = 0.0


def _rank_normalize(candidates: Sequence[MixCandidate]) -> list[MixCandidate]:
    valid = [
        candidate
        for candidate in candidates
        if candidate.eligible
        and math.isfinite(
            float(candidate.raw_score if candidate.raw_score is not None else candidate.score)
        )
        and not (
            candidate.source == "item_cf"
            and (
                float(candidate.raw_score if candidate.raw_score is not None else candidate.score)
                <= 0
                or candidate.support <= 0
            )
        )
    ]
    valid.sort(
        key=lambda candidate: (
            -float(candidate.raw_score if candidate.raw_score is not None else candidate.score),
            candidate.item_id,
        )
    )
    denominator = max(1, len(valid) - 1)
    return [
        replace(
            candidate,
            raw_score=float(
                candidate.raw_score if candidate.raw_score is not None else candidate.score
            ),
            normalized_score=1.0 - rank / denominator,
            rank_in_source=rank,
        )
        for rank, candidate in enumerate(valid)
    ]


def _policy_pattern(policy_version: str, context: MixContext) -> tuple[str, ...]:
    if policy_version == SAFE_POLICY_VERSION:
        if context.warm_user:
            return ("model",) * 7 + ("content_profile",) * 3
        return ("content_profile",) * 7 + ("popular",) * 2 + ("explore",)
    if policy_version != DYNAMIC_POLICY_VERSION:
        raise ValueError(f"unknown mix policy: {policy_version}")
    if not context.warm_user:
        return ("content_profile",) * 7 + ("popular",) * 2 + ("explore",)
    if context.history_count <= 2:
        return ("model",) * 7 + ("content_profile",) * 2 + ("popular",)
    if context.cf_support >= 2 and context.content_confidence >= 0.05:
        return ("model",) * 7 + ("content_profile",) * 2 + ("item_cf",)
    if context.cf_support >= 2:
        return ("model",) * 8 + ("item_cf",) + ("popular",)
    return ("model",) * 7 + ("content_profile",) * 2 + ("popular",)


def _jaccard(source_items: Mapping[str, set[str]]) -> dict[str, float]:
    overlaps: dict[str, float] = {}
    for left_index, left in enumerate(SOURCE_ORDER):
        for right in SOURCE_ORDER[left_index + 1 :]:
            union = source_items[left] | source_items[right]
            overlaps[f"{left}:{right}"] = (
                len(source_items[left] & source_items[right]) / len(union) if union else 0.0
            )
    return overlaps


def mix_candidates(
    policy_version: str,
    sources: Mapping[str, Sequence[MixCandidate]],
    limit: int,
    context: MixContext | None = None,
) -> tuple[list[MixCandidate], dict[str, Any]]:
    """Deterministically mix source-ranked candidates without business-rule side effects."""
    if limit < 0:
        raise ValueError("limit must be non-negative")
    context = context or MixContext()
    normalized = {
        name: _rank_normalize(list(sources.get(name) or ())) for name in SOURCE_ORDER
    }
    if policy_version == SAFE_POLICY_VERSION and context.warm_user:
        normalized["content_profile"] = [
            candidate for candidate in normalized["content_profile"] if candidate.is_cold
        ]
        for name in ("item_cf", "popular", "explore"):
            normalized[name] = []
    elif policy_version == SAFE_POLICY_VERSION:
        normalized["model"] = []
        normalized["item_cf"] = []

    pattern = _policy_pattern(policy_version, context)
    cursors = {name: 0 for name in SOURCE_ORDER}
    selected: list[MixCandidate] = []
    used: set[str] = set()
    selected_counts = {name: 0 for name in SOURCE_ORDER}
    requested_counts = {name: 0 for name in SOURCE_ORDER}
    relaxations: list[dict[str, Any]] = []

    def take(source_name: str) -> MixCandidate | None:
        values = normalized[source_name]
        while cursors[source_name] < len(values):
            candidate = values[cursors[source_name]]
            cursors[source_name] += 1
            if candidate.item_id in used:
                continue
            return candidate
        return None

    for slot in range(limit):
        requested = pattern[slot % len(pattern)]
        requested_counts[requested] += 1
        candidate = take(requested)
        actual_source = requested
        if candidate is None:
            ranked_sources = sorted(
                SOURCE_ORDER,
                key=lambda name: (
                    -(
                        float(normalized[name][cursors[name]].normalized_score or 0.0)
                        if cursors[name] < len(normalized[name])
                        else -1.0
                    ),
                    SOURCE_ORDER.index(name),
                ),
            )
            for fallback_source in ranked_sources:
                candidate = take(fallback_source)
                if candidate is not None:
                    actual_source = fallback_source
                    relaxations.append(
                        {
                            "slot": slot,
                            "requested_source": requested,
                            "used_source": fallback_source,
                            "reason": "source_exhausted_ineligible_or_duplicate",
                        }
                    )
                    break
        if candidate is None:
            break
        used.add(candidate.item_id)
        selected_counts[actual_source] += 1
        selected.append(
            replace(
                candidate,
                score=float(candidate.normalized_score or 0.0),
                explanation=(
                    f"{candidate.explanation}; bucket={actual_source}; "
                    f"source_rank={candidate.rank_in_source}"
                ),
            )
        )

    source_items = {
        name: {candidate.item_id for candidate in values} for name, values in normalized.items()
    }
    return selected, {
        "strategy": "shared_candidate_mixer",
        "mix_policy_version": policy_version,
        "context": {
            "warm_user": context.warm_user,
            "history_count": context.history_count,
            "cf_support": context.cf_support,
            "content_confidence": context.content_confidence,
        },
        "pattern": list(pattern),
        "requested": {key: value for key, value in requested_counts.items() if value},
        "available": {key: len(value) for key, value in normalized.items()},
        "selected": {key: value for key, value in selected_counts.items() if value},
        "relaxations": relaxations[:100],
        "fallback_count": len(relaxations),
        "source_jaccard": _jaccard(source_items),
        "deduplicated": True,
    }
