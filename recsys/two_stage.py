from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


UNIFIED_SOURCE_ORDER = (
    "svd",
    "dssm",
    "content",
    "visual",
    "item_cf",
    "popular",
    "explore",
)

DEFAULT_RETRIEVAL_LIMITS = {
    "svd": 150,
    "dssm": 200,
    "content": 150,
    "visual": 150,
    "item_cf": 100,
    "popular": 50,
    "explore": 50,
}

DEFAULT_SOURCE_MIN_CONFIDENCE = {
    "svd": 0.0,
    "dssm": 0.0,
    "content": 0.0,
    "visual": 0.0,
    "item_cf": 0.0,
    "popular": 0.0,
    "explore": 0.0,
}

DEFAULT_SOURCE_CAPS_AT_10 = {
    "svd": 7,
    "dssm": 6,
    "content": 4,
    "visual": 3,
    "item_cf": 3,
    "popular": 2,
    "explore": 1,
}

UNIFIED_FEATURE_SCHEMA_VERSION = "unified-seven-source-v3"


@dataclass(frozen=True)
class UnifiedCandidate:
    item_id: str
    source_scores: tuple[float, ...]
    source_mask: tuple[float, ...]
    primary_source: str
    source_raw_scores: tuple[float | None, ...]
    source_calibrated_scores: tuple[float, ...] = ()
    source_ranks: tuple[int | None, ...] = ()
    source_confidences: tuple[float | None, ...] = ()
    source_supports: tuple[int | None, ...] = ()
    source_feature_versions: tuple[str | None, ...] = ()
    source_model_versions: tuple[str | None, ...] = ()
    source_explanations: tuple[str | None, ...] = ()
    ranker_score: float | None = None
    ranker_rank: int | None = None

    def feature_values(self) -> tuple[float, ...]:
        calibrated = self.source_calibrated_scores or tuple(
            calibrate_source_score(source, raw)
            for source, raw in zip(UNIFIED_SOURCE_ORDER, self.source_raw_scores)
        )
        return (*self.source_scores, *calibrated, *self.source_mask)

    def source_evidence(self) -> dict[str, dict[str, Any]]:
        return {
            source: {
                "source": source,
                "raw_score": self.source_raw_scores[index],
                "normalized_score": self.source_scores[index],
                "rank_in_source": (
                    self.source_ranks[index] if self.source_ranks else None
                ),
                "confidence": (
                    self.source_confidences[index] if self.source_confidences else None
                ),
                "support": self.source_supports[index] if self.source_supports else None,
                "eligible": bool(self.source_mask[index]),
                "feature_version": (
                    self.source_feature_versions[index]
                    if self.source_feature_versions
                    else None
                ),
                "model_version": (
                    self.source_model_versions[index]
                    if self.source_model_versions
                    else None
                ),
                "explanation": (
                    self.source_explanations[index] if self.source_explanations else None
                ),
            }
            for index, source in enumerate(UNIFIED_SOURCE_ORDER)
            if bool(self.source_mask[index])
        }


def calibrate_source_score(source: str, raw_score: float | None) -> float:
    """Apply a fixed bounded transform shared by training and online serving."""
    if raw_score is None:
        return 0.0
    value = float(raw_score)
    if not math.isfinite(value):
        return 0.0
    if source == "svd":
        return 0.5 + 0.5 * math.tanh(value)
    if source in {"dssm", "content", "visual"}:
        return max(0.0, min(1.0, 0.5 + 0.5 * value))
    if source in {"item_cf", "popular"}:
        positive = max(0.0, value)
        compressed = math.log1p(positive)
        return compressed / (1.0 + compressed)
    return max(0.0, min(1.0, value))


def _valid_score(candidate: Any) -> float | None:
    if not bool(getattr(candidate, "eligible", True)):
        return None
    score = getattr(candidate, "raw_score", None)
    if score is None:
        score = getattr(candidate, "score", None)
    try:
        value = float(score)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    if getattr(candidate, "source", "") == "item_cf" and (
        value <= 0 or int(getattr(candidate, "support", 0)) <= 0
    ):
        return None
    return value


def merge_candidate_sources(
    sources: Mapping[str, Sequence[Any]],
    *,
    source_limits: Mapping[str, int] | None = None,
    source_min_confidence: Mapping[str, float] | None = None,
    excluded_item_ids: set[str] | None = None,
    eligible_item_ids: set[str] | None = None,
) -> tuple[list[UnifiedCandidate], dict[str, Any]]:
    """Build a deduplicated union; source limits cap generation and never force placement."""
    limits = {**DEFAULT_RETRIEVAL_LIMITS, **dict(source_limits or {})}
    thresholds = {
        **DEFAULT_SOURCE_MIN_CONFIDENCE,
        **dict(source_min_confidence or {}),
    }
    excluded = excluded_item_ids or set()
    by_item: dict[str, dict[str, tuple[float, float, int, float, int, str, str | None, str]]] = {}
    available: dict[str, int] = {}
    admitted: dict[str, int] = {}
    filtered: dict[str, int] = {}

    for source_name in UNIFIED_SOURCE_ORDER:
        valid: list[tuple[str, float, Any]] = []
        filtered[source_name] = 0
        for candidate in sources.get(source_name, ()):
            item_id = str(candidate.item_id)
            value = _valid_score(candidate)
            confidence = float(getattr(candidate, "confidence", 1.0))
            if (
                value is None
                or not math.isfinite(confidence)
                or confidence < float(thresholds[source_name])
                or item_id in excluded
            ):
                filtered[source_name] += 1
                continue
            if eligible_item_ids is not None and item_id not in eligible_item_ids:
                filtered[source_name] += 1
                continue
            valid.append((item_id, value, candidate))
        valid.sort(key=lambda row: (-row[1], row[0]))
        available[source_name] = len(valid)
        selected = valid[: max(0, int(limits[source_name]))]
        admitted[source_name] = len(selected)
        denominator = max(1, len(selected) - 1)
        for rank, (item_id, raw_score, candidate) in enumerate(selected):
            normalized = 1.0 - rank / denominator
            by_item.setdefault(item_id, {})[source_name] = (
                raw_score,
                normalized,
                rank,
                float(getattr(candidate, "confidence", 1.0)),
                int(getattr(candidate, "support", 1)),
                str(
                    getattr(candidate, "feature_version", None)
                    or f"{source_name}-candidate-v1"
                ),
                getattr(candidate, "model_version", None),
                str(getattr(candidate, "explanation", "")),
            )

    merged: list[UnifiedCandidate] = []
    for item_id, memberships in by_item.items():
        scores = tuple(
            float(memberships.get(name, (0.0, 0.0, 0, 0.0, 0, "", None, ""))[1])
            for name in UNIFIED_SOURCE_ORDER
        )
        mask = tuple(float(name in memberships) for name in UNIFIED_SOURCE_ORDER)
        raw = tuple(
            memberships[name][0] if name in memberships else None
            for name in UNIFIED_SOURCE_ORDER
        )
        ranks = tuple(memberships[name][2] if name in memberships else None for name in UNIFIED_SOURCE_ORDER)
        confidences = tuple(memberships[name][3] if name in memberships else None for name in UNIFIED_SOURCE_ORDER)
        supports = tuple(memberships[name][4] if name in memberships else None for name in UNIFIED_SOURCE_ORDER)
        feature_versions = tuple(memberships[name][5] if name in memberships else None for name in UNIFIED_SOURCE_ORDER)
        model_versions = tuple(memberships[name][6] if name in memberships else None for name in UNIFIED_SOURCE_ORDER)
        explanations = tuple(memberships[name][7] if name in memberships else None for name in UNIFIED_SOURCE_ORDER)
        calibrated = tuple(
            calibrate_source_score(name, value)
            for name, value in zip(UNIFIED_SOURCE_ORDER, raw)
        )
        primary = max(
            (name for name in UNIFIED_SOURCE_ORDER if name in memberships),
            key=lambda name: (
                memberships[name][1],
                -UNIFIED_SOURCE_ORDER.index(name),
            ),
        )
        merged.append(
            UnifiedCandidate(
                item_id,
                scores,
                mask,
                primary,
                raw,
                calibrated,
                ranks,
                confidences,
                supports,
                feature_versions,
                model_versions,
                explanations,
            )
        )
    merged.sort(key=lambda value: value.item_id)
    return merged, {
        "available": available,
        "admitted": admitted,
        "union_size": len(merged),
        "source_limits": {name: int(limits[name]) for name in UNIFIED_SOURCE_ORDER},
        "source_min_confidence": {
            name: float(thresholds[name]) for name in UNIFIED_SOURCE_ORDER
        },
        "filtered": filtered,
        "deduplicated": True,
    }


def safe_quality_prior(
    candidates: Sequence[UnifiedCandidate], *, cold_item_ids: set[str]
) -> dict[str, float]:
    """Reproduce the stable 7-warm/3-cold preference over an open candidate union."""
    if not candidates:
        return {}
    svd_index = UNIFIED_SOURCE_ORDER.index("svd")
    content_index = UNIFIED_SOURCE_ORDER.index("content")
    popular_index = UNIFIED_SOURCE_ORDER.index("popular")
    warm = sorted(
        (
            candidate
            for candidate in candidates
            if candidate.item_id not in cold_item_ids and candidate.source_mask[svd_index]
        ),
        key=lambda candidate: (-candidate.source_scores[svd_index], candidate.item_id),
    )
    cold = sorted(
        (
            candidate
            for candidate in candidates
            if candidate.item_id in cold_item_ids and candidate.source_mask[content_index]
        ),
        key=lambda candidate: (-candidate.source_scores[content_index], candidate.item_id),
    )
    protected = [*warm[:7], *cold[:3]]
    protected_ids = {candidate.item_id for candidate in protected}

    def blended(candidate: UnifiedCandidate) -> float:
        return (
            0.55 * candidate.source_scores[svd_index]
            + 0.35 * candidate.source_scores[content_index]
            + 0.10 * candidate.source_scores[popular_index]
        )

    remainder = sorted(
        (candidate for candidate in candidates if candidate.item_id not in protected_ids),
        key=lambda candidate: (-blended(candidate), candidate.item_id),
    )
    ordered = [*protected, *remainder]
    denominator = max(1, len(ordered) - 1)
    return {
        candidate.item_id: 1.0 - rank / denominator
        for rank, candidate in enumerate(ordered)
    }


def apply_source_caps(
    ranked: Sequence[Any],
    *,
    limit: int,
    caps_at_10: Mapping[str, int] | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    """Apply soft source upper bounds independently to every Top-10 window."""
    if limit <= 0:
        return [], {"selected": {}, "deferred": {}, "relaxed": 0}
    base_caps = {**DEFAULT_SOURCE_CAPS_AT_10, **dict(caps_at_10 or {})}
    selected: list[Any] = []
    counts = {name: 0 for name in UNIFIED_SOURCE_ORDER}
    deferred_counts = {name: 0 for name in UNIFIED_SOURCE_ORDER}
    remaining = list(ranked)
    relaxed = 0
    while remaining and len(selected) < limit:
        window_size = min(10, limit - len(selected))
        window: list[Any] = []
        deferred: list[Any] = []
        window_counts = {name: 0 for name in UNIFIED_SOURCE_ORDER}
        consumed: set[str] = set()
        for candidate in remaining:
            source = str(getattr(candidate, "source", ""))
            if source in base_caps and window_counts[source] >= int(base_caps[source]):
                deferred.append(candidate)
                deferred_counts[source] += 1
                continue
            window.append(candidate)
            consumed.add(str(candidate.item_id))
            if source in window_counts:
                window_counts[source] += 1
            if len(window) >= window_size:
                break
        if len(window) < window_size:
            for candidate in deferred:
                item_id = str(candidate.item_id)
                if item_id in consumed:
                    continue
                window.append(candidate)
                consumed.add(item_id)
                relaxed += 1
                if len(window) >= window_size:
                    break
        if not window:
            break
        selected.extend(window)
        for candidate in window:
            source = str(getattr(candidate, "source", ""))
            if source in counts:
                counts[source] += 1
        remaining = [
            candidate for candidate in remaining if str(candidate.item_id) not in consumed
        ]
    return selected[:limit], {
        "caps_at_10": {name: int(base_caps[name]) for name in UNIFIED_SOURCE_ORDER},
        "selected": {name: count for name, count in counts.items() if count},
        "deferred": {name: count for name, count in deferred_counts.items() if count},
        "relaxed": relaxed,
        "window_size": 10,
    }
