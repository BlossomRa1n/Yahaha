from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Iterable


DAY_MS = 86_400_000
WINDOW_DAYS = (1, 7, 30)
DEFAULT_HALF_LIFE_DAYS = 7.0


@dataclass(frozen=True)
class PopularityFeatures:
    cumulative_interactions: float
    interactions_1d: float
    interactions_7d: float
    interactions_30d: float
    time_decay_score: float
    recent_growth: float


def parse_utc_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def timestamp_to_ms(value: str) -> int:
    return int(parse_utc_timestamp(value).timestamp() * 1000)


def snapshot_is_available(available_at: str | None, prediction_time_ms: int) -> bool:
    if not available_at:
        return False
    return timestamp_to_ms(available_at) <= prediction_time_ms


def build_popularity_features(
    events: Iterable[tuple[str | int, int, float]],
    *,
    feature_cutoff_ms: int,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
) -> dict[str, PopularityFeatures]:
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")
    values: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "cumulative": 0.0,
            "d1": 0.0,
            "d7": 0.0,
            "d30": 0.0,
            "decay": 0.0,
        }
    )
    half_life_ms = half_life_days * DAY_MS
    for item_id, event_timestamp_ms, weight in events:
        if event_timestamp_ms > feature_cutoff_ms:
            continue
        numeric_weight = float(weight)
        if not math.isfinite(numeric_weight) or numeric_weight < 0:
            raise ValueError("event weights must be finite and non-negative")
        age_ms = feature_cutoff_ms - event_timestamp_ms
        item = values[str(item_id)]
        item["cumulative"] += numeric_weight
        if age_ms <= DAY_MS:
            item["d1"] += numeric_weight
        if age_ms <= 7 * DAY_MS:
            item["d7"] += numeric_weight
        if age_ms <= 30 * DAY_MS:
            item["d30"] += numeric_weight
        item["decay"] += numeric_weight * math.exp(-math.log(2.0) * age_ms / half_life_ms)

    return {
        item_id: PopularityFeatures(
            cumulative_interactions=value["cumulative"],
            interactions_1d=value["d1"],
            interactions_7d=value["d7"],
            interactions_30d=value["d30"],
            time_decay_score=value["decay"],
            recent_growth=value["d1"] - (value["d7"] - value["d1"]) / 6.0,
        )
        for item_id, value in values.items()
    }
