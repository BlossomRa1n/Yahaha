from __future__ import annotations

import pytest

from recsys.popularity import (
    DAY_MS,
    build_popularity_features,
    snapshot_is_available,
    timestamp_to_ms,
)


def test_popularity_features_are_cutoff_bounded_and_window_inclusive() -> None:
    cutoff = 40 * DAY_MS
    base_events = [
        ("item", cutoff, 1.0),
        ("item", cutoff - DAY_MS, 1.0),
        ("item", cutoff - 7 * DAY_MS, 1.0),
        ("item", cutoff - 30 * DAY_MS, 1.0),
        ("item", cutoff - 30 * DAY_MS - 1, 1.0),
    ]
    first = build_popularity_features(base_events, feature_cutoff_ms=cutoff)["item"]
    with_future = build_popularity_features(
        [*base_events, ("item", cutoff + 1, 1000.0)],
        feature_cutoff_ms=cutoff,
    )["item"]

    assert with_future == first
    assert first.cumulative_interactions == 5
    assert first.interactions_1d == 2
    assert first.interactions_7d == 3
    assert first.interactions_30d == 4
    assert 1.0 < first.time_decay_score < first.cumulative_interactions
    assert first.recent_growth == pytest.approx(2 - (3 - 2) / 6)


def test_likes_views_snapshot_requires_known_available_at_before_prediction() -> None:
    prediction_ms = timestamp_to_ms("2026-09-02T04:00:00Z")
    assert snapshot_is_available(None, prediction_ms) is False
    assert snapshot_is_available("2027-01-15T08:00:00Z", prediction_ms) is False
    assert snapshot_is_available("2026-01-15T08:00:00Z", prediction_ms) is True
