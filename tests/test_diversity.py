from __future__ import annotations

import sqlite3

from app.recommendation import Candidate, RecommendationService


def _candidate(item_id: str, score: float) -> Candidate:
    return Candidate(
        item_id=item_id,
        source="model",
        score=score,
        explanation="latent score",
        model_version="model-v1",
    )


def test_title_mmr_reduces_adjacent_similarity_deterministically() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE items(item_id TEXT PRIMARY KEY, title TEXT NOT NULL)")
    conn.executemany(
        "INSERT INTO items(item_id, title) VALUES (?, ?)",
        [
            ("1", "Sports football highlights"),
            ("2", "Sports football interview"),
            ("3", "Sports football tactics"),
            ("4", "Technology laptop review"),
            ("5", "Cooking noodle recipe"),
            ("6", "Travel mountain guide"),
        ],
    )
    candidates = [_candidate(str(index), 7.0 - index) for index in range(1, 7)]

    first, metrics = RecommendationService._rerank_diverse(conn, candidates)
    second, repeated_metrics = RecommendationService._rerank_diverse(conn, candidates)

    assert [item.item_id for item in first] == [item.item_id for item in second]
    assert first[0].item_id == "1"
    assert first[1].item_id != "2"
    assert metrics == repeated_metrics
    assert (
        metrics["after"]["adjacent_title_similarity"]
        < metrics["before"]["adjacent_title_similarity"]
    )
    assert (
        metrics["after"]["intra_list_diversity"]
        == metrics["before"]["intra_list_diversity"]
    )
    assert metrics["after"]["duplicate_items"] == 0
    assert all("diversity rerank" in item.explanation for item in first)
    assert [item.score for item in sorted(first, key=lambda item: item.item_id)] == [
        item.score for item in candidates
    ]
