from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from recsys.artifacts import (
    ArtifactValidationError,
    publish_artifact,
    write_staged_artifact,
)
from recsys.model import (
    EvaluationQuery,
    _evaluation_markdown,
    analyze_badcases,
    evaluate_queries,
)


def test_ranking_metrics_match_hand_calculation() -> None:
    queries = [EvaluationQuery(1, (10, 11), (20, 21, 22))]
    scores = {10: 0.9, 20: 0.8, 11: 0.7, 21: 0.6, 22: 0.5}

    result = evaluate_queries(
        queries,
        lambda _user_id, item_ids: np.asarray([scores[int(item_id)] for item_id in item_ids]),
        k=2,
    )

    assert result["recall@2"] == pytest.approx(0.5)
    assert result["hitrate@2"] == pytest.approx(1.0)
    expected_ndcg = 1.0 / (1.0 + 1.0 / np.log2(3.0))
    assert result["ndcg@2"] == pytest.approx(expected_ndcg)


def test_badcases_are_anonymized_ranked_and_reproducible() -> None:
    queries = [
        EvaluationQuery(123, (10,), (20, 21, 22)),
        EvaluationQuery(456, (30, 31), (40, 41, 42)),
    ]
    scores = {
        10: 0.1, 20: 0.9, 21: 0.8, 22: 0.7,
        30: 0.9, 31: 0.1, 40: 0.8, 41: 0.7, 42: 0.6,
    }

    def score(_user_id: int, item_ids: np.ndarray) -> np.ndarray:
        return np.asarray([scores[int(item_id)] for item_id in item_ids])

    first = analyze_badcases(queries, score, k=2)
    second = analyze_badcases(queries, score, k=2)

    assert first == second
    assert len(first) == 2
    assert first[0]["user_id"].startswith("u-")
    assert first[0]["user_id"] != "123"
    assert first[0]["positive_item_ids"][0].startswith("i-")
    assert first[0]["positive_ranks"] == [4]
    assert first[0]["reason"] == "no_positive_in_top_k"
    assert first[1]["reason"] == "some_positives_below_top_k"


def test_evaluation_report_explains_query_miss_and_coverage() -> None:
    cohort = {
        "evaluated_users": 1,
        "item_coverage": 0.5,
        "target_positive_interactions": 4,
        "scorable_positive_interactions": 2,
    }
    model_metrics = {"recall@10": 0.0, "ndcg@10": 0.0, "hitrate@10": 0.0}
    badcase = {
        "candidate_count": 101,
        "positive_count": 1,
        "positive_item_ids": ["i-redacted"],
        "positive_ranks": [23],
        "reason": "no_positive_in_top_k",
        "top_k": 10,
        "user_id": "u-redacted",
    }
    split = {
        "cohort": cohort,
        "models": {name: model_metrics for name in ("popular", "random", "svd")},
        "badcases": {"svd": [badcase]},
    }

    report = _evaluation_markdown({"validation": split, "test": split})

    assert "u-redacted" in report
    assert "i-redacted" in report
    assert "| 23 | 101 | no_positive_in_top_k |" in report
    assert "2 target positives were not rankable" in report


def _stage(artifacts: Path, version: str) -> Path:
    return write_staged_artifact(
        artifacts,
        manifest_base={
            "algorithm": "toy",
            "data_version": "toy-data",
            "model_version": version,
            "schema_version": 1,
            "training_config": {"seed": 1},
            "training_summary": {},
        },
        user_ids=np.asarray([1, 2]),
        item_ids=np.asarray([10, 20, 30]),
        user_factors=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        item_factors=np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]),
        popularity={"items": [], "schema_version": 1},
        metrics={"schema_version": 1},
        evaluation_markdown="# Toy\n",
    )


def test_corrupt_artifact_does_not_replace_current_pointer(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    first = _stage(artifacts, "model-one")
    publish_artifact(first, artifacts)
    original_pointer = (artifacts / "current.json").read_bytes()

    corrupt = _stage(artifacts, "model-two")
    with (corrupt / "user_factors.npy").open("ab") as handle:
        handle.write(b"corrupt")

    with pytest.raises(ArtifactValidationError, match="checksum mismatch"):
        publish_artifact(corrupt, artifacts)
    assert (artifacts / "current.json").read_bytes() == original_pointer
    assert json.loads(original_pointer)["model_version"] == "model-one"
    assert not (artifacts / "model-two").exists()
