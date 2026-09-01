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
from recsys.model import EvaluationQuery, evaluate_queries


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

