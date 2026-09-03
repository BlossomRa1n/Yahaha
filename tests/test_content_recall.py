from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from app.artifacts import ArtifactStore
from recsys.artifacts import publish_artifact, write_staged_artifact
from recsys.content import build_content_features


def _write_items(path: Path) -> None:
    rows = [
        (1, "alpha cooking kitchen"),
        (2, "alpha cooking recipe"),
        (3, "beta racing car"),
        (4, "beta racing driver"),
        (5, "neutral daily story"),
        (6, "neutral daily news"),
        (7, "alpha secretzzzz cold item"),
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item_id", "title", "likes", "views"])
        for item_id, title in rows:
            writer.writerow([item_id, title, 0, 0])


def test_train_only_vocabulary_transforms_cold_items_and_profiles_differ(
    tmp_path: Path,
) -> None:
    items_path = tmp_path / "items.csv"
    _write_items(items_path)
    cutoff = 1_000
    train_rows = [
        (10, 1, 100),
        (10, 2, 200),
        (20, 3, 300),
        (20, 4, 400),
        (30, 5, 500),
        (30, 6, 600),
        (10, 7, cutoff + 1),
    ]

    features = build_content_features(
        items_path=items_path,
        train_rows=train_rows,
        user_ids=np.asarray([10, 20, 30]),
        feature_cutoff_ms=cutoff,
        max_features=512,
        min_df=2,
    )

    vocabulary = features.vectorizer_state["vocabulary"]
    assert not any("zz" in token for token in vocabulary)
    cold_index = int(np.flatnonzero(features.item_ids == 7)[0])
    assert features.item_vectors[cold_index].nnz > 0
    assert features.metadata["future_train_rows_excluded"] == 1
    user_alpha = features.user_vectors[0]
    user_beta = features.user_vectors[1]
    alpha_score = float((features.item_vectors[cold_index] @ user_alpha.T).toarray()[0, 0])
    beta_score = float((features.item_vectors[cold_index] @ user_beta.T).toarray()[0, 0])
    assert alpha_score > beta_score

    without_future = build_content_features(
        items_path=items_path,
        train_rows=train_rows[:-1],
        user_ids=np.asarray([10, 20, 30]),
        feature_cutoff_ms=cutoff,
        max_features=512,
        min_df=2,
    )
    assert features.vectorizer_state == without_future.vectorizer_state
    np.testing.assert_allclose(
        features.user_vectors.toarray(),
        without_future.user_vectors.toarray(),
    )


def test_corrupt_content_artifact_keeps_svd_available(tmp_path: Path) -> None:
    items_path = tmp_path / "items.csv"
    _write_items(items_path)
    features = build_content_features(
        items_path=items_path,
        train_rows=[(10, 1, 100), (10, 2, 200), (20, 3, 300), (20, 4, 400)],
        user_ids=np.asarray([10, 20]),
        feature_cutoff_ms=1_000,
        max_features=512,
        min_df=2,
    )
    artifacts = tmp_path / "artifacts"
    staged = write_staged_artifact(
        artifacts,
        manifest_base={
            "algorithm": "toy-svd-content",
            "content_features": {**features.metadata, "schema_version": 1},
            "data_version": "toy-data",
            "model_version": "toy-content",
            "schema_version": 1,
        },
        user_ids=np.asarray([10, 20]),
        item_ids=np.asarray([1, 2, 3, 4]),
        user_factors=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        item_factors=np.asarray(
            [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]]
        ),
        popularity={"items": []},
        metrics={"schema_version": 1},
        evaluation_markdown="# toy\n",
        extra_artifacts={
            "content_item_ids.npy": features.item_ids,
            "content_user_ids.npy": features.user_ids,
            "content_item_vectors.npz": features.item_vectors,
            "content_user_vectors.npz": features.user_vectors,
            "content_idf.npy": features.idf,
            "content_vectorizer.json": features.vectorizer_state,
        },
    )
    destination = publish_artifact(staged, artifacts)
    with (destination / "content_item_vectors.npz").open("ab") as handle:
        handle.write(b"corrupt")

    artifact = ArtifactStore(artifacts / "current.json").get()

    assert artifact is not None
    assert artifact.model_version == "toy-content"
    assert artifact.content_item_vectors is None
    assert artifact.feature_errors
