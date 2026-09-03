from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.sparse import csr_matrix, spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .data import sha256_file


CONTENT_SCHEMA_VERSION = 1
CONTENT_BEHAVIOR_WEIGHTS = {
    "click": 1.0,
    "like": 3.0,
    "favorite": 3.0,
    "not_interested": -4.0,
}


class ContentFeatureError(ValueError):
    pass


@dataclass(frozen=True)
class ContentFeatureBundle:
    item_ids: np.ndarray
    user_ids: np.ndarray
    item_vectors: csr_matrix
    user_vectors: csr_matrix
    idf: np.ndarray
    vectorizer_state: dict[str, Any]
    metadata: dict[str, Any]


def _read_titles(path: Path) -> tuple[list[int], list[str]]:
    item_ids: list[int] = []
    titles: list[str] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"item_id", "title"}.issubset(reader.fieldnames):
            raise ContentFeatureError("items.csv must contain item_id and title")
        seen: set[int] = set()
        for line_number, row in enumerate(reader, start=2):
            try:
                item_id = int(row["item_id"])
            except (TypeError, ValueError) as exc:
                raise ContentFeatureError(
                    f"items.csv line {line_number} has invalid item_id"
                ) from exc
            if item_id in seen:
                raise ContentFeatureError(f"items.csv has duplicate item_id {item_id}")
            seen.add(item_id)
            item_ids.append(item_id)
            titles.append(str(row.get("title") or "").strip())
    if not item_ids:
        raise ContentFeatureError("items.csv is empty")
    order = np.argsort(np.asarray(item_ids, dtype=np.int64), kind="stable")
    return [item_ids[int(index)] for index in order], [titles[int(index)] for index in order]


def build_content_features(
    *,
    items_path: Path,
    train_rows: Sequence[tuple[int, int, int]],
    user_ids: np.ndarray,
    feature_cutoff_ms: int,
    online_feedback: Sequence[tuple[int, int, int, str, float]] = (),
    max_features: int = 8192,
    ngram_range: tuple[int, int] = (2, 4),
    min_df: int = 2,
) -> ContentFeatureBundle:
    """Build train-fit title features and cutoff-safe user content profiles.

    The vocabulary is fitted only on titles attached to items observed in rows at or
    before the feature cutoff. All catalog titles are then transformed with that fixed
    vocabulary so future/cold items cannot influence feature fitting.
    """
    if max_features < 128:
        raise ContentFeatureError("max_features must be at least 128")
    if ngram_range[0] < 1 or ngram_range[0] > ngram_range[1]:
        raise ContentFeatureError("invalid ngram_range")
    catalog_ids, titles = _read_titles(Path(items_path))
    catalog_index = {item_id: index for index, item_id in enumerate(catalog_ids)}
    allowed_rows = [row for row in train_rows if int(row[2]) <= feature_cutoff_ms]
    train_item_ids = sorted({int(row[1]) for row in allowed_rows if int(row[1]) in catalog_index})
    fit_titles = [titles[catalog_index[item_id]] for item_id in train_item_ids]
    fit_titles = [title for title in fit_titles if title]
    if not fit_titles:
        raise ContentFeatureError("no non-empty train-visible titles are available")

    vectorizer = TfidfVectorizer(
        analyzer="char",
        lowercase=True,
        max_features=max_features,
        min_df=min_df,
        ngram_range=ngram_range,
        norm="l2",
        sublinear_tf=True,
        dtype=np.float32,
    )
    vectorizer.fit(fit_titles)
    item_vectors = vectorizer.transform(titles).tocsr().astype(np.float32)
    item_vectors.sort_indices()

    normalized_user_ids = np.asarray(user_ids, dtype=np.int64)
    user_index = {int(value): index for index, value in enumerate(normalized_user_ids)}
    pair_weights: dict[tuple[int, int], float] = {}
    for user_id, item_id, timestamp_ms in allowed_rows:
        if int(user_id) in user_index and int(item_id) in catalog_index:
            pair_weights[(int(user_id), int(item_id))] = 1.0
    for user_id, item_id, timestamp_ms, event_type, weight in online_feedback:
        if timestamp_ms > feature_cutoff_ms or weight <= 0:
            continue
        if int(user_id) not in user_index or int(item_id) not in catalog_index:
            continue
        key = (int(user_id), int(item_id))
        pair_weights[key] = min(4.0, pair_weights.get(key, 0.0) + float(weight))

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for (user_id, item_id), weight in sorted(pair_weights.items()):
        rows.append(user_index[user_id])
        columns.append(catalog_index[item_id])
        values.append(weight)
    history = csr_matrix(
        (
            np.asarray(values, dtype=np.float32),
            (np.asarray(rows, dtype=np.int64), np.asarray(columns, dtype=np.int64)),
        ),
        shape=(len(normalized_user_ids), len(catalog_ids)),
        dtype=np.float32,
    )
    user_vectors = normalize(history @ item_vectors, norm="l2", axis=1, copy=False).tocsr()
    user_vectors.sort_indices()
    vocabulary = {token: int(index) for token, index in vectorizer.vocabulary_.items()}
    vectorizer_state = {
        "schema_version": CONTENT_SCHEMA_VERSION,
        "analyzer": "char",
        "lowercase": True,
        "max_features": max_features,
        "min_df": min_df,
        "ngram_range": list(ngram_range),
        "norm": "l2",
        "sublinear_tf": True,
        "vocabulary": dict(sorted(vocabulary.items())),
    }
    metadata = {
        "schema_version": CONTENT_SCHEMA_VERSION,
        "feature_name": "title_char_ngram_tfidf",
        "feature_cutoff_ms": int(feature_cutoff_ms),
        "fit_scope": "titles for items observed in cutoff-safe training interactions",
        "transform_scope": "all catalog titles with frozen train-only vocabulary",
        "source_file": Path(items_path).name,
        "source_sha256": sha256_file(Path(items_path)),
        "catalog_items": len(catalog_ids),
        "train_visible_items": len(train_item_ids),
        "vocabulary_size": len(vocabulary),
        "nonzero_item_vectors": int(np.count_nonzero(np.diff(item_vectors.indptr))),
        "nonzero_user_vectors": int(np.count_nonzero(np.diff(user_vectors.indptr))),
        "behavior_weights": CONTENT_BEHAVIOR_WEIGHTS,
        "per_user_item_positive_weight_cap": 4.0,
        "future_train_rows_excluded": len(train_rows) - len(allowed_rows),
    }
    return ContentFeatureBundle(
        item_ids=np.asarray(catalog_ids, dtype=np.int64),
        user_ids=normalized_user_ids,
        item_vectors=item_vectors,
        user_vectors=user_vectors,
        idf=np.asarray(vectorizer.idf_, dtype=np.float32),
        vectorizer_state=vectorizer_state,
        metadata=metadata,
    )


def sparse_rows_are_finite(matrix: spmatrix) -> bool:
    return bool(np.isfinite(matrix.data).all())
