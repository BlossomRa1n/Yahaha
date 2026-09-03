from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.sparse import csr_matrix


ITEM_CF_SCHEMA_VERSION = 1


class ItemCFError(ValueError):
    pass


@dataclass(frozen=True)
class ItemCFBundle:
    neighbors: csr_matrix
    user_history: csr_matrix
    metadata: dict[str, Any]


def build_item_cf(
    *,
    train_rows: Sequence[tuple[int, int, int]],
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    feature_cutoff_ms: int,
    top_n: int = 50,
    min_support: int = 2,
) -> ItemCFBundle:
    """Build bounded train-only cosine item neighbors and user history."""
    if top_n < 1 or top_n > 500:
        raise ItemCFError("top_n must be between 1 and 500")
    if min_support < 1:
        raise ItemCFError("min_support must be positive")
    user_index = {int(value): index for index, value in enumerate(user_ids)}
    item_index = {int(value): index for index, value in enumerate(item_ids)}
    pairs = {
        (user_index[int(user_id)], item_index[int(item_id)])
        for user_id, item_id, timestamp_ms in train_rows
        if int(timestamp_ms) <= feature_cutoff_ms
        and int(user_id) in user_index
        and int(item_id) in item_index
    }
    if not pairs:
        raise ItemCFError("no cutoff-safe user-item pairs are available")
    ordered_pairs = sorted(pairs)
    history = csr_matrix(
        (
            np.ones(len(ordered_pairs), dtype=np.float32),
            (
                np.asarray([pair[0] for pair in ordered_pairs], dtype=np.int64),
                np.asarray([pair[1] for pair in ordered_pairs], dtype=np.int64),
            ),
        ),
        shape=(len(user_ids), len(item_ids)),
        dtype=np.float32,
    )
    cooccurrence = (history.T @ history).tocsr().astype(np.float32)
    supports = np.asarray(cooccurrence.diagonal(), dtype=np.float64)
    neighbor_rows: list[int] = []
    neighbor_columns: list[int] = []
    neighbor_scores: list[float] = []
    retained_supports: list[float] = []
    for item_row in range(cooccurrence.shape[0]):
        start = cooccurrence.indptr[item_row]
        end = cooccurrence.indptr[item_row + 1]
        values: list[tuple[float, int, float]] = []
        for column, support in zip(
            cooccurrence.indices[start:end], cooccurrence.data[start:end]
        ):
            column = int(column)
            support = float(support)
            if column == item_row or support < min_support:
                continue
            denominator = float(np.sqrt(supports[item_row] * supports[column]))
            if denominator <= 0:
                continue
            score = support / denominator
            if np.isfinite(score) and score > 0:
                values.append((score, column, support))
        values.sort(key=lambda row: (-row[0], int(item_ids[row[1]])))
        for score, column, support in values[:top_n]:
            neighbor_rows.append(item_row)
            neighbor_columns.append(column)
            neighbor_scores.append(score)
            retained_supports.append(support)
    neighbors = csr_matrix(
        (
            np.asarray(neighbor_scores, dtype=np.float32),
            (
                np.asarray(neighbor_rows, dtype=np.int64),
                np.asarray(neighbor_columns, dtype=np.int64),
            ),
        ),
        shape=(len(item_ids), len(item_ids)),
        dtype=np.float32,
    )
    neighbors.sort_indices()
    metadata = {
        "schema_version": ITEM_CF_SCHEMA_VERSION,
        "algorithm": "train_only_sparse_item_cosine",
        "feature_cutoff_ms": int(feature_cutoff_ms),
        "feature_rule": "event_timestamp_ms <= feature_cutoff_ms",
        "min_support": min_support,
        "top_n": top_n,
        "user_item_pairs": len(ordered_pairs),
        "items": len(item_ids),
        "users": len(user_ids),
        "neighbor_edges": int(neighbors.nnz),
        "items_with_neighbors": int(np.count_nonzero(np.diff(neighbors.indptr))),
        "minimum_retained_support": min(retained_supports) if retained_supports else None,
        "maximum_retained_support": max(retained_supports) if retained_supports else None,
        "self_similarity_retained": False,
    }
    return ItemCFBundle(neighbors=neighbors, user_history=history, metadata=metadata)
