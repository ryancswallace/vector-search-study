"""Deterministic top-k selection helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def full_sort_top_k(scores: NDArray[np.floating], k: int) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Fully sort score rows with stable index tie-breaking."""
    order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    indices = np.asarray(order, dtype=np.int64, order="C")
    selected = np.asarray(np.take_along_axis(scores, order, axis=1), dtype=np.float64, order="C")
    return indices, selected


def partition_top_k(
    scores: NDArray[np.floating],
    k: int,
    *,
    index_offset: int = 0,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Partially select rows and repair boundary ties deterministically."""
    query_count, candidate_count = scores.shape
    indices = np.empty((query_count, k), dtype=np.int64)
    selected_scores = np.empty((query_count, k), dtype=np.float64)

    for query_index in range(query_count):
        row = scores[query_index]
        if k == candidate_count:
            candidates = np.arange(candidate_count, dtype=np.int64)
        else:
            partition = np.argpartition(row, candidate_count - k)[candidate_count - k :]
            threshold = np.min(row[partition])
            higher = np.flatnonzero(row > threshold)
            tied = np.flatnonzero(row == threshold)
            candidates = np.concatenate((higher, tied[: k - higher.size]))

        global_candidates = np.asarray(candidates + index_offset, dtype=np.int64)
        candidate_scores = row[candidates]
        order = np.lexsort((global_candidates, -candidate_scores))
        ordered_candidates = candidates[order]
        indices[query_index] = ordered_candidates + index_offset
        selected_scores[query_index] = row[ordered_candidates]

    return indices, selected_scores


def merge_top_k(
    left_indices: NDArray[np.int64],
    left_scores: NDArray[np.float64],
    right_indices: NDArray[np.int64],
    right_scores: NDArray[np.float64],
    k: int,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Merge two ordered candidate sets into a deterministic top-k."""
    all_indices = np.concatenate((left_indices, right_indices), axis=1)
    all_scores = np.concatenate((left_scores, right_scores), axis=1)
    query_count = all_indices.shape[0]
    retained = min(k, all_indices.shape[1])
    indices = np.empty((query_count, retained), dtype=np.int64)
    scores = np.empty((query_count, retained), dtype=np.float64)
    for query_index in range(query_count):
        order = np.lexsort((all_indices[query_index], -all_scores[query_index]))[:retained]
        indices[query_index] = all_indices[query_index, order]
        scores[query_index] = all_scores[query_index, order]
    return indices, scores
