"""Scalar pure-Python exact search implementations."""

from __future__ import annotations

import heapq

import numpy as np

from vector_search_study._base import BaseExactSearcher
from vector_search_study._validation import FloatMatrix
from vector_search_study.api import PreparedQueries, SearchResult


class PythonSortSearcher(BaseExactSearcher):
    """Exhaustively score into a Python list and fully sort it."""

    def __init__(self, corpus: FloatMatrix) -> None:
        """Build the scalar corpus representation outside search timing."""
        super().__init__(corpus)
        self._rows = tuple(tuple(float(value) for value in row) for row in self._corpus)

    def _search_prepared(self, queries: PreparedQueries, k: int) -> SearchResult:
        """Score every vector, then sort all candidates."""
        all_indices: list[list[int]] = []
        all_scores: list[list[float]] = []
        for query_array in queries.values:
            query = tuple(float(value) for value in query_array)
            ranked: list[tuple[float, int]] = []
            for index, vector in enumerate(self._rows):
                score = _scalar_dot(vector, query)
                ranked.append((score, index))
            ranked.sort(key=lambda item: (-item[0], item[1]))
            selected = ranked[:k]
            all_indices.append([index for _, index in selected])
            all_scores.append([score for score, _ in selected])
        return SearchResult(
            indices=np.asarray(all_indices, dtype=np.int64, order="C"),
            scores=np.asarray(all_scores, dtype=np.float64, order="C"),
        )


class PythonHeapSearcher(BaseExactSearcher):
    """Stream exhaustive scalar scores through a bounded size-k heap."""

    def __init__(self, corpus: FloatMatrix) -> None:
        """Build the scalar corpus representation outside search timing."""
        super().__init__(corpus)
        self._rows = tuple(tuple(float(value) for value in row) for row in self._corpus)

    def _search_prepared(self, queries: PreparedQueries, k: int) -> SearchResult:
        """Retain only the best k candidates while streaming scores."""
        all_indices: list[list[int]] = []
        all_scores: list[list[float]] = []
        for query_array in queries.values:
            query = tuple(float(value) for value in query_array)
            heap: list[tuple[float, int, int]] = []
            for index, vector in enumerate(self._rows):
                score = _scalar_dot(vector, query)
                item = (score, -index, index)
                if len(heap) < k:
                    heapq.heappush(heap, item)
                elif item[:2] > heap[0][:2]:
                    _ = heapq.heapreplace(heap, item)
            selected = sorted(heap, key=lambda item: (-item[0], item[2]))
            all_indices.append([index for _, _, index in selected])
            all_scores.append([score for score, _, _ in selected])
        return SearchResult(
            indices=np.asarray(all_indices, dtype=np.int64, order="C"),
            scores=np.asarray(all_scores, dtype=np.float64, order="C"),
        )


def _scalar_dot(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    """Return a scalar dot product using an ordinary Python accumulator."""
    result = 0.0
    for left_value, right_value in zip(left, right, strict=True):
        result += left_value * right_value
    return result
