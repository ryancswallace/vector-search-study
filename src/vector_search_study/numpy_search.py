"""Vectorized NumPy exact search implementations."""

from __future__ import annotations

import numpy as np

from vector_search_study._base import BaseExactSearcher
from vector_search_study._selection import full_sort_top_k, merge_top_k, partition_top_k
from vector_search_study._validation import FloatMatrix, validate_positive_int
from vector_search_study.api import PreparedQueries, SearchObjective, SearchResult


def score_matrix(
    queries: FloatMatrix,
    corpus: FloatMatrix,
    objective: SearchObjective,
) -> FloatMatrix:
    """Return a higher-is-better objective score matrix."""
    products = queries @ corpus.T
    if objective is not SearchObjective.SQUARED_L2:
        return products
    query_norms = np.sum(queries * queries, axis=1, keepdims=True)
    corpus_norms = np.sum(corpus * corpus, axis=1, keepdims=True).T
    distances = np.maximum(query_norms + corpus_norms - 2.0 * products, 0.0)
    return np.asarray(-distances, dtype=queries.dtype, order="C")


class NumpySortSearcher(BaseExactSearcher):
    """Score by matrix multiplication and fully sort every score row."""

    def _search_prepared(self, queries: PreparedQueries, k: int) -> SearchResult:
        """Materialize the full score matrix and fully sort it."""
        scores_matrix = score_matrix(queries.values, self._corpus, self.objective)
        indices, scores = full_sort_top_k(scores_matrix, k)
        return SearchResult(indices=indices, scores=scores)


class NumpyArgpartitionSearcher(BaseExactSearcher):
    """Score by matrix multiplication and partially select exact top-k rows."""

    def _search_prepared(self, queries: PreparedQueries, k: int) -> SearchResult:
        """Materialize scores, partition, and repair deterministic ties."""
        scores_matrix = score_matrix(queries.values, self._corpus, self.objective)
        indices, scores = partition_top_k(scores_matrix, k)
        return SearchResult(indices=indices, scores=scores)


class NumpyBlockedSearcher(BaseExactSearcher):
    """Limit score-matrix memory by searching fixed-size corpus blocks."""

    def __init__(
        self,
        corpus: FloatMatrix,
        *,
        block_size: int = 16_384,
        objective: SearchObjective | str = SearchObjective.NORMALIZED_COSINE,
    ) -> None:
        """Build an index with a fixed number of corpus rows per block.

        Args:
            corpus: Pre-normalized corpus vectors.
            block_size: Maximum number of corpus rows scored per matrix
                multiplication.
            objective: Exact-search score convention.
        """
        super().__init__(corpus, objective=objective)
        self._block_size = validate_positive_int(block_size, name="block_size")

    @property
    def block_size(self) -> int:
        """Return the configured corpus rows per score block."""
        return self._block_size

    def _search_prepared(self, queries: PreparedQueries, k: int) -> SearchResult:
        """Select block-local candidates and merge them into global top-k."""
        best_indices = np.empty((queries.query_count, 0), dtype=np.int64)
        best_scores = np.empty((queries.query_count, 0), dtype=np.float64)
        for start in range(0, self.size, self._block_size):
            stop = min(start + self._block_size, self.size)
            block_scores = score_matrix(queries.values, self._corpus[start:stop], self.objective)
            block_k = min(k, stop - start)
            block_indices, selected_scores = partition_top_k(block_scores, block_k, index_offset=start)
            best_indices, best_scores = merge_top_k(
                best_indices,
                best_scores,
                block_indices,
                selected_scores,
                k,
            )
        return SearchResult(indices=best_indices, scores=best_scores)
