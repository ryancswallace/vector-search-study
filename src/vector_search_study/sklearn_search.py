"""Exact scikit-learn nearest-neighbor adapters."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from vector_search_study._backend_utils import canonical_result, import_optional
from vector_search_study._base import BaseExactSearcher
from vector_search_study._validation import FloatMatrix, validate_positive_int
from vector_search_study.api import PreparedQueries, SearchObjective, SearchResult, resolve_search_objective
from vector_search_study.exceptions import UnsupportedObjectiveError


class SklearnBruteSearcher(BaseExactSearcher):
    """Exact scikit-learn brute-force L2 or cosine search."""

    _supported: ClassVar[frozenset[SearchObjective]] = frozenset(
        {SearchObjective.SQUARED_L2, SearchObjective.NORMALIZED_COSINE}
    )

    def __init__(self, corpus: FloatMatrix, *, objective: SearchObjective | str) -> None:
        """Build a one-thread brute-force nearest-neighbor index."""
        resolved = _require_supported(objective, self._supported, backend="scikit-learn brute")
        super().__init__(corpus, objective=resolved)
        neighbors = import_optional("sklearn.neighbors", extra="benchmark-backends")
        metric = "cosine" if resolved is SearchObjective.NORMALIZED_COSINE else "euclidean"
        self._index: Any = neighbors.NearestNeighbors(algorithm="brute", metric=metric, n_jobs=1)
        self._index.fit(self._corpus)

    def _search_prepared(self, queries: PreparedQueries, k: int) -> SearchResult:
        """Query the fitted brute-force index."""
        distances, indices = self._index.kneighbors(queries.values, n_neighbors=k, return_distance=True)
        scores = -(distances * distances) if self.objective is SearchObjective.SQUARED_L2 else 1.0 - distances
        return canonical_result(indices, scores)


class SklearnKDTreeSearcher(BaseExactSearcher):
    """Exact scikit-learn KDTree search for L2-derived objectives."""

    _supported: ClassVar[frozenset[SearchObjective]] = frozenset(
        {SearchObjective.SQUARED_L2, SearchObjective.NORMALIZED_COSINE}
    )

    def __init__(
        self,
        corpus: FloatMatrix,
        *,
        objective: SearchObjective | str,
        leaf_size: int = 40,
    ) -> None:
        """Build a Euclidean KDTree outside search timing."""
        resolved = _require_supported(objective, self._supported, backend="scikit-learn KDTree")
        super().__init__(corpus, objective=resolved)
        neighbors = import_optional("sklearn.neighbors", extra="benchmark-backends")
        self._leaf_size = validate_positive_int(leaf_size, name="leaf_size")
        self._index: Any = neighbors.KDTree(self._corpus, leaf_size=self._leaf_size, metric="euclidean")

    @property
    def leaf_size(self) -> int:
        """Return the configured tree leaf size."""
        return self._leaf_size

    def _search_prepared(self, queries: PreparedQueries, k: int) -> SearchResult:
        """Query the Euclidean tree and convert distances to objective scores."""
        distances, indices = self._index.query(queries.values, k=k, return_distance=True, dualtree=False)
        return canonical_result(indices, _euclidean_scores(distances, self.objective))


class SklearnBallTreeSearcher(BaseExactSearcher):
    """Exact scikit-learn BallTree search for L2-derived objectives."""

    _supported: ClassVar[frozenset[SearchObjective]] = SklearnKDTreeSearcher._supported

    def __init__(
        self,
        corpus: FloatMatrix,
        *,
        objective: SearchObjective | str,
        leaf_size: int = 40,
    ) -> None:
        """Build a Euclidean BallTree outside search timing."""
        resolved = _require_supported(objective, self._supported, backend="scikit-learn BallTree")
        super().__init__(corpus, objective=resolved)
        neighbors = import_optional("sklearn.neighbors", extra="benchmark-backends")
        self._leaf_size = validate_positive_int(leaf_size, name="leaf_size")
        self._index: Any = neighbors.BallTree(self._corpus, leaf_size=self._leaf_size, metric="euclidean")

    @property
    def leaf_size(self) -> int:
        """Return the configured tree leaf size."""
        return self._leaf_size

    def _search_prepared(self, queries: PreparedQueries, k: int) -> SearchResult:
        """Query the Euclidean tree and convert distances to objective scores."""
        distances, indices = self._index.query(queries.values, k=k, return_distance=True, dualtree=False)
        return canonical_result(indices, _euclidean_scores(distances, self.objective))


def _require_supported(
    objective: SearchObjective | str,
    supported: frozenset[SearchObjective],
    *,
    backend: str,
) -> SearchObjective:
    """Resolve an objective or reject a non-native backend pairing."""
    resolved = resolve_search_objective(objective)
    if resolved not in supported:
        raise UnsupportedObjectiveError(f"{backend} does not support exact {resolved.value} search")
    return resolved


def _euclidean_scores(distances: FloatMatrix, objective: SearchObjective) -> FloatMatrix:
    """Convert Euclidean distances to the public higher-is-better score."""
    squared = distances * distances
    if objective is SearchObjective.NORMALIZED_COSINE:
        return np.asarray(1.0 - squared / 2.0, dtype=distances.dtype, order="C")
    return np.asarray(-squared, dtype=distances.dtype, order="C")
