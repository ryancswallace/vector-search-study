"""Exact SciPy cKDTree search adapter."""

from __future__ import annotations

from typing import Any, ClassVar

from vector_search_study._backend_utils import canonical_result, import_optional
from vector_search_study._base import BaseExactSearcher
from vector_search_study._validation import FloatMatrix, validate_positive_int
from vector_search_study.api import PreparedQueries, SearchObjective, SearchResult
from vector_search_study.sklearn_search import _euclidean_scores, _require_supported


class ScipyCKDTreeSearcher(BaseExactSearcher):
    """Exact one-worker SciPy cKDTree search for L2-derived objectives."""

    _supported: ClassVar[frozenset[SearchObjective]] = frozenset(
        {SearchObjective.SQUARED_L2, SearchObjective.NORMALIZED_COSINE}
    )

    def __init__(
        self,
        corpus: FloatMatrix,
        *,
        objective: SearchObjective | str,
        leaf_size: int = 16,
    ) -> None:
        """Build a balanced compact cKDTree outside search timing."""
        resolved = _require_supported(objective, self._supported, backend="SciPy cKDTree")
        super().__init__(corpus, objective=resolved)
        spatial = import_optional("scipy.spatial", extra="benchmark-backends")
        self._leaf_size = validate_positive_int(leaf_size, name="leaf_size")
        self._index: Any = spatial.cKDTree(
            self._corpus,
            leafsize=self._leaf_size,
            compact_nodes=True,
            copy_data=True,
            balanced_tree=True,
        )

    @property
    def leaf_size(self) -> int:
        """Return the configured tree leaf size."""
        return self._leaf_size

    def _search_prepared(self, queries: PreparedQueries, k: int) -> SearchResult:
        """Query the exact Euclidean tree with one worker."""
        distances, indices = self._index.query(queries.values, k=k, eps=0.0, p=2.0, workers=1)
        return canonical_result(indices, _euclidean_scores(distances, self.objective))
