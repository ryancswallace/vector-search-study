"""Exact Faiss flat-index adapters."""

from __future__ import annotations

from typing import Any

import numpy as np

from vector_search_study._backend_utils import canonical_result, import_optional
from vector_search_study._base import BaseExactSearcher
from vector_search_study._validation import FloatMatrix
from vector_search_study.api import PreparedQueries, SearchObjective, SearchResult, resolve_search_objective
from vector_search_study.exceptions import InvalidVectorDataError, UnsupportedObjectiveError


class FaissFlatL2Searcher(BaseExactSearcher):
    """Exact Faiss IndexFlatL2 search with negative squared-distance scores."""

    def __init__(self, corpus: FloatMatrix) -> None:
        """Build a float32 flat L2 index outside search timing."""
        _require_float32(corpus)
        super().__init__(corpus, objective=SearchObjective.SQUARED_L2)
        faiss = import_optional("faiss", extra="benchmark-backends")
        faiss.omp_set_num_threads(1)
        self._index: Any = faiss.IndexFlatL2(self.dimension)
        self._index.add(self._corpus)

    def _search_prepared(self, queries: PreparedQueries, k: int) -> SearchResult:
        """Search the flat L2 index and invert its distances."""
        distances, indices = self._index.search(queries.values, k)
        return canonical_result(indices, -distances)


class FaissFlatIPSearcher(BaseExactSearcher):
    """Exact Faiss IndexFlatIP search for inner product or cosine."""

    def __init__(
        self,
        corpus: FloatMatrix,
        *,
        objective: SearchObjective | str = SearchObjective.INNER_PRODUCT,
    ) -> None:
        """Build a float32 flat inner-product index outside search timing."""
        resolved = resolve_search_objective(objective)
        if resolved not in {SearchObjective.INNER_PRODUCT, SearchObjective.NORMALIZED_COSINE}:
            raise UnsupportedObjectiveError("Faiss IndexFlatIP supports only inner product and normalized cosine")
        _require_float32(corpus)
        super().__init__(corpus, objective=resolved)
        faiss = import_optional("faiss", extra="benchmark-backends")
        faiss.omp_set_num_threads(1)
        self._index: Any = faiss.IndexFlatIP(self.dimension)
        self._index.add(self._corpus)

    def _search_prepared(self, queries: PreparedQueries, k: int) -> SearchResult:
        """Search the flat inner-product index."""
        scores, indices = self._index.search(queries.values, k)
        return canonical_result(indices, scores)


def _require_float32(values: FloatMatrix) -> None:
    """Enforce Faiss flat indexes' native scalar representation."""
    if values.dtype != np.dtype(np.float32):
        raise InvalidVectorDataError("Faiss flat indexes require float32 vectors")
