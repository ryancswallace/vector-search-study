"""Shared exact-searcher lifecycle behavior."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from vector_search_study._validation import FloatMatrix, validate_search_k, validate_vector_matrix
from vector_search_study.api import (
    FloatDType,
    PreparedQueries,
    SearchObjective,
    SearchResult,
    resolve_search_objective,
)
from vector_search_study.exceptions import InvalidVectorDataError


class BaseExactSearcher(ABC):
    """Own a validated corpus and implement the common search entry points."""

    def __init__(
        self,
        corpus: FloatMatrix,
        *,
        objective: SearchObjective | str = SearchObjective.NORMALIZED_COSINE,
    ) -> None:
        """Validate, copy, and freeze a corpus for one objective."""
        self._objective = resolve_search_objective(objective)
        validated = validate_vector_matrix(
            corpus,
            name="corpus",
            require_normalized=self._objective.requires_normalization,
        )
        owned = np.array(validated, dtype=validated.dtype, order="C", copy=True)
        owned.flags.writeable = False
        self._corpus: FloatMatrix = owned

    @property
    def size(self) -> int:
        """Return the number of indexed corpus vectors."""
        return self._corpus.shape[0]

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._corpus.shape[1]

    @property
    def dtype(self) -> FloatDType:
        """Return the indexed scalar dtype."""
        return self._corpus.dtype

    @property
    def objective(self) -> SearchObjective:
        """Return the search objective."""
        return self._objective

    def prepare_queries(self, queries: FloatMatrix) -> PreparedQueries:
        """Validate and copy queries for repeated searches."""
        return PreparedQueries(queries, objective=self._objective)

    def search(self, queries: FloatMatrix, k: int) -> SearchResult:
        """Validate, prepare, and search a raw query matrix."""
        return self.search_prepared(self.prepare_queries(queries), k)

    def search_prepared(self, queries: PreparedQueries, k: int) -> SearchResult:
        """Search an immutable query batch with preparation excluded."""
        if not isinstance(queries, PreparedQueries):
            raise InvalidVectorDataError("queries must be a PreparedQueries instance")
        if queries.objective is not self._objective:
            raise InvalidVectorDataError(
                f"query objective {queries.objective.value} does not match search objective {self._objective.value}"
            )
        if queries.dimension != self.dimension:
            raise InvalidVectorDataError(
                f"query dimension {queries.dimension} does not match corpus dimension {self.dimension}"
            )
        if queries.dtype != self.dtype:
            raise InvalidVectorDataError(f"query dtype {queries.dtype} does not match corpus dtype {self.dtype}")
        resolved_k = validate_search_k(k, corpus_size=self.size)
        return self._search_prepared(queries, resolved_k)

    @abstractmethod
    def _search_prepared(self, queries: PreparedQueries, k: int) -> SearchResult:
        """Execute one validated exact-search operation."""
