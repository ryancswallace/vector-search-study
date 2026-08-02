"""Public types and preparation helpers for exact vector search."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

from vector_search_study._validation import FloatMatrix, validate_vector_matrix
from vector_search_study.exceptions import InvalidVectorDataError

FloatDType: TypeAlias = np.dtype[np.floating[Any]]


class SearchObjective(StrEnum):
    """Exact-search score convention used throughout the study."""

    SQUARED_L2 = "squared_l2"
    INNER_PRODUCT = "inner_product"
    NORMALIZED_COSINE = "normalized_cosine"

    @property
    def requires_normalization(self) -> bool:
        """Return whether corpus and query rows must have unit L2 norm."""
        return self is SearchObjective.NORMALIZED_COSINE


def resolve_search_objective(value: SearchObjective | str) -> SearchObjective:
    """Return a validated search objective.

    Args:
        value: An objective enum value or its string representation.

    Returns:
        The resolved objective.

    Raises:
        InvalidVectorDataError: If ``value`` is not a supported objective.
    """
    try:
        return SearchObjective(value)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(objective.value for objective in SearchObjective)
        raise InvalidVectorDataError(f"objective must be one of: {choices}") from exc


@dataclass(frozen=True, slots=True)
class PreparedQueries:
    """Validated, immutable queries ready for repeated timed searches.

    Constructing this object validates normalization and copies the query
    matrix. Benchmarks can therefore prepare it outside the timed operation.

    Args:
        values: C-contiguous float32 or float64 query matrix.
        objective: Score convention for which the queries were prepared.
    """

    values: FloatMatrix
    objective: SearchObjective = SearchObjective.NORMALIZED_COSINE
    backend_name: str | None = None
    backend_payload: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate, own, and freeze the query matrix."""
        objective = resolve_search_objective(self.objective)
        validated = validate_vector_matrix(
            self.values,
            name="queries",
            require_normalized=objective.requires_normalization,
        )
        owned = np.array(validated, dtype=validated.dtype, order="C", copy=True)
        owned.flags.writeable = False
        object.__setattr__(self, "values", owned)
        object.__setattr__(self, "objective", objective)
        if self.backend_name is not None and not self.backend_name:
            raise InvalidVectorDataError("backend_name must be non-empty when provided")

    @property
    def query_count(self) -> int:
        """Return the number of queries in the batch."""
        return self.values.shape[0]

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self.values.shape[1]

    @property
    def dtype(self) -> FloatDType:
        """Return the query scalar dtype."""
        return self.values.dtype


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Ordered exact top-k indices and scores for a query batch.

    Rows are ordered by decreasing score, with smaller corpus indices winning
    exact score ties. Scores are float64 and always use a higher-is-better
    convention: negative squared distance, inner product, or normalized cosine.

    Args:
        indices: Corpus indices with shape ``(query_count, k)``.
        scores: Objective scores with the same shape.
    """

    indices: NDArray[np.int64]
    scores: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Enforce the public result representation and ordering contract."""
        if not isinstance(self.indices, np.ndarray) or self.indices.dtype != np.dtype(np.int64):
            raise InvalidVectorDataError("result indices must be an int64 NumPy array")
        if not isinstance(self.scores, np.ndarray) or self.scores.dtype != np.dtype(np.float64):
            raise InvalidVectorDataError("result scores must be a float64 NumPy array")
        if self.indices.ndim != 2 or self.scores.ndim != 2 or self.indices.shape != self.scores.shape:
            raise InvalidVectorDataError("result indices and scores must have the same two-dimensional shape")
        if self.indices.shape[0] == 0 or self.indices.shape[1] == 0:
            raise InvalidVectorDataError("result arrays must not have an empty axis")
        if not self.indices.flags.c_contiguous or not self.scores.flags.c_contiguous:
            raise InvalidVectorDataError("result arrays must be C-contiguous")
        if bool(np.any(self.indices < 0)):
            raise InvalidVectorDataError("result indices must be non-negative")
        if not bool(np.isfinite(self.scores).all()):
            raise InvalidVectorDataError("result scores must be finite")

        if self.scores.shape[1] > 1:
            left_scores = self.scores[:, :-1]
            right_scores = self.scores[:, 1:]
            if bool(np.any(left_scores < right_scores)):
                raise InvalidVectorDataError("result scores must be ordered from greatest to least")
            tied = left_scores == right_scores
            if bool(np.any(tied & (self.indices[:, :-1] > self.indices[:, 1:]))):
                raise InvalidVectorDataError("result index ties must be ordered from least to greatest")

        self.indices.flags.writeable = False
        self.scores.flags.writeable = False


class ExactSearcher(Protocol):
    """Common synchronous interface implemented by every exact searcher."""

    @property
    def size(self) -> int:
        """Return the number of indexed corpus vectors."""
        ...

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...

    @property
    def dtype(self) -> FloatDType:
        """Return the indexed scalar dtype."""
        ...

    @property
    def objective(self) -> SearchObjective:
        """Return the search objective."""
        ...

    def prepare_queries(self, queries: FloatMatrix) -> PreparedQueries:
        """Prepare queries outside the timed search operation."""
        ...

    def search(self, queries: FloatMatrix, k: int) -> SearchResult:
        """Validate and search a raw query matrix."""
        ...

    def search_prepared(self, queries: PreparedQueries, k: int) -> SearchResult:
        """Search queries prepared outside the timed operation."""
        ...


def prepare_queries(
    queries: FloatMatrix,
    *,
    objective: SearchObjective | str = SearchObjective.NORMALIZED_COSINE,
) -> PreparedQueries:
    """Validate and copy queries for repeated search.

    Args:
        queries: C-contiguous float32 or float64 matrix. Rows must be
            normalized for normalized cosine.
        objective: Score convention for the prepared queries.

    Returns:
        An immutable prepared query batch.
    """
    return PreparedQueries(queries, objective=resolve_search_objective(objective))


def normalize_rows(values: FloatMatrix) -> FloatMatrix:
    """Return a C-contiguous copy with every row L2-normalized.

    Args:
        values: A finite, non-empty float32 or float64 matrix.

    Returns:
        A normalized matrix with the input dtype.

    Raises:
        InvalidVectorDataError: If the input violates the matrix contract or
            contains a zero row.
    """
    matrix = validate_vector_matrix(values, name="values", require_normalized=False)
    scales = np.max(np.abs(matrix), axis=1, keepdims=True)
    if bool(np.any(scales == 0.0)):
        raise InvalidVectorDataError("values must not contain a zero row")
    scaled = matrix / scales
    norms = np.sqrt(np.sum(scaled * scaled, axis=1, keepdims=True))
    return np.asarray(scaled / norms, dtype=matrix.dtype, order="C")
