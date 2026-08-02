"""Tests for vector, result, and search-parameter validation."""

from __future__ import annotations

import numpy as np
import pytest

from vector_search_study import (
    InvalidSearchParameterError,
    InvalidVectorDataError,
    NumpySortSearcher,
    SearchResult,
    normalize_rows,
)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ([[1.0, 0.0]], "NumPy array"),
        (np.asarray([1.0, 0.0], dtype=np.float32), "shape"),
        (np.empty((0, 2), dtype=np.float32), "empty axis"),
        (np.empty((2, 0), dtype=np.float32), "empty axis"),
        (np.asarray([[1, 0]], dtype=np.int64), "dtype"),
        (np.asarray([[np.nan, 0.0]], dtype=np.float32), "finite"),
        (np.asarray([[np.inf, 0.0]], dtype=np.float32), "finite"),
        (np.asarray([[0.5, 0.0]], dtype=np.float32), "L2 norm one"),
    ],
)
def test_index_rejects_invalid_corpora(value: object, message: str) -> None:
    """Corpus validation rejects every unsupported matrix category."""
    with pytest.raises(InvalidVectorDataError, match=message):
        _ = NumpySortSearcher(value)  # type: ignore[arg-type]


def test_index_rejects_noncontiguous_corpus() -> None:
    """The input-layout contract is explicit rather than silently copying views."""
    corpus = np.eye(3, dtype=np.float32)[:, ::-1]

    with pytest.raises(InvalidVectorDataError, match="C-contiguous"):
        _ = NumpySortSearcher(corpus)


@pytest.mark.parametrize(
    ("queries", "message"),
    [
        (np.asarray([1.0, 0.0], dtype=np.float32), "shape"),
        (np.empty((0, 2), dtype=np.float32), "empty axis"),
        (np.asarray([[1, 0]], dtype=np.int64), "dtype"),
        (np.asarray([[np.nan, 0.0]], dtype=np.float32), "finite"),
        (np.asarray([[0.25, 0.0]], dtype=np.float32), "L2 norm one"),
    ],
)
def test_search_rejects_invalid_query_matrices(queries: np.ndarray, message: str) -> None:
    """Raw search applies the same vector contract to query batches."""
    searcher = NumpySortSearcher(np.eye(2, dtype=np.float32))

    with pytest.raises(InvalidVectorDataError, match=message):
        _ = searcher.search(queries, 1)


def test_search_rejects_noncontiguous_queries() -> None:
    """Query preparation rejects noncontiguous views."""
    searcher = NumpySortSearcher(np.eye(3, dtype=np.float32))
    queries = np.eye(3, dtype=np.float32)[:, ::-1]

    with pytest.raises(InvalidVectorDataError, match="C-contiguous"):
        _ = searcher.search(queries, 1)


@pytest.mark.parametrize("k", [True, 1.0, "1", None])
def test_search_rejects_noninteger_k(k: object) -> None:
    """Boolean and coercible values do not masquerade as integer k."""
    searcher = NumpySortSearcher(np.eye(2, dtype=np.float32))

    with pytest.raises(InvalidSearchParameterError, match="integer"):
        _ = searcher.search(np.asarray([[1.0, 0.0]], dtype=np.float32), k)  # type: ignore[arg-type]


@pytest.mark.parametrize("k", [0, -1, 3])
def test_search_rejects_out_of_range_k(k: int) -> None:
    """Top-k must select at least one and no more than the corpus size."""
    searcher = NumpySortSearcher(np.eye(2, dtype=np.float32))

    with pytest.raises(InvalidSearchParameterError, match="1 <= k <= 2"):
        _ = searcher.search(np.asarray([[1.0, 0.0]], dtype=np.float32), k)


def test_normalize_rows_is_stable_for_large_values(float_dtype: type[np.float32] | type[np.float64]) -> None:
    """Scale-first normalization avoids overflowing intermediate squares."""
    maximum = np.finfo(float_dtype).max / 4
    values = np.asarray([[maximum, maximum], [3.0, 4.0]], dtype=float_dtype)

    normalized = normalize_rows(values)

    np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), 1.0, rtol=1e-5, atol=1e-6)
    assert normalized.dtype == values.dtype
    assert normalized.flags.c_contiguous
    assert not np.shares_memory(normalized, values)


def test_normalize_rows_rejects_zero_rows() -> None:
    """A zero vector has no defined cosine normalization."""
    with pytest.raises(InvalidVectorDataError, match="zero row"):
        _ = normalize_rows(np.asarray([[0.0, 0.0]], dtype=np.float32))


def _valid_result_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Return fresh valid arrays because result construction freezes them."""
    return np.asarray([[0, 1]], dtype=np.int64), np.asarray([[1.0, 0.5]], dtype=np.float64)


@pytest.mark.parametrize(
    ("indices", "scores", "message"),
    [
        ([[0]], np.asarray([[1.0]], dtype=np.float64), "indices"),
        (np.asarray([[0]], dtype=np.int32), np.asarray([[1.0]], dtype=np.float64), "indices"),
        (np.asarray([[0]], dtype=np.int64), [[1.0]], "scores"),
        (np.asarray([[0]], dtype=np.int64), np.asarray([[1.0]], dtype=np.float32), "scores"),
        (np.asarray([0], dtype=np.int64), np.asarray([[1.0]], dtype=np.float64), "two-dimensional"),
        (np.asarray([[0, 1]], dtype=np.int64), np.asarray([[1.0]], dtype=np.float64), "two-dimensional"),
        (np.empty((0, 1), dtype=np.int64), np.empty((0, 1), dtype=np.float64), "empty axis"),
        (np.asarray([[-1]], dtype=np.int64), np.asarray([[1.0]], dtype=np.float64), "non-negative"),
        (np.asarray([[0]], dtype=np.int64), np.asarray([[np.nan]], dtype=np.float64), "finite"),
        (np.asarray([[0, 1]], dtype=np.int64), np.asarray([[0.5, 1.0]], dtype=np.float64), "greatest"),
        (np.asarray([[1, 0]], dtype=np.int64), np.asarray([[1.0, 1.0]], dtype=np.float64), "index ties"),
    ],
)
def test_search_result_rejects_invalid_representation(indices: object, scores: object, message: str) -> None:
    """Result objects cannot violate the shape, type, or ordering contract."""
    with pytest.raises(InvalidVectorDataError, match=message):
        _ = SearchResult(indices=indices, scores=scores)  # type: ignore[arg-type]


def test_search_result_rejects_noncontiguous_arrays() -> None:
    """Result arrays use a predictable dense layout."""
    indices = np.asarray([[0, 2], [1, 3]], dtype=np.int64).T
    scores = np.asarray([[1.0, 0.5], [0.75, 0.25]], dtype=np.float64).T

    with pytest.raises(InvalidVectorDataError, match="C-contiguous"):
        _ = SearchResult(indices=indices, scores=scores)


def test_valid_search_result_is_frozen() -> None:
    """Successful result construction makes both arrays immutable."""
    indices, scores = _valid_result_arrays()

    result = SearchResult(indices=indices, scores=scores)

    assert not result.indices.flags.writeable
    assert not result.scores.flags.writeable
