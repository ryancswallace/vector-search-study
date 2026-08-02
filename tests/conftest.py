"""Shared correctness fixtures for exact vector search."""

from collections.abc import Callable

import numpy as np
import pytest

from vector_search_study import (
    NumpyArgpartitionSearcher,
    NumpyBlockedSearcher,
    NumpySortSearcher,
    PythonHeapSearcher,
    PythonSortSearcher,
)
from vector_search_study.api import ExactSearcher

SearcherFactory = Callable[[np.ndarray], ExactSearcher]


@pytest.fixture(params=[np.float32, np.float64], ids=["float32", "float64"])
def float_dtype(request: pytest.FixtureRequest) -> type[np.float32] | type[np.float64]:
    """Return each supported floating-point scalar type."""
    return request.param


@pytest.fixture(
    params=[
        PythonSortSearcher,
        PythonHeapSearcher,
        NumpySortSearcher,
        NumpyArgpartitionSearcher,
        lambda corpus: NumpyBlockedSearcher(corpus, block_size=2),
    ],
    ids=["python-sort", "python-heap", "numpy-sort", "numpy-partition", "numpy-blocked"],
)
def searcher_factory(request: pytest.FixtureRequest) -> SearcherFactory:
    """Return every non-compiled searcher constructor."""
    return request.param


@pytest.fixture
def tied_corpus(float_dtype: type[np.float32] | type[np.float64]) -> np.ndarray:
    """Return normalized vectors with exact ties crossing block boundaries."""
    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float_dtype,
    )


@pytest.fixture
def tied_queries(float_dtype: type[np.float32] | type[np.float64]) -> np.ndarray:
    """Return a normalized two-query batch for deterministic-order tests."""
    return np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float_dtype)
