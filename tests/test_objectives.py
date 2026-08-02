"""Cross-implementation tests for all exact-search objectives."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from vector_search_study import (
    InvalidVectorDataError,
    NumpyArgpartitionSearcher,
    NumpyBlockedSearcher,
    NumpySortSearcher,
    PreparedQueries,
    PythonHeapSearcher,
    PythonSortSearcher,
    SearchObjective,
    make_gaussian_dataset,
    normalize_rows,
    prepare_queries,
    reference_search,
)
from vector_search_study.api import ExactSearcher

NATIVE_FACTORIES: tuple[type | Callable[..., ExactSearcher], ...] = (
    PythonSortSearcher,
    PythonHeapSearcher,
    NumpySortSearcher,
    NumpyArgpartitionSearcher,
    lambda corpus, *, objective: NumpyBlockedSearcher(corpus, objective=objective, block_size=2),
)


@pytest.mark.parametrize("objective", list(SearchObjective))
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_native_implementations_match_reference_for_every_objective(
    objective: SearchObjective,
    dtype: type[np.float32] | type[np.float64],
) -> None:
    """Every native implementation preserves identity, order, and scores."""
    corpus = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=dtype)
    queries = np.asarray([[2.0, 0.0], [0.0, 0.5]], dtype=dtype)
    if objective is SearchObjective.NORMALIZED_COSINE:
        corpus = normalize_rows(corpus)
        queries = normalize_rows(queries)
    expected = reference_search(corpus, queries, 3, objective=objective)

    for factory in NATIVE_FACTORIES:
        result = factory(corpus, objective=objective).search(queries, 3)
        np.testing.assert_array_equal(result.indices, expected.indices)
        tolerance = 3e-5 if dtype is np.float32 else 3e-12
        np.testing.assert_allclose(result.scores, expected.scores, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("objective", [SearchObjective.SQUARED_L2, SearchObjective.INNER_PRODUCT])
def test_random_unnormalized_objectives_match_reference(objective: SearchObjective) -> None:
    """Unnormalized Gaussian embeddings exercise non-cosine ranking paths."""
    dataset = make_gaussian_dataset(19, 7, 3, objective=objective, seed=17)
    expected = reference_search(dataset.corpus, dataset.queries, 5, objective=objective)

    for factory in NATIVE_FACTORIES:
        result = factory(dataset.corpus, objective=objective).search(dataset.queries, 5)
        np.testing.assert_array_equal(result.indices, expected.indices)
        np.testing.assert_allclose(result.scores, expected.scores, rtol=3e-5, atol=3e-5)


def test_objective_specific_validation_and_preparation() -> None:
    """Only normalized cosine requires unit rows and prepared objectives match indexes."""
    values = np.asarray([[2.0, 0.0]], dtype=np.float32)
    l2 = NumpySortSearcher(values, objective=SearchObjective.SQUARED_L2)
    prepared = prepare_queries(values, objective=SearchObjective.SQUARED_L2)

    assert l2.objective is SearchObjective.SQUARED_L2
    assert prepared.objective is SearchObjective.SQUARED_L2
    _ = l2.search_prepared(prepared, 1)
    with pytest.raises(InvalidVectorDataError, match="query objective"):
        _ = l2.search_prepared(PreparedQueries(normalize_rows(values)), 1)
    with pytest.raises(InvalidVectorDataError, match="objective must be one of"):
        _ = prepare_queries(values, objective="not-an-objective")
    with pytest.raises(InvalidVectorDataError, match="L2 norm one"):
        _ = NumpySortSearcher(values, objective=SearchObjective.NORMALIZED_COSINE)
