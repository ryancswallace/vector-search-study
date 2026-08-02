"""Correctness tests shared by every exact search implementation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from vector_search_study import (
    NumpyArgpartitionSearcher,
    NumpyBlockedSearcher,
    NumpySortSearcher,
    PreparedQueries,
    PythonHeapSearcher,
    PythonSortSearcher,
    prepare_queries,
    reference_search,
)
from vector_search_study.api import ExactSearcher

SearcherFactory = Callable[[np.ndarray], ExactSearcher]


def test_every_searcher_matches_ordering_and_tie_contract(
    searcher_factory: SearcherFactory,
    tied_corpus: np.ndarray,
    tied_queries: np.ndarray,
) -> None:
    """Scores descend and exact ties prefer smaller corpus indices."""
    searcher = searcher_factory(tied_corpus)

    result = searcher.search(tied_queries, 3)

    np.testing.assert_array_equal(result.indices, [[0, 1, 2], [2, 0, 1]])
    np.testing.assert_allclose(result.scores, [[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    assert not result.indices.flags.writeable
    assert not result.scores.flags.writeable


def test_every_searcher_supports_k_equal_to_corpus_size(
    searcher_factory: SearcherFactory,
    tied_corpus: np.ndarray,
    tied_queries: np.ndarray,
) -> None:
    """A full result retains deterministic ordering for every corpus row."""
    searcher = searcher_factory(tied_corpus)

    result = searcher.search(tied_queries[:1].copy(), tied_corpus.shape[0])

    np.testing.assert_array_equal(result.indices, [[0, 1, 2, 4, 3]])
    np.testing.assert_allclose(result.scores, [[1.0, 1.0, 0.0, 0.0, -1.0]])


def test_prepared_and_raw_search_paths_agree(
    searcher_factory: SearcherFactory,
    tied_corpus: np.ndarray,
    tied_queries: np.ndarray,
) -> None:
    """Preparation changes lifecycle cost rather than search semantics."""
    searcher = searcher_factory(tied_corpus)
    prepared = prepare_queries(tied_queries)

    raw = searcher.search(tied_queries, 2)
    reused = searcher.search_prepared(prepared, 2)

    np.testing.assert_array_equal(reused.indices, raw.indices)
    np.testing.assert_allclose(reused.scores, raw.scores)
    assert prepared.query_count == 2
    assert prepared.dimension == 3
    assert prepared.dtype == tied_queries.dtype
    assert not prepared.values.flags.writeable
    assert not np.shares_memory(prepared.values, tied_queries)


def test_indexes_own_corpus_state_and_do_not_mutate_inputs(
    searcher_factory: SearcherFactory,
    tied_corpus: np.ndarray,
    tied_queries: np.ndarray,
) -> None:
    """Index construction isolates later caller mutation and search is read-only."""
    original_corpus = tied_corpus.copy()
    original_queries = tied_queries.copy()
    searcher = searcher_factory(tied_corpus)
    tied_corpus[0] = [0.0, 0.0, 1.0]

    result = searcher.search(tied_queries, 2)

    expected = reference_search(original_corpus, original_queries, 2)
    np.testing.assert_array_equal(result.indices, expected.indices)
    np.testing.assert_allclose(result.scores, expected.scores)
    np.testing.assert_array_equal(tied_queries, original_queries)


def test_searcher_metadata_describes_owned_index(
    searcher_factory: SearcherFactory,
    tied_corpus: np.ndarray,
) -> None:
    """Common index metadata is available without implementation inspection."""
    searcher = searcher_factory(tied_corpus)

    assert searcher.size == 5
    assert searcher.dimension == 3
    assert searcher.dtype == tied_corpus.dtype


@pytest.mark.property
@settings(max_examples=25, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    corpus_size=st.integers(min_value=2, max_value=14),
    dimension=st.integers(min_value=2, max_value=10),
    query_count=st.integers(min_value=1, max_value=4),
    dtype_name=st.sampled_from(["float32", "float64"]),
    requested_k=st.integers(min_value=1, max_value=14),
)
def test_randomized_implementations_match_trusted_reference(
    seed: int,
    corpus_size: int,
    dimension: int,
    query_count: int,
    dtype_name: str,
    requested_k: int,
) -> None:
    """Randomized normalized inputs retain reference identities and scores."""
    from vector_search_study import make_uniform_sphere_dataset

    dtype = np.float32 if dtype_name == "float32" else np.float64
    dataset = make_uniform_sphere_dataset(corpus_size, dimension, query_count, dtype=dtype, seed=seed)
    k = min(requested_k, corpus_size)
    expected = reference_search(dataset.corpus, dataset.queries, k)
    factories: tuple[Callable[[np.ndarray], ExactSearcher], ...] = (
        PythonSortSearcher,
        PythonHeapSearcher,
        NumpySortSearcher,
        NumpyArgpartitionSearcher,
        lambda corpus: NumpyBlockedSearcher(corpus, block_size=3),
    )

    for factory in factories:
        result = factory(dataset.corpus).search(dataset.queries, k)
        np.testing.assert_array_equal(result.indices, expected.indices)
        tolerance = 2e-5 if dtype is np.float32 else 2e-12
        np.testing.assert_allclose(result.scores, expected.scores, rtol=tolerance, atol=tolerance)


def test_prepared_queries_type_is_required(
    tied_corpus: np.ndarray,
    tied_queries: np.ndarray,
) -> None:
    """The prepared fast path cannot silently accept an unvalidated matrix."""
    searcher = NumpySortSearcher(tied_corpus)

    with pytest.raises(ValueError, match="PreparedQueries"):
        _ = searcher.search_prepared(tied_queries, 1)  # type: ignore[arg-type]


def test_prepared_queries_must_match_index_dimension_and_dtype(
    tied_corpus: np.ndarray,
    float_dtype: type[np.float32] | type[np.float64],
) -> None:
    """Prepared query reuse cannot cross incompatible indexes."""
    searcher = NumpySortSearcher(tied_corpus)
    wrong_dimension = PreparedQueries(np.asarray([[1.0, 0.0]], dtype=float_dtype))
    other_dtype = np.float64 if float_dtype is np.float32 else np.float32
    wrong_dtype = PreparedQueries(np.asarray([[1.0, 0.0, 0.0]], dtype=other_dtype))

    with pytest.raises(ValueError, match="query dimension"):
        _ = searcher.search_prepared(wrong_dimension, 1)
    with pytest.raises(ValueError, match="query dtype"):
        _ = searcher.search_prepared(wrong_dtype, 1)


def test_blocked_searcher_exposes_and_validates_block_size(tied_corpus: np.ndarray) -> None:
    """Blocking is a fixed, inspectable positive configuration."""
    searcher = NumpyBlockedSearcher(tied_corpus, block_size=2)

    assert searcher.block_size == 2
    with pytest.raises(ValueError, match="block_size must be positive"):
        _ = NumpyBlockedSearcher(tied_corpus, block_size=0)
    with pytest.raises(ValueError, match="block_size must be an integer"):
        _ = NumpyBlockedSearcher(tied_corpus, block_size=True)
