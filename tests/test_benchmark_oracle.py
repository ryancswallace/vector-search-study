"""Tests for scalable benchmark reference construction."""

from __future__ import annotations

import numpy as np
import pytest

from vector_search_study import SearchObjective, SearchResult, make_gaussian_dataset, make_uniform_sphere_dataset
from vector_search_study._benchmark_oracle import (
    benchmark_reference_digest,
    build_benchmark_reference,
    validate_benchmark_result,
)
from vector_search_study.exceptions import InvalidSearchParameterError


@pytest.mark.parametrize("objective", list(SearchObjective))
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_blocked_float64_oracle_matches_scalar_reference(
    objective: SearchObjective,
    dtype: type[np.float32] | type[np.float64],
) -> None:
    """The scalable path agrees with the independent scalar oracle."""
    if objective.requires_normalization:
        dataset = make_uniform_sphere_dataset(37, 9, 5, dtype=dtype, seed=101)
    else:
        dataset = make_gaussian_dataset(37, 9, 5, objective=objective, dtype=dtype, seed=101)
    scalar = build_benchmark_reference(
        dataset.corpus,
        dataset.queries,
        7,
        objective=objective,
        max_scalar_coordinates=10**9,
    )
    blocked = build_benchmark_reference(
        dataset.corpus,
        dataset.queries,
        7,
        objective=objective,
        max_scalar_coordinates=0,
        corpus_block_size=11,
        query_block_size=2,
    )

    assert scalar.method == "scalar_math_fsum_v1"
    assert blocked.method == "blocked_float64_v1"
    np.testing.assert_array_equal(blocked.result.indices, scalar.result.indices)
    tolerance = 3e-5 if dtype is np.float32 else 3e-12
    np.testing.assert_allclose(blocked.result.scores, scalar.result.scores, rtol=tolerance, atol=tolerance)
    assert blocked.boundary_margin > 0.0
    assert blocked.digest == benchmark_reference_digest(blocked.result)
    assert blocked.metadata()["reference_digest"] == blocked.digest


def test_reference_digest_is_deterministic_and_result_sensitive() -> None:
    """Canonical digests are stable and change with selected identities."""
    dataset = make_gaussian_dataset(19, 4, 2, objective=SearchObjective.INNER_PRODUCT, seed=8)
    first = build_benchmark_reference(dataset.corpus, dataset.queries, 3, objective=dataset.objective)
    second = build_benchmark_reference(dataset.corpus, dataset.queries, 3, objective=dataset.objective)
    different = build_benchmark_reference(dataset.corpus, dataset.queries, 4, objective=dataset.objective)

    assert first.digest == second.digest
    assert first.digest != different.digest


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_scalar_coordinates": -1}, "non-negative"),
        ({"max_scalar_coordinates": True}, "integer"),
        ({"corpus_block_size": 0}, "corpus_block_size"),
        ({"query_block_size": False}, "query_block_size"),
    ],
)
def test_reference_settings_are_validated(kwargs: dict[str, object], message: str) -> None:
    """Invalid oracle controls fail before benchmark collection."""
    dataset = make_gaussian_dataset(5, 3, 1, objective=SearchObjective.SQUARED_L2)
    with pytest.raises(InvalidSearchParameterError, match=message):
        _ = build_benchmark_reference(
            dataset.corpus,
            dataset.queries,
            2,
            objective=dataset.objective,
            **kwargs,  # type: ignore[arg-type]
        )


def test_reference_requires_an_observable_selection_boundary() -> None:
    """Top-k validation reserves rank k+1 for a strict margin check."""
    dataset = make_gaussian_dataset(5, 3, 1, objective=SearchObjective.SQUARED_L2)
    with pytest.raises(InvalidSearchParameterError, match="smaller than corpus_size"):
        _ = build_benchmark_reference(
            dataset.corpus,
            dataset.queries,
            5,
            objective=dataset.objective,
        )


def test_result_validation_accepts_only_numerically_ambiguous_permutations() -> None:
    """Float32 ordering may vary only inside its documented error scale."""
    expected = SearchResult(
        indices=np.asarray([[0, 1, 2]], dtype=np.int64),
        scores=np.asarray([[1.0, 0.50000001, 0.5]], dtype=np.float64),
    )
    ambiguous = SearchResult(
        indices=np.asarray([[0, 2, 1]], dtype=np.int64),
        scores=np.asarray([[1.0, 0.50000006, 0.5]], dtype=np.float64),
    )

    validate_benchmark_result(ambiguous, expected, dtype=np.float32)
    with pytest.raises(AssertionError, match="ordering inversion"):
        validate_benchmark_result(
            SearchResult(
                indices=np.asarray([[0, 2, 1]], dtype=np.int64),
                scores=np.asarray([[1.0, 0.50005, 0.50004]], dtype=np.float64),
            ),
            SearchResult(
                indices=np.asarray([[0, 1, 2]], dtype=np.int64),
                scores=np.asarray([[1.0, 0.5001, 0.5]], dtype=np.float64),
            ),
            dtype=np.float32,
        )


def test_result_validation_rejects_identity_and_shape_changes() -> None:
    """Numerical tolerance never permits a different selected neighbor set."""
    expected = SearchResult(
        indices=np.asarray([[0, 1]], dtype=np.int64),
        scores=np.asarray([[1.0, 0.5]], dtype=np.float64),
    )
    with pytest.raises(AssertionError, match="Arrays are not equal"):
        validate_benchmark_result(
            SearchResult(
                indices=np.asarray([[0, 2]], dtype=np.int64),
                scores=np.asarray([[1.0, 0.5]], dtype=np.float64),
            ),
            expected,
            dtype=np.float32,
        )
    with pytest.raises(AssertionError, match="shape differs"):
        validate_benchmark_result(
            SearchResult(
                indices=np.asarray([[0]], dtype=np.int64),
                scores=np.asarray([[1.0]], dtype=np.float64),
            ),
            expected,
            dtype=np.float64,
        )
