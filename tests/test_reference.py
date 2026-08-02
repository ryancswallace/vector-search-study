"""Tests for the independent trusted reference implementation."""

import numpy as np
import pytest

from vector_search_study import InvalidVectorDataError, reference_search


def test_reference_uses_accurate_summation_and_canonical_order() -> None:
    """The reference provides deterministic high-accuracy expected values."""
    corpus = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, -1.0]], dtype=np.float64)
    queries = np.asarray([[1.0, 0.0]], dtype=np.float64)

    result = reference_search(corpus, queries, 3)

    np.testing.assert_array_equal(result.indices, [[0, 1, 2]])
    np.testing.assert_allclose(result.scores, [[1.0, 1.0, 0.0]])


def test_reference_rejects_mismatched_dimension() -> None:
    """Reference validation is independent of production searchers."""
    with pytest.raises(InvalidVectorDataError, match="query dimension"):
        _ = reference_search(
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            1,
        )


def test_reference_rejects_mismatched_dtype() -> None:
    """Reference scores use the same dtype compatibility contract."""
    with pytest.raises(InvalidVectorDataError, match="query dtype"):
        _ = reference_search(
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            np.asarray([[1.0, 0.0]], dtype=np.float64),
            1,
        )
