"""Tests for deterministic synthetic embedding generators."""

import numpy as np
import pytest

from vector_search_study import (
    InvalidSearchParameterError,
    InvalidVectorDataError,
    SearchObjective,
    make_clustered_dataset,
    make_gaussian_dataset,
    make_uniform_sphere_dataset,
)


@pytest.mark.parametrize("factory", [make_uniform_sphere_dataset, make_clustered_dataset])
def test_synthetic_datasets_are_reproducible_normalized_and_immutable(factory: object) -> None:
    """A fixed configuration produces reusable vectors and provenance."""
    first = factory(12, 6, 4, dtype=np.float32, seed=42)  # type: ignore[operator]
    second = factory(12, 6, 4, dtype=np.float32, seed=42)  # type: ignore[operator]

    np.testing.assert_array_equal(first.corpus, second.corpus)
    np.testing.assert_array_equal(first.queries, second.queries)
    np.testing.assert_allclose(np.linalg.norm(first.corpus, axis=1), 1.0, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(first.queries, axis=1), 1.0, rtol=1e-5, atol=1e-6)
    assert first.seed == 42
    assert first.corpus.dtype == np.float32
    assert first.corpus.flags.c_contiguous
    assert not first.corpus.flags.writeable
    assert not first.queries.flags.writeable


def test_synthetic_distributions_and_seeds_change_data() -> None:
    """Generator family and seed are material experimental inputs."""
    uniform = make_uniform_sphere_dataset(10, 5, 3, seed=1)
    other_seed = make_uniform_sphere_dataset(10, 5, 3, seed=2)
    clustered = make_clustered_dataset(10, 5, 3, cluster_count=2, noise=0.05, seed=1)

    assert uniform.distribution == "uniform_sphere"
    assert clustered.distribution == "clustered"
    assert not np.array_equal(uniform.corpus, other_seed.corpus)
    assert not np.array_equal(uniform.corpus, clustered.corpus)


@pytest.mark.parametrize("objective", [SearchObjective.SQUARED_L2, SearchObjective.INNER_PRODUCT])
def test_gaussian_datasets_are_reproducible_and_unnormalized(objective: SearchObjective) -> None:
    """Non-cosine workloads preserve natural vector magnitudes."""
    first = make_gaussian_dataset(10, 5, 3, objective=objective, seed=7)
    second = make_gaussian_dataset(10, 5, 3, objective=objective, seed=7)

    np.testing.assert_array_equal(first.corpus, second.corpus)
    assert first.distribution == "gaussian"
    assert first.objective is objective
    assert not np.allclose(np.linalg.norm(first.corpus, axis=1), 1.0)
    assert not first.corpus.flags.writeable


def test_clustered_dataset_obeys_objective_normalization() -> None:
    """Clustered data can support all objectives without implicit metric changes."""
    inner_product = make_clustered_dataset(10, 5, 3, objective=SearchObjective.INNER_PRODUCT)
    cosine = make_clustered_dataset(10, 5, 3)

    assert inner_product.objective is SearchObjective.INNER_PRODUCT
    assert not np.allclose(np.linalg.norm(inner_product.corpus, axis=1), 1.0)
    np.testing.assert_allclose(np.linalg.norm(cosine.corpus, axis=1), 1.0, rtol=1e-5, atol=1e-6)


def test_gaussian_generator_rejects_cosine() -> None:
    """Normalized cosine uses the explicitly normalized generator."""
    with pytest.raises(InvalidSearchParameterError, match="uniform_sphere"):
        _ = make_gaussian_dataset(4, 3, 2, objective=SearchObjective.NORMALIZED_COSINE)


@pytest.mark.parametrize("field", ["corpus_size", "dimension", "query_count"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_synthetic_generators_validate_positive_sizes(field: str, value: object) -> None:
    """Matrix shape configuration uses strict positive integers."""
    kwargs: dict[str, object] = {"corpus_size": 4, "dimension": 3, "query_count": 2}
    kwargs[field] = value

    with pytest.raises(InvalidSearchParameterError, match=field):
        _ = make_uniform_sphere_dataset(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("seed", [-1, True, 1.5])
def test_synthetic_generators_validate_seed(seed: object) -> None:
    """Seeds are explicit non-negative integers."""
    with pytest.raises(InvalidSearchParameterError, match="seed"):
        _ = make_uniform_sphere_dataset(4, 3, 2, seed=seed)  # type: ignore[arg-type]


@pytest.mark.parametrize("dtype", [np.int64, "int32", object()])
def test_synthetic_generators_validate_dtype(dtype: object) -> None:
    """Generated benchmark matrices use only supported floating types."""
    with pytest.raises(InvalidVectorDataError, match="dtype"):
        _ = make_uniform_sphere_dataset(4, 3, 2, dtype=dtype)


@pytest.mark.parametrize("cluster_count", [0, -1, True, 1.5])
def test_clustered_generator_validates_cluster_count(cluster_count: object) -> None:
    """Cluster configuration uses a strict positive integer."""
    with pytest.raises(InvalidSearchParameterError, match="cluster_count"):
        _ = make_clustered_dataset(4, 3, 2, cluster_count=cluster_count)  # type: ignore[arg-type]


@pytest.mark.parametrize("noise", [0.0, -0.1, float("nan"), float("inf"), True, "0.1"])
def test_clustered_generator_validates_noise(noise: object) -> None:
    """Cluster dispersion must be a positive finite number."""
    with pytest.raises(InvalidSearchParameterError, match="noise"):
        _ = make_clustered_dataset(4, 3, 2, noise=noise)  # type: ignore[arg-type]
