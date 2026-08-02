"""Deterministic synthetic embedding corpora for tests and experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vector_search_study._validation import FloatMatrix, resolve_float_dtype, validate_positive_int
from vector_search_study.api import normalize_rows
from vector_search_study.exceptions import InvalidSearchParameterError


@dataclass(frozen=True, slots=True)
class SyntheticDataset:
    """An immutable generated corpus/query pair and its provenance.

    Attributes:
        corpus: Pre-normalized corpus vectors.
        queries: Pre-normalized query vectors.
        distribution: Generator family name.
        seed: PCG64 seed.
    """

    corpus: FloatMatrix
    queries: FloatMatrix
    distribution: str
    seed: int


def make_uniform_sphere_dataset(
    corpus_size: int,
    dimension: int,
    query_count: int,
    *,
    dtype: object = np.float32,
    seed: int = 20_260_801,
) -> SyntheticDataset:
    """Generate independent corpus and query vectors on the unit sphere.

    Args:
        corpus_size: Number of corpus vectors.
        dimension: Embedding dimension.
        query_count: Number of query vectors.
        dtype: Either float32 or float64.
        seed: Non-negative PCG64 seed.

    Returns:
        A deterministic normalized synthetic dataset.
    """
    size, dimensions, queries, resolved_dtype, resolved_seed = _validate_generator_inputs(
        corpus_size,
        dimension,
        query_count,
        dtype=dtype,
        seed=seed,
    )
    generator = np.random.Generator(np.random.PCG64(resolved_seed))
    corpus = _normal_matrix(generator, size, dimensions, resolved_dtype)
    query_matrix = _normal_matrix(generator, queries, dimensions, resolved_dtype)
    return _dataset(corpus, query_matrix, distribution="uniform_sphere", seed=resolved_seed)


def make_clustered_dataset(
    corpus_size: int,
    dimension: int,
    query_count: int,
    *,
    cluster_count: int = 8,
    noise: float = 0.15,
    dtype: object = np.float32,
    seed: int = 20_260_801,
) -> SyntheticDataset:
    """Generate normalized vectors around shared random cluster centroids.

    Args:
        corpus_size: Number of corpus vectors.
        dimension: Embedding dimension.
        query_count: Number of query vectors.
        cluster_count: Number of latent centroids.
        noise: Positive standard deviation around each centroid.
        dtype: Either float32 or float64.
        seed: Non-negative PCG64 seed.

    Returns:
        A deterministic normalized clustered dataset.
    """
    size, dimensions, queries, resolved_dtype, resolved_seed = _validate_generator_inputs(
        corpus_size,
        dimension,
        query_count,
        dtype=dtype,
        seed=seed,
    )
    clusters = validate_positive_int(cluster_count, name="cluster_count")
    if isinstance(noise, bool) or not isinstance(noise, (int, float)) or not np.isfinite(noise) or noise <= 0:
        raise InvalidSearchParameterError("noise must be a positive finite number")

    generator = np.random.Generator(np.random.PCG64(resolved_seed))
    centroids = _normal_matrix(generator, clusters, dimensions, resolved_dtype)
    corpus_assignments = generator.integers(0, clusters, size=size)
    query_assignments = generator.integers(0, clusters, size=queries)
    corpus_noise = generator.standard_normal((size, dimensions)).astype(resolved_dtype, copy=False)
    query_noise = generator.standard_normal((queries, dimensions)).astype(resolved_dtype, copy=False)
    corpus = normalize_rows(np.asarray(centroids[corpus_assignments] + noise * corpus_noise, dtype=resolved_dtype))
    query_matrix = normalize_rows(np.asarray(centroids[query_assignments] + noise * query_noise, dtype=resolved_dtype))
    return _dataset(corpus, query_matrix, distribution="clustered", seed=resolved_seed)


def _validate_generator_inputs(
    corpus_size: object,
    dimension: object,
    query_count: object,
    *,
    dtype: object,
    seed: object,
) -> tuple[int, int, int, np.dtype[np.float32] | np.dtype[np.float64], int]:
    """Validate and resolve shared generator settings."""
    size = validate_positive_int(corpus_size, name="corpus_size")
    dimensions = validate_positive_int(dimension, name="dimension")
    queries = validate_positive_int(query_count, name="query_count")
    resolved_dtype = resolve_float_dtype(dtype)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise InvalidSearchParameterError("seed must be an integer")
    if seed < 0:
        raise InvalidSearchParameterError("seed must be non-negative")
    return size, dimensions, queries, resolved_dtype, seed


def _normal_matrix(
    generator: np.random.Generator,
    rows: int,
    dimensions: int,
    dtype: np.dtype[np.float32] | np.dtype[np.float64],
) -> FloatMatrix:
    """Generate and normalize a Gaussian matrix."""
    values = generator.standard_normal((rows, dimensions)).astype(dtype, copy=False)
    return normalize_rows(values)


def _dataset(corpus: FloatMatrix, queries: FloatMatrix, *, distribution: str, seed: int) -> SyntheticDataset:
    """Freeze generated arrays and return their provenance wrapper."""
    corpus.flags.writeable = False
    queries.flags.writeable = False
    return SyntheticDataset(corpus=corpus, queries=queries, distribution=distribution, seed=seed)
