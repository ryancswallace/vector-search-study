"""Scalable trusted-result construction for benchmark validation."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from vector_search_study._selection import merge_top_k, partition_top_k
from vector_search_study._validation import FloatMatrix, resolve_float_dtype, validate_search_k, validate_vector_matrix
from vector_search_study.api import SearchObjective, SearchResult, resolve_search_objective
from vector_search_study.exceptions import InvalidSearchParameterError, InvalidVectorDataError
from vector_search_study.reference import reference_search

DEFAULT_SCALAR_REFERENCE_LIMIT = 1_000_000
DEFAULT_REFERENCE_CORPUS_BLOCK_SIZE = 8_192
DEFAULT_REFERENCE_QUERY_BLOCK_SIZE = 64
_DIGEST_SCORE_DECIMALS = 12
_REFERENCE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class BenchmarkReference:
    """Expected top-k result and deterministic validation provenance."""

    result: SearchResult
    method: str
    digest: str
    boundary_margin: float

    def metadata(self) -> dict[str, object]:
        """Return strict-JSON-safe reference metadata."""
        return {
            "reference_schema_version": _REFERENCE_SCHEMA_VERSION,
            "reference_method": self.method,
            "reference_digest": self.digest,
            "reference_digest_method": f"sha256_indices_scores_round{_DIGEST_SCORE_DECIMALS}_v1",
            "reference_boundary_margin": self.boundary_margin,
        }


def build_benchmark_reference(
    corpus: FloatMatrix,
    queries: FloatMatrix,
    k: int,
    *,
    objective: SearchObjective | str,
    max_scalar_coordinates: int = DEFAULT_SCALAR_REFERENCE_LIMIT,
    corpus_block_size: int = DEFAULT_REFERENCE_CORPUS_BLOCK_SIZE,
    query_block_size: int = DEFAULT_REFERENCE_QUERY_BLOCK_SIZE,
) -> BenchmarkReference:
    """Build a strict-boundary top-k reference without timing oracle work.

    Args:
        corpus: Corpus matrix under the exact-search contract.
        queries: Query matrix under the exact-search contract.
        k: Number of neighbors required by the benchmark.
        objective: Exact-search score convention.
        max_scalar_coordinates: Largest workload evaluated by the scalar
            ``math.fsum`` reference.
        corpus_block_size: Corpus rows per float64 oracle block.
        query_block_size: Query rows per float64 oracle block.

    Returns:
        Expected results and deterministic reference provenance.

    Raises:
        InvalidSearchParameterError: If no rank ``k + 1`` boundary exists or
            an oracle limit is invalid.
        AssertionError: If the generated workload has no strict top-k margin.
    """
    resolved_objective = resolve_search_objective(objective)
    validated_corpus = validate_vector_matrix(
        corpus,
        name="corpus",
        require_normalized=resolved_objective.requires_normalization,
    )
    validated_queries = validate_vector_matrix(
        queries,
        name="queries",
        require_normalized=resolved_objective.requires_normalization,
    )
    if validated_corpus.shape[1] != validated_queries.shape[1]:
        raise InvalidVectorDataError("query dimension does not match corpus dimension")
    resolved_k = validate_search_k(k, corpus_size=validated_corpus.shape[0])
    if resolved_k == validated_corpus.shape[0]:
        raise InvalidSearchParameterError("benchmark validation requires k to be smaller than corpus_size")
    if isinstance(max_scalar_coordinates, bool) or not isinstance(max_scalar_coordinates, int):
        raise InvalidSearchParameterError("max_scalar_coordinates must be an integer")
    if max_scalar_coordinates < 0:
        raise InvalidSearchParameterError("max_scalar_coordinates must be non-negative")
    resolved_corpus_block_size = _positive_int(corpus_block_size, name="corpus_block_size")
    resolved_query_block_size = _positive_int(query_block_size, name="query_block_size")

    coordinate_evaluations = int(validated_corpus.shape[0] * validated_corpus.shape[1] * validated_queries.shape[0])
    if coordinate_evaluations <= max_scalar_coordinates:
        ranked = reference_search(
            validated_corpus,
            validated_queries,
            resolved_k + 1,
            objective=resolved_objective,
        )
        method = "scalar_math_fsum_v1"
    else:
        ranked = _blocked_float64_reference_search(
            validated_corpus,
            validated_queries,
            resolved_k + 1,
            objective=resolved_objective,
            corpus_block_size=resolved_corpus_block_size,
            query_block_size=resolved_query_block_size,
        )
        method = "blocked_float64_v1"

    boundary_margin = float(np.min(ranked.scores[:, resolved_k - 1] - ranked.scores[:, resolved_k]))
    if boundary_margin <= 0.0:
        raise AssertionError("benchmark workload lacks a strict top-k boundary margin")
    result = SearchResult(
        indices=np.array(ranked.indices[:, :resolved_k], dtype=np.int64, order="C", copy=True),
        scores=np.array(ranked.scores[:, :resolved_k], dtype=np.float64, order="C", copy=True),
    )
    return BenchmarkReference(
        result=result,
        method=method,
        digest=benchmark_reference_digest(result),
        boundary_margin=boundary_margin,
    )


def benchmark_reference_digest(result: SearchResult) -> str:
    """Return a stable digest over canonical indices and rounded scores."""
    indices = np.asarray(result.indices, dtype="<i8", order="C")
    rounded_scores = np.asarray(np.round(result.scores, decimals=_DIGEST_SCORE_DECIMALS), dtype="<f8", order="C")
    digest = sha256()
    digest.update(b"vector-search-study-reference-v1\0")
    digest.update(np.asarray(indices.shape, dtype="<i8").tobytes(order="C"))
    digest.update(indices.tobytes(order="C"))
    digest.update(rounded_scores.tobytes(order="C"))
    return f"sha256:{digest.hexdigest()}"


def validate_benchmark_result(
    actual: SearchResult,
    expected: SearchResult,
    *,
    dtype: object,
) -> None:
    """Validate identities, scores, and numerically resolvable ordering.

    Float32 BLAS implementations can round two nearly equal scores in opposite
    directions. The selected identity set must still match exactly, while an
    internal permutation is accepted only when the float64 reference scores
    differ by no more than the dtype-aware numerical ranking tolerance.

    Args:
        actual: Result returned by a measured implementation.
        expected: Canonical float64 oracle result.
        dtype: Workload dtype used by the measured implementation.

    Raises:
        AssertionError: If identities, scores, or resolvable ordering differ.
    """
    resolved_dtype = resolve_float_dtype(dtype)
    if actual.indices.shape != expected.indices.shape:
        raise AssertionError("benchmark result shape differs from the reference")
    np.testing.assert_array_equal(np.sort(actual.indices, axis=1), np.sort(expected.indices, axis=1))

    aligned_expected_scores = np.empty_like(expected.scores)
    for query_index in range(actual.indices.shape[0]):
        expected_by_index = dict(
            zip(
                expected.indices[query_index].tolist(),
                expected.scores[query_index].tolist(),
                strict=True,
            )
        )
        aligned_expected_scores[query_index] = [expected_by_index[int(index)] for index in actual.indices[query_index]]

    if resolved_dtype == np.dtype(np.float32):
        score_rtol, score_atol = 2e-4, 2e-5
        rank_rtol, rank_atol = 8 * np.finfo(np.float32).eps, 8 * np.finfo(np.float32).eps
    else:
        score_rtol, score_atol = 2e-10, 2e-10
        rank_rtol, rank_atol = 8 * np.finfo(np.float64).eps, 8 * np.finfo(np.float64).eps
    np.testing.assert_allclose(
        actual.scores,
        aligned_expected_scores,
        rtol=score_rtol,
        atol=score_atol,
    )

    left = aligned_expected_scores[:, :-1]
    right = aligned_expected_scores[:, 1:]
    scale = np.maximum(np.maximum(np.abs(left), np.abs(right)), 1.0)
    tolerance = rank_atol + rank_rtol * scale
    if bool(np.any(left + tolerance < right)):
        raise AssertionError("benchmark result contains an ordering inversion beyond numerical tolerance")


def _blocked_float64_reference_search(
    corpus: FloatMatrix,
    queries: FloatMatrix,
    k: int,
    *,
    objective: SearchObjective,
    corpus_block_size: int,
    query_block_size: int,
) -> SearchResult:
    """Search bounded float64 query and corpus blocks."""
    query_count = queries.shape[0]
    all_indices = np.empty((query_count, k), dtype=np.int64)
    all_scores = np.empty((query_count, k), dtype=np.float64)
    for query_start in range(0, query_count, query_block_size):
        query_stop = min(query_start + query_block_size, query_count)
        query_values = _float64_values(queries[query_start:query_stop], normalize=objective.requires_normalization)
        best_indices = np.empty((query_stop - query_start, 0), dtype=np.int64)
        best_scores = np.empty((query_stop - query_start, 0), dtype=np.float64)
        for corpus_start in range(0, corpus.shape[0], corpus_block_size):
            corpus_stop = min(corpus_start + corpus_block_size, corpus.shape[0])
            corpus_values = _float64_values(
                corpus[corpus_start:corpus_stop],
                normalize=objective.requires_normalization,
            )
            block_scores = _float64_scores(query_values, corpus_values, objective)
            block_k = min(k, corpus_stop - corpus_start)
            block_indices, selected_scores = partition_top_k(
                block_scores,
                block_k,
                index_offset=corpus_start,
            )
            best_indices, best_scores = merge_top_k(
                best_indices,
                best_scores,
                block_indices,
                selected_scores,
                k,
            )
        all_indices[query_start:query_stop] = best_indices
        all_scores[query_start:query_stop] = best_scores
    return SearchResult(indices=all_indices, scores=all_scores)


def _float64_values(values: FloatMatrix, *, normalize: bool) -> FloatMatrix:
    """Copy one block to float64 and optionally restore exact unit norms."""
    converted = np.array(values, dtype=np.float64, order="C", copy=True)
    if normalize:
        converted /= np.linalg.norm(converted, axis=1, keepdims=True)
    return converted


def _float64_scores(queries: FloatMatrix, corpus: FloatMatrix, objective: SearchObjective) -> FloatMatrix:
    """Return one float64 higher-is-better score block."""
    products = queries @ corpus.T
    if objective is not SearchObjective.SQUARED_L2:
        return np.asarray(products, dtype=np.float64, order="C")
    query_norms = np.sum(queries * queries, axis=1, keepdims=True)
    corpus_norms = np.sum(corpus * corpus, axis=1, keepdims=True).T
    distances = np.maximum(query_norms + corpus_norms - 2.0 * products, 0.0)
    return np.asarray(-distances, dtype=np.float64, order="C")


def _positive_int(value: object, *, name: str) -> int:
    """Return a positive integer oracle setting."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidSearchParameterError(f"{name} must be an integer")
    if value <= 0:
        raise InvalidSearchParameterError(f"{name} must be positive")
    return value
