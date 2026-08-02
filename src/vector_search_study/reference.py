"""High-accuracy trusted reference for exact vector search."""

from __future__ import annotations

import math

import numpy as np

from vector_search_study._validation import FloatMatrix, validate_search_k, validate_vector_matrix
from vector_search_study.api import PreparedQueries, SearchResult
from vector_search_study.exceptions import InvalidVectorDataError


def reference_search(corpus: FloatMatrix, queries: FloatMatrix, k: int) -> SearchResult:
    """Compute canonical exact top-k results with accurate scalar summation.

    This intentionally slow implementation is designed for correctness tests
    and untimed benchmark validation, not performance measurement.

    Args:
        corpus: Pre-normalized corpus matrix with shape ``(N, D)``.
        queries: Pre-normalized query matrix with shape ``(Q, D)``.
        k: Number of ordered neighbors to return.

    Returns:
        Canonically ordered exact results.

    Raises:
        InvalidVectorDataError: If corpus and query contracts do not match.
    """
    validated_corpus = validate_vector_matrix(corpus, name="corpus")
    prepared = PreparedQueries(queries)
    if prepared.dimension != validated_corpus.shape[1]:
        raise InvalidVectorDataError(
            f"query dimension {prepared.dimension} does not match corpus dimension {validated_corpus.shape[1]}"
        )
    if prepared.dtype != validated_corpus.dtype:
        raise InvalidVectorDataError(
            f"query dtype {prepared.dtype} does not match corpus dtype {validated_corpus.dtype}"
        )
    resolved_k = validate_search_k(k, corpus_size=validated_corpus.shape[0])

    indices = np.empty((prepared.query_count, resolved_k), dtype=np.int64)
    scores = np.empty((prepared.query_count, resolved_k), dtype=np.float64)
    for query_index, query in enumerate(prepared.values):
        ranked: list[tuple[float, int]] = []
        for corpus_index, vector in enumerate(validated_corpus):
            score = math.fsum(float(left) * float(right) for left, right in zip(vector, query, strict=True))
            ranked.append((score, corpus_index))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        for result_index, (score, corpus_index) in enumerate(ranked[:resolved_k]):
            indices[query_index, result_index] = corpus_index
            scores[query_index, result_index] = score
    return SearchResult(indices=indices, scores=scores)
