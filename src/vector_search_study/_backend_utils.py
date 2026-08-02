"""Shared utilities for optional exact-search backends."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

import numpy as np
from numpy.typing import NDArray

from vector_search_study.api import SearchResult
from vector_search_study.exceptions import BackendUnavailableError


def import_optional(module_name: str, *, extra: str) -> ModuleType:
    """Import an optional module or raise an actionable package exception."""
    try:
        return import_module(module_name)
    except ImportError as exc:
        raise BackendUnavailableError(
            f"optional backend {module_name!r} is unavailable; install the {extra!r} dependency group"
        ) from exc


def canonical_result(
    indices: NDArray[np.integer],
    scores: NDArray[np.floating],
) -> SearchResult:
    """Convert native candidates to the canonical result order and dtypes."""
    resolved_indices = np.asarray(indices, dtype=np.int64, order="C")
    resolved_scores = np.asarray(scores, dtype=np.float64, order="C")
    if resolved_indices.ndim == 1:
        resolved_indices = resolved_indices[:, None]
        resolved_scores = resolved_scores[:, None]
    ordered_indices = np.empty_like(resolved_indices)
    ordered_scores = np.empty_like(resolved_scores)
    for query_index in range(resolved_indices.shape[0]):
        order = np.lexsort((resolved_indices[query_index], -resolved_scores[query_index]))
        ordered_indices[query_index] = resolved_indices[query_index, order]
        ordered_scores[query_index] = resolved_scores[query_index, order]
    return SearchResult(
        indices=np.asarray(ordered_indices, dtype=np.int64, order="C"),
        scores=np.asarray(ordered_scores, dtype=np.float64, order="C"),
    )
