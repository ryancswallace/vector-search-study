"""Validate vector matrices and search parameters."""

from __future__ import annotations

from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import DTypeLike, NDArray

from vector_search_study.exceptions import InvalidSearchParameterError, InvalidVectorDataError

FloatMatrix: TypeAlias = NDArray[np.floating[Any]]

_FLOAT32_NORMALIZATION_RTOL = 5e-5
_FLOAT32_NORMALIZATION_ATOL = 5e-6
_FLOAT64_NORMALIZATION_RTOL = 1e-12
_FLOAT64_NORMALIZATION_ATOL = 1e-12
_SUPPORTED_DTYPES = (np.dtype(np.float32), np.dtype(np.float64))


def validate_vector_matrix(
    value: object,
    *,
    name: str,
    require_normalized: bool = True,
) -> FloatMatrix:
    """Return a matrix after enforcing the study's dense-vector contract."""
    if not isinstance(value, np.ndarray):
        raise InvalidVectorDataError(f"{name} must be a NumPy array")
    if value.ndim != 2:
        raise InvalidVectorDataError(f"{name} must have shape (rows, dimensions)")
    if value.shape[0] == 0 or value.shape[1] == 0:
        raise InvalidVectorDataError(f"{name} must not have an empty axis")
    if value.dtype not in _SUPPORTED_DTYPES:
        raise InvalidVectorDataError(f"{name} must have dtype float32 or float64")
    if not value.flags.c_contiguous:
        raise InvalidVectorDataError(f"{name} must be C-contiguous")
    if not bool(np.isfinite(value).all()):
        raise InvalidVectorDataError(f"{name} must contain only finite values")

    matrix = cast(FloatMatrix, value)
    if require_normalized:
        _validate_normalized(matrix, name=name)
    return matrix


def validate_search_k(k: object, *, corpus_size: int) -> int:
    """Return a validated top-k value."""
    if isinstance(k, bool) or not isinstance(k, int):
        raise InvalidSearchParameterError("k must be an integer")
    if not 1 <= k <= corpus_size:
        raise InvalidSearchParameterError(f"k must satisfy 1 <= k <= {corpus_size}")
    return k


def validate_positive_int(value: object, *, name: str) -> int:
    """Return a positive integer configuration value."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidSearchParameterError(f"{name} must be an integer")
    if value <= 0:
        raise InvalidSearchParameterError(f"{name} must be positive")
    return value


def resolve_float_dtype(value: object) -> np.dtype[np.float32] | np.dtype[np.float64]:
    """Return a supported concrete floating-point dtype."""
    try:
        dtype = np.dtype(cast(DTypeLike, value))
    except (TypeError, ValueError) as exc:
        raise InvalidVectorDataError("dtype must be float32 or float64") from exc
    if dtype == np.dtype(np.float32):
        return np.dtype(np.float32)
    if dtype == np.dtype(np.float64):
        return np.dtype(np.float64)
    raise InvalidVectorDataError("dtype must be float32 or float64")


def _validate_normalized(matrix: FloatMatrix, *, name: str) -> None:
    """Reject rows whose L2 norm is not approximately one."""
    norms = np.linalg.norm(matrix, axis=1)
    if matrix.dtype == np.dtype(np.float32):
        rtol = _FLOAT32_NORMALIZATION_RTOL
        atol = _FLOAT32_NORMALIZATION_ATOL
    else:
        rtol = _FLOAT64_NORMALIZATION_RTOL
        atol = _FLOAT64_NORMALIZATION_ATOL
    if not bool(np.allclose(norms, 1.0, rtol=rtol, atol=atol)):
        raise InvalidVectorDataError(f"every row in {name} must have L2 norm one")
