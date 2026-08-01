"""Public package interface for Vector Search Study."""

from importlib.metadata import PackageNotFoundError, version

from vector_search_study._core import greet
from vector_search_study.exceptions import VectorSearchStudyError

try:
    __version__ = version("vector-search-study")
except PackageNotFoundError:  # pragma: no cover - source tree fallback before installation
    __version__ = "0+unknown"

__all__ = [
    "VectorSearchStudyError",
    "__version__",
    "greet",
]
