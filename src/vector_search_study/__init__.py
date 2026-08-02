"""Public package interface for Vector Search Study."""

from importlib.metadata import PackageNotFoundError, version

from vector_search_study.api import ExactSearcher, PreparedQueries, SearchResult, normalize_rows, prepare_queries
from vector_search_study.exceptions import (
    InvalidSearchParameterError,
    InvalidVectorDataError,
    VectorSearchStudyError,
)
from vector_search_study.numpy_search import NumpyArgpartitionSearcher, NumpyBlockedSearcher, NumpySortSearcher
from vector_search_study.python_search import PythonHeapSearcher, PythonSortSearcher
from vector_search_study.reference import reference_search
from vector_search_study.synthetic import SyntheticDataset, make_clustered_dataset, make_uniform_sphere_dataset

try:
    __version__ = version("vector-search-study")
except PackageNotFoundError:  # pragma: no cover - source tree fallback before installation
    __version__ = "0+unknown"

__all__ = [
    "ExactSearcher",
    "InvalidSearchParameterError",
    "InvalidVectorDataError",
    "NumpyArgpartitionSearcher",
    "NumpyBlockedSearcher",
    "NumpySortSearcher",
    "PreparedQueries",
    "PythonHeapSearcher",
    "PythonSortSearcher",
    "SearchResult",
    "SyntheticDataset",
    "VectorSearchStudyError",
    "__version__",
    "make_clustered_dataset",
    "make_uniform_sphere_dataset",
    "normalize_rows",
    "prepare_queries",
    "reference_search",
]
