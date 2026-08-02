"""Public package interface for Vector Search Study."""

from importlib.metadata import PackageNotFoundError, version

from vector_search_study.api import (
    ExactSearcher,
    PreparedQueries,
    SearchObjective,
    SearchResult,
    normalize_rows,
    prepare_queries,
)
from vector_search_study.exceptions import (
    BackendUnavailableError,
    InvalidSearchParameterError,
    InvalidVectorDataError,
    UnsupportedObjectiveError,
    VectorSearchStudyError,
)
from vector_search_study.faiss_search import FaissFlatIPSearcher, FaissFlatL2Searcher
from vector_search_study.numpy_search import NumpyArgpartitionSearcher, NumpyBlockedSearcher, NumpySortSearcher
from vector_search_study.python_search import PythonHeapSearcher, PythonSortSearcher
from vector_search_study.reference import reference_search
from vector_search_study.scipy_search import ScipyCKDTreeSearcher
from vector_search_study.sklearn_search import SklearnBallTreeSearcher, SklearnBruteSearcher, SklearnKDTreeSearcher
from vector_search_study.synthetic import (
    SyntheticDataset,
    make_clustered_dataset,
    make_gaussian_dataset,
    make_uniform_sphere_dataset,
)
from vector_search_study.torch_search import TorchTopKSearcher

try:
    __version__ = version("vector-search-study")
except PackageNotFoundError:  # pragma: no cover - source tree fallback before installation
    __version__ = "0+unknown"

__all__ = [
    "BackendUnavailableError",
    "ExactSearcher",
    "FaissFlatIPSearcher",
    "FaissFlatL2Searcher",
    "InvalidSearchParameterError",
    "InvalidVectorDataError",
    "NumpyArgpartitionSearcher",
    "NumpyBlockedSearcher",
    "NumpySortSearcher",
    "PreparedQueries",
    "PythonHeapSearcher",
    "PythonSortSearcher",
    "ScipyCKDTreeSearcher",
    "SearchObjective",
    "SearchResult",
    "SklearnBallTreeSearcher",
    "SklearnBruteSearcher",
    "SklearnKDTreeSearcher",
    "SyntheticDataset",
    "TorchTopKSearcher",
    "UnsupportedObjectiveError",
    "VectorSearchStudyError",
    "__version__",
    "make_clustered_dataset",
    "make_gaussian_dataset",
    "make_uniform_sphere_dataset",
    "normalize_rows",
    "prepare_queries",
    "reference_search",
]
