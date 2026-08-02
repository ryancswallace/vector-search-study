"""Tests for the public package API."""

import vector_search_study


def test_public_exports_are_available() -> None:
    """The package exports only its documented milestone objects."""
    expected_exports = {
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
    }

    assert set(vector_search_study.__all__) == expected_exports
    for name in vector_search_study.__all__:
        assert hasattr(vector_search_study, name)


def test_version_is_available() -> None:
    """A version string is always exposed."""
    assert vector_search_study.__version__
