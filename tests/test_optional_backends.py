"""Isolated correctness tests for optional exact-search adapters."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from vector_search_study import (
    BackendUnavailableError,
    FaissFlatIPSearcher,
    FaissFlatL2Searcher,
    ScipyCKDTreeSearcher,
    SearchObjective,
    SklearnBallTreeSearcher,
    SklearnBruteSearcher,
    SklearnKDTreeSearcher,
    TorchTopKSearcher,
    UnsupportedObjectiveError,
    _backend_utils,
    faiss_search,
    normalize_rows,
    reference_search,
    scipy_search,
    sklearn_search,
    torch_search,
)


def _rank(corpus: np.ndarray, queries: np.ndarray, objective: SearchObjective, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return stable native candidates for fake backend modules."""
    if objective is SearchObjective.SQUARED_L2:
        differences = queries[:, None, :] - corpus[None, :, :]
        scores = -np.sum(differences * differences, axis=2)
    else:
        scores = queries @ corpus.T
    order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    return order, np.take_along_axis(scores, order, axis=1)


class _FakeNearestNeighbors:
    """Minimal NearestNeighbors implementation backed by NumPy."""

    def __init__(self, *, metric: str, **_kwargs: object) -> None:
        self.metric = metric
        self.corpus = np.empty((0, 0), dtype=np.float32)

    def fit(self, corpus: np.ndarray) -> _FakeNearestNeighbors:
        """Retain the corpus and emulate sklearn's fitted return value."""
        self.corpus = corpus
        return self

    def kneighbors(
        self, queries: np.ndarray, *, n_neighbors: int, return_distance: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return exact Euclidean or cosine distances."""
        objective = SearchObjective.NORMALIZED_COSINE if self.metric == "cosine" else SearchObjective.SQUARED_L2
        indices, scores = _rank(self.corpus, queries, objective, n_neighbors)
        distances = 1.0 - scores if objective is SearchObjective.NORMALIZED_COSINE else np.sqrt(-scores)
        assert return_distance
        return distances, indices


class _FakeTree:
    """Minimal KDTree/BallTree/cKDTree implementation backed by NumPy."""

    def __init__(self, corpus: np.ndarray, **_kwargs: object) -> None:
        self.corpus = corpus

    def query(self, queries: np.ndarray, k: int, **_kwargs: object) -> tuple[np.ndarray, np.ndarray]:
        """Return exact Euclidean nearest neighbors."""
        indices, scores = _rank(self.corpus, queries, SearchObjective.SQUARED_L2, k)
        distances = np.sqrt(-scores)
        if k == 1:
            return distances[:, 0], indices[:, 0]
        return distances, indices


class _FakeFaissL2:
    """Minimal Faiss IndexFlatL2 implementation."""

    def __init__(self, _dimension: int) -> None:
        self.corpus = np.empty((0, 0), dtype=np.float32)

    def add(self, corpus: np.ndarray) -> None:
        """Retain indexed vectors."""
        self.corpus = corpus

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return squared distances followed by indices."""
        indices, scores = _rank(self.corpus, queries, SearchObjective.SQUARED_L2, k)
        return -scores, indices


class _FakeFaissIP(_FakeFaissL2):
    """Minimal Faiss IndexFlatIP implementation."""

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return inner products followed by indices."""
        indices, scores = _rank(self.corpus, queries, SearchObjective.INNER_PRODUCT, k)
        return scores, indices


class _FakeTensor:
    """Small NumPy-backed tensor implementing the adapter's used surface."""

    def __init__(self, values: np.ndarray) -> None:
        self.values = np.asarray(values)

    def __mul__(self, other: object) -> _FakeTensor:
        values = other.values if isinstance(other, _FakeTensor) else other
        return _FakeTensor(self.values * values)

    __rmul__ = __mul__

    def __add__(self, other: object) -> _FakeTensor:
        values = other.values if isinstance(other, _FakeTensor) else other
        return _FakeTensor(self.values + values)

    __radd__ = __add__

    def __sub__(self, other: object) -> _FakeTensor:
        values = other.values if isinstance(other, _FakeTensor) else other
        return _FakeTensor(self.values - values)

    def __rsub__(self, other: object) -> _FakeTensor:
        values = other.values if isinstance(other, _FakeTensor) else other
        return _FakeTensor(values - self.values)

    def __neg__(self) -> _FakeTensor:
        return _FakeTensor(-self.values)

    def sum(self, *, dim: int) -> _FakeTensor:
        """Sum over one dimension."""
        return _FakeTensor(np.sum(self.values, axis=dim))

    def unsqueeze(self, dim: int) -> _FakeTensor:
        """Insert one singleton dimension."""
        return _FakeTensor(np.expand_dims(self.values, axis=dim))

    def transpose(self, first: int, second: int) -> _FakeTensor:
        """Swap two dimensions."""
        return _FakeTensor(np.swapaxes(self.values, first, second))

    def clamp_min(self, minimum: float) -> _FakeTensor:
        """Clamp values from below."""
        return _FakeTensor(np.maximum(self.values, minimum))

    def detach(self) -> _FakeTensor:
        """Return this non-autograd tensor."""
        return self

    def cpu(self) -> _FakeTensor:
        """Return this already-CPU tensor."""
        return self

    def numpy(self) -> np.ndarray:
        """Expose the backing array."""
        return self.values


class _FakeTorch:
    """Minimal module-like PyTorch surface."""

    @staticmethod
    def set_num_threads(_threads: int) -> None:
        """Accept deterministic thread configuration."""

    @staticmethod
    def from_numpy(values: np.ndarray) -> _FakeTensor:
        """Wrap a NumPy array."""
        return _FakeTensor(values)

    @staticmethod
    def matmul(left: _FakeTensor, right: _FakeTensor) -> _FakeTensor:
        """Multiply two tensors."""
        return _FakeTensor(left.values @ right.values)

    @staticmethod
    def topk(
        scores: _FakeTensor,
        k: int,
        *,
        dim: int,
        largest: bool,
        sorted: bool,
    ) -> tuple[_FakeTensor, _FakeTensor]:
        """Return deliberately reversed exact top-k candidates."""
        assert dim == 1
        assert largest
        assert not sorted
        order = np.argsort(-scores.values, axis=1, kind="stable")[:, :k][:, ::-1]
        selected = np.take_along_axis(scores.values, order, axis=1)
        return _FakeTensor(selected), _FakeTensor(order)


def _assert_matches_reference(searcher: Any, corpus: np.ndarray, queries: np.ndarray, k: int = 2) -> None:
    """Compare an adapter against the trusted scalar reference."""
    expected = reference_search(corpus, queries, k, objective=searcher.objective)
    result = searcher.search(queries, k)
    np.testing.assert_array_equal(result.indices, expected.indices)
    np.testing.assert_allclose(result.scores, expected.scores, rtol=3e-5, atol=3e-5)


def test_sklearn_and_scipy_adapters_match_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    """Brute force and both tree families preserve their native objective semantics."""
    sklearn_module = SimpleNamespace(
        NearestNeighbors=_FakeNearestNeighbors,
        KDTree=_FakeTree,
        BallTree=_FakeTree,
    )
    monkeypatch.setattr(sklearn_search, "import_optional", lambda *_args, **_kwargs: sklearn_module)
    monkeypatch.setattr(
        scipy_search,
        "import_optional",
        lambda *_args, **_kwargs: SimpleNamespace(cKDTree=_FakeTree),
    )
    raw_corpus = np.asarray([[2.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    raw_queries = np.asarray([[1.0, 0.2]], dtype=np.float32)

    for objective in (SearchObjective.SQUARED_L2, SearchObjective.NORMALIZED_COSINE):
        corpus = normalize_rows(raw_corpus) if objective.requires_normalization else raw_corpus
        queries = normalize_rows(raw_queries) if objective.requires_normalization else raw_queries
        searchers = (
            SklearnBruteSearcher(corpus, objective=objective),
            SklearnKDTreeSearcher(corpus, objective=objective, leaf_size=2),
            SklearnBallTreeSearcher(corpus, objective=objective, leaf_size=2),
            ScipyCKDTreeSearcher(corpus, objective=objective, leaf_size=2),
        )
        for searcher in searchers:
            _assert_matches_reference(searcher, corpus, queries, k=1)
        assert searchers[1].leaf_size == 2
        assert searchers[2].leaf_size == 2
        assert searchers[3].leaf_size == 2


def test_faiss_adapters_match_reference_and_validate_capabilities(monkeypatch: pytest.MonkeyPatch) -> None:
    """Faiss flat indexes expose only their exact native objective pairings."""
    module = SimpleNamespace(
        IndexFlatL2=_FakeFaissL2,
        IndexFlatIP=_FakeFaissIP,
        omp_set_num_threads=lambda _threads: None,
    )
    monkeypatch.setattr(faiss_search, "import_optional", lambda *_args, **_kwargs: module)
    corpus = np.asarray([[2.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    queries = np.asarray([[1.0, 0.2]], dtype=np.float32)

    _assert_matches_reference(FaissFlatL2Searcher(corpus), corpus, queries)
    _assert_matches_reference(FaissFlatIPSearcher(corpus), corpus, queries)
    normalized_corpus = normalize_rows(corpus)
    normalized_queries = normalize_rows(queries)
    _assert_matches_reference(
        FaissFlatIPSearcher(normalized_corpus, objective=SearchObjective.NORMALIZED_COSINE),
        normalized_corpus,
        normalized_queries,
    )
    with pytest.raises(UnsupportedObjectiveError, match="supports only"):
        _ = FaissFlatIPSearcher(corpus, objective=SearchObjective.SQUARED_L2)
    with pytest.raises(ValueError, match="float32"):
        _ = FaissFlatL2Searcher(corpus.astype(np.float64))


@pytest.mark.parametrize("objective", list(SearchObjective))
def test_torch_adapter_matches_reference(monkeypatch: pytest.MonkeyPatch, objective: SearchObjective) -> None:
    """PyTorch query tensor preparation remains outside the timed search path."""
    monkeypatch.setattr(torch_search, "import_optional", lambda *_args, **_kwargs: _FakeTorch())
    corpus = np.asarray([[2.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    queries = np.asarray([[1.0, 0.2]], dtype=np.float32)
    if objective.requires_normalization:
        corpus = normalize_rows(corpus)
        queries = normalize_rows(queries)
    searcher = TorchTopKSearcher(corpus, objective=objective)

    _assert_matches_reference(searcher, corpus, queries)
    prepared = searcher.prepare_queries(queries)
    assert prepared.backend_name == "torch_cpu"
    assert prepared.backend_payload is not None
    with pytest.raises(ValueError, match="prepared by this PyTorch"):
        _ = searcher.search_prepared(
            type(prepared)(prepared.values, objective=objective),
            1,
        )


def test_unsupported_tree_objectives_fail_before_import() -> None:
    """Tree adapters reject inner product instead of silently changing its meaning."""
    corpus = np.asarray([[1.0, 0.0]], dtype=np.float32)

    for searcher_type in (SklearnBruteSearcher, SklearnKDTreeSearcher, SklearnBallTreeSearcher, ScipyCKDTreeSearcher):
        with pytest.raises(UnsupportedObjectiveError, match="does not support"):
            _ = searcher_type(corpus, objective=SearchObjective.INNER_PRODUCT)


def test_optional_import_failure_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing heavy dependencies produce a domain-specific installation hint."""

    def fail(_module_name: str) -> None:
        raise ImportError("missing")

    monkeypatch.setattr(_backend_utils, "import_module", fail)
    with pytest.raises(BackendUnavailableError, match="benchmark-backends"):
        _ = _backend_utils.import_optional("missing_backend", extra="benchmark-backends")
