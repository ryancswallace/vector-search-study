"""Correctness checks against real installed optional backend packages."""

from __future__ import annotations

from collections.abc import Callable
from importlib.util import find_spec

import numpy as np
import pytest

from vector_search_study import (
    FaissFlatIPSearcher,
    FaissFlatL2Searcher,
    ScipyCKDTreeSearcher,
    SearchObjective,
    SklearnBallTreeSearcher,
    SklearnBruteSearcher,
    SklearnKDTreeSearcher,
    TorchTopKSearcher,
    make_gaussian_dataset,
    make_uniform_sphere_dataset,
    reference_search,
)
from vector_search_study.api import ExactSearcher

BackendFactory = Callable[[np.ndarray, SearchObjective], ExactSearcher]


def _sklearn_brute(corpus: np.ndarray, objective: SearchObjective) -> ExactSearcher:
    """Build sklearn brute for an integration parameter."""
    return SklearnBruteSearcher(corpus, objective=objective)


def _sklearn_kdtree(corpus: np.ndarray, objective: SearchObjective) -> ExactSearcher:
    """Build sklearn KDTree for an integration parameter."""
    return SklearnKDTreeSearcher(corpus, objective=objective, leaf_size=4)


def _sklearn_balltree(corpus: np.ndarray, objective: SearchObjective) -> ExactSearcher:
    """Build sklearn BallTree for an integration parameter."""
    return SklearnBallTreeSearcher(corpus, objective=objective, leaf_size=4)


def _scipy_ckdtree(corpus: np.ndarray, objective: SearchObjective) -> ExactSearcher:
    """Build SciPy cKDTree for an integration parameter."""
    return ScipyCKDTreeSearcher(corpus, objective=objective, leaf_size=4)


def _faiss_l2(corpus: np.ndarray, _objective: SearchObjective) -> ExactSearcher:
    """Build Faiss Flat L2 for an integration parameter."""
    return FaissFlatL2Searcher(corpus)


def _faiss_ip(corpus: np.ndarray, objective: SearchObjective) -> ExactSearcher:
    """Build Faiss Flat IP for an integration parameter."""
    return FaissFlatIPSearcher(corpus, objective=objective)


def _torch(corpus: np.ndarray, objective: SearchObjective) -> ExactSearcher:
    """Build CPU PyTorch for an integration parameter."""
    return TorchTopKSearcher(corpus, objective=objective)


_BACKENDS: tuple[tuple[str, str, BackendFactory, tuple[SearchObjective, ...]], ...] = (
    (
        "sklearn-brute",
        "sklearn",
        _sklearn_brute,
        (SearchObjective.SQUARED_L2, SearchObjective.NORMALIZED_COSINE),
    ),
    (
        "sklearn-kdtree",
        "sklearn",
        _sklearn_kdtree,
        (SearchObjective.SQUARED_L2, SearchObjective.NORMALIZED_COSINE),
    ),
    (
        "sklearn-balltree",
        "sklearn",
        _sklearn_balltree,
        (SearchObjective.SQUARED_L2, SearchObjective.NORMALIZED_COSINE),
    ),
    (
        "scipy-ckdtree",
        "scipy",
        _scipy_ckdtree,
        (SearchObjective.SQUARED_L2, SearchObjective.NORMALIZED_COSINE),
    ),
    ("faiss-flat-l2", "faiss", _faiss_l2, (SearchObjective.SQUARED_L2,)),
    (
        "faiss-flat-ip",
        "faiss",
        _faiss_ip,
        (SearchObjective.INNER_PRODUCT, SearchObjective.NORMALIZED_COSINE),
    ),
    ("torch-topk", "torch", _torch, tuple(SearchObjective)),
)


@pytest.mark.integration
@pytest.mark.parametrize(("_name", "module", "factory", "objectives"), _BACKENDS, ids=[row[0] for row in _BACKENDS])
def test_installed_backend_matches_reference(
    _name: str,
    module: str,
    factory: BackendFactory,
    objectives: tuple[SearchObjective, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real optional packages agree on ordered identities and tolerant scores."""
    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "1")
    if find_spec(module) is None:
        pytest.skip(f"optional module {module} is not installed")
    for objective in objectives:
        if objective.requires_normalization:
            dataset = make_uniform_sphere_dataset(31, 7, 3, seed=91)
        else:
            dataset = make_gaussian_dataset(31, 7, 3, objective=objective, seed=91)
        expected = reference_search(dataset.corpus, dataset.queries, 5, objective=objective)
        searcher = factory(dataset.corpus, objective)

        result = searcher.search(dataset.queries, 5)

        np.testing.assert_array_equal(result.indices, expected.indices)
        np.testing.assert_allclose(result.scores, expected.scores, rtol=3e-4, atol=3e-5)


@pytest.mark.integration
@pytest.mark.parametrize(("_name", "module", "factory", "objectives"), _BACKENDS, ids=[row[0] for row in _BACKENDS])
def test_installed_backend_canonicalizes_ties(
    _name: str,
    module: str,
    factory: BackendFactory,
    objectives: tuple[SearchObjective, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Returned ties are ordered, while native boundary-tie choices stay equivalent."""
    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "1")
    if find_spec(module) is None:
        pytest.skip(f"optional module {module} is not installed")
    objective = objectives[0]
    corpus = np.asarray([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
    queries = np.asarray([[1.0, 0.0]], dtype=np.float32)
    searcher = factory(corpus, objective)

    both = searcher.search(queries, 2)
    boundary = searcher.search(queries, 1)

    np.testing.assert_array_equal(both.indices, [[0, 1]])
    assert int(boundary.indices[0, 0]) in {0, 1}
    np.testing.assert_allclose(both.scores, [[both.scores[0, 0], both.scores[0, 0]]], rtol=3e-4, atol=3e-5)
