"""Tiny benchmatrix discovery matrix that validates the full benchmark path.

Run with ``make benchmark-smoke``. This is intentionally outside the default
unit-test path and does not constitute a performance result.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib.util import find_spec

import numpy as np
from benchmatrix import BenchmarkCase, BenchmarkConfig, BenchmarkHookContext, make_benchmark_test

from vector_search_study import PreparedQueries, SearchResult, reference_search
from vector_search_study.api import ExactSearcher, SearchObjective
from vector_search_study.benchmarking import IMPLEMENTATIONS, WorkloadSpec, smoke_specs


@dataclass(frozen=True, slots=True)
class _ActiveCell:
    """Resources prepared outside a timed benchmark invocation."""

    searcher: ExactSearcher
    queries: PreparedQueries
    expected: SearchResult
    k: int


_ACTIVE: dict[tuple[str, str], _ActiveCell] = {}


def _target(implementation_name: str) -> Callable[[WorkloadSpec], SearchResult]:
    """Bind one implementation name to the common timed operation."""

    def search(spec: WorkloadSpec) -> SearchResult:
        active = _ACTIVE[(implementation_name, spec.name)]
        return active.searcher.search_prepared(active.queries, active.k)

    return search


def _before_benchmark(context: BenchmarkHookContext) -> None:
    """Build data, reference results, index, and prepared queries untimed."""
    args, _ = context.case.make_call()
    spec = args[0]
    if not isinstance(spec, WorkloadSpec):
        raise TypeError("benchmark case must contain a WorkloadSpec")
    dataset = spec.make_dataset()
    reference = reference_search(dataset.corpus, dataset.queries, spec.k + 1, objective=spec.objective)
    minimum_margin = float(np.min(reference.scores[:, spec.k - 1] - reference.scores[:, spec.k]))
    if minimum_margin <= 1e-6:
        raise AssertionError(f"{spec.name} lacks a strict top-k boundary margin: {minimum_margin}")
    expected = SearchResult(
        indices=np.array(reference.indices[:, : spec.k], dtype=np.int64, order="C", copy=True),
        scores=np.array(reference.scores[:, : spec.k], dtype=np.float64, order="C", copy=True),
    )
    searcher = IMPLEMENTATIONS[context.implementation_name].build(dataset.corpus, spec.objective)
    prepared = searcher.prepare_queries(dataset.queries)
    _ACTIVE[(context.implementation_name, spec.name)] = _ActiveCell(searcher, prepared, expected, spec.k)


def _validate_result(context: BenchmarkHookContext, result: object) -> None:
    """Reject a measured result that differs from the trusted reference."""
    args, _ = context.case.make_call()
    spec = args[0]
    if not isinstance(spec, WorkloadSpec) or not isinstance(result, SearchResult):
        raise TypeError("benchmark target returned an invalid result type")
    expected = _ACTIVE[(context.implementation_name, spec.name)].expected
    np.testing.assert_array_equal(result.indices, expected.indices)
    np.testing.assert_allclose(result.scores, expected.scores, rtol=2e-4, atol=2e-5)


def _after_benchmark(context: BenchmarkHookContext) -> None:
    """Release the per-cell index and corpus after validation."""
    args, _ = context.case.make_call()
    spec = args[0]
    if isinstance(spec, WorkloadSpec):
        _ACTIVE.pop((context.implementation_name, spec.name), None)


def _backend_is_installed(implementation_name: str) -> bool:
    """Return whether an implementation's optional import is available."""
    dependency = IMPLEMENTATIONS[implementation_name].optional_dependency
    module = {"scikit-learn": "sklearn", "faiss-cpu": "faiss"}.get(dependency, dependency)
    return module is None or find_spec(module) is not None


def _case(spec: WorkloadSpec) -> BenchmarkCase:
    """Convert a semantic workload into strict benchmatrix metadata."""
    return BenchmarkCase.from_values(
        spec.name,
        spec,
        work_units=spec.query_count,
        work_unit_name="queries",
        metadata=spec.metadata(),
    )


def _implementations(objective: SearchObjective) -> dict[str, Callable[[WorkloadSpec], SearchResult]]:
    """Return the installed native capability slice for one objective."""
    return {
        name: _target(name)
        for name, implementation in IMPLEMENTATIONS.items()
        if objective in implementation.objectives and _backend_is_installed(name)
    }


CONFIG = BenchmarkConfig(
    pedantic_rounds=10,
    warmup_rounds=2,
    pedantic_iterations=1,
    before_benchmark=_before_benchmark,
    validate_result=_validate_result,
    after_benchmark=_after_benchmark,
)
_SMOKE = {spec.objective: spec for spec in smoke_specs()}
_METRICS = ("single_call_latency", "batch_throughput", "tail_latency")

test_squared_l2_smoke = make_benchmark_test(
    _implementations(SearchObjective.SQUARED_L2),
    [_case(_SMOKE[SearchObjective.SQUARED_L2])],
    metrics=_METRICS,
    config=CONFIG,
)
test_inner_product_smoke = make_benchmark_test(
    _implementations(SearchObjective.INNER_PRODUCT),
    [_case(_SMOKE[SearchObjective.INNER_PRODUCT])],
    metrics=_METRICS,
    config=CONFIG,
)
test_normalized_cosine_smoke = make_benchmark_test(
    _implementations(SearchObjective.NORMALIZED_COSINE),
    [_case(_SMOKE[SearchObjective.NORMALIZED_COSINE])],
    metrics=_METRICS,
    config=CONFIG,
)
