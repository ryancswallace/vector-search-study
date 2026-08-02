"""Shared correctness-guarded benchmatrix discovery lifecycle."""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable, MutableMapping
from dataclasses import dataclass
from typing import cast

from benchmatrix import BenchmarkCase, BenchmarkConfig, BenchmarkHookContext, MetricName, make_benchmark_test

from vector_search_study import PreparedQueries, SearchResult
from vector_search_study._benchmark_oracle import (
    DEFAULT_SCALAR_REFERENCE_LIMIT,
    BenchmarkReference,
    build_benchmark_reference,
    validate_benchmark_result,
)
from vector_search_study.api import ExactSearcher
from vector_search_study.benchmarking import (
    DEFAULT_MEMORY_BUDGET_BYTES,
    IMPLEMENTATIONS,
    ImplementationSpec,
    WorkloadSpec,
    assess_feasibility,
    dependency_is_installed,
    selected_metrics,
)
from vector_search_study.synthetic import SyntheticDataset

_TEST_NAME_PATTERN = re.compile(r"[^a-zA-Z0-9_]+")


@dataclass(frozen=True, slots=True)
class _CachedWorkload:
    """One generated workload and its untimed expected result."""

    dataset: SyntheticDataset
    reference: BenchmarkReference


@dataclass(frozen=True, slots=True)
class _ActiveCell:
    """Backend resources prepared outside one timed invocation."""

    searcher: ExactSearcher
    queries: PreparedQueries
    expected: SearchResult
    k: int
    dtype: str


class _SingleWorkloadCache:
    """Retain only the current workload across implementations and metrics."""

    def __init__(self) -> None:
        self._spec: WorkloadSpec | None = None
        self._resource: _CachedWorkload | None = None
        self.hits = 0
        self.misses = 0

    def get(self, spec: WorkloadSpec) -> _CachedWorkload:
        """Return cached data or materialize one bounded workload."""
        if self._spec == spec and self._resource is not None:
            self.hits += 1
            return self._resource
        dataset = spec.make_dataset()
        reference = build_benchmark_reference(
            dataset.corpus,
            dataset.queries,
            spec.k,
            objective=spec.objective,
        )
        self._spec = spec
        self._resource = _CachedWorkload(dataset=dataset, reference=reference)
        self.misses += 1
        return self._resource

    def clear(self) -> None:
        """Release retained arrays and reset cache counters."""
        self._spec = None
        self._resource = None
        self.hits = 0
        self.misses = 0


_ACTIVE: dict[tuple[str, str], _ActiveCell] = {}
_CACHE = _SingleWorkloadCache()
_REFERENCE_METADATA: dict[str, dict[str, object]] = {}


def discovery_config() -> BenchmarkConfig:
    """Build deterministic timing controls from validated environment values."""
    return BenchmarkConfig(
        pedantic_rounds=_environment_nonnegative_int("VECTOR_SEARCH_BENCHMARK_ROUNDS", default=10, positive=True),
        warmup_rounds=_environment_nonnegative_int("VECTOR_SEARCH_BENCHMARK_WARMUP_ROUNDS", default=2),
        pedantic_iterations=1,
        stream_progress=False,
        before_benchmark=_before_benchmark,
        validate_result=_validate_result,
        after_benchmark=_after_benchmark,
    )


def tail_discovery_config() -> BenchmarkConfig:
    """Build timing controls that satisfy the predeclared tail-sample policy."""
    return BenchmarkConfig(
        pedantic_rounds=_environment_nonnegative_int(
            "VECTOR_SEARCH_BENCHMARK_TAIL_ROUNDS",
            default=100,
            positive=True,
        ),
        warmup_rounds=_environment_nonnegative_int("VECTOR_SEARCH_BENCHMARK_WARMUP_ROUNDS", default=2),
        pedantic_iterations=1,
        stream_progress=False,
        before_benchmark=_before_benchmark,
        validate_result=_validate_result,
        after_benchmark=_after_benchmark,
    )


def install_discovery_tests(
    namespace: MutableMapping[str, object],
    specs: Iterable[WorkloadSpec],
    *,
    config: BenchmarkConfig | None = None,
    memory_budget_bytes: int = DEFAULT_MEMORY_BUDGET_BYTES,
) -> None:
    """Install one bounded benchmark matrix per workload into a module."""
    resolved_config = discovery_config() if config is None else config
    for spec in specs:
        implementations = _implementations(spec, memory_budget_bytes=memory_budget_bytes)
        if not implementations:
            continue
        metrics = _metrics(spec)
        central_metrics = tuple(metric for metric in metrics if metric != "tail_latency")
        if central_metrics:
            _install_test(
                namespace,
                spec,
                implementations,
                central_metrics,
                resolved_config,
                suffix="central",
            )
        if "tail_latency" in metrics:
            _install_test(
                namespace,
                spec,
                implementations,
                (cast(MetricName, "tail_latency"),),
                resolved_config if config is not None else tail_discovery_config(),
                suffix="tail",
            )


def annotate_reference_metadata(output_json: object) -> None:
    """Inject oracle provenance into raw pytest-benchmark JSON records."""
    if not isinstance(output_json, dict):
        return
    benchmarks = output_json.get("benchmarks")
    if not isinstance(benchmarks, list):
        return
    for entry in benchmarks:
        if not isinstance(entry, dict):
            continue
        extra_info = entry.get("extra_info")
        if not isinstance(extra_info, dict):
            continue
        case_name = extra_info.get("case_name")
        if not isinstance(case_name, str):
            continue
        metadata = _REFERENCE_METADATA.get(case_name)
        if metadata is None:
            continue
        for key, value in metadata.items():
            extra_info[f"case_{key}"] = value
    output_json["vector_search_reference_records"] = [
        {"case_name": case_name, **metadata} for case_name, metadata in sorted(_REFERENCE_METADATA.items())
    ]


def benchmark_runtime_state() -> dict[str, object]:
    """Return cache and reference state for lifecycle tests."""
    return {
        "cache_hits": _CACHE.hits,
        "cache_misses": _CACHE.misses,
        "active_cells": len(_ACTIVE),
        "reference_cases": sorted(_REFERENCE_METADATA),
    }


def reset_benchmark_runtime_state() -> None:
    """Release all global benchmark lifecycle state."""
    _ACTIVE.clear()
    _CACHE.clear()
    _REFERENCE_METADATA.clear()


def _install_test(
    namespace: MutableMapping[str, object],
    spec: WorkloadSpec,
    implementations: dict[str, Callable[[WorkloadSpec], SearchResult]],
    metrics: tuple[MetricName, ...],
    config: BenchmarkConfig,
    *,
    suffix: str,
) -> None:
    """Install one metric-specific matrix for a workload."""
    function = make_benchmark_test(
        implementations,
        [_case(spec)],
        metrics=metrics,
        config=config,
    )
    test_name = f"test_{_TEST_NAME_PATTERN.sub('_', spec.name)}_{suffix}"
    function.__name__ = test_name
    function.__qualname__ = test_name
    namespace[test_name] = function


def _target(implementation_name: str) -> Callable[[WorkloadSpec], SearchResult]:
    """Bind one implementation to the common timed search identity."""

    def search(spec: WorkloadSpec) -> SearchResult:
        active = _ACTIVE[(implementation_name, spec.name)]
        return active.searcher.search_prepared(active.queries, active.k)

    return search


def _before_benchmark(context: BenchmarkHookContext) -> None:
    """Build the oracle, index, and backend queries outside timing."""
    spec = _context_spec(context)
    resource = _CACHE.get(spec)
    implementation = IMPLEMENTATIONS[context.implementation_name]
    searcher = implementation.build(resource.dataset.corpus, spec.objective)
    prepared = searcher.prepare_queries(resource.dataset.queries)
    _ACTIVE[(context.implementation_name, spec.name)] = _ActiveCell(
        searcher=searcher,
        queries=prepared,
        expected=resource.reference.result,
        k=spec.k,
        dtype=spec.dtype,
    )
    _REFERENCE_METADATA[spec.name] = resource.reference.metadata()


def _validate_result(context: BenchmarkHookContext, result: object) -> None:
    """Reject any measured result that differs from the untimed oracle."""
    spec = _context_spec(context)
    if not isinstance(result, SearchResult):
        raise TypeError("benchmark target returned an invalid result type")
    active = _ACTIVE[(context.implementation_name, spec.name)]
    validate_benchmark_result(result, active.expected, dtype=active.dtype)


def _after_benchmark(context: BenchmarkHookContext) -> None:
    """Release the implementation index while retaining one workload."""
    spec = _context_spec(context)
    _ACTIVE.pop((context.implementation_name, spec.name), None)


def _context_spec(context: BenchmarkHookContext) -> WorkloadSpec:
    """Return the workload carried by one benchmark case."""
    args, _ = context.case.make_call()
    spec = args[0]
    if not isinstance(spec, WorkloadSpec):
        raise TypeError("benchmark case must contain a WorkloadSpec")
    return spec


def _case(spec: WorkloadSpec) -> BenchmarkCase:
    """Convert a workload to a strict benchmark case."""
    metadata = {
        **spec.metadata(),
        "reference_policy": "scalar_then_blocked_float64_v1",
        "reference_scalar_coordinate_limit": DEFAULT_SCALAR_REFERENCE_LIMIT,
        "selected_metrics": list(selected_metrics(spec)),
    }
    return BenchmarkCase.from_values(
        spec.name,
        spec,
        work_units=spec.query_count,
        work_unit_name="queries",
        metadata=metadata,
    )


def _implementations(
    spec: WorkloadSpec,
    *,
    memory_budget_bytes: int,
) -> dict[str, Callable[[WorkloadSpec], SearchResult]]:
    """Return installed implementations admitted by deterministic policy."""
    return {
        name: _target(name)
        for name, implementation in IMPLEMENTATIONS.items()
        if _implementation_is_included(
            spec,
            implementation,
            memory_budget_bytes=memory_budget_bytes,
        )
    }


def _implementation_is_included(
    spec: WorkloadSpec,
    implementation: ImplementationSpec,
    *,
    memory_budget_bytes: int,
) -> bool:
    """Return whether one implementation/workload cell can run here."""
    decision = assess_feasibility(spec, implementation, memory_budget_bytes=memory_budget_bytes)
    return decision.feasible and dependency_is_installed(implementation)


def _metrics(spec: WorkloadSpec) -> tuple[MetricName, ...]:
    """Return selected metric names with benchmatrix's public type."""
    return tuple(cast(MetricName, metric) for metric in selected_metrics(spec))


def _environment_nonnegative_int(name: str, *, default: int, positive: bool = False) -> int:
    """Return a validated integer timing control from the environment."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    minimum = 1 if positive else 0
    if value < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return value
