"""Correctness-guarded common-identity harness for paired experiments."""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Final

from benchmatrix import BenchmarkCase, BenchmarkConfig, BenchmarkHookContext, make_benchmark_test

from benchmarks.confirmatory_experiments import ConfirmatoryExperiment, get_experiment
from vector_search_study import PreparedQueries, SearchResult
from vector_search_study._benchmark_oracle import (
    BenchmarkReference,
    build_benchmark_reference,
    validate_benchmark_result,
)
from vector_search_study.api import ExactSearcher
from vector_search_study.benchmarking import IMPLEMENTATIONS, WorkloadSpec
from vector_search_study.synthetic import SyntheticDataset

COMMON_IMPLEMENTATION_NAME: Final = "algorithm_under_test"
_EXPERIMENT_ENV: Final = "VECTOR_SEARCH_CONFIRMATORY_EXPERIMENT"
_VARIANT_ENV: Final = "VECTOR_SEARCH_CONFIRMATORY_VARIANT"


@dataclass(frozen=True, slots=True)
class _Resource:
    """One untimed confirmatory workload and trusted result."""

    dataset: SyntheticDataset
    reference: BenchmarkReference


@dataclass(frozen=True, slots=True)
class _Active:
    """Backend state prepared outside timing."""

    searcher: ExactSearcher
    queries: PreparedQueries
    expected: SearchResult
    k: int
    dtype: str


_ACTIVE: dict[str, _Active] = {}
_CACHE: dict[WorkloadSpec, _Resource] = {}
_REFERENCE_METADATA: dict[str, dict[str, object]] = {}


def install_confirmatory_test(namespace: MutableMapping[str, object]) -> None:
    """Install the selected variant behind one common benchmark identity."""
    experiment, _implementation = selected_variant()
    function = make_benchmark_test(
        {COMMON_IMPLEMENTATION_NAME: _target},
        [_case(experiment, workload) for workload in experiment.workloads],
        metrics=("single_call_latency",),
        config=_config(),
    )
    function.__name__ = "test_confirmatory_matrix"
    function.__qualname__ = function.__name__
    namespace[function.__name__] = function


def selected_variant() -> tuple[ConfirmatoryExperiment, str]:
    """Resolve one baseline or candidate from explicit environment metadata."""
    experiment_name = os.environ.get(_EXPERIMENT_ENV)
    variant = os.environ.get(_VARIANT_ENV)
    if experiment_name is None or variant is None:
        raise ValueError(f"{_EXPERIMENT_ENV} and {_VARIANT_ENV} must be set together")
    experiment = get_experiment(experiment_name)
    if variant == "baseline":
        return experiment, experiment.baseline
    if variant == "candidate":
        return experiment, experiment.candidate
    raise ValueError(f"{_VARIANT_ENV} must be baseline or candidate")


def annotate_reference_metadata(output_json: object) -> None:
    """Inject oracle provenance into confirmatory raw JSON records."""
    if not isinstance(output_json, dict):
        return
    benchmarks = output_json.get("benchmarks")
    if not isinstance(benchmarks, list):
        return
    for entry in benchmarks:
        if not isinstance(entry, dict):
            continue
        extra = entry.get("extra_info")
        if not isinstance(extra, dict):
            continue
        case_name = extra.get("case_name")
        if not isinstance(case_name, str) or case_name not in _REFERENCE_METADATA:
            continue
        for key, value in _REFERENCE_METADATA[case_name].items():
            extra[f"case_{key}"] = value
    if _REFERENCE_METADATA:
        output_json["vector_search_reference_records"] = [
            {"case_name": name, **metadata} for name, metadata in sorted(_REFERENCE_METADATA.items())
        ]


def reset_confirmatory_runtime_state() -> None:
    """Release global backend and dataset state."""
    _ACTIVE.clear()
    _CACHE.clear()
    _REFERENCE_METADATA.clear()


def _target(workload: WorkloadSpec) -> SearchResult:
    """Run the selected algorithm through a common timed identity."""
    active = _ACTIVE[workload.name]
    return active.searcher.search_prepared(active.queries, active.k)


def _before_benchmark(context: BenchmarkHookContext) -> None:
    """Build data, oracle, index, and prepared queries outside timing."""
    workload = _context_workload(context)
    resource = _CACHE.get(workload)
    if resource is None:
        dataset = workload.make_dataset()
        resource = _Resource(
            dataset=dataset,
            reference=build_benchmark_reference(
                dataset.corpus,
                dataset.queries,
                workload.k,
                objective=workload.objective,
            ),
        )
        _CACHE[workload] = resource
    _experiment, implementation_name = selected_variant()
    searcher = IMPLEMENTATIONS[implementation_name].build(resource.dataset.corpus, workload.objective)
    _ACTIVE[workload.name] = _Active(
        searcher=searcher,
        queries=searcher.prepare_queries(resource.dataset.queries),
        expected=resource.reference.result,
        k=workload.k,
        dtype=workload.dtype,
    )
    _REFERENCE_METADATA[workload.name] = resource.reference.metadata()


def _validate_result(context: BenchmarkHookContext, result: object) -> None:
    """Require exact selected identities and tolerance-bounded scores."""
    workload = _context_workload(context)
    if not isinstance(result, SearchResult):
        raise TypeError("confirmatory target returned an invalid result type")
    active = _ACTIVE[workload.name]
    validate_benchmark_result(result, active.expected, dtype=active.dtype)


def _after_benchmark(context: BenchmarkHookContext) -> None:
    """Release the active backend index after each cell."""
    _ACTIVE.pop(_context_workload(context).name, None)


def _context_workload(context: BenchmarkHookContext) -> WorkloadSpec:
    """Return the workload carried by a benchmark case."""
    args, _kwargs = context.case.make_call()
    workload = args[0]
    if not isinstance(workload, WorkloadSpec):
        raise TypeError("confirmatory benchmark case must contain a WorkloadSpec")
    return workload


def _case(experiment: ConfirmatoryExperiment, workload: WorkloadSpec) -> BenchmarkCase:
    """Build variant-neutral case metadata for strict paired compatibility."""
    return BenchmarkCase.from_values(
        workload.name,
        workload,
        work_units=workload.query_count,
        work_unit_name="queries",
        metadata={
            **workload.metadata(),
            "confirmatory_experiment": experiment.identifier,
            "confirmatory_question": experiment.question,
            "common_benchmark_identity": COMMON_IMPLEMENTATION_NAME,
            "equivalence_margin_percent": experiment.equivalence_margin_percent,
            "precision_target_half_width_percent": experiment.precision_target_half_width_percent,
            "reference_policy": "scalar_then_blocked_float64_v1",
        },
    )


def _config() -> BenchmarkConfig:
    """Build deterministic central-latency timing controls."""
    return BenchmarkConfig(
        pedantic_rounds=_environment_int("VECTOR_SEARCH_BENCHMARK_ROUNDS", 10, positive=True),
        warmup_rounds=_environment_int("VECTOR_SEARCH_BENCHMARK_WARMUP_ROUNDS", 2, positive=False),
        pedantic_iterations=1,
        stream_progress=False,
        before_benchmark=_before_benchmark,
        validate_result=_validate_result,
        after_benchmark=_after_benchmark,
    )


def _environment_int(name: str, default: int, *, positive: bool) -> int:
    """Read one bounded integer environment control."""
    raw = os.environ.get(name)
    value = default if raw is None else int(raw)
    if value < (1 if positive else 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'non-negative'}")
    return value
