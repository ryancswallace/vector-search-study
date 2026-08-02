"""Deterministic workload and implementation definitions for benchmark matrices."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from importlib.util import find_spec
from typing import Final

import numpy as np

from vector_search_study._validation import FloatMatrix, resolve_float_dtype, validate_positive_int
from vector_search_study.api import ExactSearcher, SearchObjective, resolve_search_objective
from vector_search_study.faiss_search import FaissFlatIPSearcher, FaissFlatL2Searcher
from vector_search_study.numpy_search import NumpyArgpartitionSearcher, NumpyBlockedSearcher, NumpySortSearcher
from vector_search_study.python_search import PythonHeapSearcher, PythonSortSearcher
from vector_search_study.scipy_search import ScipyCKDTreeSearcher
from vector_search_study.sklearn_search import SklearnBallTreeSearcher, SklearnBruteSearcher, SklearnKDTreeSearcher
from vector_search_study.synthetic import SyntheticDataset, make_gaussian_dataset, make_uniform_sphere_dataset
from vector_search_study.torch_search import TorchTopKSearcher

DIMENSIONS: Final[tuple[int, ...]] = (8, 32, 128, 768)
CORPUS_SIZES: Final[tuple[int, ...]] = (1_000, 10_000, 100_000, 1_000_000)
QUERY_COUNTS: Final[tuple[int, ...]] = (1, 32, 1_024)
K_VALUES: Final[tuple[int, ...]] = (1, 10, 100)
OBJECTIVES: Final[tuple[SearchObjective, ...]] = tuple(SearchObjective)
DEFAULT_SEED: Final = 20_260_801
DEFAULT_MEMORY_BUDGET_BYTES: Final = 8 * 1_024**3
DISCOVERY_METRICS: Final[tuple[str, ...]] = ("single_call_latency", "batch_throughput")
TAIL_LATENCY_METRIC: Final = "tail_latency"
_SCHEMA_VERSION: Final = 1


@dataclass(frozen=True, slots=True)
class WorkloadSpec:
    """One deterministic semantic exact-search workload."""

    objective: SearchObjective
    corpus_size: int
    dimension: int
    query_count: int
    k: int
    dtype: str = "float32"
    seed: int = DEFAULT_SEED
    profile: str = "discovery-core"

    def __post_init__(self) -> None:
        """Validate and canonicalize workload fields."""
        object.__setattr__(self, "objective", resolve_search_objective(self.objective))
        object.__setattr__(self, "corpus_size", validate_positive_int(self.corpus_size, name="corpus_size"))
        object.__setattr__(self, "dimension", validate_positive_int(self.dimension, name="dimension"))
        object.__setattr__(self, "query_count", validate_positive_int(self.query_count, name="query_count"))
        resolved_k = validate_positive_int(self.k, name="k")
        if resolved_k > self.corpus_size:
            raise ValueError("k must not exceed corpus_size")
        object.__setattr__(self, "k", resolved_k)
        object.__setattr__(self, "dtype", resolve_float_dtype(self.dtype).name)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not self.profile:
            raise ValueError("profile must be non-empty")

    @property
    def name(self) -> str:
        """Return a stable human-readable case name."""
        return (
            f"{self.profile}__{self.objective.value}__n{self.corpus_size}__d{self.dimension}"
            f"__q{self.query_count}__k{self.k}__{self.dtype}"
        )

    @property
    def coordinate_evaluations(self) -> int:
        """Return scalar query-corpus coordinate evaluations per search."""
        return self.corpus_size * self.dimension * self.query_count

    def metadata(self) -> dict[str, object]:
        """Return strict-JSON-safe deterministic benchmark metadata."""
        distribution = "uniform_sphere" if self.objective.requires_normalization else "gaussian"
        return {
            "schema_version": _SCHEMA_VERSION,
            "study": "exact_top_k_vector_search",
            "profile": self.profile,
            "objective": self.objective.value,
            "score_convention": _score_convention(self.objective),
            "corpus_size": self.corpus_size,
            "dimension": self.dimension,
            "query_count": self.query_count,
            "k": self.k,
            "dtype": self.dtype,
            "normalization": "l2_unit_rows" if self.objective.requires_normalization else "none",
            "dataset_family": "synthetic",
            "distribution": distribution,
            "generator": "numpy.random.PCG64",
            "generator_revision": 1,
            "seed": self.seed,
            "dataset_id": (
                f"synthetic-pcg64-v1:{distribution}:{self.corpus_size}:{self.dimension}:"
                f"{self.query_count}:{self.dtype}:{self.seed}"
            ),
            "boundary_policy": "strict_top_k_margin",
            "threads": 1,
        }

    def make_dataset(self) -> SyntheticDataset:
        """Materialize the deterministic corpus and queries."""
        if self.objective.requires_normalization:
            return make_uniform_sphere_dataset(
                self.corpus_size,
                self.dimension,
                self.query_count,
                dtype=self.dtype,
                seed=self.seed,
            )
        return make_gaussian_dataset(
            self.corpus_size,
            self.dimension,
            self.query_count,
            objective=self.objective,
            dtype=self.dtype,
            seed=self.seed,
        )


@dataclass(frozen=True, slots=True)
class NaturalDatasetSpec:
    """Pinned manifest for the approved natural-embedding slice."""

    dataset: str = "BeIR/scifact"
    dataset_revision: str = "a75ae049398addde9b70f6b268875f5cbce99089"  # pragma: allowlist secret
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_revision: str = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"  # pragma: allowlist secret
    dimension: int = 384
    corpus_size: int = 5_183
    query_count: int = 1_109

    def metadata(self) -> dict[str, object]:
        """Return the immutable natural-data provenance manifest."""
        return {
            "schema_version": _SCHEMA_VERSION,
            "dataset_family": "natural",
            "dataset": self.dataset,
            "dataset_revision": self.dataset_revision,
            "model": self.model,
            "model_revision": self.model_revision,
            "dimension": self.dimension,
            "corpus_size": self.corpus_size,
            "query_count": self.query_count,
            "dtype": "float32",
            "normalization": "l2_unit_rows",
        }


SearcherBuilder = Callable[[FloatMatrix, SearchObjective], ExactSearcher]


@dataclass(frozen=True, slots=True)
class ImplementationSpec:
    """A named implementation and its exact native capabilities."""

    name: str
    objectives: frozenset[SearchObjective]
    builder: SearcherBuilder
    optional_dependency: str | None = None
    float32_only: bool = False

    def build(self, corpus: FloatMatrix, objective: SearchObjective) -> ExactSearcher:
        """Build the search index after validating semantic capability."""
        resolved = resolve_search_objective(objective)
        if resolved not in self.objectives:
            raise ValueError(f"{self.name} does not support {resolved.value}")
        if self.float32_only and corpus.dtype != np.dtype(np.float32):
            raise ValueError(f"{self.name} supports only float32")
        return self.builder(corpus, resolved)

    def metadata(self) -> dict[str, object]:
        """Return deterministic implementation metadata."""
        return {
            "implementation": self.name,
            "objectives": sorted(objective.value for objective in self.objectives),
            "optional_dependency": self.optional_dependency,
            "float32_only": self.float32_only,
        }


def _generic_builder(searcher_type: type, **settings: object) -> SearcherBuilder:
    """Create a typed builder for an objective-aware searcher class."""

    def build(corpus: FloatMatrix, objective: SearchObjective) -> ExactSearcher:
        return searcher_type(corpus, objective=objective, **settings)

    return build


def _faiss_l2_builder(corpus: FloatMatrix, _objective: SearchObjective) -> ExactSearcher:
    """Build the objective-specific Faiss L2 adapter."""
    return FaissFlatL2Searcher(corpus)


IMPLEMENTATIONS: Final[Mapping[str, ImplementationSpec]] = {
    "python_sort": ImplementationSpec("python_sort", frozenset(SearchObjective), _generic_builder(PythonSortSearcher)),
    "python_heap": ImplementationSpec("python_heap", frozenset(SearchObjective), _generic_builder(PythonHeapSearcher)),
    "numpy_full": ImplementationSpec("numpy_full", frozenset(SearchObjective), _generic_builder(NumpySortSearcher)),
    "numpy_argpartition": ImplementationSpec(
        "numpy_argpartition", frozenset(SearchObjective), _generic_builder(NumpyArgpartitionSearcher)
    ),
    "numpy_blocked": ImplementationSpec(
        "numpy_blocked", frozenset(SearchObjective), _generic_builder(NumpyBlockedSearcher, block_size=16_384)
    ),
    "sklearn_brute": ImplementationSpec(
        "sklearn_brute",
        frozenset({SearchObjective.SQUARED_L2, SearchObjective.NORMALIZED_COSINE}),
        _generic_builder(SklearnBruteSearcher),
        optional_dependency="scikit-learn",
    ),
    "sklearn_kdtree": ImplementationSpec(
        "sklearn_kdtree",
        frozenset({SearchObjective.SQUARED_L2, SearchObjective.NORMALIZED_COSINE}),
        _generic_builder(SklearnKDTreeSearcher),
        optional_dependency="scikit-learn",
    ),
    "sklearn_balltree": ImplementationSpec(
        "sklearn_balltree",
        frozenset({SearchObjective.SQUARED_L2, SearchObjective.NORMALIZED_COSINE}),
        _generic_builder(SklearnBallTreeSearcher),
        optional_dependency="scikit-learn",
    ),
    "scipy_ckdtree": ImplementationSpec(
        "scipy_ckdtree",
        frozenset({SearchObjective.SQUARED_L2, SearchObjective.NORMALIZED_COSINE}),
        _generic_builder(ScipyCKDTreeSearcher),
        optional_dependency="scipy",
    ),
    "faiss_flat_l2": ImplementationSpec(
        "faiss_flat_l2",
        frozenset({SearchObjective.SQUARED_L2}),
        _faiss_l2_builder,
        optional_dependency="faiss-cpu",
        float32_only=True,
    ),
    "faiss_flat_ip": ImplementationSpec(
        "faiss_flat_ip",
        frozenset({SearchObjective.INNER_PRODUCT, SearchObjective.NORMALIZED_COSINE}),
        _generic_builder(FaissFlatIPSearcher),
        optional_dependency="faiss-cpu",
        float32_only=True,
    ),
    "torch_matmul_topk": ImplementationSpec(
        "torch_matmul_topk",
        frozenset(SearchObjective),
        _generic_builder(TorchTopKSearcher),
        optional_dependency="torch",
    ),
}


@dataclass(frozen=True, slots=True)
class FeasibilityDecision:
    """Deterministic reason for including or excluding a matrix cell."""

    feasible: bool
    reason: str
    estimated_peak_bytes: int

    def metadata(self) -> dict[str, object]:
        """Return strict-JSON-safe inclusion metadata."""
        return {
            "feasible": self.feasible,
            "reason": self.reason,
            "estimated_peak_bytes": self.estimated_peak_bytes,
        }


@dataclass(frozen=True, slots=True)
class DiscoveryCellPlan:
    """One implementation/workload decision in a discovery profile."""

    workload: WorkloadSpec
    implementation: ImplementationSpec
    decision: FeasibilityDecision
    dependency_installed: bool
    metrics: tuple[str, ...]

    @property
    def included(self) -> bool:
        """Return whether this cell can be collected on the current host."""
        return self.decision.feasible and self.dependency_installed

    def metadata(self) -> dict[str, object]:
        """Return strict-JSON-safe plan metadata."""
        reason = self.decision.reason
        if self.decision.feasible and not self.dependency_installed:
            reason = "dependency_not_installed"
        return {
            "workload": self.workload.name,
            "implementation": self.implementation.name,
            "included": self.included,
            "reason": reason,
            "estimated_peak_bytes": self.decision.estimated_peak_bytes,
            "dependency": self.implementation.optional_dependency,
            "dependency_installed": self.dependency_installed,
            "metrics": list(self.metrics),
        }


def assess_feasibility(
    workload: WorkloadSpec,
    implementation: ImplementationSpec,
    *,
    memory_budget_bytes: int = DEFAULT_MEMORY_BUDGET_BYTES,
) -> FeasibilityDecision:
    """Apply semantic, runtime, and conservative memory feasibility rules."""
    budget = validate_positive_int(memory_budget_bytes, name="memory_budget_bytes")
    peak = estimate_peak_bytes(workload, implementation.name)
    if workload.objective not in implementation.objectives:
        return FeasibilityDecision(False, "objective_not_supported", peak)
    if implementation.float32_only and workload.dtype != "float32":
        return FeasibilityDecision(False, "dtype_not_supported", peak)
    if implementation.name.startswith("python_") and workload.coordinate_evaluations > 1_000_000:
        return FeasibilityDecision(False, "scalar_runtime_limit", peak)
    if peak > budget:
        return FeasibilityDecision(False, "memory_budget_exceeded", peak)
    return FeasibilityDecision(True, "included", peak)


def estimate_peak_bytes(workload: WorkloadSpec, implementation_name: str) -> int:
    """Estimate dominant corpus, score, selection, and index allocations."""
    item_size = np.dtype(workload.dtype).itemsize
    corpus = workload.corpus_size * workload.dimension * item_size
    queries = workload.query_count * workload.dimension * item_size
    scores = workload.corpus_size * workload.query_count * item_size
    output = workload.query_count * workload.k * (item_size + np.dtype(np.int64).itemsize)
    cached_dataset_and_index = corpus * 2 + queries + output
    if implementation_name == "numpy_full":
        return cached_dataset_and_index + scores + workload.corpus_size * workload.query_count * 8
    if implementation_name in {"numpy_argpartition", "torch_matmul_topk"}:
        return cached_dataset_and_index + scores + workload.corpus_size * 8
    if implementation_name == "numpy_blocked":
        return cached_dataset_and_index + min(workload.corpus_size, 16_384) * workload.query_count * item_size + output
    if implementation_name.startswith(("sklearn_", "scipy_")):
        return cached_dataset_and_index + corpus * 2
    if implementation_name.startswith("faiss_"):
        return cached_dataset_and_index + corpus
    return cached_dataset_and_index


def discovery_core_specs() -> tuple[WorkloadSpec, ...]:
    """Return the 33-case one-factor-at-a-time discovery core."""
    anchor = (10_000, 128, 32, 10)
    cases: list[WorkloadSpec] = []
    for objective in OBJECTIVES:
        cases.append(WorkloadSpec(objective, *anchor))
        for dimension in DIMENSIONS:
            if dimension != anchor[1]:
                cases.append(WorkloadSpec(objective, anchor[0], dimension, anchor[2], anchor[3]))
        for corpus_size in CORPUS_SIZES:
            if corpus_size != anchor[0]:
                cases.append(WorkloadSpec(objective, corpus_size, anchor[1], anchor[2], anchor[3]))
        for query_count in QUERY_COUNTS:
            if query_count != anchor[2]:
                cases.append(WorkloadSpec(objective, anchor[0], anchor[1], query_count, anchor[3]))
        for k in K_VALUES:
            if k != anchor[3]:
                cases.append(WorkloadSpec(objective, anchor[0], anchor[1], anchor[2], k))
    return tuple(cases)


def small_specs() -> tuple[WorkloadSpec, ...]:
    """Return small cases on which scalar Python implementations are credible."""
    return tuple(
        WorkloadSpec(objective, 1_000, dimension, 1, 10, profile="discovery-small")
        for objective in OBJECTIVES
        for dimension in DIMENSIONS
    )


def stress_specs() -> tuple[WorkloadSpec, ...]:
    """Return core cases that isolate a high-cost factor."""
    return tuple(
        replace(spec, profile="discovery-stress")
        for spec in discovery_core_specs()
        if spec.corpus_size == 1_000_000 or spec.query_count == 1_024 or spec.dimension == 768
    )


def standard_specs() -> tuple[WorkloadSpec, ...]:
    """Return discovery-core cases with stress factors collected separately."""
    stress_coordinates = {
        (spec.objective, spec.corpus_size, spec.dimension, spec.query_count, spec.k) for spec in stress_specs()
    }
    return tuple(
        spec
        for spec in discovery_core_specs()
        if (spec.objective, spec.corpus_size, spec.dimension, spec.query_count, spec.k) not in stress_coordinates
    )


def smoke_specs() -> tuple[WorkloadSpec, ...]:
    """Return tiny deterministic cases for benchmark-harness validation."""
    return tuple(WorkloadSpec(objective, 1_000, 8, 1, 10, profile="smoke") for objective in OBJECTIVES)


def profile_specs(profile: str) -> tuple[WorkloadSpec, ...]:
    """Return the workload slice collected by a named benchmark target."""
    profiles = {
        "smoke": smoke_specs,
        "small": small_specs,
        "core": standard_specs,
        "stress": stress_specs,
    }
    try:
        return profiles[profile]()
    except KeyError as exc:
        choices = ", ".join(sorted(profiles))
        raise ValueError(f"profile must be one of: {choices}") from exc


def selected_metrics(workload: WorkloadSpec) -> tuple[str, ...]:
    """Return latency views selected before discovery collection."""
    include_tail = workload.profile in {"smoke", "discovery-small"} or (
        workload.profile == "discovery-core"
        and workload.corpus_size == 10_000
        and workload.dimension == 128
        and workload.query_count == 32
        and workload.k == 10
    )
    if include_tail:
        return (*DISCOVERY_METRICS, TAIL_LATENCY_METRIC)
    return DISCOVERY_METRICS


def dependency_is_installed(implementation: ImplementationSpec) -> bool:
    """Return whether an implementation's optional import is available."""
    dependency = implementation.optional_dependency
    if dependency is None:
        return True
    module = {"scikit-learn": "sklearn", "faiss-cpu": "faiss"}.get(dependency, dependency)
    return find_spec(module) is not None


def discovery_cell_plans(
    profile: str,
    *,
    memory_budget_bytes: int = DEFAULT_MEMORY_BUDGET_BYTES,
    availability: Callable[[ImplementationSpec], bool] = dependency_is_installed,
) -> tuple[DiscoveryCellPlan, ...]:
    """Return every included and excluded cell for one discovery profile."""
    cells: list[DiscoveryCellPlan] = []
    for workload in profile_specs(profile):
        metrics = selected_metrics(workload)
        for implementation in IMPLEMENTATIONS.values():
            cells.append(
                DiscoveryCellPlan(
                    workload=workload,
                    implementation=implementation,
                    decision=assess_feasibility(
                        workload,
                        implementation,
                        memory_budget_bytes=memory_budget_bytes,
                    ),
                    dependency_installed=availability(implementation),
                    metrics=metrics,
                )
            )
    return tuple(cells)


def discovery_plan_metadata(
    profile: str,
    *,
    memory_budget_bytes: int = DEFAULT_MEMORY_BUDGET_BYTES,
    availability: Callable[[ImplementationSpec], bool] = dependency_is_installed,
) -> dict[str, object]:
    """Return a deterministic manifest of all discovery cell decisions."""
    workloads = profile_specs(profile)
    cells = discovery_cell_plans(
        profile,
        memory_budget_bytes=memory_budget_bytes,
        availability=availability,
    )
    return {
        "schema_version": _SCHEMA_VERSION,
        "study": "exact_top_k_vector_search",
        "profile": profile,
        "memory_budget_bytes": memory_budget_bytes,
        "workload_count": len(workloads),
        "included_cell_count": sum(cell.included for cell in cells),
        "excluded_cell_count": sum(not cell.included for cell in cells),
        "workloads": [workload.metadata() for workload in workloads],
        "cells": [cell.metadata() for cell in cells],
    }


def _score_convention(objective: SearchObjective) -> str:
    """Return an unambiguous metadata label for public scores."""
    if objective is SearchObjective.SQUARED_L2:
        return "negative_squared_l2_higher_is_better"
    if objective is SearchObjective.INNER_PRODUCT:
        return "inner_product_higher_is_better"
    return "cosine_similarity_higher_is_better"
