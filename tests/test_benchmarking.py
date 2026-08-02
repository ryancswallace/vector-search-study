"""Tests for deterministic benchmark definitions and feasibility rules."""

from __future__ import annotations

import json

import numpy as np
import pytest

from vector_search_study import SearchObjective
from vector_search_study.benchmarking import (
    CORPUS_SIZES,
    DIMENSIONS,
    IMPLEMENTATIONS,
    K_VALUES,
    OBJECTIVES,
    QUERY_COUNTS,
    NaturalDatasetSpec,
    WorkloadSpec,
    assess_feasibility,
    discovery_cell_plans,
    discovery_core_specs,
    discovery_plan_metadata,
    estimate_peak_bytes,
    profile_specs,
    selected_metrics,
    small_specs,
    smoke_specs,
    standard_specs,
    stress_specs,
)


def test_requested_factor_levels_and_one_factor_core_are_complete() -> None:
    """The manageable core still covers every approved factor level."""
    specs = discovery_core_specs()

    assert DIMENSIONS == (8, 32, 128, 768)
    assert CORPUS_SIZES == (1_000, 10_000, 100_000, 1_000_000)
    assert QUERY_COUNTS == (1, 32, 1_024)
    assert K_VALUES == (1, 10, 100)
    assert tuple(SearchObjective) == OBJECTIVES
    assert len(specs) == 33
    assert len({spec.name for spec in specs}) == 33
    assert {spec.dimension for spec in specs} == set(DIMENSIONS)
    assert {spec.corpus_size for spec in specs} == set(CORPUS_SIZES)
    assert {spec.query_count for spec in specs} == set(QUERY_COUNTS)
    assert {spec.k for spec in specs} == set(K_VALUES)


def test_profile_catalogs_are_distinct_and_bounded() -> None:
    """Small, stress, and smoke profiles have explicit experimental roles."""
    small = small_specs()
    standard = standard_specs()
    stress = stress_specs()
    smoke = smoke_specs()

    assert len(small) == 12
    assert len(standard) == 24
    assert len(stress) == 9
    assert len(smoke) == 3
    assert all(spec.profile == "discovery-small" for spec in small)
    assert all(spec.profile == "discovery-core" for spec in standard)
    assert all(spec.profile == "discovery-stress" for spec in stress)
    assert all(spec.corpus_size == 1_000_000 or spec.query_count == 1_024 or spec.dimension == 768 for spec in stress)
    assert {spec.objective for spec in smoke} == set(SearchObjective)
    assert len({spec.name for spec in (*standard, *stress)}) == 33
    assert profile_specs("core") == standard


def test_discovery_plan_records_every_inclusion_and_exclusion() -> None:
    """Planning is deterministic, JSON-safe, and retains excluded cells."""

    def availability(_implementation: object) -> bool:
        return True

    cells = discovery_cell_plans("small", availability=availability)
    metadata = discovery_plan_metadata("small", availability=availability)

    assert len(cells) == len(small_specs()) * len(IMPLEMENTATIONS)
    assert any(cell.included for cell in cells)
    assert any(not cell.included for cell in cells)
    assert metadata["included_cell_count"] == sum(cell.included for cell in cells)
    assert metadata["excluded_cell_count"] == sum(not cell.included for cell in cells)
    recorded_cells = metadata["cells"]
    assert isinstance(recorded_cells, list)
    assert len(recorded_cells) == len(cells)
    _ = json.dumps(metadata, sort_keys=True, allow_nan=False)


def test_selected_tail_latency_is_limited_to_predeclared_cases() -> None:
    """Tail sampling is broad for small data and selected for the core anchor."""
    small = small_specs()[0]
    anchor = WorkloadSpec(SearchObjective.INNER_PRODUCT, 10_000, 128, 32, 10)
    non_anchor = WorkloadSpec(SearchObjective.INNER_PRODUCT, 100_000, 128, 32, 10)

    assert "tail_latency" in selected_metrics(small)
    assert "tail_latency" in selected_metrics(anchor)
    assert "tail_latency" not in selected_metrics(non_anchor)


def test_unknown_profile_is_rejected() -> None:
    """Collection targets cannot silently fall back to another profile."""
    with pytest.raises(ValueError, match="profile must be one of"):
        _ = profile_specs("unknown")


def test_workload_metadata_is_deterministic_json_and_materializes_matching_data() -> None:
    """Case provenance contains no paths, timestamps, or non-JSON objects."""
    spec = WorkloadSpec(SearchObjective.SQUARED_L2, 1_000, 8, 1, 10, seed=123, profile="test")

    first = spec.metadata()
    second = spec.metadata()
    encoded = json.dumps(first, sort_keys=True, allow_nan=False)
    dataset = spec.make_dataset()

    assert first == second
    assert "synthetic-pcg64-v1" in encoded
    assert first["score_convention"] == "negative_squared_l2_higher_is_better"
    assert spec.coordinate_evaluations == 8_000
    assert dataset.corpus.shape == (1_000, 8)
    assert dataset.queries.shape == (1, 8)
    assert dataset.objective is SearchObjective.SQUARED_L2


def test_natural_manifest_pins_dataset_and_model_revisions() -> None:
    """The natural-data slice is separate from the synthetic dimension grid."""
    metadata = NaturalDatasetSpec().metadata()

    assert metadata["dataset"] == "BeIR/scifact"
    assert len(str(metadata["dataset_revision"])) == 40
    assert metadata["model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert len(str(metadata["model_revision"])) == 40
    assert metadata["dimension"] == 384
    _ = json.dumps(metadata, sort_keys=True, allow_nan=False)


def test_implementation_capabilities_are_sparse_and_deterministic() -> None:
    """Trees are not silently repurposed for non-native inner product."""
    all_objectives = frozenset(SearchObjective)

    assert IMPLEMENTATIONS["numpy_full"].objectives == all_objectives
    assert SearchObjective.INNER_PRODUCT not in IMPLEMENTATIONS["sklearn_kdtree"].objectives
    assert IMPLEMENTATIONS["faiss_flat_l2"].objectives == frozenset({SearchObjective.SQUARED_L2})
    assert IMPLEMENTATIONS["faiss_flat_ip"].objectives == frozenset(
        {SearchObjective.INNER_PRODUCT, SearchObjective.NORMALIZED_COSINE}
    )
    for implementation in IMPLEMENTATIONS.values():
        _ = json.dumps(implementation.metadata(), sort_keys=True, allow_nan=False)


def test_feasibility_records_semantic_runtime_and_memory_exclusions() -> None:
    """Large sparse matrices are filtered with stable machine-readable reasons."""
    anchor = WorkloadSpec(SearchObjective.INNER_PRODUCT, 10_000, 128, 32, 10)
    unsupported = assess_feasibility(anchor, IMPLEMENTATIONS["sklearn_kdtree"])
    slow = assess_feasibility(anchor, IMPLEMENTATIONS["python_sort"])
    huge = WorkloadSpec(SearchObjective.INNER_PRODUCT, 1_000_000, 128, 1_024, 10)
    memory = assess_feasibility(huge, IMPLEMENTATIONS["numpy_full"])
    included = assess_feasibility(anchor, IMPLEMENTATIONS["numpy_blocked"])

    assert (unsupported.feasible, unsupported.reason) == (False, "objective_not_supported")
    assert (slow.feasible, slow.reason) == (False, "scalar_runtime_limit")
    assert (memory.feasible, memory.reason) == (False, "memory_budget_exceeded")
    assert (included.feasible, included.reason) == (True, "included")
    assert estimate_peak_bytes(huge, "numpy_blocked") < estimate_peak_bytes(huge, "numpy_full")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"corpus_size": 0}, "corpus_size"),
        ({"k": 1_001}, "k must not exceed"),
        ({"dtype": "int32"}, "dtype"),
        ({"seed": -1}, "seed"),
        ({"profile": ""}, "profile"),
    ],
)
def test_workload_validation_rejects_ambiguous_metadata(kwargs: dict[str, object], message: str) -> None:
    """Invalid workload fields fail before any large allocation."""
    values: dict[str, object] = {
        "objective": SearchObjective.INNER_PRODUCT,
        "corpus_size": 1_000,
        "dimension": 8,
        "query_count": 1,
        "k": 10,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        _ = WorkloadSpec(**values)  # type: ignore[arg-type]


def test_implementation_builder_rejects_unsupported_objective_and_dtype() -> None:
    """Capability failures happen before optional backend imports."""
    corpus64 = np.ones((2, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="does not support"):
        _ = IMPLEMENTATIONS["faiss_flat_l2"].build(corpus64, SearchObjective.INNER_PRODUCT)
    with pytest.raises(ValueError, match="float32"):
        _ = IMPLEMENTATIONS["faiss_flat_l2"].build(corpus64, SearchObjective.SQUARED_L2)
