"""Predeclared confirmatory comparison families selected by discovery questions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from vector_search_study import SearchObjective
from vector_search_study.benchmarking import IMPLEMENTATIONS, WorkloadSpec


@dataclass(frozen=True, slots=True)
class ConfirmatoryExperiment:
    """One fixed paired family with common benchmark identities."""

    identifier: str
    question: str
    baseline: str
    candidate: str
    workloads: tuple[WorkloadSpec, ...]
    equivalence_margin_percent: float = 5.0
    precision_target_half_width_percent: float = 2.0

    def __post_init__(self) -> None:
        """Reject ambiguous or unsupported predeclarations."""
        if not self.identifier or not self.question:
            raise ValueError("confirmatory identifier and question must be non-empty")
        if self.baseline == self.candidate:
            raise ValueError("confirmatory variants must differ")
        if self.baseline not in IMPLEMENTATIONS or self.candidate not in IMPLEMENTATIONS:
            raise ValueError("confirmatory variants must name registered implementations")
        if not self.workloads or len({workload.name for workload in self.workloads}) != len(self.workloads):
            raise ValueError("confirmatory workloads must be non-empty and unique")
        if any(workload.profile != f"confirmatory-{self.identifier}" for workload in self.workloads):
            raise ValueError("confirmatory workload profiles must match the experiment identifier")
        if self.equivalence_margin_percent <= 0.0 or self.precision_target_half_width_percent <= 0.0:
            raise ValueError("confirmatory margins must be positive")

    def metadata(self) -> dict[str, object]:
        """Return strict JSON-safe predeclaration metadata."""
        return {
            "schema_version": 1,
            "status": "predeclared",
            "identifier": self.identifier,
            "question": self.question,
            "baseline": self.baseline,
            "candidate": self.candidate,
            "common_benchmark_identity": "algorithm_under_test",
            "metrics": ["single_call_latency"],
            "equivalence_margin_percent": self.equivalence_margin_percent,
            "precision_target_half_width_percent": self.precision_target_half_width_percent,
            "multiplicity_family": "all workload cells in this experiment",
            "multiplicity_control": "bonferroni",
            "inference": "stratified_paired_bca_bootstrap",
            "workloads": [workload.metadata() for workload in self.workloads],
        }


def _workload(
    experiment: str,
    objective: SearchObjective,
    corpus_size: int,
    dimension: int,
    query_count: int,
    k: int,
) -> WorkloadSpec:
    """Build one workload carrying its fixed confirmatory family identity."""
    return WorkloadSpec(
        objective,
        corpus_size,
        dimension,
        query_count,
        k,
        profile=f"confirmatory-{experiment}",
    )


EXPERIMENTS: Final[dict[str, ConfirmatoryExperiment]] = {
    "argpartition_vs_full_sort": ConfirmatoryExperiment(
        identifier="argpartition_vs_full_sort",
        question="How does k change the latency advantage of partial selection over full sorting?",
        baseline="numpy_full",
        candidate="numpy_argpartition",
        workloads=tuple(
            _workload(
                "argpartition_vs_full_sort",
                SearchObjective.NORMALIZED_COSINE,
                100_000,
                128,
                32,
                k,
            )
            for k in (1, 10, 100)
        ),
    ),
    "blocked_vs_argpartition": ConfirmatoryExperiment(
        identifier="blocked_vs_argpartition",
        question="At what corpus scale does blocked search justify its bounded temporary memory?",
        baseline="numpy_argpartition",
        candidate="numpy_blocked",
        workloads=tuple(
            _workload(
                "blocked_vs_argpartition",
                SearchObjective.NORMALIZED_COSINE,
                corpus_size,
                128,
                32,
                10,
            )
            for corpus_size in (100_000, 1_000_000)
        ),
    ),
    "python_heap_vs_argpartition": ConfirmatoryExperiment(
        identifier="python_heap_vs_argpartition",
        question="At small N and Q, where does NumPy overcome scalar Python heap overhead?",
        baseline="python_heap",
        candidate="numpy_argpartition",
        workloads=tuple(
            _workload(
                "python_heap_vs_argpartition",
                SearchObjective.NORMALIZED_COSINE,
                1_000,
                dimension,
                1,
                10,
            )
            for dimension in (8, 32, 128)
        ),
    ),
}


def get_experiment(identifier: str) -> ConfirmatoryExperiment:
    """Return a named predeclared experiment or reject it."""
    try:
        return EXPERIMENTS[identifier]
    except KeyError as exc:
        choices = ", ".join(sorted(EXPERIMENTS))
        raise ValueError(f"unknown confirmatory experiment; choose one of: {choices}") from exc


def predeclaration_manifest() -> dict[str, object]:
    """Return the complete immutable confirmatory registry."""
    return {
        "schema_version": 1,
        "study": "exact_top_k_vector_search",
        "selection_stage": "post_discovery_pre_confirmation",
        "pilot_reuse_prohibited": True,
        "experiments": [experiment.metadata() for experiment in EXPERIMENTS.values()],
    }
