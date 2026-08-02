"""Tests for predeclared paired confirmatory workflow controls."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from benchmarks._confirmatory_harness import COMMON_IMPLEMENTATION_NAME, install_confirmatory_test, selected_variant
from benchmarks.confirmatory_experiments import EXPERIMENTS, get_experiment, predeclaration_manifest
from scripts.confirmatory_reporting import render_confirmatory_artifacts
from scripts.run_confirmatory import main as confirmatory_main
from scripts.run_discovery import _child_resource_usage
from scripts.run_discovery_study import discovery_tasks


def test_confirmatory_registry_is_predeclared_json_and_uses_common_identity() -> None:
    """Every family freezes variants, workloads, margins, and multiplicity scope."""
    manifest = predeclaration_manifest()

    assert set(EXPERIMENTS) == {
        "argpartition_vs_full_sort",
        "blocked_vs_argpartition",
        "python_heap_vs_argpartition",
    }
    assert manifest["pilot_reuse_prohibited"] is True
    assert all(
        experiment.metadata()["common_benchmark_identity"] == COMMON_IMPLEMENTATION_NAME
        for experiment in EXPERIMENTS.values()
    )
    assert all(experiment.metadata()["multiplicity_control"] == "bonferroni" for experiment in EXPERIMENTS.values())
    _ = json.dumps(manifest, sort_keys=True, allow_nan=False)


def test_variant_selection_requires_explicit_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Paired sides differ only through the collector-recorded variant command."""
    monkeypatch.delenv("VECTOR_SEARCH_CONFIRMATORY_EXPERIMENT", raising=False)
    monkeypatch.delenv("VECTOR_SEARCH_CONFIRMATORY_VARIANT", raising=False)
    with pytest.raises(ValueError, match="must be set together"):
        _ = selected_variant()

    monkeypatch.setenv("VECTOR_SEARCH_CONFIRMATORY_EXPERIMENT", "argpartition_vs_full_sort")
    monkeypatch.setenv("VECTOR_SEARCH_CONFIRMATORY_VARIANT", "baseline")
    experiment, implementation = selected_variant()
    assert implementation == experiment.baseline == "numpy_full"

    monkeypatch.setenv("VECTOR_SEARCH_CONFIRMATORY_VARIANT", "candidate")
    _experiment, implementation = selected_variant()
    assert implementation == "numpy_argpartition"

    monkeypatch.setenv("VECTOR_SEARCH_CONFIRMATORY_VARIANT", "unknown")
    with pytest.raises(ValueError, match="baseline or candidate"):
        _ = selected_variant()


def test_confirmatory_harness_installs_one_common_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Variant implementation names never enter the benchmatrix cell identity."""
    monkeypatch.setenv("VECTOR_SEARCH_CONFIRMATORY_EXPERIMENT", "python_heap_vs_argpartition")
    monkeypatch.setenv("VECTOR_SEARCH_CONFIRMATORY_VARIANT", "baseline")
    namespace: dict[str, object] = {}

    install_confirmatory_test(namespace)

    assert set(namespace) == {"test_confirmatory_matrix"}
    assert callable(namespace["test_confirmatory_matrix"])


def test_unknown_experiment_and_invalid_predeclaration_are_rejected(tmp_path: Path) -> None:
    """Selection cannot silently fall back or detach from discovery evidence."""
    with pytest.raises(ValueError, match="choose one of"):
        _ = get_experiment("unknown")
    analysis = tmp_path / "analysis.json"
    analysis.write_text(json.dumps({"analysis_kind": "other"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="must be a discovery analysis"):
        _ = confirmatory_main(
            ["predeclare", "--discovery-analysis", str(analysis), "--output", str(tmp_path / "plan.json")]
        )


def test_predeclaration_binds_registry_to_discovery_digest(tmp_path: Path) -> None:
    """The selected families retain the exact exploratory input identity."""
    analysis = tmp_path / "analysis.json"
    analysis.write_text(
        json.dumps(
            {
                "analysis_kind": "discovery",
                "input_digest": "sha256:" + "a" * 64,
                "audit": {"formal_ready": False},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "predeclaration.json"

    assert confirmatory_main(["predeclare", "--discovery-analysis", str(analysis), "--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["discovery_analysis"]["input_digest"] == "sha256:" + "a" * 64
    assert payload["discovery_analysis"]["formal_ready"] is False
    with pytest.raises(SystemExit, match="already exists"):
        _ = confirmatory_main(["predeclare", "--discovery-analysis", str(analysis), "--output", str(output)])


def test_discovery_study_expands_each_stress_workload_separately(tmp_path: Path) -> None:
    """High-cost factors cannot accidentally run as one monolithic process."""
    tasks = discovery_tasks(tmp_path, ["small", "core", "stress"])

    assert len(tasks) == 11
    assert [task.profile for task in tasks[:2]] == ["small", "core"]
    assert all(task.pytest_filter is not None for task in tasks[2:])
    assert len({task.output for task in tasks}) == len(tasks)


def test_discovery_resource_metadata_is_portable_to_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing discovery helpers must not require the POSIX resource module."""
    monkeypatch.setattr(sys, "platform", "win32")

    assert _child_resource_usage() == {
        "maximum_resident_set_bytes": None,
        "measurement_scope": "unavailable",
    }


def test_confirmatory_reporting_renders_adjusted_interval_artifacts(tmp_path: Path) -> None:
    """Final reports pair exact decisions with a readable interval plot."""
    comparison = tmp_path / "comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "passed": True,
                "paired_collections": [{"complete_pairs": 12}],
                "comparisons": [
                    {
                        "case_name": (
                            "confirmatory-argpartition_vs_full_sort__normalized_cosine"
                            "__n100000__d128__q32__k10__float32"
                        ),
                        "improvement_percent": 20.0,
                        "improvement_low_percent": 15.0,
                        "improvement_high_percent": 25.0,
                        "threshold_percent": 5.0,
                        "regression": "improved",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    render_confirmatory_artifacts(comparison, get_experiment("argpartition_vs_full_sort"), tmp_path)

    assert (tmp_path / "effect-intervals.png").is_file()
    assert (tmp_path / "effect-intervals.svg").is_file()
    assert (tmp_path / "chart-map.json").is_file()
    assert "## Technical summary" in (tmp_path / "technical-report.md").read_text(encoding="utf-8")
