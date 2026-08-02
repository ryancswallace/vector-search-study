"""Tests for deterministic discovery analysis and report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from benchmatrix import BenchmarkRunGroup
from scripts.discovery_analysis import (
    DiscoveryAnalysis,
    DiscoveryRecord,
    analyze_discovery,
    audit_evidence,
    discover_collection_directories,
    pairwise_results,
    render_technical_report,
    select_winners,
    summarize_records,
    write_analysis,
)


def _record(
    implementation: str,
    case_name: str,
    *,
    value: float,
    objective: str = "normalized_cosine",
    profile: str = "discovery-core",
    corpus_size: int = 10_000,
    dimension: int = 128,
    query_count: int = 32,
    k: int = 10,
    run_index: int = 1,
) -> DiscoveryRecord:
    """Build one central-latency record."""
    return DiscoveryRecord(
        collection="collection",
        run_index=run_index,
        source=f"run-{run_index:03d}.json",
        implementation=implementation,
        case_name=case_name,
        metric="single_call_latency",
        value=value,
        unit="seconds",
        objective=objective,
        profile=profile,
        corpus_size=corpus_size,
        dimension=dimension,
        query_count=query_count,
        k=k,
        dtype="float32",
        rounds=5,
        iterations=1,
        sample_count=5,
        p50_seconds=None,
        p90_seconds=None,
        p95_seconds=None,
        p99_seconds=None,
    )


def test_summaries_winners_and_predeclared_pairwise_contrasts() -> None:
    """Analysis uses independent run statistics and fixed comparison slices."""
    records: list[DiscoveryRecord] = []
    for run_index, scale in ((1, 1.0), (2, 1.1)):
        records.extend(
            (
                _record("numpy_full", "k-case", value=0.010 * scale, k=1, run_index=run_index),
                _record("numpy_argpartition", "k-case", value=0.008 * scale, k=1, run_index=run_index),
                _record(
                    "numpy_argpartition",
                    "n-case",
                    value=0.020 * scale,
                    corpus_size=100_000,
                    run_index=run_index,
                ),
                _record(
                    "numpy_blocked",
                    "n-case",
                    value=0.018 * scale,
                    corpus_size=100_000,
                    run_index=run_index,
                ),
                _record(
                    "python_heap",
                    "d-case",
                    value=0.006 * scale,
                    profile="discovery-small",
                    corpus_size=1_000,
                    dimension=8,
                    query_count=1,
                    run_index=run_index,
                ),
                _record(
                    "numpy_argpartition",
                    "d-case",
                    value=0.004 * scale,
                    profile="discovery-small",
                    corpus_size=1_000,
                    dimension=8,
                    query_count=1,
                    run_index=run_index,
                ),
            )
        )

    summaries = summarize_records(records)
    winners = select_winners(summaries)
    contrasts = pairwise_results(summaries)

    assert all(summary.run_count == 2 for summary in summaries)
    assert {winner.implementation for winner in winners} == {"numpy_argpartition", "numpy_blocked"}
    assert {contrast.question for contrast in contrasts} == {
        "argpartition_vs_full_sort",
        "blocked_vs_argpartition",
        "python_heap_vs_argpartition",
    }
    assert all(contrast.candidate_improvement_percent > 0.0 for contrast in contrasts)


class _FakeGroup:
    """Minimal run-group evidence needed by the study audit."""

    is_complete = True


def test_evidence_audit_distinguishes_formal_and_exploratory_inputs(tmp_path: Path) -> None:
    """Clean five-run inputs pass while shallow or dirty evidence remains exploratory."""
    records = tuple(_record("numpy_full", "case", value=0.01, run_index=index) for index in range(1, 6))
    plan = {
        "collection": {
            "source_provenance": {
                "git_revision": "a" * 40,
                "git_dirty": False,
                "source_tree_digest": "sha256:" + "b" * 64,
            }
        }
    }
    groups = [(tmp_path, cast(BenchmarkRunGroup, _FakeGroup()), plan)]

    formal = audit_evidence(records, groups)
    dirty_plan = json.loads(json.dumps(plan))
    dirty_plan["collection"]["source_provenance"]["git_dirty"] = True
    exploratory = audit_evidence(records[:2], [(tmp_path, cast(BenchmarkRunGroup, _FakeGroup()), dirty_plan)])

    assert formal.formal_ready is True
    assert formal.minimum_runs_per_cell == 5
    assert exploratory.formal_ready is False
    assert any("dirty source" in issue for issue in exploratory.issues)
    assert any("independent runs" in issue for issue in exploratory.issues)


def test_collection_discovery_rejects_empty_roots_and_finds_nested_manifests(tmp_path: Path) -> None:
    """Only ordinary benchmatrix run-group manifests are selected."""
    nested = tmp_path / "core"
    nested.mkdir()
    (nested / "benchmatrix-manifest.json").write_text(
        json.dumps({"kind": "benchmark_run_group"}),
        encoding="utf-8",
    )
    paired = tmp_path / "paired"
    paired.mkdir()
    (paired / "benchmatrix-manifest.json").write_text(
        json.dumps({"kind": "benchmark_paired_run_group"}),
        encoding="utf-8",
    )

    assert discover_collection_directories([tmp_path]) == (nested.resolve(),)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no discovery"):
        _ = discover_collection_directories([empty])


def test_analysis_writer_emits_tables_plots_contract_and_technical_report(tmp_path: Path) -> None:
    """Static research artifacts share one deterministic analysis bundle."""
    records = (
        _record("numpy_full", "case", value=0.010),
        _record("numpy_argpartition", "case", value=0.008),
    )
    summaries = summarize_records(records)
    audit = audit_evidence(
        records,
        [
            (
                tmp_path,
                cast(BenchmarkRunGroup, _FakeGroup()),
                {
                    "collection": {
                        "source_provenance": {
                            "git_revision": "a" * 40,
                            "git_dirty": False,
                            "source_tree_digest": "sha256:" + "b" * 64,
                        }
                    }
                },
            )
        ],
    )
    analysis = DiscoveryAnalysis(
        records=records,
        summaries=summaries,
        winners=select_winners(summaries),
        pairwise=pairwise_results(summaries),
        audit=audit,
        input_digest="sha256:" + "c" * 64,
    )
    output = tmp_path / "analysis"

    write_analysis(analysis, output)
    report = (output / "technical-report.md").read_text(encoding="utf-8")
    chart_map = json.loads((output / "chart-map.json").read_text(encoding="utf-8"))

    assert (output / "summary.csv").is_file()
    assert (output / "plots" / "latency-winner-counts.png").is_file()
    assert chart_map[0]["chart_type"] == "sorted horizontal bar"
    assert "## Technical summary" in report
    assert "exploratory only" in render_technical_report(analysis)


def test_analyze_discovery_requires_collections() -> None:
    """An empty evidence set cannot silently produce a report."""
    with pytest.raises(ValueError, match="at least one"):
        _ = analyze_discovery([])
