"""Pytest hooks for preserving vector-search reference provenance."""

from __future__ import annotations

from benchmarks._discovery_harness import annotate_reference_metadata


def pytest_benchmark_update_json(config: object, benchmarks: object, output_json: object) -> None:
    """Attach deterministic oracle metadata to each raw benchmark record."""
    del config, benchmarks
    annotate_reference_metadata(output_json)
