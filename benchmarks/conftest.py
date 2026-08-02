"""Pytest hooks for preserving vector-search reference provenance."""

from __future__ import annotations

from benchmarks._confirmatory_harness import annotate_reference_metadata as annotate_confirmatory_metadata
from benchmarks._discovery_harness import annotate_reference_metadata as annotate_discovery_metadata


def pytest_benchmark_update_json(config: object, benchmarks: object, output_json: object) -> None:
    """Attach deterministic oracle metadata to each raw benchmark record."""
    del config, benchmarks
    annotate_discovery_metadata(output_json)
    annotate_confirmatory_metadata(output_json)
