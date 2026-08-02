"""Tiny correctness-guarded matrix for validating the benchmark lifecycle."""

from benchmarks._discovery_harness import install_discovery_tests
from vector_search_study.benchmarking import smoke_specs

install_discovery_tests(globals(), smoke_specs())
