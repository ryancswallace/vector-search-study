"""High-cost one-factor discovery cells intended for filtered collection."""

from benchmarks._discovery_harness import install_discovery_tests
from vector_search_study.benchmarking import stress_specs

install_discovery_tests(globals(), stress_specs())
