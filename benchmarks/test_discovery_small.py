"""Small discovery matrix retaining credible scalar Python implementations."""

from benchmarks._discovery_harness import install_discovery_tests
from vector_search_study.benchmarking import small_specs

install_discovery_tests(globals(), small_specs())
