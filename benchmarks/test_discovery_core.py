"""Standard-cost discovery matrix with stress factors collected separately."""

from benchmarks._discovery_harness import install_discovery_tests
from vector_search_study.benchmarking import standard_specs

install_discovery_tests(globals(), standard_specs())
