# Development

Use uv for all Python dependency management.

```bash
make ready
```

Run the full local validation suite before submitting changes:

```bash
make check
```

The package source lives in `src/vector_search_study/`, tests live in `tests/`,
and documentation lives in `docs/`.

## Complete backend environment

The devcontainer runs as Linux `amd64` with Python 3.12 and installs the locked
CPU builds of PyTorch, Faiss, SciPy, and scikit-learn. The container stores
`.venv` in a persistent Docker volume so Linux wheels do not overwrite the
native macOS environment and large Torch installs avoid bind-mount filesystem
overhead.

After creating or rebuilding the devcontainer, verify real-backend correctness
and the benchmatrix harness:

```bash
make benchmark-backend-test
make benchmark-smoke BENCHMARK_OUTPUT=benchmark-results/devcontainer-smoke
```

Use a fresh benchmark output directory for each smoke collection. Preserve its
manifest and run JSON with the host/container provenance. Container and native
OS timing results are separate environments and should not be pooled.

## Discovery pilots

Discovery collection has separate small, standard-cost core, and filtered
stress targets. Always choose a fresh artifact root:

```bash
make benchmark-discovery-small DISCOVERY_OUTPUT=benchmark-results/pilot-001
make benchmark-discovery-core DISCOVERY_OUTPUT=benchmark-results/pilot-001
make benchmark-discovery-stress \
  DISCOVERY_OUTPUT=benchmark-results/pilot-001-stress-d768 \
  BENCHMARK_FILTER='n10000__d768__q32__k10'
```

The default is one independent run with ten central-latency rounds, 100 tail
rounds, and two warmup rounds. Override `BENCHMARK_RUNS`, `BENCHMARK_ROUNDS`,
`BENCHMARK_TAIL_ROUNDS`, or `BENCHMARK_WARMUP_ROUNDS` for an explicitly named
pilot. A stress filter is mandatory. These pilot collections are exploratory
and do not meet the five-run evidence policy configured for formal
comparisons.

## Analysis and paired confirmation

The full workflow is intentionally staged:

```bash
make benchmark-discovery-study \
  DISCOVERY_OUTPUT=benchmark-results/discovery-001 \
  BENCHMARK_RUNS=2
make benchmark-analyze-discovery \
  DISCOVERY_OUTPUT=benchmark-results/discovery-001
make benchmark-confirmatory-predeclare \
  DISCOVERY_OUTPUT=benchmark-results/discovery-001 \
  CONFIRMATORY_OUTPUT=benchmark-results/confirmatory/selection-k
```

After committing the predeclared harness and returning to a clean tree, collect
a paired pilot, create a precision plan, and collect a fresh fixed-size design:

```bash
make benchmark-confirmatory-pilot \
  CONFIRMATORY_EXPERIMENT=argpartition_vs_full_sort \
  CONFIRMATORY_OUTPUT=benchmark-results/confirmatory/selection-k
make benchmark-confirmatory-plan \
  CONFIRMATORY_EXPERIMENT=argpartition_vs_full_sort \
  CONFIRMATORY_OUTPUT=benchmark-results/confirmatory/selection-k
make benchmark-confirmatory-final \
  CONFIRMATORY_EXPERIMENT=argpartition_vs_full_sort \
  CONFIRMATORY_OUTPUT=benchmark-results/confirmatory/selection-k
make benchmark-confirmatory-report \
  CONFIRMATORY_EXPERIMENT=argpartition_vs_full_sort \
  CONFIRMATORY_OUTPUT=benchmark-results/confirmatory/selection-k
```

The precision pilot is never appended to the final sample. Final collection
rejects dirty sources and pair-count plans above the configured safety cap.
