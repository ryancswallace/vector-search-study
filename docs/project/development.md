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
CPU builds of PyTorch, Faiss, SciPy, and scikit-learn. This provides the backend
coverage unavailable on Intel macOS without compiling PyTorch locally.
Benchmatrix 1.2.1 is installed from PyPI.
The container stores `.venv` in a persistent Docker volume so Linux wheels do
not overwrite the native macOS environment and large Torch installs avoid
bind-mount filesystem overhead.

After creating or rebuilding the devcontainer, verify real-backend correctness
and the benchmatrix harness:

```bash
make benchmark-backend-test
make benchmark-smoke BENCHMARK_OUTPUT=benchmark-results/devcontainer-smoke
```

Use a fresh benchmark output directory for each smoke collection. Preserve its
manifest and run JSON with the host/container provenance. Container and native
macOS timing results are separate environments and should not be pooled.

`make audit` keeps the locked CPU-only Torch wheel in the dependency export but
removes its PEP 440 local `+cpu` label from the temporary pip-audit lookup file.
PyPI vulnerability records use Torch's public release version, while the
PyTorch wheel index uses the local label; this lookup-only normalization allows
Torch and its transitive dependencies to remain in the audit.

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

The default is one independent run with ten measured and two warmup rounds.
Override `BENCHMARK_RUNS`, `BENCHMARK_ROUNDS`, or
`BENCHMARK_WARMUP_ROUNDS` for an explicitly named pilot. A stress filter is
mandatory. These pilot collections are exploratory and do not meet the
five-run evidence policy configured for formal comparisons.
