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
Benchmatrix is pinned to its 1.2.0 Git release commit because the configured
package index does not yet expose that version.
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
