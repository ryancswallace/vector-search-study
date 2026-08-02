# Devcontainer

This devcontainer is the reproducible contributor environment for the project.

It provides:

* Python 3.12 on Debian Bookworm under `linux/amd64`, matching the benchmark
    environment even on an Intel macOS host.
* uv 0.11.23.
* CPU-only PyTorch, Faiss, SciPy, and scikit-learn benchmark backends.
* Benchmatrix 1.2.1 from PyPI, without relying on an adjacent host checkout.
* A persistent Docker volume for `.venv`, keeping Linux packages off the slow
    macOS bind mount and isolated from the native host environment.
* Node.js 24 for Markdown, spelling, workflow, and Dockerfile checks.
* GitHub CLI for pull request and release workflows.
* Docker-outside-of-Docker for optional local container checks.
* VS Code recommendations for Ruff, basedpyright, Markdown, CSpell, GitHub
    Actions, containers, TOML, YAML, and Makefiles.

The create hook installs the locked Python, optional benchmark backends, and
Node dependencies; verifies every backend import; and installs the pre-commit
and pre-push hooks:

```bash
make devcontainer-ready
```

Run the full validation suite explicitly before submitting changes:

```bash
make check
```

Run correctness tests against the real optional packages, including CPU
PyTorch, with:

```bash
make benchmark-backend-test
```

Run the artifact-preserving benchmatrix smoke suite with a new output path:

```bash
make benchmark-smoke BENCHMARK_OUTPUT=benchmark-results/devcontainer-smoke
```

The devcontainer uses CPU backends only. It does not require GPU passthrough or
CUDA, and its benchmark artifacts must not be compared directly with native
macOS artifacts as if they came from the same host environment.

Docker-outside-of-Docker exposes the host Docker socket inside the container.
That is useful for `make docker-check`, but it means the devcontainer should be
treated as a trusted development environment.
