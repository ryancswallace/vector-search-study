# Vector Search Study

[![CI](https://github.com/ryancswallace/vector-search-study/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ryancswallace/vector-search-study/actions/workflows/ci.yml)
[![Documentation](https://github.com/ryancswallace/vector-search-study/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/ryancswallace/vector-search-study/actions/workflows/docs.yml)
[![Docker](https://github.com/ryancswallace/vector-search-study/actions/workflows/docker.yml/badge.svg?branch=main)](https://github.com/ryancswallace/vector-search-study/actions/workflows/docker.yml)
[![CodeQL](https://github.com/ryancswallace/vector-search-study/actions/workflows/codeql.yml/badge.svg?branch=main)](https://github.com/ryancswallace/vector-search-study/actions/workflows/codeql.yml)
[![OpenSSF Scorecard](https://github.com/ryancswallace/vector-search-study/actions/workflows/scorecard.yml/badge.svg?branch=main)](https://github.com/ryancswallace/vector-search-study/actions/workflows/scorecard.yml)
[![Python 3.11-3.14](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-3776AB?logo=python&logoColor=white)](https://github.com/ryancswallace/vector-search-study/blob/main/pyproject.toml)
[![Typed with basedpyright](https://img.shields.io/badge/types-basedpyright-2f6fdd)](https://github.com/DetachHead/basedpyright)
[![Linted with Ruff](https://img.shields.io/badge/lint-Ruff-46a2f1)](https://docs.astral.sh/ruff/)
[![Coverage gate: 95%](https://img.shields.io/badge/coverage%20gate-%E2%89%A595%25-2e7d32)](https://github.com/ryancswallace/vector-search-study/blob/main/pyproject.toml)
[![SBOM: CycloneDX 1.6](https://img.shields.io/badge/SBOM-CycloneDX%201.6-6f42c1)](https://cyclonedx.org/)

Benchmarks and analysis of exact vector search algorithms and implementations.

## Quick Start

Install the package:

```bash
uv add vector-search-study
```

Build a normalized corpus and run deterministic exact top-k search:

```python
import numpy as np

from vector_search_study import NumpyArgpartitionSearcher, normalize_rows

corpus = normalize_rows(np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
queries = normalize_rows(np.asarray([[0.8, 0.2]], dtype=np.float32))

index = NumpyArgpartitionSearcher(corpus)
result = index.search(queries, k=1)
print(result.indices)  # [[0]]
```

Every implementation returns scores in descending order and resolves exact
ties by the smaller corpus index. Prepare queries once with `prepare_queries`
and call `search_prepared` when query validation and copying must stay outside
a timed operation.

The correctness milestone includes pure-Python full-sort and heap search plus
NumPy full-sort, argpartition, and blocked implementations. Benchmark matrices,
natural-data preparation, and compiled implementations follow in later stages.

For local development from this repository:

```bash
make ready
```

## Documentation

The documentation source lives under [`docs/`](docs/). The MkDocs site builds in
strict mode and generates API reference pages from package docstrings.

| Start here | Use it for |
| --- | --- |
| [Development](docs/project/development.md) | Local setup, test commands, and repository layout. |
| [Tooling](docs/explanation/tooling.md) | The validation, release, and automation stack. |
| [API reference](docs/reference/index.md) | Generated package API pages. |
| [Release runbook](docs/runbooks/release.md) | Release metadata and publishing workflow. |

## Project Links

* [Contributing](CONTRIBUTING.md)
* [Changelog](CHANGELOG.md)
* [Security policy](SECURITY.md)
* [Release policy](RELEASE.md)
* [Code of conduct](CODE_OF_CONDUCT.md)
* [Citation metadata](CITATION.cff)

## License

Vector Search Study is distributed under the [MIT license](LICENSE).
