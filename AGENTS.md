# Vector Search Study Project Instructions

## Project

This repository contains `vector-search-study`, a Python package generated from
the Copier project template.

## Package Layout

* `src/vector_search_study/__init__.py`: public package exports.
* `src/vector_search_study/_core.py`: core example functionality.
* `src/vector_search_study/exceptions.py`: package-specific exceptions.
* `tests/`: tests.

## Tooling

Use uv. The authoritative local check command is:

```bash
make check
```

It checks the uv lockfile, Ruff, Markdown, the documentation site, GitHub
Actions workflows, CSpell, secrets, Bandit, deptry, pip-audit, pytest and
coverage, basedpyright, built distributions, and CycloneDX SBOM generation.
Run it after edits.

## Code Style

* Python 3.11+.
* Type hints should pass basedpyright in standard mode.
* Use Ruff for linting and formatting.
* Public functions/classes use Google-style docstrings.
* Private helpers use concise one-line PEP 257 docstrings.
* Keep metadata strict-JSON-safe when adding generated metadata.
