# Changelog

All notable changes to Vector Search Study will be documented in this file.

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and [Semantic Versioning](https://semver.org/).

## Unreleased

### Added

* Start the project from the Copier Python package template.
* Add a validated synchronous exact-search interface over pre-normalized
    float32 and float64 vectors.
* Add a high-accuracy scalar reference and pure-Python sort, pure-Python heap,
    NumPy full-sort, NumPy argpartition, and blocked NumPy implementations.
* Add deterministic uniform-sphere and clustered synthetic datasets.
* Add adversarial, invalid-input, numerical, and property-based correctness
    tests for every implementation.
* Add squared-L2 and inner-product objectives alongside normalized cosine.
* Add optional exact adapters for scikit-learn, SciPy, Faiss, and CPU PyTorch.
* Add deterministic discovery, small, stress, natural-data, and benchmatrix
    smoke definitions with explicit capability and feasibility policies.
* Add a Linux amd64 devcontainer environment for CPU PyTorch and all optional
    benchmark backends.
* Add bounded full discovery runners, scalable float64 oracle validation,
    deterministic result digests, cell exclusion manifests, and pilot resource
    records.
