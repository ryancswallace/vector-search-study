# Benchmark design

The discovery study spans three objectives, dimensions 8/32/128/768, corpus
sizes 1k/10k/100k/1M, query batches 1/32/1024, and `k` values 1/10/100. The
full Cartesian product would create 432 semantic cases before implementations
and timing views, and includes allocations that exceed practical host memory.

## Profiles

The discovery core anchors on `N=10k`, `D=128`, `Q=32`, and `k=10`, then
varies one factor at a time for each objective. This produces 33 semantic
cases. A 12-case small profile keeps scalar Python implementations in scope. A
stress profile isolates `N=1M`, `Q=1024`, or `D=768`; deterministic feasibility
rules exclude cells that exceed an 8 GiB dominant-allocation budget or a scalar
runtime ceiling.

Synthetic normalized-cosine data uses unit-normalized Gaussian vectors from a
pinned NumPy PCG64 seed. Squared-L2 and inner-product data uses unnormalized
Gaussian vectors. Natural embeddings are a separate 384-dimensional slice:
SciFact corpus and queries embedded with `all-MiniLM-L6-v2`. Both Hugging Face
repository revisions are pinned in `NaturalDatasetSpec`; downloading and
embedding are not part of this milestone.

## Capability matrix

| Implementation | Squared L2 | Inner product | Normalized cosine |
| --- | --- | --- | --- |
| Python sort/heap | yes | yes | yes |
| NumPy full/argpartition/blocked | yes | yes | yes |
| scikit-learn brute | yes | no | yes |
| scikit-learn KDTree/BallTree | yes | no | yes, via normalized L2 |
| SciPy cKDTree | yes | no | yes, via normalized L2 |
| Faiss Flat L2 | yes | no | no |
| Faiss Flat IP | no | yes | yes |
| PyTorch matmul/topk | yes | yes | yes |

Tree backends are not assigned inner product because they do not implement it
natively. Faiss is restricted to float32. Current PyTorch releases do not ship
an Intel-macOS wheel, so that cell is absent on that host and remains available
on supported CPU hosts.

## Timing and provenance

Each benchmark cell builds one index and prepares backend-specific query data
before timing. Timed work includes exact scoring, selection, native-result
conversion, and canonical ordering. Benchmatrix records latency, query
throughput, and tail-latency views. The smoke matrix validates a strict top-k
boundary margin and compares every returned result with the trusted scalar
reference after timing.

Case metadata is strict JSON and contains objective, score convention, shapes,
dtype, normalization, generator revision, seed, dataset identity, profile, and
thread policy. It excludes timestamps, local paths, and Python object
representations. Collection should set BLAS and backend thread counts to one,
retain raw pytest-benchmark JSON, and capture the lockfile, Git revision, host,
CPU, OS, and package versions alongside artifacts.
