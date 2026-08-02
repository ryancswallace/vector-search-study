# Benchmark design

The discovery study spans three objectives, dimensions 8/32/128/768, corpus
sizes 1k/10k/100k/1M, query batches 1/32/1024, and `k` values 1/10/100. The
full Cartesian product would create 432 semantic cases before implementations
and timing views, and includes allocations that exceed practical host memory.

## Profiles

The discovery core anchors on `N=10k`, `D=128`, `Q=32`, and `k=10`, then
varies one factor at a time for each objective. This produces 33 semantic
cases. Collection separates 24 standard-cost cases from nine stress cases that
isolate `N=1M`, `Q=1024`, or `D=768`. A 12-case small profile keeps scalar
Python implementations in scope. Deterministic feasibility rules record every
included and excluded implementation/workload cell and reject cells that
exceed an 8 GiB conservative peak-allocation budget or a scalar runtime ceiling.

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
conversion, and canonical ordering. Benchmatrix records latency and query
throughput for every included cell. Tail latency is preselected for all small
workloads and the three objective-specific core anchors rather than duplicated
across every costly cell.

Every returned result is validated after timing. Workloads at or below one
million scalar coordinate evaluations use the independent `math.fsum`
reference. Larger workloads use bounded float64 query/corpus blocks after that
path has been checked against the scalar oracle across objectives and dtypes.
Only one generated workload and reference result is retained at a time, while
all implementations and metrics for that workload reuse it. Raw benchmark JSON
records the oracle method, strict top-k boundary margin, and a canonical SHA-256
digest of expected identities and rounded scores. Selected identity sets must
match exactly. An implementation-specific internal ordering may differ from
the float64 oracle only when the corresponding reference scores are
indistinguishable at an explicit dtype-scaled numerical tolerance; resolvable
ordering inversions still fail validation.

Case metadata is strict JSON and contains objective, score convention, shapes,
dtype, normalization, generator revision, seed, dataset identity, profile, and
thread policy. It excludes timestamps, local paths, and Python object
representations. Collection should set BLAS and backend thread counts to one,
retain raw pytest-benchmark JSON, and capture the lockfile, Git revision, host,
CPU, OS, and package versions alongside artifacts.

## Collection workflow

The checked-in benchmatrix policy requires five independent runs for formal
evidence, raw samples, 50,000 deterministic BCa resamples, Bonferroni
multiplicity control, and stricter tail-latency evidence. Discovery pilots use
one or two runs only for feasibility and exploratory crossover inspection; they
do not satisfy that evidence policy and cannot support confirmatory claims.

Use a fresh output root for every collection:

```bash
make benchmark-discovery-small DISCOVERY_OUTPUT=benchmark-results/pilot-001
make benchmark-discovery-core DISCOVERY_OUTPUT=benchmark-results/pilot-001
make benchmark-discovery-stress \
  DISCOVERY_OUTPUT=benchmark-results/pilot-001-stress-n1m \
  BENCHMARK_FILTER='n1000000__d128__q32__k10'
```

Stress collection requires a filter so only one high-cost workload runs at a
time. Each successful output directory contains the benchmatrix manifest and
raw run JSON plus `discovery-plan.json`, which retains excluded cells, and
`resource-usage.json`, which records elapsed collection time and child-process
peak resident memory. The plan also records the Git revision and dirty state,
the uv lockfile digest, and a digest over every tracked or non-ignored source
file, so an exploratory dirty-tree pilot remains identifiable. Use
`BENCHMARK_RUNS`, `BENCHMARK_ROUNDS`, and
`BENCHMARK_WARMUP_ROUNDS` to control pilot repetition without changing case
identity.
