# Exact-search contract

The study compares exhaustive top-k search over a corpus with shape `(N, D)`
and a query batch with shape `(Q, D)`. Corpus and query rows are finite,
C-contiguous `float32` or `float64` vectors. Normalized-cosine inputs must have
unit L2 norm, making that objective equivalent to maximum inner-product search.
Squared L2 and inner-product inputs retain their original magnitudes.

`search(queries, k)` validates and copies raw queries. For benchmarking,
`prepare_queries(queries)` performs that work once and `search_prepared`
measures only the implementation's scoring, selection, and result assembly.
Index construction and corpus conversion happen in each searcher's constructor
and are likewise outside the search operation.

Results contain `(Q, k)` corpus indices and higher-is-better scores: negative
squared distance, inner product, or normalized cosine. Each row is ordered by
decreasing score; exact score ties prefer the smaller corpus index. Native
implementations repair selection-boundary ties. Third-party performance cells
require a strict boundary margin and canonicalize returned candidates because
several native APIs do not guarantee which index wins a tie.

The trusted small-workload reference uses scalar objective calculations with
`math.fsum` and a complete lexicographic ordering. Ordinary tests compare every
implementation with that reference across both supported dtypes, batch shapes,
values of `k`, exact ties, invalid inputs, and randomized datasets. Larger
benchmark workloads use a bounded float64 oracle that is itself checked against
the scalar path. Untimed validation surrounds every measured cell, and raw
artifacts retain the oracle method, strict boundary margin, and result digest.
The selected identity set must match exactly. Internal ordering differences are
accepted only for reference scores indistinguishable at a dtype-scaled
numerical tolerance; all resolvable ordering differences remain failures.

Index construction, corpus conversion, query normalization, and query tensor
creation stay outside the timed operation. Confirmatory variants are exposed
under the same `algorithm_under_test` benchmark identity, so the paired
collector changes only the selected implementation. The current milestone
deliberately excludes Numba, natural-embedding generation, and broad
performance claims beyond the recorded environments.
