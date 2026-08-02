# Exact-search contract

The study compares exhaustive top-k search over a corpus with shape `(N, D)`
and a query batch with shape `(Q, D)`. Corpus and query rows are finite,
C-contiguous `float32` or `float64` vectors normalized to unit L2 norm. Cosine
similarity is therefore equivalent to maximum inner-product search.

`search(queries, k)` validates and copies raw queries. For benchmarking,
`prepare_queries(queries)` performs that work once and `search_prepared`
measures only the implementation's scoring, selection, and result assembly.
Index construction and corpus conversion happen in each searcher's constructor
and are likewise outside the search operation.

Results contain `(Q, k)` corpus indices and scores. Each row is ordered by
decreasing score; exact score ties prefer the smaller corpus index. The
argpartition and blocked implementations explicitly repair selection-boundary
ties rather than relying on unstable partition order.

The trusted reference uses scalar products with `math.fsum` and a complete
lexicographic ordering. Ordinary tests compare every implementation with that
reference across both supported dtypes, batch shapes, values of `k`, exact
ties, invalid inputs, and randomized normalized datasets. Future benchmark
matrices will also run an untimed reference validator around every measured
cell.

The current milestone deliberately excludes embedding generation, Numba or
other compiled implementations, benchmark collection, and performance claims.
