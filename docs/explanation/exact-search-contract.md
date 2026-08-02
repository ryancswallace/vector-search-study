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

The trusted reference uses scalar objective calculations with `math.fsum` and a complete
lexicographic ordering. Ordinary tests compare every implementation with that
reference across both supported dtypes, batch shapes, values of `k`, exact
ties, invalid inputs, and randomized normalized datasets. Future benchmark
matrices run an untimed reference validator around every measured cell.

Index construction, corpus conversion, query normalization, and query tensor
creation stay outside the timed operation. The current milestone deliberately
excludes Numba, natural-embedding generation, full-scale collection, and
performance claims.
