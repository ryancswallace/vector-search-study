# Statistical workflow

The study separates question discovery from confirmatory claims. Raw timing
rounds estimate a statistic within one launched process; only separate process
runs or complete paired blocks count as independent evidence.

## Discovery asks where to look

Discovery covers the approved one-factor slices and answers descriptive
questions about scaling, crossovers, and memory trade-offs. The analysis
pipeline retains one run-level statistic per matrix cell, generates exact CSV
tables, and plots winner counts and predeclared pairwise contrasts. It audits:

* collection completion and matching matrix cells;
* clean Git revision and source-tree digests;
* independent run counts and measured rounds;
* tail sample counts; and
* strict JSON provenance.

An exploratory label is mandatory if any configured evidence gate is missing.
Discovery ranks are not hypothesis tests and do not establish equivalence.

## Predeclaration freezes the comparison family

The registry contains three initial families:

| Family | Baseline | Candidate | Varied factor |
| --- | --- | --- | --- |
| `argpartition_vs_full_sort` | NumPy full sort | NumPy argpartition | `k` |
| `blocked_vs_argpartition` | NumPy argpartition | blocked NumPy | corpus size |
| `python_heap_vs_argpartition` | Python heap | NumPy argpartition | dimension |

Each side is exposed as `algorithm_under_test` with identical case and metric
identities. The predeclaration fixes the workload family, single-call latency
outcome, 5% practical-equivalence margin, 2% precision target, and Bonferroni
family. It also binds the selection to the discovery input digest.

## The paired pilot plans a different experiment

Benchmatrix alternates adjacent AB/BA blocks. Both members use the same
balanced Williams-style matrix order, and complete pairs are stratified by
orientation during bootstrap resampling. The automatic pilot target completes
the joint orientation-by-order supercycle.

The paired log-ratio precision calculation estimates a fixed pair count for a
fresh future design. It is neither power analysis nor a sequential stopping
rule. The pilot's outcomes are never appended to the confirmatory sample.

## Fresh confirmation classifies intervals

Final collection requires a clean tree and the frozen pair count. Benchmatrix
uses the ratio of marginal run medians, a deterministic paired BCa bootstrap,
orientation-stratified resampling, and Bonferroni multiplicity control.

An adjusted interval wholly beyond the practical threshold is an improvement
or regression. An interval wholly inside the ±5% region is practically
equivalent. Every interval crossing a boundary is inconclusive; absence of a
detected regression is not evidence of equivalence.

The generated report preserves comparison JSON and Markdown, source and
predeclaration digests, a forest plot with the equivalence region, and a
technical narrative. Performance conclusions remain scoped to the recorded
host, dependencies, thread policy, datasets, and timing boundary.
