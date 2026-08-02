"""Deterministic discovery-result analysis and technical-report rendering."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, cast

from benchmatrix import BenchmarkRunGroup, ParsedBenchmarkRow, load_benchmark_run_group

_MANIFEST_NAME: Final = "benchmatrix-manifest.json"
_PLAN_NAME: Final = "discovery-plan.json"
_METRIC_DIRECTIONS: Final = {
    "single_call_latency": "lower",
    "batch_throughput": "higher",
    "tail_latency": "lower",
}
_METRIC_UNITS: Final = {
    "single_call_latency": "seconds",
    "batch_throughput": "queries_per_second",
    "tail_latency": "p95_seconds",
}


@dataclass(frozen=True, slots=True)
class DiscoveryQuestion:
    """One predeclared descriptive crossover comparison."""

    identifier: str
    baseline: str
    candidate: str
    factor: str
    question: str


DISCOVERY_QUESTIONS: Final = (
    DiscoveryQuestion(
        "argpartition_vs_full_sort",
        "numpy_full",
        "numpy_argpartition",
        "k",
        "When does partial selection outperform full sorting?",
    ),
    DiscoveryQuestion(
        "blocked_vs_argpartition",
        "numpy_argpartition",
        "numpy_blocked",
        "corpus_size",
        "When does bounded temporary memory justify blocked search?",
    ),
    DiscoveryQuestion(
        "python_heap_vs_argpartition",
        "python_heap",
        "numpy_argpartition",
        "dimension",
        "Where does vectorized computation overcome Python overhead?",
    ),
)


@dataclass(frozen=True, slots=True)
class DiscoveryRecord:
    """One run-level statistic for one benchmark matrix cell."""

    collection: str
    run_index: int
    source: str
    implementation: str
    case_name: str
    metric: str
    value: float
    unit: str
    objective: str
    profile: str
    corpus_size: int
    dimension: int
    query_count: int
    k: int
    dtype: str
    rounds: int
    iterations: int
    sample_count: int
    p50_seconds: float | None
    p90_seconds: float | None
    p95_seconds: float | None
    p99_seconds: float | None

    @property
    def cell(self) -> tuple[str, str, str]:
        """Return the stable benchmatrix cell identity."""
        return (self.implementation, self.case_name, self.metric)


@dataclass(frozen=True, slots=True)
class DiscoverySummary:
    """Across-run descriptive summary for one matrix cell."""

    implementation: str
    case_name: str
    metric: str
    unit: str
    objective: str
    profile: str
    corpus_size: int
    dimension: int
    query_count: int
    k: int
    dtype: str
    run_count: int
    median_value: float
    minimum_value: float
    maximum_value: float
    run_cv: float | None

    @property
    def cell(self) -> tuple[str, str, str]:
        """Return the stable benchmatrix cell identity."""
        return (self.implementation, self.case_name, self.metric)


@dataclass(frozen=True, slots=True)
class DiscoveryWinner:
    """Fastest implementation for one case and metric."""

    case_name: str
    metric: str
    implementation: str
    value: float
    unit: str
    runner_up: str | None
    advantage_percent: float | None


@dataclass(frozen=True, slots=True)
class PairwiseResult:
    """One descriptive candidate-versus-baseline comparison."""

    question: str
    baseline: str
    candidate: str
    case_name: str
    objective: str
    corpus_size: int
    dimension: int
    query_count: int
    k: int
    baseline_seconds: float
    candidate_seconds: float
    candidate_improvement_percent: float


@dataclass(frozen=True, slots=True)
class EvidenceAudit:
    """Qualification of discovery inputs for descriptive or formal use."""

    collection_count: int
    record_count: int
    cell_count: int
    minimum_runs_per_cell: int
    minimum_rounds_per_run: int
    minimum_tail_samples_per_run: int | None
    all_collections_complete: bool
    all_sources_clean: bool
    single_revision: bool
    single_source_tree_digest: bool
    formal_ready: bool
    revisions: tuple[str, ...]
    source_tree_digests: tuple[str, ...]
    issues: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DiscoveryAnalysis:
    """Complete in-memory discovery analysis."""

    records: tuple[DiscoveryRecord, ...]
    summaries: tuple[DiscoverySummary, ...]
    winners: tuple[DiscoveryWinner, ...]
    pairwise: tuple[PairwiseResult, ...]
    audit: EvidenceAudit
    input_digest: str


def discover_collection_directories(roots: Iterable[Path]) -> tuple[Path, ...]:
    """Find ordinary benchmatrix collection directories below input roots."""
    discovered: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        candidates = [resolved / _MANIFEST_NAME] if resolved.is_dir() else []
        if resolved.is_dir():
            candidates.extend(resolved.rglob(_MANIFEST_NAME))
        elif resolved.name == _MANIFEST_NAME:
            candidates.append(resolved)
        for manifest in candidates:
            if not manifest.is_file():
                continue
            payload = _load_mapping(manifest)
            if payload.get("kind") == "benchmark_run_group":
                discovered.add(manifest.parent)
    if not discovered:
        raise ValueError("no discovery benchmatrix manifests found")
    return tuple(sorted(discovered))


def analyze_discovery(collections: Sequence[Path]) -> DiscoveryAnalysis:
    """Load, validate, and summarize discovery collections."""
    if not collections:
        raise ValueError("at least one discovery collection is required")
    records: list[DiscoveryRecord] = []
    groups: list[tuple[Path, BenchmarkRunGroup, Mapping[str, object]]] = []
    source_paths: list[Path] = []
    for collection in sorted(path.resolve() for path in collections):
        group = load_benchmark_run_group(collection)
        plan_path = collection / _PLAN_NAME
        if not plan_path.is_file():
            raise ValueError(f"missing discovery plan: {plan_path}")
        plan = _load_mapping(plan_path)
        groups.append((collection, group, plan))
        source_paths.extend((collection / _MANIFEST_NAME, plan_path))
        for record in group.records:
            if record.path.is_file():
                source_paths.append(record.path)
        for run_index, run in enumerate(group.runs, start=1):
            for row in run.rows:
                records.append(_record_from_row(collection, run_index, row, str(run.source or "")))
    summaries = summarize_records(records)
    return DiscoveryAnalysis(
        records=tuple(records),
        summaries=summaries,
        winners=select_winners(summaries),
        pairwise=pairwise_results(summaries),
        audit=audit_evidence(records, groups),
        input_digest=_digest_files(source_paths),
    )


def summarize_records(records: Sequence[DiscoveryRecord]) -> tuple[DiscoverySummary, ...]:
    """Aggregate run-level values without treating timing rounds as replicates."""
    grouped: dict[tuple[str, str, str], list[DiscoveryRecord]] = defaultdict(list)
    for record in records:
        grouped[record.cell].append(record)
    summaries: list[DiscoverySummary] = []
    for key in sorted(grouped):
        group = grouped[key]
        exemplar = group[0]
        values = [record.value for record in group]
        mean = statistics.fmean(values)
        cv = statistics.stdev(values) / mean if len(values) >= 2 and mean != 0.0 else None
        summaries.append(
            DiscoverySummary(
                implementation=exemplar.implementation,
                case_name=exemplar.case_name,
                metric=exemplar.metric,
                unit=exemplar.unit,
                objective=exemplar.objective,
                profile=exemplar.profile,
                corpus_size=exemplar.corpus_size,
                dimension=exemplar.dimension,
                query_count=exemplar.query_count,
                k=exemplar.k,
                dtype=exemplar.dtype,
                run_count=len(values),
                median_value=statistics.median(values),
                minimum_value=min(values),
                maximum_value=max(values),
                run_cv=cv,
            )
        )
    return tuple(summaries)


def select_winners(summaries: Sequence[DiscoverySummary]) -> tuple[DiscoveryWinner, ...]:
    """Select a descriptive winner for every observed case and metric."""
    grouped: dict[tuple[str, str], list[DiscoverySummary]] = defaultdict(list)
    for summary in summaries:
        grouped[(summary.case_name, summary.metric)].append(summary)
    winners: list[DiscoveryWinner] = []
    for key in sorted(grouped):
        case_name, metric = key
        direction = _METRIC_DIRECTIONS[metric]
        ordered = sorted(
            grouped[key],
            key=lambda item: item.median_value,
            reverse=direction == "higher",
        )
        best = ordered[0]
        runner_up = ordered[1] if len(ordered) > 1 else None
        advantage: float | None = None
        if runner_up is not None and runner_up.median_value != 0.0:
            if direction == "lower":
                advantage = 100.0 * (runner_up.median_value - best.median_value) / runner_up.median_value
            else:
                advantage = 100.0 * (best.median_value - runner_up.median_value) / runner_up.median_value
        winners.append(
            DiscoveryWinner(
                case_name=case_name,
                metric=metric,
                implementation=best.implementation,
                value=best.median_value,
                unit=best.unit,
                runner_up=None if runner_up is None else runner_up.implementation,
                advantage_percent=advantage,
            )
        )
    return tuple(winners)


def pairwise_results(summaries: Sequence[DiscoverySummary]) -> tuple[PairwiseResult, ...]:
    """Calculate predeclared exploratory latency contrasts."""
    latency = {
        (summary.case_name, summary.implementation): summary
        for summary in summaries
        if summary.metric == "single_call_latency"
    }
    results: list[PairwiseResult] = []
    for question in DISCOVERY_QUESTIONS:
        case_names = sorted(case for case, implementation in latency if implementation == question.baseline)
        for case_name in case_names:
            baseline = latency[(case_name, question.baseline)]
            candidate = latency.get((case_name, question.candidate))
            if candidate is None or not _question_case_is_relevant(question, baseline):
                continue
            improvement = 100.0 * (baseline.median_value - candidate.median_value) / baseline.median_value
            results.append(
                PairwiseResult(
                    question=question.identifier,
                    baseline=question.baseline,
                    candidate=question.candidate,
                    case_name=case_name,
                    objective=baseline.objective,
                    corpus_size=baseline.corpus_size,
                    dimension=baseline.dimension,
                    query_count=baseline.query_count,
                    k=baseline.k,
                    baseline_seconds=baseline.median_value,
                    candidate_seconds=candidate.median_value,
                    candidate_improvement_percent=improvement,
                )
            )
    return tuple(results)


def audit_evidence(
    records: Sequence[DiscoveryRecord],
    groups: Sequence[tuple[Path, BenchmarkRunGroup, Mapping[str, object]]],
) -> EvidenceAudit:
    """Audit collection lifecycle, source provenance, and evidence depth."""
    runs_per_cell = Counter(record.cell for record in records)
    minimum_runs = min(runs_per_cell.values(), default=0)
    minimum_rounds = min((record.rounds for record in records), default=0)
    tail_counts = [record.sample_count for record in records if record.metric == "tail_latency"]
    minimum_tail = min(tail_counts) if tail_counts else None
    revisions: set[str] = set()
    digests: set[str] = set()
    clean_flags: list[bool] = []
    complete_flags: list[bool] = []
    issues: list[str] = []
    for collection, group, plan in groups:
        complete_flags.append(group.is_complete)
        provenance = _source_provenance(plan, collection)
        revision = _required_str(provenance, "git_revision")
        digest = _required_str(provenance, "source_tree_digest")
        revisions.add(revision)
        digests.add(digest)
        clean = not _required_bool(provenance, "git_dirty")
        clean_flags.append(clean)
        if not group.is_complete:
            issues.append(f"incomplete collection: {collection}")
        if not clean:
            issues.append(f"dirty source tree: {collection}")
    if minimum_runs < 5:
        issues.append(f"minimum independent runs per cell is {minimum_runs}; formal policy requires 5")
    if minimum_rounds < 5:
        issues.append(f"minimum measured rounds is {minimum_rounds}; formal policy requires 5")
    if minimum_tail is not None and minimum_tail < 100:
        issues.append(f"minimum tail samples is {minimum_tail}; formal policy requires 100")
    if len(revisions) != 1:
        issues.append("collections do not share one Git revision")
    if len(digests) != 1:
        issues.append("collections do not share one source-tree digest")
    all_complete = bool(complete_flags) and all(complete_flags)
    all_clean = bool(clean_flags) and all(clean_flags)
    formal_ready = not issues and all_complete and all_clean
    return EvidenceAudit(
        collection_count=len(groups),
        record_count=len(records),
        cell_count=len(runs_per_cell),
        minimum_runs_per_cell=minimum_runs,
        minimum_rounds_per_run=minimum_rounds,
        minimum_tail_samples_per_run=minimum_tail,
        all_collections_complete=all_complete,
        all_sources_clean=all_clean,
        single_revision=len(revisions) == 1,
        single_source_tree_digest=len(digests) == 1,
        formal_ready=formal_ready,
        revisions=tuple(sorted(revisions)),
        source_tree_digests=tuple(sorted(digests)),
        issues=tuple(dict.fromkeys(issues)),
    )


def write_analysis(analysis: DiscoveryAnalysis, output: Path) -> None:
    """Write strict machine-readable tables, plots, and a technical report."""
    output.mkdir(parents=True, exist_ok=False)
    plots = output / "plots"
    plots.mkdir()
    _write_csv(output / "measurements.csv", analysis.records)
    _write_csv(output / "summary.csv", analysis.summaries)
    _write_csv(output / "winners.csv", analysis.winners)
    _write_csv(output / "pairwise.csv", analysis.pairwise)
    chart_map = render_plots(analysis, plots)
    _write_json(output / "chart-map.json", chart_map)
    _write_json(
        output / "analysis.json",
        {
            "schema_version": 1,
            "study": "exact_top_k_vector_search",
            "analysis_kind": "discovery",
            "input_digest": analysis.input_digest,
            "audit": asdict(analysis.audit),
            "counts": {
                "records": len(analysis.records),
                "summaries": len(analysis.summaries),
                "winners": len(analysis.winners),
                "pairwise": len(analysis.pairwise),
            },
            "questions": [asdict(question) for question in DISCOVERY_QUESTIONS],
        },
    )
    (output / "technical-report.md").write_text(render_technical_report(analysis), encoding="utf-8")


def render_plots(analysis: DiscoveryAnalysis, output: Path) -> list[dict[str, object]]:
    """Render deterministic, source-backed static charts."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "axes.edgecolor": "#475569",
            "axes.labelcolor": "#1f2937",
            "axes.titlecolor": "#111827",
            "figure.facecolor": "white",
            "font.family": "DejaVu Sans",
            "savefig.facecolor": "white",
            "svg.hashsalt": "vector-search-study-v1",
            "text.color": "#1f2937",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
        }
    )
    chart_map: list[dict[str, object]] = []
    counts = Counter(winner.implementation for winner in analysis.winners if winner.metric == "single_call_latency")
    if counts:
        labels, values = zip(*sorted(counts.items(), key=lambda item: (item[1], item[0])), strict=True)
        figure, axis = plt.subplots(figsize=(9.5, max(4.5, len(labels) * 0.42)))
        axis.barh(labels, values, color="#2563eb", edgecolor="#1e3a8a", linewidth=0.8)
        axis.set_title("Discovery latency winner counts", loc="left", fontweight="bold")
        axis.set_xlabel("Number of observed workload cases")
        axis.grid(axis="x", color="#e2e8f0", linewidth=0.8)
        axis.set_axisbelow(True)
        for index, value in enumerate(values):
            axis.text(value + 0.15, index, str(value), va="center", fontsize=9)
        figure.tight_layout()
        paths = _save_figure(figure, output / "latency-winner-counts")
        plt.close(figure)
        chart_map.append(
            _chart_contract(
                "Latency winner counts",
                "Which implementation is descriptively fastest most often?",
                "Comparison & Ranking",
                "sorted horizontal bar",
                len(counts),
                paths,
            )
        )
    marker_by_objective = {"squared_l2": "o", "inner_product": "s", "normalized_cosine": "^"}
    color_by_objective = {"squared_l2": "#2563eb", "inner_product": "#d97706", "normalized_cosine": "#6b7f2a"}
    for question in DISCOVERY_QUESTIONS:
        rows = [row for row in analysis.pairwise if row.question == question.identifier]
        if not rows:
            continue
        rows.sort(key=lambda row: (row.objective, _factor_value(row, question.factor)))
        labels = [f"{row.objective} · {question.factor}={_factor_value(row, question.factor)}" for row in rows]
        figure, axis = plt.subplots(figsize=(10.5, max(4.8, len(rows) * 0.42)))
        axis.axvline(0.0, color="#334155", linewidth=1.2, linestyle="--")
        for index, row in enumerate(rows):
            axis.scatter(
                row.candidate_improvement_percent,
                index,
                color=color_by_objective[row.objective],
                edgecolor="#1f2937",
                marker=marker_by_objective[row.objective],
                s=62,
                linewidth=0.7,
                zorder=3,
            )
        axis.set_yticks(range(len(labels)), labels)
        axis.set_xlabel(f"{question.candidate} latency improvement over {question.baseline} (%)")
        axis.set_title(
            f"{question.candidate} vs {question.baseline} latency",
            loc="left",
            fontweight="bold",
            pad=26,
        )
        axis.text(
            0.0,
            1.015,
            f"{question.question} Descriptive across-run medians; zero marks parity.",
            transform=axis.transAxes,
            color="#475569",
            fontsize=9,
        )
        axis.grid(axis="x", color="#e2e8f0", linewidth=0.8)
        axis.set_axisbelow(True)
        figure.tight_layout()
        paths = _save_figure(figure, output / question.identifier)
        plt.close(figure)
        chart_map.append(
            _chart_contract(
                question.question,
                "Where does the candidate cross the zero-improvement reference?",
                "Uncertainty & Benchmark",
                "faceted dot with parity reference",
                len(rows),
                paths,
            )
        )
    return chart_map


def render_technical_report(analysis: DiscoveryAnalysis) -> str:
    """Render an answer-first technical Markdown report."""
    status = "formal-ready" if analysis.audit.formal_ready else "exploratory only"
    latency_winners = Counter(
        winner.implementation for winner in analysis.winners if winner.metric == "single_call_latency"
    )
    leading = latency_winners.most_common(3)
    leader_text = ", ".join(f"`{name}` ({count})" for name, count in leading) or "none"
    pairwise_lines = []
    for question in DISCOVERY_QUESTIONS:
        rows = [row for row in analysis.pairwise if row.question == question.identifier]
        if not rows:
            continue
        improvements = [row.candidate_improvement_percent for row in rows]
        pairwise_lines.append(
            f"* `{question.identifier}`: median candidate improvement "
            f"{statistics.median(improvements):+.1f}% across {len(rows)} observed contrasts."
        )
    issue_lines = [f"* {issue}." for issue in analysis.audit.issues] or ["* No evidence-policy gaps were detected."]
    revision = analysis.audit.revisions[0] if len(analysis.audit.revisions) == 1 else "multiple revisions"
    lines = [
        "# Exact vector search discovery report",
        "",
        "## Technical summary",
        "",
        f"This collection is **{status}**. It contains {analysis.audit.record_count} run-level records across "
        f"{analysis.audit.cell_count} matrix cells from revision `{revision}`. Descriptive latency winners are led "
        f"by {leader_text}. These ranks reveal candidates for paired experiments; they are not inferential claims.",
        "",
        "## Key findings and visual evidence",
        "",
        "Winner counts summarize how often each implementation has the lowest across-run median latency. A win count "
        "does not weight workload importance or quantify uncertainty.",
        "",
        "![Discovery latency winner counts](plots/latency-winner-counts.png)",
        "",
        *pairwise_lines,
        "",
    ]
    for question in DISCOVERY_QUESTIONS:
        if any(row.question == question.identifier for row in analysis.pairwise):
            lines.extend(
                (
                    f"The `{question.identifier}` dot plot shows descriptive candidate improvement; points to the "
                    "right favor the candidate and the zero line marks parity.",
                    "",
                    f"![{question.question}](plots/{question.identifier}.png)",
                    "",
                )
            )
    lines.extend(
        (
            "## Scope, data, and metric definitions",
            "",
            "The study measures exact top-k search over deterministic synthetic float32 data for squared L2, inner "
            "product, and normalized cosine. Single-call latency is the median elapsed search time within each "
            "process run. Throughput is queries per second. Tail latency is the within-run p95 of measured calls and "
            "is not a production service-level percentile.",
            "",
            "## Methodology",
            "",
            "Every timed result is checked against the untimed scalable oracle. This analysis aggregates one statistic "
            "per independently launched run; raw timing rounds are never treated as independent replicates. Winners "
            "and pairwise percentages are descriptive. Formal claims require benchmatrix's paired AB/BA design, "
            "stratified run-level BCa bootstrap, practical-equivalence classification, and Bonferroni control.",
            "",
            "## Limitations, uncertainty, and robustness checks",
            "",
            *issue_lines,
            "",
            "The one-factor-at-a-time design reveals interpretable scaling slices but cannot estimate interactions "
            "among corpus size, dimension, query batch, and k. Index construction and preprocessing remain outside "
            "the timed search operation.",
            "",
            "## Recommended next steps",
            "",
            "Use `pairwise.csv` only to choose and freeze confirmatory families. Run a separate paired pilot for "
            "precision planning, then collect a fresh fixed-size experiment; do not append pilot outcomes to the "
            "confirmatory sample.",
            "",
            "## Further questions",
            "",
            "* Do the observed crossovers persist on native Linux hardware without x86 virtualization?",
            "* How do natural embedding distributions shift the selected comparisons?",
            "* Which apparent wins survive a fresh paired design and the practical-equivalence margin?",
            "",
            f"Input digest: `{analysis.input_digest}`.",
            "",
        )
    )
    return "\n".join(lines)


def _record_from_row(collection: Path, run_index: int, row: ParsedBenchmarkRow, source: str) -> DiscoveryRecord:
    """Convert one validated benchmatrix row to the study schema."""
    metric = row.metric_name
    derived_key = {
        "single_call_latency": "latency_median",
        "batch_throughput": "throughput_median",
        "tail_latency": "p95",
    }[metric]
    value = _required_float(row.derived, derived_key)
    p50 = _optional_float(row.derived.get("p50"))
    p90 = _optional_float(row.derived.get("p90"))
    p95 = _optional_float(row.derived.get("p95"))
    p99 = _optional_float(row.derived.get("p99"))
    return DiscoveryRecord(
        collection=str(collection),
        run_index=run_index,
        source=source,
        implementation=row.implementation_name,
        case_name=row.case_name,
        metric=metric,
        value=value,
        unit=_METRIC_UNITS[metric],
        objective=_required_str(row.extra_info, "case_objective"),
        profile=_required_str(row.extra_info, "case_profile"),
        corpus_size=_required_int(row.extra_info, "case_corpus_size"),
        dimension=_required_int(row.extra_info, "case_dimension"),
        query_count=_required_int(row.extra_info, "case_query_count"),
        k=_required_int(row.extra_info, "case_k"),
        dtype=_required_str(row.extra_info, "case_dtype"),
        rounds=_required_int(row.stats, "rounds"),
        iterations=_required_int(row.stats, "iterations"),
        sample_count=len(row.samples),
        p50_seconds=p50,
        p90_seconds=p90,
        p95_seconds=p95,
        p99_seconds=p99,
    )


def _question_case_is_relevant(question: DiscoveryQuestion, summary: DiscoverySummary) -> bool:
    """Apply fixed one-factor slices for each descriptive question."""
    if question.factor == "k":
        return summary.corpus_size == 10_000 and summary.dimension == 128 and summary.query_count == 32
    if question.factor == "corpus_size":
        return summary.dimension == 128 and summary.query_count == 32 and summary.k == 10
    return summary.profile == "discovery-small" and summary.corpus_size == 1_000 and summary.query_count == 1


def _factor_value(row: PairwiseResult, factor: str) -> int:
    """Return one fixed factor value from a pairwise row."""
    return cast(int, getattr(row, factor))


def _source_provenance(plan: Mapping[str, object], collection: Path) -> Mapping[str, object]:
    """Return the source-provenance mapping from a discovery plan."""
    collection_meta = plan.get("collection")
    if not isinstance(collection_meta, Mapping):
        raise ValueError(f"missing collection metadata in {collection / _PLAN_NAME}")
    provenance = collection_meta.get("source_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"missing source provenance in {collection / _PLAN_NAME}")
    return cast(Mapping[str, object], provenance)


def _chart_contract(
    title: str,
    question: str,
    family: str,
    chart_type: str,
    rows: int,
    paths: Sequence[str],
) -> dict[str, object]:
    """Return a compact auditable visualization contract."""
    return {
        "title": title,
        "analytical_question": question,
        "family": family,
        "chart_type": chart_type,
        "row_count": rows,
        "palette_policy": "hard_two_root_cap_plus_neutrals",
        "non_color_distinction": "direct labels, ordering, marker shapes, and parity reference",
        "outputs": list(paths),
        "source": "summary.csv and pairwise.csv",
    }


def _save_figure(figure: object, stem: Path) -> tuple[str, str]:
    """Save one Matplotlib figure as deterministic PNG and SVG outputs."""
    savefig = cast(object, figure).__getattribute__("savefig")
    png = stem.with_suffix(".png")
    svg = stem.with_suffix(".svg")
    savefig(png, dpi=160, metadata={"Software": "vector-search-study"})
    savefig(svg, metadata={"Date": None, "Creator": "vector-search-study"})
    return (png.name, svg.name)


def _write_csv(path: Path, rows: Sequence[object]) -> None:
    """Write dataclass rows using a stable column order."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    dictionaries = [cast(dict[str, object], asdict(cast(Any, row))) for row in rows]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dictionaries[0]))
        writer.writeheader()
        writer.writerows(dictionaries)


def _write_json(path: Path, value: object) -> None:
    """Write strict deterministic JSON."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _load_mapping(path: Path) -> Mapping[str, object]:
    """Load one JSON object."""
    payload = cast(object, json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected a JSON object: {path}")
    return cast(Mapping[str, object], payload)


def _digest_files(paths: Iterable[Path]) -> str:
    """Hash every unique input artifact with canonical path ordering."""
    digest = hashlib.sha256(b"vector-search-study-discovery-analysis-v1\0")
    resolved = sorted({path.resolve() for path in paths})
    for path in resolved:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def _required_str(mapping: Mapping[str, object], key: str) -> str:
    """Return one required non-empty string."""
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _required_int(mapping: Mapping[str, object], key: str) -> int:
    """Return one required non-boolean integer."""
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _required_bool(mapping: Mapping[str, object], key: str) -> bool:
    """Return one required Boolean."""
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _required_float(mapping: Mapping[str, object], key: str) -> float:
    """Return one required finite numeric value."""
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{key} must be finite")
    return numeric


def _optional_float(value: object) -> float | None:
    """Return an optional finite float."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("optional value must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("optional value must be finite")
    return numeric
