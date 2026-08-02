"""Render paired confirmatory effect plots and technical reports."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from benchmarks.confirmatory_experiments import ConfirmatoryExperiment


def render_confirmatory_artifacts(
    comparison_path: Path,
    experiment: ConfirmatoryExperiment,
    output: Path,
) -> None:
    """Render a source-backed interval chart, chart contract, and report."""
    payload = _load_mapping(comparison_path)
    comparisons = _comparisons(payload)
    plot_paths = _render_effect_plot(comparisons, experiment, output)
    _write_json(
        output / "chart-map.json",
        [
            {
                "title": "Paired latency effects and adjusted intervals",
                "analytical_question": experiment.question,
                "family": "Uncertainty & Benchmark",
                "chart_type": "forest plot with practical-equivalence region",
                "row_count": len(comparisons),
                "fields": [
                    "case_name",
                    "improvement_percent",
                    "improvement_low_percent",
                    "improvement_high_percent",
                    "threshold_percent",
                ],
                "palette_policy": "single_root_preferred_plus_neutrals",
                "non_color_distinction": "intervals, markers, direct labels, and equivalence-region shading",
                "outputs": [path.name for path in plot_paths],
                "source": comparison_path.name,
            }
        ],
    )
    (output / "technical-report.md").write_text(
        _technical_report(payload, comparisons, experiment),
        encoding="utf-8",
    )


def _render_effect_plot(
    comparisons: list[Mapping[str, object]],
    experiment: ConfirmatoryExperiment,
    output: Path,
) -> tuple[Path, Path]:
    """Render an honest adjusted-interval forest plot."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "font.family": "DejaVu Sans",
            "savefig.facecolor": "white",
            "svg.hashsalt": "vector-search-study-confirmatory-v1",
            "text.color": "#1f2937",
        }
    )
    labels: list[str] = []
    estimates: list[float] = []
    lower_errors: list[float] = []
    upper_errors: list[float] = []
    for comparison in comparisons:
        estimate = _required_float(comparison, "improvement_percent")
        low = _required_float(comparison, "improvement_low_percent")
        high = _required_float(comparison, "improvement_high_percent")
        labels.append(_short_case_label(_required_str(comparison, "case_name")))
        estimates.append(estimate)
        lower_errors.append(estimate - low)
        upper_errors.append(high - estimate)
    threshold = experiment.equivalence_margin_percent
    figure, axis = plt.subplots(figsize=(10.5, max(4.6, len(labels) * 0.8)))
    axis.axvspan(-threshold, threshold, color="#e2e8f0", alpha=0.8, label="practical equivalence")
    axis.axvline(0.0, color="#334155", linestyle="--", linewidth=1.1)
    positions = list(range(len(labels)))
    axis.errorbar(
        estimates,
        positions,
        xerr=[lower_errors, upper_errors],
        fmt="o",
        color="#2563eb",
        ecolor="#1e3a8a",
        markeredgecolor="#172554",
        capsize=4,
        linewidth=1.5,
    )
    axis.set_yticks(positions, labels)
    axis.set_xlabel(f"{experiment.candidate} improvement over {experiment.baseline} (%)")
    axis.set_title("Paired latency effects and adjusted intervals", loc="left", fontweight="bold")
    axis.grid(axis="x", color="#e2e8f0", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="best")
    figure.tight_layout()
    png = output / "effect-intervals.png"
    svg = output / "effect-intervals.svg"
    figure.savefig(png, dpi=160, metadata={"Software": "vector-search-study"})
    figure.savefig(svg, metadata={"Date": None, "Creator": "vector-search-study"})
    plt.close(figure)
    return (png, svg)


def _technical_report(
    payload: Mapping[str, object],
    comparisons: list[Mapping[str, object]],
    experiment: ConfirmatoryExperiment,
) -> str:
    """Render the required technical report structure."""
    classifications = Counter(_required_str(comparison, "regression") for comparison in comparisons)
    summary_text = ", ".join(f"{count} {name}" for name, count in sorted(classifications.items()))
    findings = [
        (
            f"* `{_short_case_label(_required_str(comparison, 'case_name'))}`: "
            f"{_required_float(comparison, 'improvement_percent'):+.2f}% "
            f"[{_required_float(comparison, 'improvement_low_percent'):+.2f}%, "
            f"{_required_float(comparison, 'improvement_high_percent'):+.2f}%], "
            f"classified **{_required_str(comparison, 'regression')}**."
        )
        for comparison in comparisons
    ]
    paired = payload.get("paired_collections")
    pair_count = "unknown"
    if isinstance(paired, list) and paired and isinstance(paired[0], dict):
        value = paired[0].get("complete_pairs")
        if isinstance(value, int):
            pair_count = str(value)
    overall = "passed" if payload.get("passed") is True else "did not pass every decision gate"
    return "\n".join(
        (
            f"# {experiment.identifier} confirmatory report",
            "",
            "## Technical summary",
            "",
            f"The fresh paired experiment {overall}. Across {len(comparisons)} predeclared cells, the classifications "
            f"were {summary_text}. The design contains {pair_count} complete adjacent pairs and does not reuse the "
            "precision pilot.",
            "",
            "## Adjusted effects answer the predeclared question",
            "",
            f"{experiment.question} Positive effects favor `{experiment.candidate}`. Intervals are "
            "multiplicity-adjusted, "
            f"and the shaded ±{experiment.equivalence_margin_percent:.1f}% region is the practical-equivalence margin.",
            "",
            "![Paired latency effects and adjusted intervals](effect-intervals.png)",
            "",
            *findings,
            "",
            "## Scope, data, and metric definitions",
            "",
            "The outcome is median single-call exact-search latency. Every process run reconstructs the index and "
            "prepares queries outside timing, then validates the measured result against the trusted oracle. Each "
            "workload cell belongs to one predeclared multiplicity family.",
            "",
            "## Paired experimental and inferential design",
            "",
            "Benchmatrix alternates AB/BA blocks, gives both members the same balanced matrix-cell order, and "
            "resamples "
            "complete matched process-run pairs stratified by orientation. It reports a direction-aware ratio of "
            "marginal medians with a deterministic BCa interval and Bonferroni family-wise control.",
            "",
            "## Limitations and robustness",
            "",
            "The result applies to the recorded CPU, software environment, synthetic distribution, dtype, and fixed "
            "thread policy. It does not include index construction or production concurrency. Failure to establish an "
            "improvement is not equivalence unless the complete adjusted interval lies inside the equivalence region.",
            "",
            "## Recommended next steps",
            "",
            "Replicate any decision-relevant result on native Linux hardware and on the pinned natural-embedding slice "
            "before making a broad systems recommendation.",
            "",
            "## Further questions",
            "",
            "* Does the classification persist under float64 where every backend supports it?",
            "* How sensitive is the result to BLAS implementation and CPU cache topology?",
            "* Does natural embedding structure move the observed crossover?",
            "",
        )
    )


def _comparisons(payload: Mapping[str, object]) -> list[Mapping[str, object]]:
    """Return matched comparison rows with complete intervals."""
    value = payload.get("comparisons")
    if not isinstance(value, list) or not value:
        raise ValueError("comparison report must contain cells")
    rows: list[Mapping[str, object]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("comparison cells must be objects")
        rows.append(cast(Mapping[str, object], item))
    return rows


def _short_case_label(case_name: str) -> str:
    """Remove the long confirmatory profile prefix from a case label."""
    parts = case_name.split("__")
    return " · ".join(parts[2:]) if len(parts) > 2 else case_name


def _load_mapping(path: Path) -> Mapping[str, object]:
    """Load one JSON object."""
    payload = cast(object, json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return cast(Mapping[str, object], payload)


def _required_str(mapping: Mapping[str, object], key: str) -> str:
    """Return one required string."""
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _required_float(mapping: Mapping[str, object], key: str) -> float:
    """Return one required numeric value."""
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def _write_json(path: Path, value: object) -> None:
    """Write strict deterministic JSON."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
