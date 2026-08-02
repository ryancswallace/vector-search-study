"""Predeclare, collect, precision-plan, and report paired experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Final, cast

from benchmarks.confirmatory_experiments import EXPERIMENTS, get_experiment, predeclaration_manifest
from benchmatrix import format_comparison_report_markdown, load_comparison_report
from scripts.confirmatory_reporting import render_confirmatory_artifacts
from scripts.run_discovery import source_provenance_record

_RANDOM_SEED: Final = 20_260_801


def build_parser() -> argparse.ArgumentParser:
    """Build the staged confirmatory-workflow parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    predeclare = subparsers.add_parser("predeclare", help="Bind the registry to one discovery analysis.")
    predeclare.add_argument("--discovery-analysis", type=Path, required=True)
    predeclare.add_argument("--output", type=Path, required=True)

    pilot = subparsers.add_parser("collect-pilot", help="Collect a paired pilot for precision planning.")
    _add_collection_arguments(pilot)
    pilot.add_argument("--pairs", type=_positive_int, default=None)

    plan = subparsers.add_parser("plan", help="Create a fresh-design pair-count plan from a paired pilot.")
    _add_experiment_argument(plan)
    plan.add_argument("--pilot", type=Path, required=True)
    plan.add_argument("--predeclaration", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)

    final = subparsers.add_parser("collect-final", help="Collect the fresh fixed-size confirmatory design.")
    _add_collection_arguments(final)
    final.add_argument("--precision-plan", type=Path, required=True)
    final.add_argument("--maximum-pairs", type=_positive_int, default=200)

    compare = subparsers.add_parser("compare-final", help="Apply paired BCa inference and write reports.")
    _add_experiment_argument(compare)
    compare.add_argument("--collection", type=Path, required=True)
    compare.add_argument("--predeclaration", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute one explicit confirmatory workflow stage."""
    args = build_parser().parse_args(argv)
    if args.command == "predeclare":
        return _predeclare(args.discovery_analysis, args.output)
    if args.command == "collect-pilot":
        return _collect(
            experiment_name=args.experiment,
            predeclaration=args.predeclaration,
            output=args.output,
            pair_count=args.pairs,
            rounds=args.rounds,
            warmup_rounds=args.warmup_rounds,
            allow_dirty=args.allow_dirty,
            stage="precision_pilot",
        )
    if args.command == "plan":
        return _plan(args.experiment, args.pilot, args.predeclaration, args.output)
    if args.command == "collect-final":
        plan = _load_mapping(args.precision_plan)
        pair_count = _required_int(plan, "required_pairs")
        if pair_count > args.maximum_pairs:
            raise SystemExit(
                f"precision plan requires {pair_count} pairs, above --maximum-pairs={args.maximum_pairs}; "
                "revise resources or the predeclared precision target"
            )
        return _collect(
            experiment_name=args.experiment,
            predeclaration=args.predeclaration,
            output=args.output,
            pair_count=pair_count,
            rounds=args.rounds,
            warmup_rounds=args.warmup_rounds,
            allow_dirty=args.allow_dirty,
            stage="fresh_confirmatory",
        )
    return _compare_final(args.experiment, args.collection, args.predeclaration, args.output)


def _add_experiment_argument(parser: argparse.ArgumentParser) -> None:
    """Add the registered experiment selector."""
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)


def _add_collection_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common paired-collection controls."""
    _add_experiment_argument(parser)
    parser.add_argument("--predeclaration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=_positive_int, default=10)
    parser.add_argument("--warmup-rounds", type=_nonnegative_int, default=2)
    parser.add_argument("--allow-dirty", action="store_true")


def _predeclare(discovery_analysis: Path, output: Path) -> int:
    """Freeze comparison families against one immutable discovery analysis."""
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    analysis = _load_mapping(discovery_analysis)
    if analysis.get("analysis_kind") != "discovery":
        raise SystemExit("predeclaration input must be a discovery analysis")
    input_digest = _required_str(analysis, "input_digest")
    manifest = predeclaration_manifest()
    manifest["discovery_analysis"] = {
        "path": str(discovery_analysis.resolve()),
        "input_digest": input_digest,
        "formal_ready": _analysis_formal_ready(analysis),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, manifest)
    return 0


def _collect(
    *,
    experiment_name: str,
    predeclaration: Path,
    output: Path,
    pair_count: int | None,
    rounds: int,
    warmup_rounds: int,
    allow_dirty: bool,
    stage: str,
) -> int:
    """Collect one atomic, balanced paired design."""
    experiment = get_experiment(experiment_name)
    _validate_predeclaration(predeclaration, experiment_name)
    root = Path.cwd().resolve()
    provenance = source_provenance_record(root)
    if provenance["git_dirty"] and not allow_dirty:
        raise SystemExit("paired collection requires a clean Git tree; use --allow-dirty only for development pilots")
    if stage == "fresh_confirmatory" and allow_dirty:
        raise SystemExit("fresh confirmatory collection never permits a dirty source tree")
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    common = [
        "env",
        "LOKY_MAX_CPU_COUNT=1",
        "OMP_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
        "VECLIB_MAXIMUM_THREADS=1",
        f"VECTOR_SEARCH_CONFIRMATORY_EXPERIMENT={experiment_name}",
        f"VECTOR_SEARCH_BENCHMARK_ROUNDS={rounds}",
        f"VECTOR_SEARCH_BENCHMARK_WARMUP_ROUNDS={warmup_rounds}",
    ]
    pytest_command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--benchmark-quiet",
        "-o",
        "addopts=",
        "benchmarks/test_confirmatory.py",
    ]
    baseline = [*common, "VECTOR_SEARCH_CONFIRMATORY_VARIANT=baseline", *pytest_command]
    candidate = [*common, "VECTOR_SEARCH_CONFIRMATORY_VARIANT=candidate", *pytest_command]
    command = [
        sys.executable,
        "-m",
        "benchmatrix",
        "collect-paired",
        "--random-seed",
        str(_RANDOM_SEED),
        "--output",
        str(output),
        "--baseline-cwd",
        str(root),
        "--candidate-cwd",
        str(root),
    ]
    if pair_count is not None:
        command.extend(("--pairs", str(pair_count)))
    command.extend(("--", *baseline, ":::", *candidate))
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        return completed.returncode
    _write_json(
        output / "confirmatory-collection.json",
        {
            "schema_version": 1,
            "stage": stage,
            "experiment": experiment.metadata(),
            "source_provenance": provenance,
            "predeclaration": {
                "path": str(predeclaration.resolve()),
                "digest": _digest_file(predeclaration),
            },
            "requested_pairs": pair_count,
            "rounds": rounds,
            "warmup_rounds": warmup_rounds,
        },
    )
    return 0


def _plan(experiment_name: str, pilot: Path, predeclaration: Path, output: Path) -> int:
    """Estimate a fresh fixed-size design from paired pilot variability."""
    experiment = get_experiment(experiment_name)
    _validate_predeclaration(predeclaration, experiment_name)
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir(parents=True)
    report_path = output / "pilot-comparison.json"
    _run_comparison(
        pilot,
        report_path,
        precision_target=experiment.precision_target_half_width_percent,
    )
    report = _load_mapping(report_path)
    plans = []
    for comparison in _required_list(report, "comparisons"):
        if not isinstance(comparison, dict):
            raise SystemExit("comparison report contains an invalid cell")
        precision = comparison.get("precision")
        if not isinstance(precision, dict):
            raise SystemExit("paired pilot did not produce a precision plan")
        required = precision.get("required_pairs")
        if isinstance(required, bool) or not isinstance(required, int):
            issues = precision.get("issues")
            raise SystemExit(f"paired pilot precision is not estimable: {issues}")
        plans.append((cast(dict[str, object], comparison), cast(dict[str, object], precision)))
    required_pairs = max(_required_int(precision, "required_pairs") for _comparison, precision in plans)
    supercycles = {_required_int(precision, "pair_count_multiple") for _comparison, precision in plans}
    if len(supercycles) != 1:
        raise SystemExit("precision plans disagree on the paired design supercycle")
    plan_payload = {
        "schema_version": 1,
        "stage": "fresh_design_precision_plan",
        "experiment": experiment_name,
        "pilot_reuse_prohibited": True,
        "pilot_collection": str(pilot.resolve()),
        "pilot_comparison_digest": _digest_file(report_path),
        "predeclaration_digest": _digest_file(predeclaration),
        "target_half_width_percent": experiment.precision_target_half_width_percent,
        "required_pairs": required_pairs,
        "pair_count_multiple": next(iter(supercycles)),
        "cell_plans": [
            {
                "case_name": _required_str(comparison, "case_name"),
                "required_pairs": _required_int(precision, "required_pairs"),
                "unconstrained_required_pairs": _required_int(precision, "unconstrained_required_pairs"),
                "pilot_log_ratio_standard_deviation": precision.get("pilot_log_ratio_standard_deviation"),
            }
            for comparison, precision in plans
        ],
    }
    _write_json(output / "precision-plan.json", plan_payload)
    typed_report = load_comparison_report(report_path)
    (output / "pilot-comparison.md").write_text(
        format_comparison_report_markdown(typed_report),
        encoding="utf-8",
    )
    return 0


def _compare_final(experiment_name: str, collection: Path, predeclaration: Path, output: Path) -> int:
    """Run final paired inference without changing the fixed design."""
    experiment = get_experiment(experiment_name)
    _validate_predeclaration(predeclaration, experiment_name)
    collection_metadata = _load_mapping(collection / "confirmatory-collection.json")
    if collection_metadata.get("stage") != "fresh_confirmatory":
        raise SystemExit("final inference requires a fresh_confirmatory collection")
    provenance = collection_metadata.get("source_provenance")
    if not isinstance(provenance, dict) or provenance.get("git_dirty") is not False:
        raise SystemExit("final inference requires clean-source collection provenance")
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir(parents=True)
    report_path = output / "comparison.json"
    _run_comparison(collection, report_path, precision_target=None)
    report = load_comparison_report(report_path)
    (output / "comparison.md").write_text(format_comparison_report_markdown(report), encoding="utf-8")
    _write_json(
        output / "confirmatory-analysis.json",
        {
            "schema_version": 1,
            "stage": "final_confirmatory_analysis",
            "experiment": experiment_name,
            "collection": str(collection.resolve()),
            "collection_digest": _digest_file(collection / "benchmatrix-manifest.json"),
            "predeclaration_digest": _digest_file(predeclaration),
            "comparison_digest": _digest_file(report_path),
            "design": "paired_ab_ba_balanced_cell_order",
            "pilot_reused": False,
            "passed": report.passed,
            "is_comparable": report.is_comparable,
            "summary": report.to_dict()["summary"],
        },
    )
    render_confirmatory_artifacts(report_path, experiment, output)
    return 0


def _run_comparison(collection: Path, output: Path, *, precision_target: float | None) -> None:
    """Run benchmatrix paired inference and validate the emitted report."""
    command = [
        sys.executable,
        "-m",
        "benchmatrix",
        "compare",
        str(collection),
        "--paired",
        "--format",
        "json",
    ]
    if precision_target is not None:
        command.extend(("--precision-target", str(precision_target)))
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise SystemExit(completed.stderr or completed.stdout)
    payload = cast(object, json.loads(completed.stdout))
    _write_json(output, payload)
    _ = load_comparison_report(output)


def _validate_predeclaration(path: Path, experiment_name: str) -> None:
    """Require a discovery-bound plan containing the selected experiment."""
    payload = _load_mapping(path)
    discovery = payload.get("discovery_analysis")
    if not isinstance(discovery, dict) or not discovery.get("input_digest"):
        raise SystemExit("predeclaration is not bound to a discovery analysis")
    experiments = _required_list(payload, "experiments")
    identifiers = {
        item.get("identifier")
        for item in experiments
        if isinstance(item, dict) and isinstance(item.get("identifier"), str)
    }
    if experiment_name not in identifiers:
        raise SystemExit(f"experiment {experiment_name!r} is absent from the predeclaration")


def _analysis_formal_ready(analysis: Mapping[str, object]) -> bool:
    """Return the discovery evidence qualification recorded by analysis."""
    audit = analysis.get("audit")
    return isinstance(audit, dict) and audit.get("formal_ready") is True


def _digest_file(path: Path) -> str:
    """Return a content digest for one workflow artifact."""
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _load_mapping(path: Path) -> dict[str, object]:
    """Load one strict JSON object."""
    payload = cast(object, json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(payload, dict):
        raise SystemExit(f"expected a JSON object: {path}")
    return cast(dict[str, object], payload)


def _write_json(path: Path, value: object) -> None:
    """Write strict deterministic JSON."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _required_str(mapping: Mapping[str, object], key: str) -> str:
    """Return one required string."""
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise SystemExit(f"{key} must be a non-empty string")
    return value


def _required_int(mapping: Mapping[str, object], key: str) -> int:
    """Return one required integer."""
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SystemExit(f"{key} must be an integer")
    return value


def _required_list(mapping: Mapping[str, object], key: str) -> list[object]:
    """Return one required list."""
    value = mapping.get(key)
    if not isinstance(value, list):
        raise SystemExit(f"{key} must be a list")
    return cast(list[object], value)


def _positive_int(value: str) -> int:
    """Parse a positive integer."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    """Parse a non-negative integer."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
