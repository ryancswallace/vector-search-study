"""Collect the complete discovery study with clean-source and stress isolation gates."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from scripts.run_discovery import source_provenance_record
from vector_search_study.benchmarking import profile_specs

_PROFILES: Final = ("small", "core", "stress")


@dataclass(frozen=True, slots=True)
class DiscoveryTask:
    """One independently collected discovery profile or stress workload."""

    profile: str
    output: str
    pytest_filter: str | None


def build_parser() -> argparse.ArgumentParser:
    """Build the full-discovery command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profiles", nargs="+", choices=_PROFILES, default=list(_PROFILES))
    parser.add_argument("--runs", type=_positive_int, default=2)
    parser.add_argument("--rounds", type=_positive_int, default=10)
    parser.add_argument("--tail-rounds", type=_positive_int, default=100)
    parser.add_argument("--warmup-rounds", type=_nonnegative_int, default=2)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Collect every requested profile sequentially and preserve one study manifest."""
    args = build_parser().parse_args(argv)
    root = Path.cwd().resolve()
    provenance = source_provenance_record(root)
    if provenance["git_dirty"] and not args.allow_dirty:
        raise SystemExit("discovery collection requires a clean Git tree; use --allow-dirty only for pilots")
    tasks = discovery_tasks(args.output, args.profiles)
    if args.dry_run:
        print(json.dumps([asdict(task) for task in tasks], indent=2, sort_keys=True))
        return 0
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    args.output.mkdir(parents=True)
    environment = os.environ.copy()
    environment.update(
        {
            "VECTOR_SEARCH_BENCHMARK_ROUNDS": str(args.rounds),
            "VECTOR_SEARCH_BENCHMARK_TAIL_ROUNDS": str(args.tail_rounds),
            "VECTOR_SEARCH_BENCHMARK_WARMUP_ROUNDS": str(args.warmup_rounds),
        }
    )
    results: list[dict[str, object]] = []
    for index, task in enumerate(tasks, start=1):
        command = [
            sys.executable,
            "scripts/run_discovery.py",
            "--profile",
            task.profile,
            "--runs",
            str(args.runs),
            "--output",
            task.output,
        ]
        if task.pytest_filter is not None:
            command.extend(("--pytest-filter", task.pytest_filter))
        started = time.monotonic()
        completed = subprocess.run(command, cwd=root, env=environment, check=False)
        result = {
            "index": index,
            **asdict(task),
            "returncode": completed.returncode,
            "elapsed_seconds": time.monotonic() - started,
        }
        results.append(result)
        _write_study_manifest(
            args.output,
            provenance=provenance,
            requested_runs=args.runs,
            rounds=args.rounds,
            tail_rounds=args.tail_rounds,
            warmup_rounds=args.warmup_rounds,
            tasks=tasks,
            results=results,
            complete=False,
        )
        if completed.returncode != 0:
            return completed.returncode
    after = source_provenance_record(root)
    if after["source_tree_digest"] != provenance["source_tree_digest"]:
        raise RuntimeError("source tree changed during discovery collection")
    _write_study_manifest(
        args.output,
        provenance=provenance,
        requested_runs=args.runs,
        rounds=args.rounds,
        tail_rounds=args.tail_rounds,
        warmup_rounds=args.warmup_rounds,
        tasks=tasks,
        results=results,
        complete=True,
    )
    return 0


def discovery_tasks(output: Path, profiles: list[str]) -> tuple[DiscoveryTask, ...]:
    """Expand profiles into sequential standard and isolated stress tasks."""
    tasks: list[DiscoveryTask] = []
    for profile in profiles:
        if profile != "stress":
            tasks.append(DiscoveryTask(profile, str(output / profile), None))
            continue
        for spec in profile_specs("stress"):
            slug = (
                f"{spec.objective.value}-n{spec.corpus_size}-d{spec.dimension}"
                f"-q{spec.query_count}-k{spec.k}-{spec.dtype}"
            )
            tasks.append(
                DiscoveryTask(
                    profile,
                    str(output / "stress" / slug),
                    spec.name.split("__", maxsplit=1)[1],
                )
            )
    return tuple(tasks)


def _write_study_manifest(
    output: Path,
    *,
    provenance: dict[str, object],
    requested_runs: int,
    rounds: int,
    tail_rounds: int,
    warmup_rounds: int,
    tasks: tuple[DiscoveryTask, ...],
    results: list[dict[str, object]],
    complete: bool,
) -> None:
    """Persist strict lifecycle state after every isolated task."""
    payload = {
        "schema_version": 1,
        "study": "exact_top_k_vector_search",
        "kind": "discovery_study",
        "complete": complete,
        "source_provenance": provenance,
        "collection": {
            "requested_runs": requested_runs,
            "rounds": rounds,
            "tail_rounds": tail_rounds,
            "warmup_rounds": warmup_rounds,
        },
        "tasks": [asdict(task) for task in tasks],
        "results": results,
    }
    (output / "discovery-study-manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


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
