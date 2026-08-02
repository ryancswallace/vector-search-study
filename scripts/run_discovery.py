"""Collect one discovery profile and preserve its plan and resource usage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Final

from vector_search_study.benchmarking import discovery_plan_metadata

_TARGETS: Final = {
    "small": "benchmarks/test_discovery_small.py",
    "core": "benchmarks/test_discovery_core.py",
    "stress": "benchmarks/test_discovery_stress.py",
}


def build_parser() -> argparse.ArgumentParser:
    """Build the discovery collection argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(_TARGETS), required=True)
    parser.add_argument("--runs", type=_positive_int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pytest-filter", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run benchmatrix and write deterministic planning and resource records."""
    args = build_parser().parse_args(argv)
    command = [
        sys.executable,
        "-m",
        "benchmatrix",
        "measure",
        "--runs",
        str(args.runs),
        "--output",
        str(args.output),
        _TARGETS[args.profile],
    ]
    if args.pytest_filter:
        command.extend(("--", "-k", args.pytest_filter))

    source_provenance = source_provenance_record(Path.cwd())
    started = time.monotonic()
    completed = subprocess.run(command, check=False)
    elapsed_seconds = time.monotonic() - started
    if completed.returncode != 0:
        return completed.returncode

    plan = discovery_plan_metadata(args.profile)
    plan["collection"] = {
        "requested_runs": args.runs,
        "pytest_filter": args.pytest_filter,
        "source_provenance": source_provenance,
    }
    _write_json(args.output / "discovery-plan.json", plan)
    _write_json(
        args.output / "resource-usage.json",
        {
            "schema_version": 1,
            "profile": args.profile,
            "elapsed_seconds": elapsed_seconds,
            "maximum_resident_set_bytes": _child_maximum_resident_set_bytes(),
            "measurement_scope": "resource.RUSAGE_CHILDREN",
            "source_tree_digest": source_provenance["source_tree_digest"],
        },
    )
    return 0


def _child_maximum_resident_set_bytes() -> int:
    """Normalize child-process peak RSS to bytes on Linux and macOS."""
    maximum = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    multiplier = 1 if sys.platform == "darwin" else 1_024
    return int(maximum * multiplier)


def _positive_int(value: str) -> int:
    """Parse a positive command-line integer."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def source_provenance_record(root: Path) -> dict[str, object]:
    """Return revision, lock, and complete non-ignored source-tree digests."""
    revision = _git_output(root, "rev-parse", "HEAD").decode().strip()
    status = _git_output(root, "status", "--porcelain=v1").decode().splitlines()
    listed = _git_output(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
    paths = sorted(path for path in listed.split(b"\0") if path)
    digest = hashlib.sha256(b"vector-search-study-source-tree-v1\0")
    for encoded_path in paths:
        path = root / os.fsdecode(encoded_path)
        if not path.is_file():
            continue
        digest.update(encoded_path)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    lock_digest = hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()
    return {
        "git_revision": revision,
        "git_dirty": bool(status),
        "git_status": status,
        "source_tree_digest": f"sha256:{digest.hexdigest()}",
        "uv_lock_digest": f"sha256:{lock_digest}",
    }


def _git_output(root: Path, *args: str) -> bytes:
    """Run one fixed Git provenance query."""
    completed = subprocess.run(
        ("git", *args),
        cwd=root,
        check=True,
        capture_output=True,
    )
    return completed.stdout


def _write_json(path: Path, value: object) -> None:
    """Write one strict, stable JSON artifact."""
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
