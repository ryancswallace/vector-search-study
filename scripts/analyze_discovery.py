"""Analyze benchmatrix discovery collections and render research artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.discovery_analysis import analyze_discovery, discover_collection_directories, write_analysis


def build_parser() -> argparse.ArgumentParser:
    """Build the discovery-analysis command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Collection directories or roots containing them.")
    parser.add_argument("--output", type=Path, required=True, help="New analysis output directory.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run deterministic discovery analysis."""
    args = build_parser().parse_args(argv)
    collections = discover_collection_directories(args.inputs)
    analysis = analyze_discovery(collections)
    write_analysis(analysis, args.output)
    readiness = "formal-ready" if analysis.audit.formal_ready else "exploratory"
    print(f"Analyzed {len(collections)} collection(s) and {len(analysis.records)} run-level rows ({readiness}).")
    print(f"Report: {args.output / 'technical-report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
