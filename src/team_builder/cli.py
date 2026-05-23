"""Command-line interface for the team formation prototype."""

from __future__ import annotations

import argparse
from pathlib import Path

from team_builder.models import PipelineConfig
from team_builder.pipeline import run_pipeline
from team_builder.reporting import print_run_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the team formation prototype pipeline."
    )

    parser.add_argument(
        "--participants",
        type=Path,
        default=None,
        help=(
            "Path to the filtered participant file. "
            "Supported formats: .csv, .jsonl, .json. "
            "If omitted, the pipeline searches default locations."
        ),
    )

    parser.add_argument(
        "--projects",
        type=Path,
        default=None,
        help=(
            "Path to a project brief JSON file. "
            "If omitted, the pipeline searches default locations."
        ),
    )

    parser.add_argument(
        "--project-set",
        choices=["a", "b"],
        default="a",
        help="Default project set to use when --projects is omitted.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config = PipelineConfig(
        participants_path=args.participants,
        projects_path=args.projects,
        project_set=args.project_set,
    )

    try:
        result = run_pipeline(config)
    except Exception as exc:
        print(f"\nPipeline failed: {exc}")
        return 1

    print_run_report(result)
    return 0
