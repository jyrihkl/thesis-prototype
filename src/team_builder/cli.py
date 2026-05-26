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
        "--participant-set",
        default="080",
        help=(
            "Named participant set to use when --participants is omitted. "
            "Valid values: 080, 120, 240, 480, 1200, 2400. "
            "Aliases: 80, default, legacy."
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

    parser.add_argument(
        "--no-save-run",
        action="store_true",
        help="Do not write run outputs to disk.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs"),
        help="Directory where timestamped run folders are written.",
    )

    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run folder name. If omitted, a timestamped run ID is generated.",
    )

    parser.add_argument(
        "--no-local-improvement",
        action="store_true",
        help="Disable the local improvement phase.",
    )

    parser.add_argument(
        "--max-local-improvement-iterations",
        type=int,
        default=100,
        help="Maximum number of accepted local-improvement iterations.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config = PipelineConfig(
        participants_path=args.participants,
        participant_set=args.participant_set,
        projects_path=args.projects,
        project_set=args.project_set,
        enable_local_improvement=not args.no_local_improvement,
        max_local_improvement_iterations=args.max_local_improvement_iterations,
        save_run=not args.no_save_run,
        output_dir=args.output_dir,
        run_id=args.run_id,
    )

    try:
        result = run_pipeline(config)
    except Exception as exc:
        print(f"\nPipeline failed: {exc}")
        return 1

    print_run_report(result)
    
    if result.saved_run_dir is not None:
        print(f"\nSaved run outputs to: {result.saved_run_dir}")
    
    return 0
