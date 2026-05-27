"""Command-line interface for batch evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from team_builder.batch import (
    DEFAULT_PROJECT_SETS,
    BatchEvaluationConfig,
    run_batch_evaluation,
)
from team_builder.weights import WEIGHT_PROFILES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batch evaluations."
    )

    parser.add_argument(
        "--participant-set",
        default="080",
        help=(
            "Named participant set, for example 080, 120, 240, 480, 1200, or 2400."
        ),
    )

    parser.add_argument(
        "--project-sets",
        nargs="+",
        default=list(DEFAULT_PROJECT_SETS),
        choices=list(DEFAULT_PROJECT_SETS),
        help="Project sets to evaluate.",
    )

    parser.add_argument(
        "--weight-profiles",
        nargs="+",
        default=["default"],
        choices=sorted(WEIGHT_PROFILES),
        help="Named score-weight profiles to evaluate.",
    )

    parser.add_argument(
        "--all-weight-profiles",
        action="store_true",
        help="Run all available score-weight profiles.",
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/batches"),
        help="Directory where batch folders are written.",
    )

    parser.add_argument(
        "--batch-id",
        default=None,
        help="Optional batch folder name. If omitted, a timestamped ID is generated.",
    )

    parser.add_argument(
        "--no-save-individual-runs",
        action="store_true",
        help="Only write aggregate batch outputs, not individual run folders.",
    )

    parser.add_argument(
        "--fairness-penalty",
        type=float,
        default=0.25,
        help="Fairness deviation penalty used by allocation methods.",
    )

    parser.add_argument(
        "--no-local-improvement",
        action="store_true",
        help="Disable local improvement for the main method.",
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

    weight_profiles = (
        tuple(sorted(WEIGHT_PROFILES))
        if args.all_weight_profiles
        else tuple(args.weight_profiles)
    )
    
    config = BatchEvaluationConfig(
        participant_set=args.participant_set,
        project_sets=tuple(args.project_sets),
        weight_profiles=weight_profiles,
        output_dir=args.output_dir,
        batch_id=args.batch_id,
        save_individual_runs=not args.no_save_individual_runs,
        fairness_penalty=args.fairness_penalty,
        enable_local_improvement=not args.no_local_improvement,
        max_local_improvement_iterations=args.max_local_improvement_iterations,
    )

    try:
        result = run_batch_evaluation(config)
    except Exception as exc:
        print(f"\nBatch evaluation failed: {exc}")
        return 1

    print("\nTeam formation batch evaluation")
    print("=" * 40)
    print(f"Batch ID:         {result.batch_id}")
    print(f"Batch directory:  {result.batch_dir}")
    print(f"Participant set:  {result.participant_set}")
    print(f"Project sets:     {', '.join(result.project_sets)}")
    print(f"Weight profiles:  {', '.join(result.weight_profiles)}")
    print(f"Runs completed:   {result.run_count}")
    print("\nOutputs:")
    print(f"  - {result.batch_runs_csv}")
    print(f"  - {result.batch_methods_csv}")
    print(f"  - {result.batch_summary_json}")
    print(f"  - {result.batch_report_txt}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
