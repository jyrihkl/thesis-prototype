#!/usr/bin/env python3
"""Create a human-review CSV from a saved allocation.json file.

This helper is intentionally separate from the main pipeline. It converts a
saved recommendation output into a review form that an evaluator can fill
in manually.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REVIEW_FIELDS = [
    "run_id",
    "project_set",
    "participant_set",
    "weight_profile",
    "method",
    "project_id",
    "project_title",
    "team_member_ids",
    "team_score",
    "covered_required_skills",
    "missing_required_skills",
    "role_families",
    "perceived_usefulness_1_5",
    "perceived_fairness_1_5",
    "perceived_transparency_1_5",
    "perceived_team_fit_1_5",
    "would_accept_team_yes_no",
    "suggested_changes",
    "comments",
    "reviewer_id_optional",
    "review_date",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a review CSV from allocation.json."
    )
    parser.add_argument(
        "--allocation",
        type=Path,
        required=True,
        help="Path to a saved allocation.json file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path where the review CSV should be written.",
    )
    parser.add_argument("--run-id", default="", help="Run identifier to include.")
    parser.add_argument("--project-set", default="", help="Project set label.")
    parser.add_argument("--participant-set", default="", help="Participant set label.")
    parser.add_argument("--weight-profile", default="", help="Weight profile label.")
    return parser.parse_args()


def join_values(values: Any) -> str:
    """Format list-like allocation fields as semicolon-separated strings."""

    if values is None:
        return ""
    if isinstance(values, list):
        return "; ".join(str(value) for value in values)
    return str(values)


def main() -> int:
    args = parse_args()

    with args.allocation.open("r", encoding="utf-8") as file:
        allocation = json.load(file)

    method = allocation.get("method", "")
    teams = allocation.get("teams", [])

    rows: list[dict[str, str]] = []

    for team in teams:
        rows.append(
            {
                "run_id": args.run_id,
                "project_set": args.project_set,
                "participant_set": args.participant_set,
                "weight_profile": args.weight_profile,
                "method": method,
                "project_id": str(team.get("project_id", "")),
                "project_title": str(team.get("project_title", "")),
                "team_member_ids": join_values(team.get("member_ids", [])),
                "team_score": str(team.get("team_score", "")),
                "covered_required_skills": join_values(
                    team.get("covered_required_skills", [])
                ),
                "missing_required_skills": join_values(
                    team.get("missing_required_skills", [])
                ),
                "role_families": join_values(team.get("role_families", [])),
                "perceived_usefulness_1_5": "",
                "perceived_fairness_1_5": "",
                "perceived_transparency_1_5": "",
                "perceived_team_fit_1_5": "",
                "would_accept_team_yes_no": "",
                "suggested_changes": "",
                "comments": "",
                "reviewer_id_optional": "",
                "review_date": "",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote review template: {args.output}")
    print(f"Rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
