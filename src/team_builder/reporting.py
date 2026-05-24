"""Reporting utilities for pipeline runs.

For now, reports are printed to standard output.
"""

from __future__ import annotations

from pathlib import Path

from team_builder.models import (
    PipelineRunResult,
    ScoringReport,
    ValidationCheck,
    ValidationReport,
)


_STATUS_LABELS = {
    "pass": "OK",
    "warn": "WARN",
    "fail": "FAIL",
}


def _format_validation_check(check: ValidationCheck) -> str:
    """Format one validation check for human-readable output."""

    label = _STATUS_LABELS.get(check.status, check.status.upper())
    return f"  [{label}] {check.name}: {check.message}"


def _format_float(value: float | None) -> str:
    """Format optional floats in a compact report-friendly form."""

    if value is None:
        return "n/a"
    return f"{value:.3f}"


def format_validation_report(report: ValidationReport) -> str:
    """Format validation findings as a multi-line string."""

    lines = [
        "Validation",
        "-" * 40,
        (
            "Summary: "
            f"{report.passed_count} passed, "
            f"{report.warning_count} warning(s), "
            f"{report.failure_count} failure(s)"
        ),
    ]

    for check in report.checks:
        lines.append(_format_validation_check(check))

    return "\n".join(lines)


def format_scoring_report(report: ScoringReport | None) -> str:
    """Format candidate-to-project scoring information."""

    if report is None:
        return "\n".join(
            [
                "Candidate-to-project scoring",
                "-" * 40,
                "Status: scoring has not been run.",
            ]
        )

    lines = [
        "Candidate-to-project scoring",
        "-" * 40,
        f"Total pairs scored:   {report.total_pairs}",
        f"Feasible pairs:       {report.feasible_pairs}",
        "Weights:",
    ]

    for name, weight in report.weights.items():
        lines.append(f"  - {name}: {weight:.2f}")

    lines.append("")
    lines.append("Project score summaries:")

    for summary in report.project_summaries:
        top_candidates = ", ".join(summary.top_candidate_ids) or "n/a"
        lines.extend(
            [
                f"  - {summary.project_id} | {summary.project_title}",
                (
                    f"    candidates: {summary.scored_candidates}, "
                    f"feasible: {summary.feasible_candidates}, "
                    f"min/mean/max: "
                    f"{_format_float(summary.min_score)} / "
                    f"{_format_float(summary.mean_score)} / "
                    f"{_format_float(summary.max_score)}"
                ),
                f"    top candidates: {top_candidates}",
            ]
        )

    return "\n".join(lines)


def format_run_report(result: PipelineRunResult) -> str:
    """Format the current pipeline result as a report string."""

    lines = [
        "",
        "Team formation prototype pipeline",
        "=" * 40,
        f"Participants file:     {result.participants_path}",
        f"Projects file:         {result.projects_path}",
        f"Participants read:     {result.participant_count}",
        f"Projects read:         {result.project_count}",
        f"Required team slots:   {result.required_slots}",
        f"Available candidates:  {result.available_candidates}",
    ]

    if result.project_titles:
        lines.extend(["", "Projects:"])
        for title in result.project_titles:
            lines.append(f"  - {title}")

    lines.extend(
        [
            "",
            format_validation_report(result.validation_report),
            "",
            format_scoring_report(result.scoring_report),
        ]
    )

    if result.validation_report.has_failures:
        lines.extend(["", "Status: input loading completed with validation failures."])
    elif result.validation_report.has_warnings:
        lines.extend(["", "Status: scoring completed with validation warnings."])
    else:
        lines.extend(["", "Status: input loading, validation, and scoring completed successfully."])

    return "\n".join(lines)


def print_run_report(result: PipelineRunResult) -> None:
    """Print a pipeline report to standard output."""

    print(format_run_report(result))


def write_run_report(result: PipelineRunResult, path: Path) -> None:
    """Write a pipeline report to a text file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_run_report(result) + "\n", encoding="utf-8")
