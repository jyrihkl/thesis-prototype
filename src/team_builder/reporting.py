"""Reporting utilities for pipeline runs.

For now, reports are printed to standard output.
"""

from __future__ import annotations

from pathlib import Path

from team_builder.models import PipelineRunResult, ValidationCheck, ValidationReport


_STATUS_LABELS = {
    "pass": "OK",
    "warn": "WARN",
    "fail": "FAIL",
}


def _format_validation_check(check: ValidationCheck) -> str:
    """Format one validation check for human-readable output."""

    label = _STATUS_LABELS.get(check.status, check.status.upper())
    return f"  [{label}] {check.name}: {check.message}"


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

    lines.extend(["", format_validation_report(result.validation_report)])

    if result.validation_report.has_failures:
        lines.extend(["", "Status: input loading completed with validation failures."])
    elif result.validation_report.has_warnings:
        lines.extend(["", "Status: input loading completed with validation warnings."])
    else:
        lines.extend(["", "Status: input loading and validation completed successfully."])

    return "\n".join(lines)


def print_run_report(result: PipelineRunResult) -> None:
    """Print a pipeline report to standard output."""

    print(format_run_report(result))


def write_run_report(result: PipelineRunResult, path: Path) -> None:
    """Write a pipeline report to a text file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_run_report(result) + "\n", encoding="utf-8")
