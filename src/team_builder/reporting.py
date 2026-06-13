"""Reporting utilities for pipeline runs.

For now, reports are printed to standard output.
"""

from __future__ import annotations

from pathlib import Path

from team_builder.models import (
    AllocationReport,
    BaselineComparisonReport,
    LocalImprovementReport,
    PipelineRunResult,
    ScoringReport,
    TimingReport,
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


def _format_mean_std(value: float | None, std: float | None) -> str:
    """Format a point estimate and optional standard deviation."""

    if value is None:
        return "n/a"
    if std is None:
        return _format_float(value)
    return f"{value:.3f} ± {std:.3f}"


def _format_component_dict(components: dict[str, float | None]) -> str:
    """Format transparent score components on one line."""

    parts = [
        f"{name}={_format_float(value)}"
        for name, value in components.items()
        if value is not None
    ]
    return ", ".join(parts) if parts else "n/a"

def _shorten_id(full_id: str, max_length: int = 8) -> str:
    """Shorten long IDs for more compact report formatting."""

    if len(full_id) <= max_length:
        return full_id
    return full_id[:max_length]

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


def format_timing_report(report: TimingReport | None) -> str:
    """Format pipeline runtime information."""

    if report is None:
        return "\n".join(
            [
                "Timing",
                "-" * 40,
                "Status: timing information is not available.",
            ]
        )

    lines = [
        "Timing",
        "-" * 40,
    ]

    for name, seconds in report.stages.items():
        lines.append(f"  - {name}: {seconds:.4f}s")

    return "\n".join(lines)


def format_local_improvement_report(report: LocalImprovementReport | None) -> str:
    """Format local improvement information."""

    if report is None:
        return "Local improvement: n/a"

    if not report.enabled:
        return "Local improvement: disabled"

    lines = [
        "Local improvement",
        "-" * 40,
        (
            "Objective before/after: "
            f"{_format_float(report.initial_objective_score)} / "
            f"{_format_float(report.final_objective_score)}"
        ),
        f"Improvement gain:       {_format_float(report.improvement_gain)}",
        f"Accepted swaps:         {report.accepted_swaps}",
        f"Accepted replacements:  {report.accepted_replacements}",
        f"Evaluated swaps:        {report.evaluated_swaps}",
        f"Evaluated replacements: {report.evaluated_replacements}",
        f"Iterations:             {report.iterations}",
        f"Stop reason:            {report.stop_reason}",
    ]

    if report.accepted_moves:
        lines.extend(["", "Accepted moves:"])
        for move in report.accepted_moves:
            lines.append(
                f"  - iteration {move.iteration}: {move.move_type}, "
                f"gain={_format_float(move.gain)} | {move.description}"
            )
    else:
        lines.extend(["", "Accepted moves: none"])

    return "\n".join(lines)



def format_allocation_report(report: AllocationReport | None) -> str:
    """Format round-based allocation information."""

    if report is None:
        return "\n".join(
            [
                "Round-based allocation",
                "-" * 40,
                "Status: allocation has not been run.",
            ]
        )

    lines = [
        "Round-based allocation",
        "-" * 40,
        f"Method:              {report.method}",
        f"Feasible allocation: {'yes' if report.feasible else 'no'}",
        f"Assigned candidates: {report.assigned_count} / {report.required_slots}",
        f"Unassigned pool:     {report.unassigned_count}",
        f"Objective score:     {_format_float(report.objective_score)}",
        f"Fairness deviation:  {_format_float(report.fairness_deviation)}",
        (
            "Team score min/mean/max: "
            f"{_format_float(report.min_team_score)} / "
            f"{_format_float(report.mean_team_score)} / "
            f"{_format_float(report.max_team_score)}"
        ),
        "",
        format_local_improvement_report(report.local_improvement),
        "",
        "Teams:",
    ]

    for summary in report.project_summaries:
        members = ", ".join(summary.member_ids) or "none"
        missing = ", ".join(summary.missing_required_skills) or "none"
        covered_required = ", ".join(summary.covered_required_skills) or "none"
        covered_preferred = ", ".join(summary.covered_preferred_skills) or "none"
        roles = ", ".join(summary.role_families) or "none"

        lines.extend(
            [
                f"  - {summary.project_id} | {summary.project_title}",
                (
                    f"    members: {len(summary.member_ids)} / "
                    f"{summary.target_team_size} | score: {_format_float(summary.team_score)}"
                ),
                f"    member IDs: {members}",
                f"    components: {_format_component_dict(summary.components)}",
                f"    covered required skills: {covered_required}",
                f"    missing required skills: {missing}",
                f"    covered preferred skills: {covered_preferred}",
                f"    role families: {roles}",
            ]
        )

    if report.warnings:
        lines.extend(["", "Allocation warnings:"])
        for warning in report.warnings:
            lines.append(f"  - {warning}")

    return "\n".join(lines)


def format_baseline_comparison_report(report: BaselineComparisonReport | None) -> str:
    """Format comparative evaluation against baseline methods."""

    if report is None:
        return "\n".join(
            [
                "Baseline comparison",
                "-" * 40,
                "Status: baseline comparison has not been run.",
            ]
        )

    lines = [
        "Baseline comparison",
        "-" * 40,
        f"Main method:              {report.main_method}",
        f"Best objective score:     {report.best_by_objective or 'n/a'}",
        f"Best minimum team score:  {report.best_by_min_team_score or 'n/a'}",
        f"Best fairness deviation:  {report.best_by_fairness or 'n/a'}",
        "",
        "Method summaries:",
        (
            "  method | feasible runs | objective | mean team | min team | "
            "max team | fairness dev. | assigned"
        ),
    ]

    for summary in report.method_summaries:
        lines.append(
            "  "
            f"{summary.method} | "
            f"{summary.feasible_count}/{summary.sample_count} | "
            f"{_format_mean_std(summary.objective_score, summary.objective_score_std)} | "
            f"{_format_mean_std(summary.mean_team_score, summary.mean_team_score_std)} | "
            f"{_format_mean_std(summary.min_team_score, summary.min_team_score_std)} | "
            f"{_format_mean_std(summary.max_team_score, summary.max_team_score_std)} | "
            f"{_format_mean_std(summary.fairness_deviation, summary.fairness_deviation_std)} | "
            f"{summary.assigned_count}/{summary.required_slots}"
        )

    baseline_warnings = [
        (baseline.method, baseline.warnings)
        for baseline in report.baseline_reports
        if baseline.warnings
    ]

    if baseline_warnings:
        lines.extend(["", "Baseline warnings:"])
        for method, warnings in baseline_warnings:
            lines.append(f"  - {method}:")
            for warning in warnings:
                lines.append(f"      - {warning}")

    random_warnings = [
        run for run in report.random_run_summaries if run.warnings
    ]
    if random_warnings:
        lines.extend(["", "Random baseline warnings:"])
        for run in random_warnings:
            lines.append(f"  - seed {run.seed}:")
            for warning in run.warnings:
                lines.append(f"      - {warning}")

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

    if result.saved_run_dir is not None:
        lines.append(f"Saved run directory:   {result.saved_run_dir}")

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
            "",
            format_allocation_report(result.allocation_report),
            "",
            format_baseline_comparison_report(result.baseline_comparison_report),
            "",
            format_timing_report(result.timing_report),
        ]
    )

    if result.validation_report.has_failures:
        lines.extend(["", "Status: input loading completed with validation failures."])
    elif result.allocation_report is not None and not result.allocation_report.feasible:
        lines.extend(["", "Status: scoring completed, but allocation is incomplete."])
    elif result.validation_report.has_warnings:
        lines.extend(["", "Status: comparative evaluation completed with validation warnings."])
    else:
        lines.extend(
            [
                "",
                "Status: input loading, validation, scoring, allocation, local improvement, and baseline comparison completed successfully.",
            ]
        )

    return "\n".join(lines)


def print_run_report(result: PipelineRunResult) -> None:
    """Print a pipeline report to standard output."""

    print(format_run_report(result))


def write_run_report(result: PipelineRunResult, path: Path) -> None:
    """Write a pipeline report to a text file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_run_report(result) + "\n", encoding="utf-8")
