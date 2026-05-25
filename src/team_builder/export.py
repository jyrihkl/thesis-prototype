"""Run export utilities.

The reporting module handles human-readable formatting. This module handles
persistent run outputs so prototype results can later be used in thesis
evaluation and comparison.

A run export currently writes:

- report.txt: the same human-readable report printed by the CLI
- run_summary.json: compact machine-readable run metadata and metrics
- allocation.json: final main-method team allocation
- baseline_comparison.json: baseline comparison summary
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TypeGuard

from team_builder.models import AllocationReport, BaselineComparisonReport, PipelineRunResult
from team_builder.reporting import format_run_report


def make_run_id() -> str:
    """Create a filesystem-friendly timestamped run identifier."""

    return datetime.now().strftime("%Y%m%d-%H%M%S")

# Marking the returned TypeGuard Any is a hacky workaround, but shouldn't matter here
def _is_dataclass_instance(value: Any) -> TypeGuard[Any]:
    """Return True only for dataclass instances, not dataclass classes."""

    return is_dataclass(value) and not isinstance(value, type)


def _json_ready(value: Any) -> Any:
    """Convert dataclasses and non-JSON-native values into JSON-ready values."""

    if _is_dataclass_instance(value):
        return _json_ready(asdict(value))

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_ready(item) for item in value]

    return value

def write_json(path: Path, data: Any) -> None:
    """Write JSON with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(data), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def allocation_to_export(report: AllocationReport | None) -> dict[str, Any]:
    """Create a compact JSON representation of the final allocation."""

    if report is None:
        return {"status": "not_run"}

    return {
        "method": report.method,
        "feasible": report.feasible,
        "assigned_count": report.assigned_count,
        "unassigned_count": report.unassigned_count,
        "required_slots": report.required_slots,
        "objective_score": report.objective_score,
        "fairness_deviation": report.fairness_deviation,
        "min_team_score": report.min_team_score,
        "mean_team_score": report.mean_team_score,
        "max_team_score": report.max_team_score,
        "local_improvement": report.local_improvement,
        "teams": [
            {
                "project_id": summary.project_id,
                "project_title": summary.project_title,
                "member_ids": summary.member_ids,
                "target_team_size": summary.target_team_size,
                "team_score": summary.team_score,
                "components": summary.components,
                "covered_required_skills": summary.covered_required_skills,
                "missing_required_skills": summary.missing_required_skills,
                "covered_preferred_skills": summary.covered_preferred_skills,
                "role_families": summary.role_families,
            }
            for summary in report.project_summaries
        ],
        "unassigned_candidate_ids": report.unassigned_candidate_ids,
        "warnings": report.warnings,
    }


def baseline_comparison_to_export(report: BaselineComparisonReport | None) -> dict[str, Any]:
    """Create a compact JSON representation of baseline comparisons."""

    if report is None:
        return {"status": "not_run"}

    return {
        "main_method": report.main_method,
        "best_by_objective": report.best_by_objective,
        "best_by_min_team_score": report.best_by_min_team_score,
        "best_by_fairness": report.best_by_fairness,
        "method_summaries": report.method_summaries,
        "baseline_reports": [
            allocation_to_export(baseline)
            for baseline in report.baseline_reports
        ],
    }


def run_summary_to_export(result: PipelineRunResult, run_id: str) -> dict[str, Any]:
    """Create a compact machine-readable summary of a pipeline run."""

    allocation = result.allocation_report
    local_improvement = allocation.local_improvement if allocation else None
    baseline_report = result.baseline_comparison_report

    return {
        "run_id": run_id,
        "participants_path": result.participants_path,
        "projects_path": result.projects_path,
        "participant_count": result.participant_count,
        "project_count": result.project_count,
        "project_titles": result.project_titles,
        "required_slots": result.required_slots,
        "available_candidates": result.available_candidates,
        "validation": {
            "passed": result.validation_report.passed_count,
            "warnings": result.validation_report.warning_count,
            "failures": result.validation_report.failure_count,
            "checks": result.validation_report.checks,
        },
        "scoring": result.scoring_report,
        "main_allocation": {
            "method": allocation.method if allocation else None,
            "feasible": allocation.feasible if allocation else None,
            "objective_score": allocation.objective_score if allocation else None,
            "fairness_deviation": allocation.fairness_deviation if allocation else None,
            "min_team_score": allocation.min_team_score if allocation else None,
            "mean_team_score": allocation.mean_team_score if allocation else None,
            "max_team_score": allocation.max_team_score if allocation else None,
            "local_improvement_gain": (
                local_improvement.improvement_gain if local_improvement else None
            ),
            "accepted_swaps": (
                local_improvement.accepted_swaps if local_improvement else None
            ),
            "accepted_replacements": (
                local_improvement.accepted_replacements if local_improvement else None
            ),
        },
        "baseline_comparison": {
            "best_by_objective": (
                baseline_report.best_by_objective if baseline_report else None
            ),
            "best_by_min_team_score": (
                baseline_report.best_by_min_team_score if baseline_report else None
            ),
            "best_by_fairness": (
                baseline_report.best_by_fairness if baseline_report else None
            ),
            "method_summaries": (
                baseline_report.method_summaries if baseline_report else []
            ),
        },
    }


def export_run_result(
    result: PipelineRunResult,
    output_dir: Path,
    run_id: str | None = None,
) -> Path:
    """Export a pipeline run and return the created run directory."""

    resolved_run_id = run_id or make_run_id()
    run_dir = output_dir / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    (run_dir / "report.txt").write_text(
        format_run_report(result) + "\n",
        encoding="utf-8",
    )

    write_json(run_dir / "run_summary.json", run_summary_to_export(result, resolved_run_id))
    write_json(run_dir / "allocation.json", allocation_to_export(result.allocation_report))
    write_json(
        run_dir / "baseline_comparison.json",
        baseline_comparison_to_export(result.baseline_comparison_report),
    )

    return run_dir
