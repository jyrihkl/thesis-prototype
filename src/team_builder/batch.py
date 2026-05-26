"""Batch evaluation.

A batch evaluation runs the normal pipeline repeatedly across selected project
sets and score-weight profiles. Each individual run can still write its normal
run outputs, while the batch writes additional aggregate files for plotting and
comparison.

Main batch outputs:

- batch_runs.csv: one row per pipeline run, focused on the main method
- batch_methods.csv: long-format method comparison, one row per method per run
- batch_summary.json: machine-readable batch metadata and aggregate information
- batch_report.txt: short human-readable batch report
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from team_builder.models import PipelineConfig, PipelineRunResult
from team_builder.pipeline import run_pipeline
from team_builder.weights import WEIGHT_PROFILES, get_weight_profile


DEFAULT_PROJECT_SETS = ("a", "b", "c", "d", "e")


@dataclass(frozen=True)
class BatchEvaluationConfig:
    """Runtime configuration for a batch evaluation."""

    participant_set: str
    project_sets: tuple[str, ...] = DEFAULT_PROJECT_SETS
    weight_profiles: tuple[str, ...] = ("default",)
    output_dir: Path = Path("runs/batches")
    batch_id: str | None = None
    save_individual_runs: bool = True
    fairness_penalty: float = 0.25
    enable_local_improvement: bool = True
    max_local_improvement_iterations: int = 100


@dataclass(frozen=True)
class BatchEvaluationResult:
    """Summary of a completed batch evaluation."""

    batch_id: str
    batch_dir: Path
    participant_set: str
    project_sets: tuple[str, ...]
    weight_profiles: tuple[str, ...]
    run_count: int
    batch_runs_csv: Path
    batch_methods_csv: Path
    batch_summary_json: Path
    batch_report_txt: Path


def make_batch_id() -> str:
    """Create a filesystem-friendly timestamped batch identifier."""

    return "batch-" + datetime.now().strftime("%Y%m%d-%H%M%S")


def participant_path_for_set(participant_set: str) -> Path:
    """Return the default processed participant path for a named participant set."""

    normalized = participant_set.strip().lower().replace("participants_", "")
    if normalized.isdigit() and len(normalized) < 3:
        normalized = normalized.zfill(3)

    return Path("data") / "processed" / "participants" / f"candidates_{normalized}.csv"


def _json_ready(value: Any) -> Any:
    """Convert paths and dataclasses into JSON-ready objects."""

    if hasattr(value, "__dataclass_fields__") and not isinstance(value, type):
        return _json_ready(asdict(value))

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_ready(item) for item in value]

    return value


def _write_json(path: Path, data: Any) -> None:
    """Write JSON with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(data), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write rows to CSV, even when the row list is empty."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _timing_value(result: PipelineRunResult, name: str) -> float | None:
    """Read one timing value from a pipeline result."""

    if result.timing_report is None:
        return None

    return result.timing_report.stages.get(name)


def _main_run_row(
    *,
    batch_id: str,
    run_id: str,
    participant_set: str,
    project_set: str,
    weight_profile: str,
    result: PipelineRunResult,
) -> dict[str, Any]:
    """Create one row for batch_runs.csv."""

    allocation = result.allocation_report
    local = allocation.local_improvement if allocation else None
    baseline = result.baseline_comparison_report

    return {
        "batch_id": batch_id,
        "run_id": run_id,
        "participant_set": participant_set,
        "project_set": project_set,
        "weight_profile": weight_profile,
        "participant_count": result.participant_count,
        "project_count": result.project_count,
        "required_slots": result.required_slots,
        "available_candidates": result.available_candidates,
        "main_method": allocation.method if allocation else "",
        "main_feasible": allocation.feasible if allocation else "",
        "main_objective_score": allocation.objective_score if allocation else "",
        "main_mean_team_score": allocation.mean_team_score if allocation else "",
        "main_min_team_score": allocation.min_team_score if allocation else "",
        "main_max_team_score": allocation.max_team_score if allocation else "",
        "main_fairness_deviation": allocation.fairness_deviation if allocation else "",
        "local_improvement_gain": local.improvement_gain if local else "",
        "accepted_swaps": local.accepted_swaps if local else "",
        "accepted_replacements": local.accepted_replacements if local else "",
        "best_by_objective": baseline.best_by_objective if baseline else "",
        "best_by_min_team_score": baseline.best_by_min_team_score if baseline else "",
        "best_by_fairness": baseline.best_by_fairness if baseline else "",
        "total_runtime_seconds": _timing_value(result, "total_runtime_seconds"),
        "load_inputs_seconds": _timing_value(result, "load_inputs"),
        "normalize_inputs_seconds": _timing_value(result, "normalize_inputs"),
        "score_seconds": _timing_value(result, "score_candidate_project_pairs"),
        "allocate_and_improve_seconds": _timing_value(result, "allocate_and_improve"),
        "compare_baselines_seconds": _timing_value(result, "compare_baselines"),
        "export_run_outputs_seconds": _timing_value(result, "export_run_outputs"),
        "saved_run_dir": result.saved_run_dir or "",
    }


def _method_rows(
    *,
    batch_id: str,
    run_id: str,
    participant_set: str,
    project_set: str,
    weight_profile: str,
    result: PipelineRunResult,
) -> list[dict[str, Any]]:
    """Create long-format method-comparison rows for batch_methods.csv."""

    report = result.baseline_comparison_report
    if report is None:
        return []

    rows: list[dict[str, Any]] = []

    for summary in report.method_summaries:
        rows.append(
            {
                "batch_id": batch_id,
                "run_id": run_id,
                "participant_set": participant_set,
                "project_set": project_set,
                "weight_profile": weight_profile,
                "method": summary.method,
                "is_main_method": summary.method == report.main_method,
                "feasible": summary.feasible,
                "objective_score": summary.objective_score,
                "mean_team_score": summary.mean_team_score,
                "min_team_score": summary.min_team_score,
                "max_team_score": summary.max_team_score,
                "fairness_deviation": summary.fairness_deviation,
                "assigned_count": summary.assigned_count,
                "required_slots": summary.required_slots,
                "participant_count": result.participant_count,
                "project_count": result.project_count,
                "total_runtime_seconds": _timing_value(result, "total_runtime_seconds"),
            }
        )

    return rows


def _format_batch_report(
    *,
    config: BatchEvaluationConfig,
    batch_id: str,
    batch_dir: Path,
    main_rows: list[dict[str, Any]],
    method_rows: list[dict[str, Any]],
) -> str:
    """Create a short human-readable batch report."""

    feasible_main_runs = sum(1 for row in main_rows if row["main_feasible"] is True)

    lines = [
        "Team formation batch evaluation",
        "=" * 40,
        f"Batch ID:          {batch_id}",
        f"Batch directory:   {batch_dir}",
        f"Participant set:   {config.participant_set}",
        f"Project sets:      {', '.join(config.project_sets)}",
        f"Weight profiles:   {', '.join(config.weight_profiles)}",
        f"Runs:              {len(main_rows)}",
        f"Feasible main runs:{feasible_main_runs} / {len(main_rows)}",
        "",
        "Outputs:",
        "  - batch_runs.csv",
        "  - batch_methods.csv",
        "  - batch_summary.json",
    ]

    if method_rows:
        best_counts: dict[str, int] = {}
        for row in main_rows:
            method = str(row["best_by_objective"])
            if method:
                best_counts[method] = best_counts.get(method, 0) + 1

        lines.extend(["", "Best-by-objective counts:"])
        for method, count in sorted(best_counts.items()):
            lines.append(f"  - {method}: {count}")

    return "\n".join(lines)


def run_batch_evaluation(config: BatchEvaluationConfig) -> BatchEvaluationResult:
    """Run a batch evaluation and write aggregate outputs."""

    batch_id = config.batch_id or make_batch_id()
    batch_dir = config.output_dir / batch_id
    individual_runs_dir = batch_dir / "runs"
    batch_dir.mkdir(parents=True, exist_ok=False)

    participant_path = participant_path_for_set(config.participant_set)

    if not participant_path.exists():
        raise FileNotFoundError(
            f"Participant set file not found: {participant_path}. "
            "Generate participant sets before running batch evaluation."
        )

    main_rows: list[dict[str, Any]] = []
    method_rows: list[dict[str, Any]] = []

    for project_set in config.project_sets:
        for weight_profile_name in config.weight_profiles:
            weights = get_weight_profile(weight_profile_name)
            run_id = (
                f"participants_{config.participant_set}"
                f"-projects_{project_set}"
                f"-weights_{weight_profile_name}"
            )

            result = run_pipeline(
                PipelineConfig(
                    participants_path=participant_path,
                    project_set=project_set,
                    score_weights=weights,
                    fairness_penalty=config.fairness_penalty,
                    enable_local_improvement=config.enable_local_improvement,
                    max_local_improvement_iterations=config.max_local_improvement_iterations,
                    save_run=config.save_individual_runs,
                    output_dir=individual_runs_dir,
                    run_id=run_id,
                )
            )

            main_rows.append(
                _main_run_row(
                    batch_id=batch_id,
                    run_id=run_id,
                    participant_set=config.participant_set,
                    project_set=project_set,
                    weight_profile=weight_profile_name,
                    result=result,
                )
            )
            method_rows.extend(
                _method_rows(
                    batch_id=batch_id,
                    run_id=run_id,
                    participant_set=config.participant_set,
                    project_set=project_set,
                    weight_profile=weight_profile_name,
                    result=result,
                )
            )

    batch_runs_csv = batch_dir / "batch_runs.csv"
    batch_methods_csv = batch_dir / "batch_methods.csv"
    batch_summary_json = batch_dir / "batch_summary.json"
    batch_report_txt = batch_dir / "batch_report.txt"

    _write_csv(
        batch_runs_csv,
        main_rows,
        [
            "batch_id",
            "run_id",
            "participant_set",
            "project_set",
            "weight_profile",
            "participant_count",
            "project_count",
            "required_slots",
            "available_candidates",
            "main_method",
            "main_feasible",
            "main_objective_score",
            "main_mean_team_score",
            "main_min_team_score",
            "main_max_team_score",
            "main_fairness_deviation",
            "local_improvement_gain",
            "accepted_swaps",
            "accepted_replacements",
            "best_by_objective",
            "best_by_min_team_score",
            "best_by_fairness",
            "total_runtime_seconds",
            "load_inputs_seconds",
            "normalize_inputs_seconds",
            "score_seconds",
            "allocate_and_improve_seconds",
            "compare_baselines_seconds",
            "export_run_outputs_seconds",
            "saved_run_dir",
        ],
    )

    _write_csv(
        batch_methods_csv,
        method_rows,
        [
            "batch_id",
            "run_id",
            "participant_set",
            "project_set",
            "weight_profile",
            "method",
            "is_main_method",
            "feasible",
            "objective_score",
            "mean_team_score",
            "min_team_score",
            "max_team_score",
            "fairness_deviation",
            "assigned_count",
            "required_slots",
            "participant_count",
            "project_count",
            "total_runtime_seconds",
        ],
    )

    _write_json(
        batch_summary_json,
        {
            "batch_id": batch_id,
            "participant_set": config.participant_set,
            "participant_path": participant_path,
            "project_sets": config.project_sets,
            "weight_profiles": config.weight_profiles,
            "available_weight_profiles": sorted(WEIGHT_PROFILES),
            "run_count": len(main_rows),
            "save_individual_runs": config.save_individual_runs,
            "fairness_penalty": config.fairness_penalty,
            "enable_local_improvement": config.enable_local_improvement,
            "max_local_improvement_iterations": config.max_local_improvement_iterations,
            "batch_runs_csv": batch_runs_csv,
            "batch_methods_csv": batch_methods_csv,
            "main_rows": main_rows,
            "method_rows": method_rows,
        },
    )

    batch_report = _format_batch_report(
        config=config,
        batch_id=batch_id,
        batch_dir=batch_dir,
        main_rows=main_rows,
        method_rows=method_rows,
    )
    batch_report_txt.write_text(batch_report + "\n", encoding="utf-8")

    return BatchEvaluationResult(
        batch_id=batch_id,
        batch_dir=batch_dir,
        participant_set=config.participant_set,
        project_sets=config.project_sets,
        weight_profiles=config.weight_profiles,
        run_count=len(main_rows),
        batch_runs_csv=batch_runs_csv,
        batch_methods_csv=batch_methods_csv,
        batch_summary_json=batch_summary_json,
        batch_report_txt=batch_report_txt,
    )
