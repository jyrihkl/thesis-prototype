"""Pipeline backbone for the team formation prototype."""

from __future__ import annotations

from team_builder.allocation import construct_round_based_allocation
from team_builder.baselines import run_baseline_comparisons
from team_builder.export import export_run_result
from team_builder.io import (
    load_participants,
    load_projects,
    resolve_participant_path,
    resolve_project_path,
)
from team_builder.models import PipelineConfig, PipelineRunResult, TimingReport
from team_builder.normalize import normalize_candidates, normalize_projects
from team_builder.reporting import format_validation_report
from team_builder.scoring import score_candidate_project_matrix, summarize_scoring
from team_builder.timing import PipelineTimer
from team_builder.validation import validate_pipeline_inputs
from team_builder.weights import score_weights_from_mapping


def run_pipeline(config: PipelineConfig) -> PipelineRunResult:
    """Run the current prototype pipeline."""

    timer = PipelineTimer()

    with timer.stage("resolve_input_paths"):
        participants_path = resolve_participant_path(
            explicit_path=config.participants_path,
            participant_set=config.participant_set,
        )
        projects_path = resolve_project_path(config.projects_path, config.project_set)

    with timer.stage("load_inputs"):
        raw_participants = load_participants(participants_path)
        raw_projects = load_projects(projects_path)

    with timer.stage("normalize_inputs"):
        candidates = normalize_candidates(raw_participants)
        projects = normalize_projects(raw_projects)

    with timer.stage("validate_inputs"):
        validation_report = validate_pipeline_inputs(candidates, projects)

    if validation_report.has_failures:
        raise ValueError(
            "Pipeline input validation failed.\n\n"
            + format_validation_report(validation_report)
        )

    score_weights = score_weights_from_mapping(config.score_weights)
    with timer.stage("score_candidate_project_pairs"):
        candidate_project_scores = score_candidate_project_matrix(
            candidates,
            projects,
            weights=score_weights,
        )
        scoring_report = summarize_scoring(
            candidate_project_scores,
            projects,
            weights=score_weights,
        )

    with timer.stage("allocate_and_improve"):
        allocation_report = construct_round_based_allocation(
            candidates=candidates,
            projects=projects,
            scores=candidate_project_scores,
            fairness_penalty=config.fairness_penalty,
            enable_local_improvement=config.enable_local_improvement,
            max_local_improvement_iterations=config.max_local_improvement_iterations,
            min_local_improvement_gain=config.min_local_improvement_gain,
        )

    with timer.stage("compare_baselines"):
        baseline_comparison_report = run_baseline_comparisons(
            main_allocation=allocation_report,
            candidates=candidates,
            projects=projects,
            scores=candidate_project_scores,
            fairness_penalty=config.fairness_penalty,
            max_local_improvement_iterations=config.max_local_improvement_iterations,
            min_local_improvement_gain=config.min_local_improvement_gain,
        )

    timing_report = TimingReport(
        stages=timer.as_ordered_dict(),
        total_runtime_seconds=timer.total_seconds(),
    )

    result = PipelineRunResult(
        participants_path=participants_path,
        projects_path=projects_path,
        participant_count=len(candidates),
        project_count=len(projects),
        project_titles=[project.title for project in projects],
        required_slots=sum(project.target_team_size for project in projects),
        available_candidates=len(candidates),
        validation_report=validation_report,
        scoring_report=scoring_report,
        candidate_project_scores=candidate_project_scores,
        allocation_report=allocation_report,
        baseline_comparison_report=baseline_comparison_report,
        timing_report=timing_report,
    )

    if config.save_run:
        with timer.stage("export_run_outputs"):
            saved_run_dir = export_run_result(
                result=result,
                output_dir=config.output_dir,
                run_id=config.run_id,
            )

        timing_report = TimingReport(
            stages=timer.as_ordered_dict(),
            total_runtime_seconds=timer.total_seconds(),
        )

        result = PipelineRunResult(
            participants_path=result.participants_path,
            projects_path=result.projects_path,
            participant_count=result.participant_count,
            project_count=result.project_count,
            project_titles=result.project_titles,
            required_slots=result.required_slots,
            available_candidates=result.available_candidates,
            validation_report=result.validation_report,
            scoring_report=result.scoring_report,
            candidate_project_scores=result.candidate_project_scores,
            allocation_report=result.allocation_report,
            baseline_comparison_report=result.baseline_comparison_report,
            timing_report=timing_report,
            saved_run_dir=saved_run_dir,
        )

    return result
