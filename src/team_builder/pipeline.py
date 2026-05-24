"""Pipeline backbone for the team formation prototype."""

from __future__ import annotations

from team_builder.io import (
    load_participants,
    load_projects,
    resolve_participant_path,
    resolve_project_path,
)
from team_builder.models import PipelineConfig, PipelineRunResult
from team_builder.normalize import normalize_candidates, normalize_projects
from team_builder.reporting import format_validation_report
from team_builder.scoring import score_candidate_project_matrix, summarize_scoring
from team_builder.validation import validate_pipeline_inputs


def run_pipeline(config: PipelineConfig) -> PipelineRunResult:
    """Run the current prototype pipeline.

    Current stage:
    1. Resolve participant and project input files.
    2. Load raw participants and project briefs.
    3. Normalize raw dictionaries into typed Candidate and Project objects.
    4. Validate normalized inputs.
    5. Score candidate-to-project fit.
    6. Return a compact run summary.

    Future stages to be added in later iterations:
    - construct teams
    - improve allocation
    - evaluate baselines
    - generate explanations
    """

    participants_path = resolve_participant_path(config.participants_path)
    projects_path = resolve_project_path(config.projects_path, config.project_set)

    raw_participants = load_participants(participants_path)
    raw_projects = load_projects(projects_path)

    candidates = normalize_candidates(raw_participants)
    projects = normalize_projects(raw_projects)

    validation_report = validate_pipeline_inputs(candidates, projects)

    if validation_report.has_failures:
        raise ValueError(
            "Pipeline input validation failed.\n\n"
            + format_validation_report(validation_report)
        )

    candidate_project_scores = score_candidate_project_matrix(candidates, projects)
    scoring_report = summarize_scoring(candidate_project_scores, projects)

    return PipelineRunResult(
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
    )
