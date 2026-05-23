"""Pipeline backbone for the team formation prototype."""

from __future__ import annotations

from team_builder.io import (
    load_participants,
    load_projects,
    project_title,
    resolve_participant_path,
    resolve_project_path,
)
from team_builder.models import PipelineConfig, PipelineLoadResult


def run_pipeline(config: PipelineConfig) -> PipelineLoadResult:
    """Run the current prototype pipeline.

    Current stage:
    1. Resolve participant and project input files.
    2. Load participants.
    3. Load project briefs.
    4. Return a compact success summary.

    Future stages to be added in later iterations:
    - normalize participants
    - score candidate-to-project fit
    - construct teams
    - improve allocation
    - evaluate baselines
    - generate explanations
    """

    participants_path = resolve_participant_path(config.participants_path)
    projects_path = resolve_project_path(config.projects_path, config.project_set)

    participants = load_participants(participants_path)
    projects = load_projects(projects_path)

    if not participants:
        raise ValueError(f"No participants found in {participants_path}")

    if not projects:
        raise ValueError(f"No projects found in {projects_path}")

    return PipelineLoadResult(
        participants_path=participants_path,
        projects_path=projects_path,
        participant_count=len(participants),
        project_count=len(projects),
        project_titles=[project_title(project) for project in projects],
    )
