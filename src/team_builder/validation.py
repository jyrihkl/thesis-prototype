"""Validation checks."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from team_builder.models import ValidationCheck, ValidationReport
from team_builder.schemas import Candidate, Project


def _duplicate_values(values: Iterable[str]) -> list[str]:
    """Return sorted duplicate values from an iterable."""

    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def validate_pipeline_inputs(
    candidates: list[Candidate],
    projects: list[Project],
) -> ValidationReport:
    """Validate normalized candidates and projects before scoring/allocation.

    These checks are intentionally limited to structural conditions that can be
    evaluated before the recommendation logic exists.
    """

    # TODO: Add scoring-specific and allocation-specific checks.

    checks: list[ValidationCheck] = []

    # TODO: Move to separate functions
    if candidates:
        checks.append(
            ValidationCheck(
                name="candidate_count",
                status="pass",
                message=f"{len(candidates)} candidate(s) available.",
            )
        )
    else:
        checks.append(
            ValidationCheck(
                name="candidate_count",
                status="fail",
                message="No candidates are available.",
            )
        )

    if projects:
        checks.append(
            ValidationCheck(
                name="project_count",
                status="pass",
                message=f"{len(projects)} project brief(s) available.",
            )
        )
    else:
        checks.append(
            ValidationCheck(
                name="project_count",
                status="fail",
                message="No project briefs are available.",
            )
        )

    duplicate_candidate_ids = _duplicate_values(candidate.id for candidate in candidates)
    if duplicate_candidate_ids:
        checks.append(
            ValidationCheck(
                name="candidate_ids_unique",
                status="fail",
                message=(
                    "Duplicate candidate ID(s): "
                    + ", ".join(duplicate_candidate_ids[:10])
                ),
            )
        )
    else:
        checks.append(
            ValidationCheck(
                name="candidate_ids_unique",
                status="pass",
                message="Candidate IDs are unique.",
            )
        )

    duplicate_project_ids = _duplicate_values(project.id for project in projects)
    if duplicate_project_ids:
        checks.append(
            ValidationCheck(
                name="project_ids_unique",
                status="fail",
                message=(
                    "Duplicate project ID(s): "
                    + ", ".join(duplicate_project_ids[:10])
                ),
            )
        )
    else:
        checks.append(
            ValidationCheck(
                name="project_ids_unique",
                status="pass",
                message="Project IDs are unique.",
            )
        )

    invalid_team_sizes = [
        project.id for project in projects if project.target_team_size <= 0
    ]
    if invalid_team_sizes:
        checks.append(
            ValidationCheck(
                name="project_team_sizes_positive",
                status="fail",
                message=(
                    "Project(s) with invalid target_team_size: "
                    + ", ".join(invalid_team_sizes[:10])
                ),
            )
        )
    else:
        checks.append(
            ValidationCheck(
                name="project_team_sizes_positive",
                status="pass",
                message="All project team sizes are positive.",
            )
        )

    required_slots = sum(project.target_team_size for project in projects)
    if required_slots <= len(candidates):
        checks.append(
            ValidationCheck(
                name="candidate_capacity",
                status="pass",
                message=(
                    f"{required_slots} project slot(s) required and "
                    f"{len(candidates)} candidate(s) available."
                ),
            )
        )
    else:
        checks.append(
            ValidationCheck(
                name="candidate_capacity",
                status="fail",
                message=(
                    f"{required_slots} project slot(s) required but only "
                    f"{len(candidates)} candidate(s) available."
                ),
            )
        )

    projects_without_required_skills = [
        project.id for project in projects if not project.required_skills
    ]
    if projects_without_required_skills:
        checks.append(
            ValidationCheck(
                name="project_required_skills_present",
                status="warn",
                message=(
                    "Project(s) without required_skills: "
                    + ", ".join(projects_without_required_skills[:10])
                ),
            )
        )
    else:
        checks.append(
            ValidationCheck(
                name="project_required_skills_present",
                status="pass",
                message="All projects define at least one required skill.",
            )
        )

    candidates_without_skills = [
        candidate.id for candidate in candidates if not candidate.skills
    ]
    if candidates_without_skills:
        checks.append(
            ValidationCheck(
                name="candidate_skills_present",
                status="warn",
                message=(
                    f"{len(candidates_without_skills)} candidate(s) have no extracted skills."
                ),
            )
        )
    else:
        checks.append(
            ValidationCheck(
                name="candidate_skills_present",
                status="pass",
                message="All candidates have at least one extracted skill.",
            )
        )

    return ValidationReport(checks=checks)
