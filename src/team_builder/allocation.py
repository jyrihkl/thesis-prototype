"""Transparent round-based team allocation.

This module implements the first team-construction procedure from the thesis
plan. It is deliberately heuristic and explainable:

1. Teams are built in rounds across the full project set.
2. Project order is rotated between rounds to reduce first-project advantage.
3. Candidates are selected by marginal contribution to the current partial team,
   not only by their individual project-fit score.
4. Completed teams are evaluated at team level.
5. The full allocation is evaluated with a simple fairness penalty based on the
   difference between the strongest and weakest team scores.

Local search and baseline comparison are left for later steps.
"""

from __future__ import annotations

from statistics import mean, pstdev

from team_builder.models import (
    AllocationReport,
    CandidateProjectScore,
    ProjectTeamSummary,
    TeamMemberAssignment,
)
from team_builder.schemas import Candidate, Project


MARGINAL_WEIGHTS: dict[str, float] = {
    "required_skill_gain": 0.35,
    "preferred_skill_gain": 0.10,
    "role_gap": 0.20,
    "experience_balance": 0.15,
    "individual_fit": 0.20,
    "redundancy_score": 0.10,
}

TEAM_QUALITY_WEIGHTS: dict[str, float] = {
    "required_skill_coverage": 0.35,
    "preferred_skill_coverage": 0.15,
    "role_balance": 0.20,
    "experience_balance": 0.15,
    "average_individual_fit": 0.10,
    "redundancy_score": 0.05,
}


def _active_weighted_average(
    components: dict[str, float | None],
    weights: dict[str, float],
) -> float:
    """Compute a weighted average over components that are not None."""

    numerator = 0.0
    denominator = 0.0

    for name, value in components.items():
        if value is None:
            continue
        weight = weights[name]
        numerator += weight * value
        denominator += weight

    if denominator == 0:
        return 0.0

    return numerator / denominator


def _candidate_score_lookup(
    scores: list[CandidateProjectScore],
) -> dict[tuple[str, str], CandidateProjectScore]:
    """Index candidate-project scores by candidate ID and project ID."""

    return {
        (score.candidate_id, score.project_id): score
        for score in scores
    }


def _skill_union(candidates: list[Candidate]) -> frozenset[str]:
    """Return the union of all candidate skills in a partial or complete team."""

    skills: set[str] = set()
    for candidate in candidates:
        skills.update(candidate.skills)
    return frozenset(skills)


def _normalized_skill_gain(
    candidate: Candidate,
    current_team: list[Candidate],
    target_skills: frozenset[str],
) -> float | None:
    """Return how much new target-skill coverage the candidate adds."""

    if not target_skills:
        return None

    already_covered = _skill_union(current_team) & target_skills
    newly_covered = (candidate.skills & target_skills) - already_covered

    return len(newly_covered) / len(target_skills)


def _role_gap_score(
    candidate: Candidate,
    project: Project,
    current_team: list[Candidate],
) -> float | None:
    """Return how well the candidate fills a desired role gap."""

    if not project.desired_roles:
        return None

    if candidate.role_family not in project.desired_roles:
        return 0.0

    current_roles = {
        member.role_family
        for member in current_team
        if member.role_family and member.role_family != "unknown"
    }
    missing_roles = project.desired_roles - current_roles

    if candidate.role_family in missing_roles:
        return 1.0

    return 0.5


def _experience_balance_after_addition(
    candidate: Candidate,
    current_team: list[Candidate],
) -> float:
    """Score experience balance after adding a candidate.

    This is intentionally modest. It rewards avoiding extreme experience spread,
    but does not treat experience as a dominant selection criterion.
    """

    years = [
        member.experience_years
        for member in current_team
        if member.experience_years is not None
    ]

    if candidate.experience_years is not None:
        years.append(candidate.experience_years)

    if not years:
        return 0.5

    if len(years) == 1:
        return 0.75

    spread = pstdev(years)

    return max(0.0, 1.0 - min(spread / 6.0, 1.0))


def _redundancy_score(candidate: Candidate, current_team: list[Candidate]) -> float:
    """Score how non-redundant the candidate is compared with the current team."""

    if not current_team or not candidate.skills:
        return 1.0

    current_skills = _skill_union(current_team)
    overlap = len(candidate.skills & current_skills) / len(candidate.skills)

    return max(0.0, 1.0 - overlap)


def marginal_contribution(
    candidate: Candidate,
    project: Project,
    current_team: list[Candidate],
    candidate_project_score: CandidateProjectScore,
) -> tuple[float, dict[str, float | None]]:
    """Compute candidate marginal contribution to a partial project team."""

    components = {
        "required_skill_gain": _normalized_skill_gain(
            candidate=candidate,
            current_team=current_team,
            target_skills=project.required_skills,
        ),
        "preferred_skill_gain": _normalized_skill_gain(
            candidate=candidate,
            current_team=current_team,
            target_skills=project.preferred_skills,
        ),
        "role_gap": _role_gap_score(
            candidate=candidate,
            project=project,
            current_team=current_team,
        ),
        "experience_balance": _experience_balance_after_addition(
            candidate=candidate,
            current_team=current_team,
        ),
        "individual_fit": candidate_project_score.total_score,
        "redundancy_score": _redundancy_score(
            candidate=candidate,
            current_team=current_team,
        ),
    }

    return _active_weighted_average(components, MARGINAL_WEIGHTS), components


def _coverage_score(team_skills: frozenset[str], target_skills: frozenset[str]) -> float | None:
    """Return normalized team-level skill coverage."""

    if not target_skills:
        return None
    return len(team_skills & target_skills) / len(target_skills)


def _role_balance_score(team: list[Candidate], project: Project) -> float | None:
    """Return project desired-role coverage at team level."""

    if not project.desired_roles:
        return None

    team_roles = {
        member.role_family
        for member in team
        if member.role_family and member.role_family != "unknown"
    }

    return len(team_roles & project.desired_roles) / len(project.desired_roles)


def _experience_balance_for_team(team: list[Candidate]) -> float:
    """Return team-level experience balance."""

    years = [
        member.experience_years
        for member in team
        if member.experience_years is not None
    ]

    if not years:
        return 0.5

    if len(years) == 1:
        return 0.75

    spread = pstdev(years)

    return max(0.0, 1.0 - min(spread / 6.0, 1.0))


def _team_redundancy_score(team: list[Candidate]) -> float:
    """Return a simple non-redundancy score based on pairwise skill overlap."""

    if len(team) < 2:
        return 1.0

    pairwise_overlaps: list[float] = []

    for left_index, left in enumerate(team):
        for right in team[left_index + 1:]:
            union = left.skills | right.skills
            if not union:
                continue
            pairwise_overlaps.append(len(left.skills & right.skills) / len(union))

    if not pairwise_overlaps:
        return 1.0

    return max(0.0, 1.0 - mean(pairwise_overlaps))


def _average_individual_fit(
    team: list[Candidate],
    project: Project,
    score_lookup: dict[tuple[str, str], CandidateProjectScore],
) -> float:
    """Return average candidate-to-project fit among assigned team members."""

    values = [
        score_lookup[(candidate.id, project.id)].total_score
        for candidate in team
        if (candidate.id, project.id) in score_lookup
    ]

    if not values:
        return 0.0

    return mean(values)


def evaluate_project_team(
    project: Project,
    team: list[Candidate],
    score_lookup: dict[tuple[str, str], CandidateProjectScore],
) -> ProjectTeamSummary:
    """Evaluate one completed or partial project team."""

    team_skills = _skill_union(team)

    components = {
        "required_skill_coverage": _coverage_score(team_skills, project.required_skills),
        "preferred_skill_coverage": _coverage_score(team_skills, project.preferred_skills),
        "role_balance": _role_balance_score(team, project),
        "experience_balance": _experience_balance_for_team(team),
        "average_individual_fit": _average_individual_fit(team, project, score_lookup),
        "redundancy_score": _team_redundancy_score(team),
    }

    team_score = _active_weighted_average(components, TEAM_QUALITY_WEIGHTS)

    covered_required = tuple(sorted(team_skills & project.required_skills))
    missing_required = tuple(sorted(project.required_skills - team_skills))
    covered_preferred = tuple(sorted(team_skills & project.preferred_skills))
    role_families = tuple(
        sorted(
            {
                member.role_family
                for member in team
                if member.role_family and member.role_family != "unknown"
            }
        )
    )

    return ProjectTeamSummary(
        project_id=project.id,
        project_title=project.title,
        member_ids=tuple(member.id for member in team),
        target_team_size=project.target_team_size,
        team_score=team_score,
        components=components,
        covered_required_skills=covered_required,
        missing_required_skills=missing_required,
        covered_preferred_skills=covered_preferred,
        role_families=role_families,
    )


def construct_round_based_allocation(
    candidates: list[Candidate],
    projects: list[Project],
    scores: list[CandidateProjectScore],
    fairness_penalty: float = 0.25,
) -> AllocationReport:
    """Construct an initial allocation with transparent round-based assignment."""

    candidates_by_id = {candidate.id: candidate for candidate in candidates}
    score_lookup = _candidate_score_lookup(scores)

    teams: dict[str, list[Candidate]] = {project.id: [] for project in projects}
    assignments: list[TeamMemberAssignment] = []
    unassigned_ids: set[str] = set(candidates_by_id)
    warnings: list[str] = []

    if not projects:
        return AllocationReport(
            method="round_based_marginal_contribution",
            feasible=False,
            assigned_count=0,
            unassigned_count=len(unassigned_ids),
            required_slots=0,
            objective_score=0.0,
            fairness_deviation=0.0,
            min_team_score=None,
            mean_team_score=None,
            max_team_score=None,
            project_summaries=[],
            assignments=[],
            unassigned_candidate_ids=tuple(sorted(unassigned_ids)),
            warnings=("No projects available for allocation.",),
        )

    max_rounds = max(project.target_team_size for project in projects)

    for round_index in range(max_rounds):
        rotation = round_index % len(projects)
        ordered_projects = projects[rotation:] + projects[:rotation]

        for project in ordered_projects:
            current_team = teams[project.id]

            if len(current_team) >= project.target_team_size:
                continue

            candidate_options: list[
                tuple[float, float, str, dict[str, float | None]]
            ] = []

            for candidate_id in sorted(unassigned_ids):
                candidate = candidates_by_id[candidate_id]
                pair_score = score_lookup.get((candidate.id, project.id))

                if pair_score is None or not pair_score.feasible:
                    continue

                marginal_score, components = marginal_contribution(
                    candidate=candidate,
                    project=project,
                    current_team=current_team,
                    candidate_project_score=pair_score,
                )

                candidate_options.append(
                    (
                        marginal_score,
                        pair_score.total_score,
                        candidate.id,
                        components,
                    )
                )

            if not candidate_options:
                warnings.append(
                    f"No feasible unassigned candidate found for project {project.id} "
                    f"during round {round_index + 1}."
                )
                continue

            marginal_score, individual_fit, selected_id, components = sorted(
                candidate_options,
                key=lambda item: (-item[0], -item[1], item[2]),
            )[0]

            selected_candidate = candidates_by_id[selected_id]
            current_team.append(selected_candidate)
            unassigned_ids.remove(selected_id)

            assignments.append(
                TeamMemberAssignment(
                    candidate_id=selected_id,
                    project_id=project.id,
                    round_number=round_index + 1,
                    marginal_score=marginal_score,
                    individual_fit_score=individual_fit,
                    marginal_components=components,
                )
            )

    project_summaries = [
        evaluate_project_team(
            project=project,
            team=teams[project.id],
            score_lookup=score_lookup,
        )
        for project in projects
    ]

    team_scores = [summary.team_score for summary in project_summaries]

    if team_scores:
        fairness_deviation = max(team_scores) - min(team_scores)
        objective_score = sum(team_scores) - fairness_penalty * fairness_deviation
        min_team_score = min(team_scores)
        mean_team_score = mean(team_scores)
        max_team_score = max(team_scores)
    else:
        fairness_deviation = 0.0
        objective_score = 0.0
        min_team_score = None
        mean_team_score = None
        max_team_score = None

    incomplete_projects = [
        summary.project_id
        for summary in project_summaries
        if len(summary.member_ids) != summary.target_team_size
    ]

    if incomplete_projects:
        warnings.append(
            "Incomplete project team(s): " + ", ".join(incomplete_projects)
        )

    feasible = not incomplete_projects

    return AllocationReport(
        method="round_based_marginal_contribution",
        feasible=feasible,
        assigned_count=len(assignments),
        unassigned_count=len(unassigned_ids),
        required_slots=sum(project.target_team_size for project in projects),
        objective_score=objective_score,
        fairness_deviation=fairness_deviation,
        min_team_score=min_team_score,
        mean_team_score=mean_team_score,
        max_team_score=max_team_score,
        project_summaries=project_summaries,
        assignments=assignments,
        unassigned_candidate_ids=tuple(sorted(unassigned_ids)),
        warnings=tuple(warnings),
    )
