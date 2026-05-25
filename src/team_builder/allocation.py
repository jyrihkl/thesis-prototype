"""Transparent round-based allocation with local improvement.

This module follows the thesis implementation plan:

1. Build an initial allocation round by round across projects.
2. Select candidates by marginal contribution to the current partial team.
3. Evaluate completed teams at team level.
4. Evaluate the full allocation with a fairness penalty.
5. Improve the allocation through transparent swaps and replacements.

The procedure remains explainable. It does not claim to predict full
team effectiveness. It optimizes observable preconditions such as skill coverage,
role coverage, experience balance, individual fit, and cross-team balance.
"""

from __future__ import annotations

from statistics import mean, pstdev

from team_builder.models import (
    AllocationReport,
    CandidateProjectScore,
    LocalImprovementMove,
    LocalImprovementReport,
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


def _candidate_feasible_for_project(
    candidate_id: str,
    project_id: str,
    score_lookup: dict[tuple[str, str], CandidateProjectScore],
) -> bool:
    """Return whether a candidate-project pair satisfies feasibility."""

    pair_score = score_lookup.get((candidate_id, project_id))
    return bool(pair_score and pair_score.feasible)


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
    """Score experience balance after adding a candidate."""

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

    score = _active_weighted_average(components, MARGINAL_WEIGHTS)
    return score, components


def _coverage_score(
    team_skills: frozenset[str],
    target_skills: frozenset[str],
) -> float | None:
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
    """Evaluate one completed or partial project team.

    This function is intentionally public because baseline methods use the same
    team-quality logic as the main method.
    """

    team_skills = _skill_union(team)

    components = {
        "required_skill_coverage": _coverage_score(
            team_skills,
            project.required_skills,
        ),
        "preferred_skill_coverage": _coverage_score(
            team_skills,
            project.preferred_skills,
        ),
        "role_balance": _role_balance_score(team, project),
        "experience_balance": _experience_balance_for_team(team),
        "average_individual_fit": _average_individual_fit(
            team,
            project,
            score_lookup,
        ),
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


def _evaluate_allocation(
    projects: list[Project],
    teams: dict[str, list[Candidate]],
    score_lookup: dict[tuple[str, str], CandidateProjectScore],
    fairness_penalty: float,
) -> tuple[
    list[ProjectTeamSummary],
    float,
    float,
    float | None,
    float | None,
    float | None,
]:
    """Evaluate all teams and return allocation-level metrics."""

    project_summaries = [
        evaluate_project_team(
            project=project,
            team=teams[project.id],
            score_lookup=score_lookup,
        )
        for project in projects
    ]

    team_scores = [summary.team_score for summary in project_summaries]

    if not team_scores:
        return project_summaries, 0.0, 0.0, None, None, None

    fairness_deviation = max(team_scores) - min(team_scores)
    objective_score = sum(team_scores) - fairness_penalty * fairness_deviation

    return (
        project_summaries,
        objective_score,
        fairness_deviation,
        min(team_scores),
        mean(team_scores),
        max(team_scores),
    )


def _copy_teams(teams: dict[str, list[Candidate]]) -> dict[str, list[Candidate]]:
    """Create a shallow copy of an allocation state."""

    return {project_id: list(team) for project_id, team in teams.items()}


def _replace_candidate_in_team(
    team: list[Candidate],
    old_candidate_id: str,
    new_candidate: Candidate,
) -> list[Candidate]:
    """Return a copy of a team with one candidate replaced."""

    return [
        new_candidate if member.id == old_candidate_id else member
        for member in team
    ]


def _initial_round_based_teams(
    candidates: list[Candidate],
    projects: list[Project],
    score_lookup: dict[tuple[str, str], CandidateProjectScore],
) -> tuple[dict[str, list[Candidate]], list[TeamMemberAssignment], set[str], list[str]]:
    """Construct the initial round-based allocation before local improvement."""

    candidates_by_id = {candidate.id: candidate for candidate in candidates}

    teams: dict[str, list[Candidate]] = {project.id: [] for project in projects}
    assignments: list[TeamMemberAssignment] = []
    unassigned_ids: set[str] = set(candidates_by_id)
    warnings: list[str] = []

    if not projects:
        warnings.append("No projects available for allocation.")
        return teams, assignments, unassigned_ids, warnings

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

    return teams, assignments, unassigned_ids, warnings


def _apply_swap(
    teams: dict[str, list[Candidate]],
    left_project_id: str,
    right_project_id: str,
    left_candidate_id: str,
    right_candidate_id: str,
) -> dict[str, list[Candidate]]:
    """Return a new team allocation after swapping two candidates."""

    new_teams = _copy_teams(teams)

    left_candidate = next(
        candidate
        for candidate in new_teams[left_project_id]
        if candidate.id == left_candidate_id
    )
    right_candidate = next(
        candidate
        for candidate in new_teams[right_project_id]
        if candidate.id == right_candidate_id
    )

    new_teams[left_project_id] = _replace_candidate_in_team(
        team=new_teams[left_project_id],
        old_candidate_id=left_candidate_id,
        new_candidate=right_candidate,
    )
    new_teams[right_project_id] = _replace_candidate_in_team(
        team=new_teams[right_project_id],
        old_candidate_id=right_candidate_id,
        new_candidate=left_candidate,
    )

    return new_teams


def _apply_replacement(
    teams: dict[str, list[Candidate]],
    project_id: str,
    removed_candidate_id: str,
    added_candidate: Candidate,
) -> dict[str, list[Candidate]]:
    """Return a new team allocation after replacing one assigned candidate."""

    new_teams = _copy_teams(teams)
    new_teams[project_id] = _replace_candidate_in_team(
        team=new_teams[project_id],
        old_candidate_id=removed_candidate_id,
        new_candidate=added_candidate,
    )

    return new_teams


def _local_improvement(
    candidates: list[Candidate],
    projects: list[Project],
    teams: dict[str, list[Candidate]],
    unassigned_ids: set[str],
    score_lookup: dict[tuple[str, str], CandidateProjectScore],
    fairness_penalty: float,
    max_iterations: int,
    min_gain: float,
) -> tuple[dict[str, list[Candidate]], set[str], LocalImprovementReport]:
    """Improve an allocation through feasible swaps and replacements."""

    candidates_by_id = {candidate.id: candidate for candidate in candidates}

    (
        _initial_project_summaries,
        initial_objective,
        _initial_fairness,
        _initial_min,
        _initial_mean,
        _initial_max,
    ) = _evaluate_allocation(
        projects=projects,
        teams=teams,
        score_lookup=score_lookup,
        fairness_penalty=fairness_penalty,
    )

    accepted_moves: list[LocalImprovementMove] = []
    accepted_swaps = 0
    accepted_replacements = 0
    evaluated_swaps = 0
    evaluated_replacements = 0
    stop_reason = "no_improving_move_found"
    iterations_completed = 0

    current_teams = _copy_teams(teams)
    current_unassigned = set(unassigned_ids)
    current_objective = initial_objective

    for iteration in range(1, max_iterations + 1):
        iterations_completed = iteration

        best_gain = min_gain
        best_move_type: str | None = None
        best_teams: dict[str, list[Candidate]] | None = None
        best_unassigned: set[str] | None = None
        best_description = ""
        best_affected_projects: tuple[str, ...] = ()

        # Candidate swaps between already assigned teams.
        for left_index, left_project in enumerate(projects):
            for right_project in projects[left_index + 1:]:
                left_team = current_teams[left_project.id]
                right_team = current_teams[right_project.id]

                for left_candidate in left_team:
                    for right_candidate in right_team:
                        if not _candidate_feasible_for_project(
                            candidate_id=left_candidate.id,
                            project_id=right_project.id,
                            score_lookup=score_lookup,
                        ):
                            continue

                        if not _candidate_feasible_for_project(
                            candidate_id=right_candidate.id,
                            project_id=left_project.id,
                            score_lookup=score_lookup,
                        ):
                            continue

                        evaluated_swaps += 1

                        trial_teams = _apply_swap(
                            teams=current_teams,
                            left_project_id=left_project.id,
                            right_project_id=right_project.id,
                            left_candidate_id=left_candidate.id,
                            right_candidate_id=right_candidate.id,
                        )

                        (
                            _project_summaries,
                            trial_objective,
                            _fairness,
                            _min_team,
                            _mean_team,
                            _max_team,
                        ) = _evaluate_allocation(
                            projects=projects,
                            teams=trial_teams,
                            score_lookup=score_lookup,
                            fairness_penalty=fairness_penalty,
                        )

                        gain = trial_objective - current_objective

                        if gain > best_gain:
                            best_gain = gain
                            best_move_type = "swap"
                            best_teams = trial_teams
                            best_unassigned = set(current_unassigned)
                            best_affected_projects = (
                                left_project.id,
                                right_project.id,
                            )
                            best_description = (
                                f"Swapped candidate {left_candidate.id} "
                                f"from {left_project.id} with candidate "
                                f"{right_candidate.id} from {right_project.id}."
                            )

        # Candidate replacement from the unassigned pool.
        for project in projects:
            team = current_teams[project.id]

            for removed_candidate in team:
                for added_candidate_id in sorted(current_unassigned):
                    if not _candidate_feasible_for_project(
                        candidate_id=added_candidate_id,
                        project_id=project.id,
                        score_lookup=score_lookup,
                    ):
                        continue

                    evaluated_replacements += 1

                    added_candidate = candidates_by_id[added_candidate_id]
                    trial_teams = _apply_replacement(
                        teams=current_teams,
                        project_id=project.id,
                        removed_candidate_id=removed_candidate.id,
                        added_candidate=added_candidate,
                    )

                    (
                        _project_summaries,
                        trial_objective,
                        _fairness,
                        _min_team,
                        _mean_team,
                        _max_team,
                    ) = _evaluate_allocation(
                        projects=projects,
                        teams=trial_teams,
                        score_lookup=score_lookup,
                        fairness_penalty=fairness_penalty,
                    )

                    gain = trial_objective - current_objective

                    if gain > best_gain:
                        trial_unassigned = set(current_unassigned)
                        trial_unassigned.remove(added_candidate_id)
                        trial_unassigned.add(removed_candidate.id)

                        best_gain = gain
                        best_move_type = "replacement"
                        best_teams = trial_teams
                        best_unassigned = trial_unassigned
                        best_affected_projects = (project.id,)
                        best_description = (
                            f"Replaced candidate {removed_candidate.id} "
                            f"with unassigned candidate {added_candidate.id} "
                            f"in project {project.id}."
                        )

        if best_move_type is None or best_teams is None or best_unassigned is None:
            stop_reason = "no_improving_move_found"
            break

        current_teams = best_teams
        current_unassigned = best_unassigned
        current_objective += best_gain

        if best_move_type == "swap":
            accepted_swaps += 1
        elif best_move_type == "replacement":
            accepted_replacements += 1

        accepted_moves.append(
            LocalImprovementMove(
                iteration=iteration,
                move_type=best_move_type,
                gain=best_gain,
                affected_projects=best_affected_projects,
                description=best_description,
            )
        )

    else:
        stop_reason = "iteration_limit_reached"

    (
        _final_project_summaries,
        final_objective,
        _final_fairness,
        _final_min,
        _final_mean,
        _final_max,
    ) = _evaluate_allocation(
        projects=projects,
        teams=current_teams,
        score_lookup=score_lookup,
        fairness_penalty=fairness_penalty,
    )

    report = LocalImprovementReport(
        enabled=True,
        initial_objective_score=initial_objective,
        final_objective_score=final_objective,
        improvement_gain=final_objective - initial_objective,
        accepted_swaps=accepted_swaps,
        accepted_replacements=accepted_replacements,
        evaluated_swaps=evaluated_swaps,
        evaluated_replacements=evaluated_replacements,
        iterations=iterations_completed,
        stop_reason=stop_reason,
        accepted_moves=tuple(accepted_moves),
    )

    return current_teams, current_unassigned, report


def construct_round_based_allocation(
    candidates: list[Candidate],
    projects: list[Project],
    scores: list[CandidateProjectScore],
    fairness_penalty: float = 0.25,
    enable_local_improvement: bool = True,
    max_local_improvement_iterations: int = 100,
    min_local_improvement_gain: float = 1e-9,
) -> AllocationReport:
    """Construct and optionally improve a transparent team allocation."""

    score_lookup = _candidate_score_lookup(scores)

    teams, assignments, unassigned_ids, warnings = _initial_round_based_teams(
        candidates=candidates,
        projects=projects,
        score_lookup=score_lookup,
    )

    (
        _initial_project_summaries,
        initial_objective_score,
        _initial_fairness_deviation,
        _initial_min_team_score,
        _initial_mean_team_score,
        _initial_max_team_score,
    ) = _evaluate_allocation(
        projects=projects,
        teams=teams,
        score_lookup=score_lookup,
        fairness_penalty=fairness_penalty,
    )

    if enable_local_improvement and projects:
        teams, unassigned_ids, local_improvement_report = _local_improvement(
            candidates=candidates,
            projects=projects,
            teams=teams,
            unassigned_ids=unassigned_ids,
            score_lookup=score_lookup,
            fairness_penalty=fairness_penalty,
            max_iterations=max_local_improvement_iterations,
            min_gain=min_local_improvement_gain,
        )
    else:
        local_improvement_report = LocalImprovementReport(
            enabled=False,
            initial_objective_score=initial_objective_score,
            final_objective_score=initial_objective_score,
            improvement_gain=0.0,
            accepted_swaps=0,
            accepted_replacements=0,
            evaluated_swaps=0,
            evaluated_replacements=0,
            iterations=0,
            stop_reason="disabled",
            accepted_moves=(),
        )

    (
        project_summaries,
        final_objective_score,
        fairness_deviation,
        min_team_score,
        mean_team_score,
        max_team_score,
    ) = _evaluate_allocation(
        projects=projects,
        teams=teams,
        score_lookup=score_lookup,
        fairness_penalty=fairness_penalty,
    )

    incomplete_projects = [
        summary.project_id
        for summary in project_summaries
        if len(summary.member_ids) != summary.target_team_size
    ]

    if incomplete_projects:
        warnings.append(
            "Incomplete project team(s): " + ", ".join(incomplete_projects)
        )

    return AllocationReport(
        method="round_based_marginal_contribution_with_local_improvement",
        feasible=not incomplete_projects,
        assigned_count=sum(len(team) for team in teams.values()),
        unassigned_count=len(unassigned_ids),
        required_slots=sum(project.target_team_size for project in projects),
        objective_score=final_objective_score,
        fairness_deviation=fairness_deviation,
        min_team_score=min_team_score,
        mean_team_score=mean_team_score,
        max_team_score=max_team_score,
        project_summaries=project_summaries,
        assignments=assignments,
        unassigned_candidate_ids=tuple(sorted(unassigned_ids)),
        warnings=tuple(warnings),
        local_improvement=local_improvement_report,
    )