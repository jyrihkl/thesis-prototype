"""Transparent local improvement for team allocations.

The initial allocation is created by the round-based marginal-contribution
heuristic in allocation.py. This module performs the next step:
it tries simple local moves and accepts only those that preserve feasibility
and improve the global allocation objective.

Move types:
- swap: exchange two already assigned candidates between two projects
- replacement: replace one assigned candidate with one candidate from the
  unassigned pool

The objective remains transparent:

    sum(team_quality_scores) - fairness_penalty * fairness_deviation

where fairness_deviation is the gap between the strongest and weakest team.
"""

from __future__ import annotations

from statistics import mean

from team_builder.allocation import evaluate_project_team
from team_builder.models import (
    AllocationReport,
    CandidateProjectScore,
    LocalImprovementMove,
    LocalImprovementReport,
)
from team_builder.schemas import Candidate, Project


def _score_lookup(
    scores: list[CandidateProjectScore],
) -> dict[tuple[str, str], CandidateProjectScore]:
    """Index candidate-project scores."""

    return {(score.candidate_id, score.project_id): score for score in scores}


def _is_feasible_pair(
    candidate_id: str,
    project_id: str,
    scores: dict[tuple[str, str], CandidateProjectScore],
) -> bool:
    """Return whether a candidate-project pair satisfies feasibility."""

    pair_score = scores.get((candidate_id, project_id))
    return bool(pair_score and pair_score.feasible)


def _copy_teams(teams: dict[str, list[Candidate]]) -> dict[str, list[Candidate]]:
    """Create a shallow copy of team membership lists."""

    return {project_id: list(team) for project_id, team in teams.items()}


def _replace_member(
    team: list[Candidate],
    removed_candidate_id: str,
    added_candidate: Candidate,
) -> list[Candidate]:
    """Return a new team list with one member replaced."""

    return [
        added_candidate if member.id == removed_candidate_id else member
        for member in team
    ]


def _teams_from_report(
    report: AllocationReport,
    candidates_by_id: dict[str, Candidate],
) -> dict[str, list[Candidate]]:
    """Reconstruct teams from an allocation report."""

    teams: dict[str, list[Candidate]] = {}
    for summary in report.project_summaries:
        teams[summary.project_id] = [
            candidates_by_id[candidate_id]
            for candidate_id in summary.member_ids
            if candidate_id in candidates_by_id
        ]
    return teams


def _evaluate(
    projects: list[Project],
    teams: dict[str, list[Candidate]],
    score_lookup: dict[tuple[str, str], CandidateProjectScore],
    fairness_penalty: float,
):
    """Evaluate project teams and the global allocation objective."""

    summaries = [
        evaluate_project_team(project, teams.get(project.id, []), score_lookup)
        for project in projects
    ]
    team_scores = [summary.team_score for summary in summaries]

    if not team_scores:
        return summaries, 0.0, 0.0, None, None, None

    fairness_deviation = max(team_scores) - min(team_scores)
    objective = sum(team_scores) - fairness_penalty * fairness_deviation

    return (
        summaries,
        objective,
        fairness_deviation,
        min(team_scores),
        mean(team_scores),
        max(team_scores),
    )


def _apply_swap(
    teams: dict[str, list[Candidate]],
    left_project_id: str,
    right_project_id: str,
    left_candidate: Candidate,
    right_candidate: Candidate,
) -> dict[str, list[Candidate]]:
    """Return a new allocation after swapping two assigned candidates."""

    trial = _copy_teams(teams)
    trial[left_project_id] = _replace_member(
        trial[left_project_id],
        removed_candidate_id=left_candidate.id,
        added_candidate=right_candidate,
    )
    trial[right_project_id] = _replace_member(
        trial[right_project_id],
        removed_candidate_id=right_candidate.id,
        added_candidate=left_candidate,
    )
    return trial


def _apply_replacement(
    teams: dict[str, list[Candidate]],
    project_id: str,
    removed_candidate: Candidate,
    added_candidate: Candidate,
) -> dict[str, list[Candidate]]:
    """Return a new allocation after replacing one assigned candidate."""

    trial = _copy_teams(teams)
    trial[project_id] = _replace_member(
        trial[project_id],
        removed_candidate_id=removed_candidate.id,
        added_candidate=added_candidate,
    )
    return trial


def improve_allocation(
    *,
    candidates: list[Candidate],
    projects: list[Project],
    scores: list[CandidateProjectScore],
    allocation: AllocationReport,
    fairness_penalty: float = 0.25,
    max_iterations: int = 100,
    min_gain: float = 1e-9,
) -> AllocationReport:
    """Improve an allocation using feasible swaps and replacements.

    The returned AllocationReport contains the final teams after local
    improvement. The original round-based construction assignments remain in the
    `assignments` field for traceability, while accepted local moves are reported
    separately in `local_improvement`.
    """

    candidates_by_id = {candidate.id: candidate for candidate in candidates}
    score_by_pair = _score_lookup(scores)

    current_teams = _teams_from_report(allocation, candidates_by_id)
    current_unassigned = set(allocation.unassigned_candidate_ids)

    (
        _initial_summaries,
        initial_objective,
        _initial_fairness,
        _initial_min,
        _initial_mean,
        _initial_max,
    ) = _evaluate(projects, current_teams, score_by_pair, fairness_penalty)

    current_objective = initial_objective
    accepted_moves: list[LocalImprovementMove] = []
    accepted_swaps = 0
    accepted_replacements = 0
    evaluated_swaps = 0
    evaluated_replacements = 0
    stop_reason = "no_improving_move_found"
    iterations_completed = 0

    for iteration in range(1, max_iterations + 1):
        iterations_completed = iteration

        best_gain = min_gain
        best_teams: dict[str, list[Candidate]] | None = None
        best_unassigned: set[str] | None = None
        best_move_type: str | None = None
        best_description = ""
        best_affected_projects: tuple[str, ...] = ()

        # Try swaps between assigned candidates in different projects.
        for left_index, left_project in enumerate(projects):
            for right_project in projects[left_index + 1:]:
                left_team = current_teams.get(left_project.id, [])
                right_team = current_teams.get(right_project.id, [])

                for left_candidate in left_team:
                    for right_candidate in right_team:
                        if not _is_feasible_pair(
                            left_candidate.id, right_project.id, score_by_pair
                        ):
                            continue
                        if not _is_feasible_pair(
                            right_candidate.id, left_project.id, score_by_pair
                        ):
                            continue

                        evaluated_swaps += 1

                        trial_teams = _apply_swap(
                            current_teams,
                            left_project.id,
                            right_project.id,
                            left_candidate,
                            right_candidate,
                        )

                        (
                            _summaries,
                            trial_objective,
                            _fairness,
                            _min_team,
                            _mean_team,
                            _max_team,
                        ) = _evaluate(projects, trial_teams, score_by_pair, fairness_penalty)

                        gain = trial_objective - current_objective

                        if gain > best_gain:
                            best_gain = gain
                            best_teams = trial_teams
                            best_unassigned = set(current_unassigned)
                            best_move_type = "swap"
                            best_affected_projects = (left_project.id, right_project.id)
                            best_description = (
                                f"Swapped {left_candidate.id} from {left_project.id} "
                                f"with {right_candidate.id} from {right_project.id}."
                            )

        # Try replacing an assigned candidate with someone unassigned.
        for project in projects:
            team = current_teams.get(project.id, [])

            for removed_candidate in team:
                for added_candidate_id in sorted(current_unassigned):
                    if not _is_feasible_pair(added_candidate_id, project.id, score_by_pair):
                        continue

                    evaluated_replacements += 1

                    added_candidate = candidates_by_id[added_candidate_id]
                    trial_teams = _apply_replacement(
                        current_teams,
                        project.id,
                        removed_candidate,
                        added_candidate,
                    )

                    (
                        _summaries,
                        trial_objective,
                        _fairness,
                        _min_team,
                        _mean_team,
                        _max_team,
                    ) = _evaluate(projects, trial_teams, score_by_pair, fairness_penalty)

                    gain = trial_objective - current_objective

                    if gain > best_gain:
                        trial_unassigned = set(current_unassigned)
                        trial_unassigned.remove(added_candidate.id)
                        trial_unassigned.add(removed_candidate.id)

                        best_gain = gain
                        best_teams = trial_teams
                        best_unassigned = trial_unassigned
                        best_move_type = "replacement"
                        best_affected_projects = (project.id,)
                        best_description = (
                            f"Replaced {removed_candidate.id} with unassigned "
                            f"{added_candidate.id} in {project.id}."
                        )

        if best_teams is None or best_unassigned is None or best_move_type is None:
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
        final_summaries,
        final_objective,
        fairness_deviation,
        min_team_score,
        mean_team_score,
        max_team_score,
    ) = _evaluate(projects, current_teams, score_by_pair, fairness_penalty)

    improvement_report = LocalImprovementReport(
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

    incomplete_projects = [
        summary.project_id
        for summary in final_summaries
        if len(summary.member_ids) != summary.target_team_size
    ]
    warnings = list(allocation.warnings)
    if incomplete_projects:
        warning = "Incomplete project team(s): " + ", ".join(incomplete_projects)
        if warning not in warnings:
            warnings.append(warning)

    return AllocationReport(
        method=f"{allocation.method}_with_local_improvement",
        feasible=not incomplete_projects,
        assigned_count=sum(len(team) for team in current_teams.values()),
        unassigned_count=len(current_unassigned),
        required_slots=allocation.required_slots,
        objective_score=final_objective,
        fairness_deviation=fairness_deviation,
        min_team_score=min_team_score,
        mean_team_score=mean_team_score,
        max_team_score=max_team_score,
        project_summaries=final_summaries,
        assignments=allocation.assignments,
        unassigned_candidate_ids=tuple(sorted(current_unassigned)),
        warnings=tuple(warnings),
        local_improvement=improvement_report,
    )
