"""Comparator methods for team allocation evaluation.

The thesis evaluation plan requires that the proposed recommendation logic is
compared against simpler alternatives under the same participant pool, project
briefs, and hard feasibility conditions. This module implements three transparent
baselines:

1. random_constrained
2. greedy_fit
3. balanced_greedy

All baselines are evaluated with the same team-quality and allocation-level
fairness objective used by the main allocation method.
"""

from __future__ import annotations

import random
from statistics import mean

from team_builder.allocation import evaluate_project_team
from team_builder.models import (
    AllocationReport,
    BaselineComparisonReport,
    BaselineMethodSummary,
    CandidateProjectScore,
    ProjectTeamSummary,
)
from team_builder.schemas import Candidate, Project


def _score_lookup(
    scores: list[CandidateProjectScore],
) -> dict[tuple[str, str], CandidateProjectScore]:
    """Index candidate-project scores by candidate ID and project ID."""

    return {
        (score.candidate_id, score.project_id): score
        for score in scores
    }


def _evaluate_teams(
    *,
    method: str,
    projects: list[Project],
    teams: dict[str, list[Candidate]],
    unassigned_ids: set[str],
    score_lookup: dict[tuple[str, str], CandidateProjectScore],
    fairness_penalty: float,
    warnings: list[str] | None = None,
) -> AllocationReport:
    """Create an AllocationReport for a baseline allocation."""

    warnings = warnings or []

    project_summaries: list[ProjectTeamSummary] = [
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

    assigned_count = sum(len(team) for team in teams.values())

    return AllocationReport(
        method=method,
        feasible=not incomplete_projects,
        assigned_count=assigned_count,
        unassigned_count=len(unassigned_ids),
        required_slots=sum(project.target_team_size for project in projects),
        objective_score=objective_score,
        fairness_deviation=fairness_deviation,
        min_team_score=min_team_score,
        mean_team_score=mean_team_score,
        max_team_score=max_team_score,
        project_summaries=project_summaries,
        assignments=[],
        unassigned_candidate_ids=tuple(sorted(unassigned_ids)),
        warnings=tuple(warnings),
        local_improvement=None,
    )


def _candidate_feasible_for_project(
    candidate: Candidate,
    project: Project,
    score_lookup: dict[tuple[str, str], CandidateProjectScore],
) -> bool:
    """Return whether a candidate-project pair satisfies hard feasibility."""

    pair_score = score_lookup.get((candidate.id, project.id))
    return bool(pair_score and pair_score.feasible)


def random_constrained_assignment(
    *,
    candidates: list[Candidate],
    projects: list[Project],
    scores: list[CandidateProjectScore],
    fairness_penalty: float = 0.25,
    random_seed: int = 42,
    max_attempts: int = 200,
) -> AllocationReport:
    """Assign candidates randomly while respecting feasibility.

    The method retries with different random orders because random assignment can
    paint itself into a corner even when a feasible allocation exists.
    """

    rng = random.Random(random_seed)
    candidates_by_id = {candidate.id: candidate for candidate in candidates}
    score_lookup = _score_lookup(scores)

    best_report: AllocationReport | None = None
    best_objective = float("-inf")

    slots = [
        project.id
        for project in projects
        for _ in range(project.target_team_size)
    ]
    projects_by_id = {project.id: project for project in projects}

    for attempt in range(1, max_attempts + 1):
        shuffled_slots = list(slots)
        rng.shuffle(shuffled_slots)

        unassigned_ids = set(candidates_by_id)
        teams: dict[str, list[Candidate]] = {project.id: [] for project in projects}
        warnings: list[str] = []

        for project_id in shuffled_slots:
            project = projects_by_id[project_id]
            feasible_ids = [
                candidate_id
                for candidate_id in unassigned_ids
                if _candidate_feasible_for_project(
                    candidate=candidates_by_id[candidate_id],
                    project=project,
                    score_lookup=score_lookup,
                )
            ]

            if not feasible_ids:
                warnings.append(
                    f"No feasible random candidate for project {project_id} "
                    f"on attempt {attempt}."
                )
                break

            selected_id = rng.choice(feasible_ids)
            teams[project_id].append(candidates_by_id[selected_id])
            unassigned_ids.remove(selected_id)

        report = _evaluate_teams(
            method="baseline_random_constrained",
            projects=projects,
            teams=teams,
            unassigned_ids=unassigned_ids,
            score_lookup=score_lookup,
            fairness_penalty=fairness_penalty,
            warnings=warnings,
        )

        if report.feasible:
            return report

        if report.objective_score > best_objective:
            best_objective = report.objective_score
            best_report = report

    if best_report is not None:
        return best_report

    return _evaluate_teams(
        method="baseline_random_constrained",
        projects=projects,
        teams={project.id: [] for project in projects},
        unassigned_ids=set(candidates_by_id),
        score_lookup=score_lookup,
        fairness_penalty=fairness_penalty,
        warnings=["Random constrained assignment could not create any allocation."],
    )


def greedy_fit_assignment(
    *,
    candidates: list[Candidate],
    projects: list[Project],
    scores: list[CandidateProjectScore],
    fairness_penalty: float = 0.25,
) -> AllocationReport:
    """Assign candidates greedily by direct candidate-to-project fit.

    This baseline intentionally ignores marginal contribution and fairness during
    construction. It tests whether the proposed method improves on simple local
    project-fit maximization.
    """

    candidates_by_id = {candidate.id: candidate for candidate in candidates}
    score_lookup = _score_lookup(scores)

    teams: dict[str, list[Candidate]] = {project.id: [] for project in projects}
    unassigned_ids = set(candidates_by_id)
    warnings: list[str] = []

    for project in projects:
        while len(teams[project.id]) < project.target_team_size:
            feasible_scores = [
                score_lookup[(candidate_id, project.id)]
                for candidate_id in unassigned_ids
                if (candidate_id, project.id) in score_lookup
                and score_lookup[(candidate_id, project.id)].feasible
            ]

            if not feasible_scores:
                warnings.append(
                    f"No feasible greedy candidate found for project {project.id}."
                )
                break

            selected_score = sorted(
                feasible_scores,
                key=lambda score: (-score.total_score, score.candidate_id),
            )[0]

            selected_candidate = candidates_by_id[selected_score.candidate_id]
            teams[project.id].append(selected_candidate)
            unassigned_ids.remove(selected_candidate.id)

    return _evaluate_teams(
        method="baseline_greedy_fit",
        projects=projects,
        teams=teams,
        unassigned_ids=unassigned_ids,
        score_lookup=score_lookup,
        fairness_penalty=fairness_penalty,
        warnings=warnings,
    )


def balanced_greedy_assignment(
    *,
    candidates: list[Candidate],
    projects: list[Project],
    scores: list[CandidateProjectScore],
    fairness_penalty: float = 0.25,
) -> AllocationReport:
    """Assign candidates greedily in rounds across projects.

    This baseline is stronger than pure greedy fit because it distributes picks
    across projects. It still does not use marginal contribution to the current
    team. Tests whether marginal contribution adds value beyond round-based balancing.
    """

    candidates_by_id = {candidate.id: candidate for candidate in candidates}
    score_lookup = _score_lookup(scores)

    teams: dict[str, list[Candidate]] = {project.id: [] for project in projects}
    unassigned_ids = set(candidates_by_id)
    warnings: list[str] = []

    if not projects:
        return _evaluate_teams(
            method="baseline_balanced_greedy",
            projects=projects,
            teams=teams,
            unassigned_ids=unassigned_ids,
            score_lookup=score_lookup,
            fairness_penalty=fairness_penalty,
            warnings=["No projects available for balanced greedy assignment."],
        )

    max_rounds = max(project.target_team_size for project in projects)

    for round_index in range(max_rounds):
        rotation = round_index % len(projects)
        ordered_projects = projects[rotation:] + projects[:rotation]

        for project in ordered_projects:
            if len(teams[project.id]) >= project.target_team_size:
                continue

            feasible_scores = [
                score_lookup[(candidate_id, project.id)]
                for candidate_id in unassigned_ids
                if (candidate_id, project.id) in score_lookup
                and score_lookup[(candidate_id, project.id)].feasible
            ]

            if not feasible_scores:
                warnings.append(
                    f"No feasible balanced-greedy candidate found for project "
                    f"{project.id} in round {round_index + 1}."
                )
                continue

            selected_score = sorted(
                feasible_scores,
                key=lambda score: (-score.total_score, score.candidate_id),
            )[0]

            selected_candidate = candidates_by_id[selected_score.candidate_id]
            teams[project.id].append(selected_candidate)
            unassigned_ids.remove(selected_candidate.id)

    return _evaluate_teams(
        method="baseline_balanced_greedy",
        projects=projects,
        teams=teams,
        unassigned_ids=unassigned_ids,
        score_lookup=score_lookup,
        fairness_penalty=fairness_penalty,
        warnings=warnings,
    )


def _summarize_method(report: AllocationReport) -> BaselineMethodSummary:
    """Convert an allocation report into a compact method-comparison summary."""

    return BaselineMethodSummary(
        method=report.method,
        feasible=report.feasible,
        objective_score=report.objective_score,
        mean_team_score=report.mean_team_score,
        min_team_score=report.min_team_score,
        max_team_score=report.max_team_score,
        fairness_deviation=report.fairness_deviation,
        assigned_count=report.assigned_count,
        required_slots=report.required_slots,
    )


def run_baseline_comparisons(
    *,
    main_allocation: AllocationReport,
    candidates: list[Candidate],
    projects: list[Project],
    scores: list[CandidateProjectScore],
    fairness_penalty: float = 0.25,
    random_seed: int = 42,
) -> BaselineComparisonReport:
    """Run comparator methods and summarize them against the main method."""

    baseline_reports = [
        random_constrained_assignment(
            candidates=candidates,
            projects=projects,
            scores=scores,
            fairness_penalty=fairness_penalty,
            random_seed=random_seed,
        ),
        greedy_fit_assignment(
            candidates=candidates,
            projects=projects,
            scores=scores,
            fairness_penalty=fairness_penalty,
        ),
        balanced_greedy_assignment(
            candidates=candidates,
            projects=projects,
            scores=scores,
            fairness_penalty=fairness_penalty,
        ),
    ]

    method_summaries = [
        _summarize_method(main_allocation),
        *[_summarize_method(report) for report in baseline_reports],
    ]

    feasible_summaries = [
        summary for summary in method_summaries if summary.feasible
    ]

    if feasible_summaries:
        best_by_objective = max(
            feasible_summaries,
            key=lambda summary: summary.objective_score,
        ).method
        best_by_min_team_score = max(
            feasible_summaries,
            key=lambda summary: (
                summary.min_team_score
                if summary.min_team_score is not None
                else float("-inf")
            ),
        ).method
        best_by_fairness = min(
            feasible_summaries,
            key=lambda summary: summary.fairness_deviation,
        ).method
    else:
        best_by_objective = None
        best_by_min_team_score = None
        best_by_fairness = None

    return BaselineComparisonReport(
        main_method=main_allocation.method,
        method_summaries=method_summaries,
        baseline_reports=baseline_reports,
        best_by_objective=best_by_objective,
        best_by_min_team_score=best_by_min_team_score,
        best_by_fairness=best_by_fairness,
    )
