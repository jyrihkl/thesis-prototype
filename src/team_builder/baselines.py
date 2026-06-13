"""Comparator methods for team allocation evaluation.

The thesis evaluation plan requires that the proposed recommendation logic is
compared against simpler alternatives under the same participant pool, project
briefs, and hard feasibility conditions. This module implements transparent
baselines:

1. random_constrained
2. greedy_fit
3. machado_k_rounds
4. thesis_no_li
5. machado_k_rounds_li

All comparators are evaluated with the same team-quality and allocation-level
fairness objective used by the main allocation method.
"""

from __future__ import annotations

import random
from statistics import mean, stdev

from team_builder.allocation import (
    construct_round_based_allocation,
    evaluate_project_team,
    improve_allocation_teams,
)
from team_builder.models import (
    AllocationReport,
    BaselineComparisonReport,
    BaselineMethodSummary,
    CandidateProjectScore,
    ProjectTeamSummary,
    RandomBaselineRunSummary,
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
    local_improvement=None,
    random_seed: int | None = None,
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
        local_improvement=local_improvement,
        random_seed=random_seed,
    )


def _candidate_feasible_for_project(
    candidate: Candidate,
    project: Project,
    score_lookup: dict[tuple[str, str], CandidateProjectScore],
) -> bool:
    """Return whether a candidate-project pair satisfies hard feasibility."""

    pair_score = score_lookup.get((candidate.id, project.id))
    return bool(pair_score and pair_score.feasible)


def _best_direct_fit_candidate_id(
    *,
    project: Project,
    unassigned_ids: set[str],
    score_lookup: dict[tuple[str, str], CandidateProjectScore],
) -> str | None:
    """Return the unassigned feasible candidate with the highest direct fit."""

    feasible_scores = [
        score_lookup[(candidate_id, project.id)]
        for candidate_id in unassigned_ids
        if (candidate_id, project.id) in score_lookup
        and score_lookup[(candidate_id, project.id)].feasible
    ]

    if not feasible_scores:
        return None

    selected_score = sorted(
        feasible_scores,
        key=lambda score: (-score.total_score, score.candidate_id),
    )[0]

    return selected_score.candidate_id


def random_constrained_assignment(
    *,
    candidates: list[Candidate],
    projects: list[Project],
    scores: list[CandidateProjectScore],
    fairness_penalty: float = 0.25,
    random_seed: int = 42,
    max_attempts: int = 200,
) -> AllocationReport:
    """Assign candidates randomly while respecting hard feasibility.

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
            feasible_ids = sorted(
                candidate_id
                for candidate_id in unassigned_ids
                if _candidate_feasible_for_project(
                    candidate=candidates_by_id[candidate_id],
                    project=project,
                    score_lookup=score_lookup,
                )
            )

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
            method="random",
            projects=projects,
            teams=teams,
            unassigned_ids=unassigned_ids,
            score_lookup=score_lookup,
            fairness_penalty=fairness_penalty,
            warnings=warnings,
            random_seed=random_seed,
        )

        if report.feasible:
            return report

        if report.objective_score > best_objective:
            best_objective = report.objective_score
            best_report = report

    if best_report is not None:
        return best_report

    return _evaluate_teams(
        method="random",
        projects=projects,
        teams={project.id: [] for project in projects},
        unassigned_ids=set(candidates_by_id),
        score_lookup=score_lookup,
        fairness_penalty=fairness_penalty,
        warnings=["Random constrained assignment could not create any allocation."],
        random_seed=random_seed,
    )


def greedy_fit_assignment(
    *,
    candidates: list[Candidate],
    projects: list[Project],
    scores: list[CandidateProjectScore],
    fairness_penalty: float = 0.25,
) -> AllocationReport:
    """Assign candidates greedily by direct candidate-to-project fit.

    This baseline fills one project before moving to the next.
    """

    candidates_by_id = {candidate.id: candidate for candidate in candidates}
    score_lookup = _score_lookup(scores)

    teams: dict[str, list[Candidate]] = {project.id: [] for project in projects}
    unassigned_ids = set(candidates_by_id)
    warnings: list[str] = []

    for project in projects:
        while len(teams[project.id]) < project.target_team_size:
            selected_id = _best_direct_fit_candidate_id(
                project=project,
                unassigned_ids=unassigned_ids,
                score_lookup=score_lookup,
            )

            if selected_id is None:
                warnings.append(
                    f"No feasible greedy candidate found for project {project.id}."
                )
                break

            teams[project.id].append(candidates_by_id[selected_id])
            unassigned_ids.remove(selected_id)

    return _evaluate_teams(
        method="greedy_fit",
        projects=projects,
        teams=teams,
        unassigned_ids=unassigned_ids,
        score_lookup=score_lookup,
        fairness_penalty=fairness_penalty,
        warnings=warnings,
    )


def machado_k_rounds_assignment(
    *,
    candidates: list[Candidate],
    projects: list[Project],
    scores: list[CandidateProjectScore],
    fairness_penalty: float = 0.25,
) -> AllocationReport:
    """Assign candidates using a Machado-inspired K-rounds baseline."""

    candidates_by_id = {candidate.id: candidate for candidate in candidates}
    score_lookup = _score_lookup(scores)

    teams: dict[str, list[Candidate]] = {project.id: [] for project in projects}
    unassigned_ids = set(candidates_by_id)
    warnings: list[str] = []

    if not projects:
        return _evaluate_teams(
            method="machado_k_rounds",
            projects=projects,
            teams=teams,
            unassigned_ids=unassigned_ids,
            score_lookup=score_lookup,
            fairness_penalty=fairness_penalty,
            warnings=["No projects available for Machado K-rounds assignment."],
        )

    max_rounds = max(project.target_team_size for project in projects)

    for round_index in range(max_rounds):
        for project in projects:
            if len(teams[project.id]) >= project.target_team_size:
                continue

            selected_id = _best_direct_fit_candidate_id(
                project=project,
                unassigned_ids=unassigned_ids,
                score_lookup=score_lookup,
            )

            if selected_id is None:
                warnings.append(
                    f"No feasible Machado K-rounds candidate found for project "
                    f"{project.id} in round {round_index + 1}."
                )
                continue

            teams[project.id].append(candidates_by_id[selected_id])
            unassigned_ids.remove(selected_id)

    return _evaluate_teams(
        method="machado_k_rounds",
        projects=projects,
        teams=teams,
        unassigned_ids=unassigned_ids,
        score_lookup=score_lookup,
        fairness_penalty=fairness_penalty,
        warnings=warnings,
    )


def thesis_without_local_improvement_assignment(
    *,
    candidates: list[Candidate],
    projects: list[Project],
    scores: list[CandidateProjectScore],
    fairness_penalty: float = 0.25,
) -> AllocationReport:
    """Run the thesis construction heuristic without local improvement."""

    return construct_round_based_allocation(
        candidates=candidates,
        projects=projects,
        scores=scores,
        fairness_penalty=fairness_penalty,
        enable_local_improvement=False,
    )


def machado_k_rounds_with_local_improvement_assignment(
    *,
    candidates: list[Candidate],
    projects: list[Project],
    scores: list[CandidateProjectScore],
    fairness_penalty: float = 0.25,
    max_local_improvement_iterations: int = 100,
    min_local_improvement_gain: float = 1e-9,
) -> AllocationReport:
    """Run Machado-inspired K-rounds and then apply local improvement."""

    initial_report = machado_k_rounds_assignment(
        candidates=candidates,
        projects=projects,
        scores=scores,
        fairness_penalty=fairness_penalty,
    )

    candidates_by_id = {candidate.id: candidate for candidate in candidates}
    teams = {
        summary.project_id: [
            candidates_by_id[candidate_id]
            for candidate_id in summary.member_ids
        ]
        for summary in initial_report.project_summaries
    }
    unassigned_ids = set(initial_report.unassigned_candidate_ids)
    score_lookup = _score_lookup(scores)

    teams, unassigned_ids, local_improvement_report = improve_allocation_teams(
        candidates=candidates,
        projects=projects,
        teams=teams,
        unassigned_ids=unassigned_ids,
        scores=scores,
        fairness_penalty=fairness_penalty,
        max_iterations=max_local_improvement_iterations,
        min_gain=min_local_improvement_gain,
    )

    improved_report = _evaluate_teams(
        method="machado_k_rounds_li",
        projects=projects,
        teams=teams,
        unassigned_ids=unassigned_ids,
        score_lookup=score_lookup,
        fairness_penalty=fairness_penalty,
        warnings=list(initial_report.warnings),
        local_improvement=local_improvement_report,
    )

    return improved_report

def _distribution_statistics(
    values: list[float],
) -> tuple[float | None, float | None, float | None, float | None]:
    """Return mean, sample standard deviation, minimum, and maximum."""

    if not values:
        return None, None, None, None

    return (
        mean(values),
        stdev(values) if len(values) > 1 else 0.0,
        min(values),
        max(values),
    )


def _optional_metric_values(
    reports: list[AllocationReport],
    attribute: str,
) -> list[float]:
    """Collect non-null numeric values for one allocation-report metric."""

    values: list[float] = []
    for report in reports:
        value = getattr(report, attribute)
        if value is not None:
            values.append(float(value))
    return values


def _summarize_random_run(report: AllocationReport) -> RandomBaselineRunSummary:
    """Convert one seeded random allocation into a compact run summary."""

    if report.random_seed is None:
        raise ValueError("Random allocation report is missing its seed.")

    return RandomBaselineRunSummary(
        seed=report.random_seed,
        feasible=report.feasible,
        objective_score=report.objective_score,
        mean_team_score=report.mean_team_score,
        min_team_score=report.min_team_score,
        max_team_score=report.max_team_score,
        fairness_deviation=report.fairness_deviation,
        assigned_count=report.assigned_count,
        required_slots=report.required_slots,
        warnings=report.warnings,
    )


def _aggregate_random_reports(
    reports: list[AllocationReport],
) -> BaselineMethodSummary:
    """Aggregate repeated random-baseline runs into one comparison row.

    Metrics are aggregated over feasible seeded runs. The summary is marked
    feasible only when every requested seed produced a feasible allocation, so
    failures remain visible rather than disappearing into the mean.
    """

    if not reports:
        raise ValueError("At least one random-baseline report is required.")

    feasible_reports = [report for report in reports if report.feasible]
    metric_reports = feasible_reports or reports

    objective = _distribution_statistics(
        [float(report.objective_score) for report in metric_reports]
    )
    mean_team = _distribution_statistics(
        _optional_metric_values(metric_reports, "mean_team_score")
    )
    min_team = _distribution_statistics(
        _optional_metric_values(metric_reports, "min_team_score")
    )
    max_team = _distribution_statistics(
        _optional_metric_values(metric_reports, "max_team_score")
    )
    fairness = _distribution_statistics(
        [float(report.fairness_deviation) for report in metric_reports]
    )

    return BaselineMethodSummary(
        method="random",
        feasible=len(feasible_reports) == len(reports),
        objective_score=objective[0] or 0.0,
        mean_team_score=mean_team[0],
        min_team_score=min_team[0],
        max_team_score=max_team[0],
        fairness_deviation=fairness[0] or 0.0,
        assigned_count=round(mean(report.assigned_count for report in metric_reports)),
        required_slots=reports[0].required_slots,
        sample_count=len(reports),
        feasible_count=len(feasible_reports),
        objective_score_std=objective[1],
        objective_score_min=objective[2],
        objective_score_max=objective[3],
        mean_team_score_std=mean_team[1],
        mean_team_score_min=mean_team[2],
        mean_team_score_max=mean_team[3],
        min_team_score_std=min_team[1],
        min_team_score_min=min_team[2],
        min_team_score_max=min_team[3],
        max_team_score_std=max_team[1],
        max_team_score_min=max_team[2],
        max_team_score_max=max_team[3],
        fairness_deviation_std=fairness[1],
        fairness_deviation_min=fairness[2],
        fairness_deviation_max=fairness[3],
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
        sample_count=1,
        feasible_count=1 if report.feasible else 0,
    )


def run_baseline_comparisons(
    *,
    main_allocation: AllocationReport,
    candidates: list[Candidate],
    projects: list[Project],
    scores: list[CandidateProjectScore],
    fairness_penalty: float = 0.25,
    random_baseline_runs: int = 30,
    random_baseline_seed_start: int = 42,
    max_local_improvement_iterations: int = 100,
    min_local_improvement_gain: float = 1e-9,
) -> BaselineComparisonReport:
    """Run comparator methods and summarize them against the main method."""

    if random_baseline_runs < 1:
        raise ValueError("random_baseline_runs must be at least 1.")

    random_reports = [
        random_constrained_assignment(
            candidates=candidates,
            projects=projects,
            scores=scores,
            fairness_penalty=fairness_penalty,
            random_seed=random_baseline_seed_start + offset,
        )
        for offset in range(random_baseline_runs)
    ]

    deterministic_reports = [
        greedy_fit_assignment(
            candidates=candidates,
            projects=projects,
            scores=scores,
            fairness_penalty=fairness_penalty,
        ),
        machado_k_rounds_assignment(
            candidates=candidates,
            projects=projects,
            scores=scores,
            fairness_penalty=fairness_penalty,
        ),
        thesis_without_local_improvement_assignment(
            candidates=candidates,
            projects=projects,
            scores=scores,
            fairness_penalty=fairness_penalty,
        ),
        machado_k_rounds_with_local_improvement_assignment(
            candidates=candidates,
            projects=projects,
            scores=scores,
            fairness_penalty=fairness_penalty,
            max_local_improvement_iterations=max_local_improvement_iterations,
            min_local_improvement_gain=min_local_improvement_gain,
        ),
    ]

    random_run_summaries = [
        _summarize_random_run(report) for report in random_reports
    ]
    baseline_reports = deterministic_reports
    method_summaries = [
        _summarize_method(main_allocation),
        _aggregate_random_reports(random_reports),
        *[_summarize_method(report) for report in deterministic_reports],
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
        random_run_summaries=random_run_summaries,
        best_by_objective=best_by_objective,
        best_by_min_team_score=best_by_min_team_score,
        best_by_fairness=best_by_fairness,
    )
