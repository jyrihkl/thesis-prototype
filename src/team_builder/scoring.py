"""Candidate-to-project scoring.

This module implements the first scoring stage from the thesis plan. It does not
construct teams yet. Instead, it computes transparent participant-to-project fit
scores that later allocation logic can use as one input.
"""

from __future__ import annotations

from statistics import mean

from team_builder.models import (
    CandidateProjectScore,
    ProjectScoreSummary,
    ScoreWeights,
    ScoringReport,
)
from team_builder.schemas import Candidate, Project


DEFAULT_SCORE_WEIGHTS = ScoreWeights(
    required_skills=0.40,
    preferred_skills=0.20,
    role=0.20,
    experience=0.10,
    language=0.05,
    interests=0.05,
)


def overlap_score(candidate_values: frozenset[str], target_values: frozenset[str]) -> float | None:
    """Return normalized overlap between candidate values and target values.

    If the project does not define the target criterion, None is returned so the
    criterion can be excluded from the active weighted average.
    """

    if not target_values:
        return None
    if not candidate_values:
        return 0.0
    return len(candidate_values & target_values) / len(target_values)


def role_match_score(candidate: Candidate, project: Project) -> float | None:
    """Score whether the candidate's role family matches the desired project roles."""

    if not project.desired_roles:
        return None
    if not candidate.role_family or candidate.role_family == "unknown":
        return 0.0
    return 1.0 if candidate.role_family in project.desired_roles else 0.0


def experience_score(candidate: Candidate) -> float:
    """Return a broad individual experience suitability score.

    Team-level experience balance will be handled later during team construction.
    This score is intentionally weak and should not dominate skill or role fit.
    """

    years = candidate.experience_years
    if years is None:
        return 0.5
    if years < 1:
        return 0.25
    if years < 3:
        return 0.55
    if years < 6:
        return 0.80
    if years < 10:
        return 0.95
    return 0.90


def language_score(candidate: Candidate, project: Project) -> tuple[float | None, bool, list[str]]:
    """Score and validate language suitability.

    The score is None when the project has no minimum language requirement.
    When a minimum exists, the score is proportional up to 1.0. The feasibility
    flag is false when the candidate does not satisfy the minimum.
    """

    if project.min_english_score is None:
        return None, True, []

    if candidate.english_score is None:
        return 0.0, False, ["missing_english_level"]

    if candidate.english_score >= project.min_english_score:
        return 1.0, True, []

    partial = max(0.0, min(1.0, candidate.english_score / project.min_english_score))
    return partial, False, ["below_minimum_english_level"]


def interest_score(candidate: Candidate, project: Project) -> float | None:
    """Score optional interest alignment using project text-derived tags.

    For now, this purely exists for possible future use.
    """

    return None


def weighted_average(
    components: dict[str, float | None],
    weights: ScoreWeights,
) -> float:
    """Compute an active weighted average over non-None components."""

    weight_map = weights.as_dict()

    numerator = 0.0
    denominator = 0.0

    for name, value in components.items():
        if value is None:
            continue # Equal to numerator += 0 and denominator += 0
        weight = weight_map[name]
        numerator += weight * value
        denominator += weight

    if denominator == 0:
        return 0.0

    return numerator / denominator


def score_candidate_for_project(
    candidate: Candidate,
    project: Project,
    weights: ScoreWeights = DEFAULT_SCORE_WEIGHTS,
) -> CandidateProjectScore:
    """Compute a transparent fit score for one candidate-project pair."""

    required = overlap_score(candidate.skills, project.required_skills)
    preferred = overlap_score(candidate.skills, project.preferred_skills)
    role = role_match_score(candidate, project)
    experience = experience_score(candidate)
    language, language_feasible, infeasibility_reasons = language_score(candidate, project)
    interests = interest_score(candidate, project)

    components = {
        "required_skills": required,
        "preferred_skills": preferred,
        "role": role,
        "experience": experience,
        "language": language,
        "interests": interests,
    }

    total_score = weighted_average(components, weights)

    return CandidateProjectScore(
        candidate_id=candidate.id,
        project_id=project.id,
        total_score=total_score,
        feasible=language_feasible,
        infeasibility_reasons=tuple(infeasibility_reasons),
        components=components,
        matched_required_skills=tuple(sorted(candidate.skills & project.required_skills)),
        missing_required_skills=tuple(sorted(project.required_skills - candidate.skills)),
        matched_preferred_skills=tuple(sorted(candidate.skills & project.preferred_skills)),
        role_family=candidate.role_family,
    )


def score_candidate_project_matrix(
    candidates: list[Candidate],
    projects: list[Project],
    weights: ScoreWeights = DEFAULT_SCORE_WEIGHTS,
) -> list[CandidateProjectScore]:
    """Score every candidate against every project."""

    return [
        score_candidate_for_project(candidate, project, weights)
        for project in projects
        for candidate in candidates
    ]


def summarize_scoring(
    scores: list[CandidateProjectScore],
    projects: list[Project],
    weights: ScoreWeights = DEFAULT_SCORE_WEIGHTS,
    top_n: int = 3,
) -> ScoringReport:
    """Summarize candidate-to-project score distributions for reporting."""

    summaries: list[ProjectScoreSummary] = []

    for project in projects:
        project_scores = [score for score in scores if score.project_id == project.id]
        feasible_scores = [score for score in project_scores if score.feasible]
        score_values = [score.total_score for score in feasible_scores]

        top_scores = sorted(
            feasible_scores,
            key=lambda score: score.total_score,
            reverse=True,
        )[:top_n]

        summaries.append(
            ProjectScoreSummary(
                project_id=project.id,
                project_title=project.title,
                scored_candidates=len(project_scores),
                feasible_candidates=len(feasible_scores),
                min_score=min(score_values) if score_values else None,
                mean_score=mean(score_values) if score_values else None,
                max_score=max(score_values) if score_values else None,
                top_candidate_ids=tuple(score.candidate_id for score in top_scores),
            )
        )

    return ScoringReport(
        total_pairs=len(scores),
        feasible_pairs=sum(1 for score in scores if score.feasible),
        weights=weights.as_dict(),
        project_summaries=summaries,
    )
