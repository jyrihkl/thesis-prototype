"""Lightweight data containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


ValidationStatus = Literal["pass", "warn", "fail"]


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime configuration for the prototype pipeline."""

    participants_path: Path | None = None
    projects_path: Path | None = None
    project_set: str = "a"

    enable_local_improvement: bool = True
    max_local_improvement_iterations: int = 100
    min_local_improvement_gain: float = 1e-9

    save_run: bool = True
    output_dir: Path = Path("runs")
    run_id: str | None = None


@dataclass(frozen=True)
class ValidationCheck:
    """One validation finding produced during a pipeline run."""

    name: str
    status: ValidationStatus
    message: str


@dataclass(frozen=True)
class ValidationReport:
    """Collection of validation findings."""

    checks: list[ValidationCheck] = field(default_factory=list)

    @property
    def has_failures(self) -> bool:
        """Return True if at least one validation check failed."""

        return any(check.status == "fail" for check in self.checks)

    @property
    def has_warnings(self) -> bool:
        """Return True if at least one validation check produced a warning."""

        return any(check.status == "warn" for check in self.checks)

    @property
    def passed_count(self) -> int:
        """Return the number of passing checks."""

        return sum(1 for check in self.checks if check.status == "pass")

    @property
    def warning_count(self) -> int:
        """Return the number of warning checks."""

        return sum(1 for check in self.checks if check.status == "warn")

    @property
    def failure_count(self) -> int:
        """Return the number of failed checks."""

        return sum(1 for check in self.checks if check.status == "fail")


@dataclass(frozen=True)
class ScoreWeights:
    """Weights used for candidate-to-project fit scoring."""

    required_skills: float = 0.40
    preferred_skills: float = 0.20
    role: float = 0.20
    experience: float = 0.10
    language: float = 0.05
    interests: float = 0.05

    def as_dict(self) -> dict[str, float]:
        """Return weights using component names as keys."""

        return {
            "required_skills": self.required_skills,
            "preferred_skills": self.preferred_skills,
            "role": self.role,
            "experience": self.experience,
            "language": self.language,
            "interests": self.interests,
        }


@dataclass(frozen=True)
class CandidateProjectScore:
    """Transparent fit score for one candidate-project pair."""

    candidate_id: str
    project_id: str
    total_score: float
    feasible: bool
    infeasibility_reasons: tuple[str, ...]
    components: dict[str, float | None]
    matched_required_skills: tuple[str, ...]
    missing_required_skills: tuple[str, ...]
    matched_preferred_skills: tuple[str, ...]
    role_family: str


@dataclass(frozen=True)
class ProjectScoreSummary:
    """Score distribution summary for one project."""

    project_id: str
    project_title: str
    scored_candidates: int
    feasible_candidates: int
    min_score: float | None
    mean_score: float | None
    max_score: float | None
    top_candidate_ids: tuple[str, ...]


@dataclass(frozen=True)
class ScoringReport:
    """Summary of the candidate-to-project scoring stage."""

    total_pairs: int
    feasible_pairs: int
    weights: dict[str, float]
    project_summaries: list[ProjectScoreSummary] = field(default_factory=list)


@dataclass(frozen=True)
class TeamMemberAssignment:
    """One selected candidate assignment and its transparent selection basis."""

    candidate_id: str
    project_id: str
    round_number: int
    marginal_score: float
    individual_fit_score: float
    marginal_components: dict[str, float | None]


@dataclass(frozen=True)
class ProjectTeamSummary:
    """Team-level summary after allocation."""

    project_id: str
    project_title: str
    member_ids: tuple[str, ...]
    target_team_size: int
    team_score: float
    components: dict[str, float | None]
    covered_required_skills: tuple[str, ...]
    missing_required_skills: tuple[str, ...]
    covered_preferred_skills: tuple[str, ...]
    role_families: tuple[str, ...]


@dataclass(frozen=True)
class LocalImprovementMove:
    """One accepted local improvement move."""

    iteration: int
    move_type: str
    gain: float
    affected_projects: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class LocalImprovementReport:
    """Summary of the local improvement phase."""

    enabled: bool
    initial_objective_score: float
    final_objective_score: float
    improvement_gain: float
    accepted_swaps: int
    accepted_replacements: int
    evaluated_swaps: int
    evaluated_replacements: int
    iterations: int
    stop_reason: str
    accepted_moves: tuple[LocalImprovementMove, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class AllocationReport:
    """Summary of a completed allocation attempt."""

    method: str
    feasible: bool
    assigned_count: int
    unassigned_count: int
    required_slots: int
    objective_score: float
    fairness_deviation: float
    min_team_score: float | None
    mean_team_score: float | None
    max_team_score: float | None
    project_summaries: list[ProjectTeamSummary] = field(default_factory=list)
    assignments: list[TeamMemberAssignment] = field(default_factory=list)
    unassigned_candidate_ids: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    local_improvement: LocalImprovementReport | None = None


@dataclass(frozen=True)
class BaselineMethodSummary:
    """Compact comparison row for one allocation method."""

    method: str
    feasible: bool
    objective_score: float
    mean_team_score: float | None
    min_team_score: float | None
    max_team_score: float | None
    fairness_deviation: float
    assigned_count: int
    required_slots: int


@dataclass(frozen=True)
class BaselineComparisonReport:
    """Comparative evaluation against baseline allocation methods."""

    main_method: str
    method_summaries: list[BaselineMethodSummary] = field(default_factory=list)
    baseline_reports: list[AllocationReport] = field(default_factory=list)
    best_by_objective: str | None = None
    best_by_min_team_score: str | None = None
    best_by_fairness: str | None = None


@dataclass(frozen=True)
class PipelineRunResult:
    """Summary returned after the current pipeline stage."""

    participants_path: Path
    projects_path: Path
    participant_count: int
    project_count: int
    project_titles: list[str]
    required_slots: int
    available_candidates: int
    validation_report: ValidationReport
    scoring_report: ScoringReport | None = None
    candidate_project_scores: list[CandidateProjectScore] = field(default_factory=list)
    allocation_report: AllocationReport | None = None
    baseline_comparison_report: BaselineComparisonReport | None = None
    saved_run_dir: Path | None = None
