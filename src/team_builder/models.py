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
