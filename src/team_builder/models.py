"""Lightweight data containers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime configuration for the prototype pipeline."""

    participants_path: Path | None = None
    projects_path: Path | None = None
    project_set: str = "a"


@dataclass(frozen=True)
class PipelineLoadResult:
    """Summary returned after the input-loading stage."""

    participants_path: Path
    projects_path: Path
    participant_count: int
    project_count: int
    project_titles: list[str]
