"""Typed internal schemas.

The loader reads raw CSV/JSON dictionaries. These dataclasses provide predictable inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Candidate:
    """Normalized participant representation."""

    id: str
    position: str = ""
    primary_keyword: str = ""
    role_family: str = "unknown"
    experience_years: float | None = None
    experience_bucket: str = "unknown"
    english_level: str | None = None
    english_score: int | None = None
    skills: frozenset[str] = field(default_factory=frozenset)
    interest_tags: frozenset[str] = field(default_factory=frozenset)
    cv_excerpt: str = ""


@dataclass(frozen=True)
class Project:
    """Normalized project-brief representation."""

    id: str
    title: str
    description: str = ""
    required_skills: frozenset[str] = field(default_factory=frozenset)
    preferred_skills: frozenset[str] = field(default_factory=frozenset)
    desired_roles: frozenset[str] = field(default_factory=frozenset)
    target_team_size: int = 0
    min_english_level: str | None = None
    min_english_score: int | None = None
    balancing_constraints: dict[str, Any] = field(default_factory=dict)