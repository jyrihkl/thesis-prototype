"""Normalization helpers for participant and project inputs."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping
from typing import Any

from team_builder.schemas import Candidate, Project


ENGLISH_LEVELS: dict[str, int] = {
    "none": 0,
    "no english": 0,
    "no-english": 0,
    "beginner": 1,
    "elementary": 1,
    "pre-intermediate": 2,
    "pre intermediate": 2,
    "intermediate": 3,
    "upper-intermediate": 4,
    "upper intermediate": 4,
    "advanced": 5,
    "fluent": 5,
    "native": 5,
}


def clean_text(value: Any) -> str:
    """Return a compact string for user-facing fields and simple matching."""

    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).replace("\u0000", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if text.lower() in {"nan", "none", "null", "n/a"}:
        return ""
    return text


def canonical_token(value: Any) -> str:
    """Normalize a skill, role, or tag token."""

    token = clean_text(value).lower()
    token = token.replace("_", " ")
    token = re.sub(r"\s+", " ", token).strip()
    return token


def parse_float(value: Any) -> float | None:
    """Parse a numeric field while tolerating empty strings."""

    text = clean_text(value)
    if not text:
        return None
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return None


def parse_int(value: Any) -> int | None:
    """Parse an integer field while tolerating empty strings."""

    number = parse_float(value)
    if number is None:
        return None
    return int(number)


def normalize_english_level(value: Any) -> tuple[str | None, int | None]:
    """Normalize English level text and return both label and ordinal score."""

    raw = clean_text(value).lower().replace("_", "-")
    raw = re.sub(r"\s+", " ", raw)
    if not raw:
        return None, None

    if raw in ENGLISH_LEVELS:
        return raw.replace(" ", "-"), ENGLISH_LEVELS[raw]

    for label, score in ENGLISH_LEVELS.items():
        if label in raw:
            return label.replace(" ", "-"), score

    return raw.replace(" ", "-"), None


def experience_bucket(years: float | None) -> str:
    """Group experience years into broad buckets used by later scoring."""

    if years is None:
        return "unknown"
    if years < 1:
        return "junior_0_1"
    if years < 3:
        return "junior_1_3"
    if years < 6:
        return "mid_3_6"
    if years < 10:
        return "senior_6_10"
    return "senior_10_plus"


def parse_token_set(value: Any) -> frozenset[str]:
    """Parse a list-like field into a normalized frozenset of strings.

    The preparation script writes CSV list fields as semicolon-separated strings
    and JSONL fields as actual arrays. This function supports both formats.
    """

    if value is None:
        return frozenset()

    if isinstance(value, str):
        text = clean_text(value)
        if not text:
            return frozenset()

        if text.startswith("[") and text.endswith("]"):
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, list):
                return parse_token_set(decoded)

        separator = ";" if ";" in text else ","
        return frozenset(
            token
            for token in (canonical_token(part) for part in text.split(separator))
            if token
        )

    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, Mapping)):
        return frozenset(token for token in (canonical_token(item) for item in value) if token)

    token = canonical_token(value)
    return frozenset({token}) if token else frozenset()


def get_first(row: Mapping[str, Any], *keys: str, default: Any = "") -> Any:
    """Return the first present and non-empty value from a raw row."""

    for key in keys:
        value = row.get(key)
        if clean_text(value):
            return value
    return default


def normalize_candidate(row: Mapping[str, Any], index: int) -> Candidate:
    """Convert one raw participant row into a Candidate."""

    candidate_id = clean_text(get_first(row, "id", "candidate_id", default=f"candidate_{index}"))
    position = clean_text(get_first(row, "position", "Position"))
    primary_keyword = clean_text(get_first(row, "primary_keyword", "Primary Keyword"))
    role_family = canonical_token(get_first(row, "role_family", "role", default="unknown")) or "unknown"

    years = parse_float(get_first(row, "experience_years", "Experience Years", default=""))
    bucket = clean_text(get_first(row, "experience_bucket", default="")) or experience_bucket(years)

    english_level, derived_english_score = normalize_english_level(
        get_first(row, "english_level", "English Level", default="")
    )
    # If English score is explicitly provided, use it. Otherwise, derive it from the English level if possible.
    english_score = parse_int(get_first(row, "english_score", default=""))
    if english_score is None:
        english_score = derived_english_score

    return Candidate(
        id=candidate_id,
        position=position,
        primary_keyword=primary_keyword,
        role_family=role_family,
        experience_years=years,
        experience_bucket=bucket,
        english_level=english_level,
        english_score=english_score,
        skills=parse_token_set(get_first(row, "skills", "Skills", default="")),
        interest_tags=parse_token_set(get_first(row, "interest_tags", "interests", "Interest Tags", default="")),
        cv_excerpt=clean_text(get_first(row, "cv_excerpt", "CV", default="")),
    )


def normalize_project(row: Mapping[str, Any], index: int) -> Project:
    """Convert one raw project brief into a Project."""

    project_id = clean_text(get_first(row, "id", "project_id", default=f"project_{index}"))
    title = clean_text(get_first(row, "title", "name", "project_title", default=project_id))
    target_team_size = parse_int(get_first(row, "target_team_size", "team_size", default=""))
    if target_team_size is None or target_team_size <= 0:
        raise ValueError(f"Project {project_id!r} must define a positive target_team_size")

    min_english_level, min_english_score = normalize_english_level(
        get_first(row, "min_english_level", "minimum_english_level", default="")
    )

    constraints = row.get("balancing_constraints") or row.get("constraints") or {}
    if not isinstance(constraints, dict):
        constraints = {"notes": clean_text(constraints)} if clean_text(constraints) else {}

    return Project(
        id=project_id,
        title=title,
        description=clean_text(get_first(row, "description", default="")),
        required_skills=parse_token_set(get_first(row, "required_skills", default="")),
        preferred_skills=parse_token_set(get_first(row, "preferred_skills", default="")),
        desired_roles=parse_token_set(get_first(row, "desired_roles", "role_requirements", default="")),
        target_team_size=target_team_size,
        min_english_level=min_english_level,
        min_english_score=min_english_score,
        balancing_constraints=constraints,
    )


def normalize_candidates(rows: Iterable[Mapping[str, Any]]) -> list[Candidate]:
    """Normalize all participant rows."""

    candidates = [normalize_candidate(row, index=i + 1) for i, row in enumerate(rows)]
    seen: set[str] = set()
    duplicates: set[str] = set()
    for candidate in candidates:
        if candidate.id in seen:
            duplicates.add(candidate.id)
        seen.add(candidate.id)
    if duplicates:
        duplicate_preview = ", ".join(sorted(duplicates)[:5])
        raise ValueError(f"Duplicate candidate IDs after normalization: {duplicate_preview}")
    return candidates


def normalize_projects(rows: Iterable[Mapping[str, Any]]) -> list[Project]:
    """Normalize all project rows."""

    projects = [normalize_project(row, index=i + 1) for i, row in enumerate(rows)]
    seen: set[str] = set()
    duplicates: set[str] = set()
    for project in projects:
        if project.id in seen:
            duplicates.add(project.id)
        seen.add(project.id)
    if duplicates:
        duplicate_preview = ", ".join(sorted(duplicates)[:5])
        raise ValueError(f"Duplicate project IDs after normalization: {duplicate_preview}")
    return projects