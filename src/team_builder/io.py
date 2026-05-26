"""Input loading utilities."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_PARTICIPANT_SET_PATHS: dict[str, tuple[Path, ...]] = {
    "080": (
        Path("data/processed/participants/candidates_080.csv"),
        Path("data/processed/participants/candidates_080.jsonl"),
        Path("data/processed/candidates_filtered.csv"),
        Path("data/processed/candidates_filtered.jsonl"),
        Path("data/candidates_filtered.csv"),
        Path("data/candidates_filtered.jsonl"),
    ),
    "120": (
        Path("data/processed/participants/candidates_120.csv"),
        Path("data/processed/participants/candidates_120.jsonl"),
        Path("data/candidates_120.csv"),
        Path("data/candidates_120.jsonl"),
    ),
    "240": (
        Path("data/processed/participants/candidates_240.csv"),
        Path("data/processed/participants/candidates_240.jsonl"),
        Path("data/candidates_240.csv"),
        Path("data/candidates_240.jsonl"),
    ),
    "480": (
        Path("data/processed/participants/candidates_480.csv"),
        Path("data/processed/participants/candidates_480.jsonl"),
        Path("data/candidates_480.csv"),
        Path("data/candidates_480.jsonl"),
    ),
    "1200": (
        Path("data/processed/participants/candidates_1200.csv"),
        Path("data/processed/participants/candidates_1200.jsonl"),
        Path("data/candidates_1200.csv"),
        Path("data/candidates_1200.jsonl"),
    ),
    "2400": (
        Path("data/processed/participants/candidates_2400.csv"),
        Path("data/processed/participants/candidates_2400.jsonl"),
        Path("data/candidates_2400.csv"),
        Path("data/candidates_2400.jsonl"),
    ),
}

# Keeping this for compatibility.
DEFAULT_PARTICIPANT_PATHS = DEFAULT_PARTICIPANT_SET_PATHS["080"]

DEFAULT_PROJECT_PATHS = {
    "a": (
        Path("data/projects_set_a.json"),
        Path("data/processed/projects_set_a.json"),
    ),
    "b": (
        Path("data/projects_set_b.json"),
        Path("data/processed/projects_set_b.json"),
    ),
}


def normalize_participant_set_name(value: str) -> str:
    """Normalize participant set names accepted by the CLI."""

    normalized = value.strip().lower()
    aliases = {
        "80": "080",
        "080": "080",
        "default": "080",
        "legacy": "080",
        "120": "120",
        "240": "240",
        "480": "480",
        "1200": "1200",
        "2400": "2400",
    }
    if normalized not in aliases:
        valid = ", ".join(sorted(aliases))
        raise ValueError(f"Unknown participant set '{value}'. Valid values: {valid}")
    return aliases[normalized]


def resolve_existing_path(
    explicit_path: Path | None,
    default_paths: tuple[Path, ...],
    label: str,
) -> Path:
    """Resolve an explicit path or the first existing default path."""

    if explicit_path is not None:
        if explicit_path.exists():
            return explicit_path
        raise FileNotFoundError(f"{label} file not found: {explicit_path}")

    for path in default_paths:
        if path.exists():
            return path

    searched = ", ".join(str(path) for path in default_paths)
    raise FileNotFoundError(
        f"No {label} file found. Provide a path explicitly or create one of: {searched}"
    )


def resolve_project_path(explicit_path: Path | None, project_set: str) -> Path:
    """Resolve the project brief path for the requested sample set."""

    try:
        defaults = DEFAULT_PROJECT_PATHS[project_set]
    except KeyError as exc:
        raise ValueError(f"Unknown project set: {project_set}") from exc

    return resolve_existing_path(
        explicit_path=explicit_path,
        default_paths=defaults,
        label="project",
    )


def resolve_participant_path(
    explicit_path: Path | None,
    participant_set: str = "080",
) -> Path:
    """Resolve the participant file path for a named candidate set."""

    if explicit_path is not None:
        return resolve_existing_path(
            explicit_path=explicit_path,
            default_paths=(),
            label="participant",
        )

    normalized_set = normalize_participant_set_name(participant_set)
    return resolve_existing_path(
        explicit_path=None,
        default_paths=DEFAULT_PARTICIPANT_SET_PATHS[normalized_set],
        label=f"participant set {normalized_set}",
    )


def load_participants(path: Path) -> list[dict[str, Any]]:
    """Load participants from CSV, JSONL, or JSON."""

    suffix = path.suffix.lower()

    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as file:
            return list(csv.DictReader(file))

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as file:
            return [json.loads(line) for line in file if line.strip()]

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            for key in ("participants", "candidates", "data"):
                value = data.get(key)
                if isinstance(value, list):
                    return value

        raise ValueError(
            f"Could not find a participant list in JSON file: {path}"
        )

    raise ValueError(
        f"Unsupported participant file format: {path.suffix}. "
        "Use .csv, .jsonl, or .json."
    )


def load_projects(path: Path) -> list[dict[str, Any]]:
    """Load project briefs from JSON.

    The function accepts either a top-level list of projects or a dictionary
    with a `projects` key.
    """

    if path.suffix.lower() != ".json":
        raise ValueError(
            f"Unsupported project file format: {path.suffix}. Use .json."
        )

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        projects = data
    elif isinstance(data, dict) and isinstance(data.get("projects"), list):
        projects = data["projects"]
    else:
        raise ValueError(
            f"Could not find a project list in JSON file: {path}"
        )

    return projects


def project_title(project: dict[str, Any]) -> str:
    """Return a human-readable title for a project brief."""

    for key in ("title", "name", "project_title", "id"):
        value = project.get(key)
        if value:
            return str(value)
    return "Untitled project"
