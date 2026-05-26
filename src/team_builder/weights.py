"""Score-weight profiles for single runs and batch evaluation."""

from __future__ import annotations

from team_builder.models import ScoreWeights


WEIGHT_PROFILES: dict[str, dict[str, float]] = {
    "default": {
        "required_skills": 0.40,
        "preferred_skills": 0.20,
        "role": 0.20,
        "experience": 0.10,
        "language": 0.05,
        "interests": 0.05,
    },
    "skill_heavy": {
        "required_skills": 0.55,
        "preferred_skills": 0.20,
        "role": 0.10,
        "experience": 0.05,
        "language": 0.05,
        "interests": 0.05,
    },
    "role_heavy": {
        "required_skills": 0.30,
        "preferred_skills": 0.15,
        "role": 0.35,
        "experience": 0.10,
        "language": 0.05,
        "interests": 0.05,
    },
    "balance_heavy": {
        "required_skills": 0.30,
        "preferred_skills": 0.15,
        "role": 0.20,
        "experience": 0.25,
        "language": 0.05,
        "interests": 0.05,
    },
}


def score_weights_from_mapping(values: dict[str, float] | None) -> ScoreWeights:
    """Create ScoreWeights from a partial or complete mapping."""

    if values is None:
        return ScoreWeights()

    defaults = ScoreWeights().as_dict()
    merged = {**defaults, **values}

    return ScoreWeights(
        required_skills=merged["required_skills"],
        preferred_skills=merged["preferred_skills"],
        role=merged["role"],
        experience=merged["experience"],
        language=merged["language"],
        interests=merged["interests"],
    )


def get_weight_profile(name: str) -> dict[str, float]:
    """Return a named weight profile."""

    try:
        return WEIGHT_PROFILES[name]
    except KeyError as exc:
        available = ", ".join(sorted(WEIGHT_PROFILES))
        raise ValueError(
            f"Unknown weight profile {name!r}. Available profiles: {available}"
        ) from exc
