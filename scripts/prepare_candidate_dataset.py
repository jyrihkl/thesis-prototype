#!/usr/bin/env python3
"""
Prepare a filtered candidate-profile dataset for the team-building prototype.

This script is intentionally separate from the main prototype pipeline. It is
meant to be run once to create a stable, reproducible participant pool from the
Hugging Face Djinni candidate-profile dataset, or from an already-downloaded
CSV/JSONL file with the same fields.

Example:
    python scripts/prepare_candidate_dataset.py \
        --sample-size 80 \
        --output data/processed/candidates_filtered.csv \
        --jsonl-output data/processed/candidates_filtered.jsonl \
        --report-output data/processed/candidates_filter_report.json

Dependencies:
    pip install pandas datasets
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_DATASET_NAME = "lang-uk/recruitment-dataset-candidate-profiles-english"
DEFAULT_SPLIT = "train"

RAW_COLUMNS = [
    "id",
    "Position",
    "Primary Keyword",
    "Experience Years",
    "English Level",
    "CV_lang",
    "CV",
    "Moreinfo",
    "Looking For",
    "Highlights",
]

# Broad but still IT/product-oriented. This keeps the prototype close to the
# dataset's recruitment-domain source while excluding profiles that are less
# useful for software/project-team examples.
DEFAULT_ALLOWED_ROLE_FAMILIES = {
    "backend",
    "frontend",
    "fullstack",
    "mobile",
    "data_analytics",
    "machine_learning",
    "devops_cloud",
    "qa_testing",
    "design_ux",
    "product_project",
}

ENGLISH_LEVELS = {
    "none": 0,
    "no english": 0,
    "no_english": 0,
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

# Canonical skill -> regex aliases. This is deliberately transparent and
# editable. Avoid inferring sensitive or personality-like attributes here.
SKILL_PATTERNS: dict[str, list[str]] = {
    "python": [r"\bpython\b"],
    "java": [r"\bjava\b(?!script)"],
    "javascript": [r"\bjavascript\b", r"\bjs\b"],
    "typescript": [r"\btypescript\b", r"\bts\b"],
    "react": [r"\breact\b", r"\breact\.js\b", r"\breactjs\b"],
    "vue": [r"\bvue\b", r"\bvue\.js\b", r"\bvuejs\b"],
    "angular": [r"\bangular\b"],
    "node.js": [r"\bnode\.js\b", r"\bnodejs\b", r"\bnode\b"],
    "django": [r"\bdjango\b"],
    "flask": [r"\bflask\b"],
    "fastapi": [r"\bfastapi\b"],
    "c#": [r"\bc#\b", r"\bc sharp\b"],
    ".net": [r"\.net\b", r"\bdotnet\b", r"\basp\.net\b"],
    "c++": [r"\bc\+\+\b", r"\bcpp\b"],
    "c": [r"\bc language\b", r"\bansi c\b"],
    "go": [r"\bgolang\b", r"\bgo language\b"],
    "rust": [r"\brust\b"],
    "ruby": [r"\bruby\b", r"\bruby on rails\b", r"\brails\b"],
    "php": [r"\bphp\b"],
    "laravel": [r"\blaravel\b"],
    "spring": [r"\bspring\b", r"\bspring boot\b"],
    "kotlin": [r"\bkotlin\b"],
    "swift": [r"\bswift\b"],
    "android": [r"\bandroid\b"],
    "ios": [r"\bios\b", r"\biOS\b"],
    "flutter": [r"\bflutter\b"],
    "react native": [r"\breact native\b"],
    "sql": [r"\bsql\b"],
    "postgresql": [r"\bpostgresql\b", r"\bpostgres\b"],
    "mysql": [r"\bmysql\b"],
    "mongodb": [r"\bmongodb\b", r"\bmongo\b"],
    "redis": [r"\bredis\b"],
    "elasticsearch": [r"\belasticsearch\b", r"\belastic search\b"],
    "aws": [r"\baws\b", r"\bamazon web services\b"],
    "azure": [r"\bazure\b"],
    "gcp": [r"\bgcp\b", r"\bgoogle cloud\b"],
    "docker": [r"\bdocker\b"],
    "kubernetes": [r"\bkubernetes\b", r"\bk8s\b"],
    "terraform": [r"\bterraform\b"],
    "linux": [r"\blinux\b"],
    "git": [r"\bgit\b", r"\bgithub\b", r"\bgitlab\b"],
    "ci/cd": [r"\bci/cd\b", r"\bcontinuous integration\b", r"\bcontinuous delivery\b"],
    "jenkins": [r"\bjenkins\b"],
    "github actions": [r"\bgithub actions\b"],
    "machine learning": [r"\bmachine learning\b", r"\bml\b"],
    "data analysis": [r"\bdata analysis\b", r"\banalytics\b"],
    "pandas": [r"\bpandas\b"],
    "numpy": [r"\bnumpy\b"],
    "pytorch": [r"\bpytorch\b", r"\btorch\b"],
    "tensorflow": [r"\btensorflow\b"],
    "scikit-learn": [r"\bscikit-learn\b", r"\bsklearn\b"],
    "nlp": [r"\bnlp\b", r"\bnatural language processing\b"],
    "computer vision": [r"\bcomputer vision\b", r"\bcv\b"],
    "tableau": [r"\btableau\b"],
    "power bi": [r"\bpower bi\b", r"\bpowerbi\b"],
    "figma": [r"\bfigma\b"],
    "ui/ux": [r"\bui/ux\b", r"\bux/ui\b", r"\bux design\b", r"\bui design\b", r"\buser experience\b"],
    "product management": [r"\bproduct management\b", r"\bproduct manager\b", r"\bproduct owner\b"],
    "project management": [r"\bproject management\b", r"\bproject manager\b"],
    "scrum": [r"\bscrum\b", r"\bscrum master\b"],
    "agile": [r"\bagile\b"],
    "qa": [r"\bqa\b", r"\bquality assurance\b"],
    "selenium": [r"\bselenium\b"],
    "cypress": [r"\bcypress\b"],
    "playwright": [r"\bplaywright\b"],
    "api": [r"\bapi\b"],
    "rest": [r"\brest\b", r"\brestful\b"],
    "graphql": [r"\bgraphql\b"],
    "microservices": [r"\bmicroservices\b", r"\bmicroservice\b"],
    "blockchain": [r"\bblockchain\b", r"\bsolidity\b", r"\bweb3\b"],
}

ROLE_PATTERNS: list[tuple[str, list[str]]] = [
    ("fullstack", [r"full[ -]?stack"]),
    ("frontend", [r"front[ -]?end", r"react", r"vue", r"angular", r"web developer"]),
    ("backend", [r"back[ -]?end", r"java developer", r"python developer", r"node", r"php", r"\.net", r"c#"]),
    ("mobile", [r"mobile", r"android", r"ios", r"flutter", r"react native"]),
    ("machine_learning", [r"machine learning", r"\bml\b", r"\bai\b", r"data scientist", r"nlp", r"computer vision"]),
    ("data_analytics", [r"data analyst", r"business analyst", r"bi analyst", r"analytics", r"sql", r"tableau", r"power bi"]),
    ("devops_cloud", [r"devops", r"sre", r"cloud", r"aws", r"azure", r"kubernetes", r"docker"]),
    ("qa_testing", [r"\bqa\b", r"quality assurance", r"tester", r"test automation", r"manual testing"]),
    ("design_ux", [r"ui/ux", r"ux", r"ui designer", r"product designer", r"graphic designer", r"figma"]),
    ("product_project", [r"product manager", r"product owner", r"project manager", r"scrum master", r"delivery manager"]),
]


@dataclass
class FilterReport:
    dataset_name: str
    split: str
    random_seed: int
    counts: dict[str, int]
    role_family_counts: dict[str, int]
    english_level_counts: dict[str, int]
    skill_count_summary: dict[str, float]
    output_columns: list[str]
    filters: dict[str, object]


def clean_text(value: object) -> str:
    """Return a normalized string suitable for matching and output."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).replace("\u0000", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if text.lower() in {"nan", "none", "null", "n/a"}:
        return ""
    return text


def normalize_english_level(value: object) -> tuple[str, int | None]:
    raw = clean_text(value).lower().replace("_", "-")
    raw = re.sub(r"\s+", " ", raw)
    if not raw:
        return "unknown", None
    if raw in ENGLISH_LEVELS:
        return raw.replace(" ", "-"), ENGLISH_LEVELS[raw]
    # Be permissive if the dataset uses labels such as "upper intermediate".
    for key, score in ENGLISH_LEVELS.items():
        if key in raw:
            return key.replace(" ", "-"), score
    return raw, None


def normalize_experience(value: object) -> float | None:
    if value is None:
        return None
    try:
        if isinstance(value, str):
            value = value.replace(",", ".").strip()
        exp = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(exp):
        return None
    return max(exp, 0.0)


def experience_bucket(years: float | None) -> str:
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


def contains_any(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def infer_role_family(position: str, primary_keyword: str, text: str) -> str:
    role_text = f"{position} {primary_keyword} {text[:600]}".lower()
    for family, patterns in ROLE_PATTERNS:
        if contains_any(role_text, patterns):
            return family
    return "other"


def extract_skills(text: str) -> list[str]:
    text = f" {text} "
    found = []
    for canonical, patterns in SKILL_PATTERNS.items():
        if contains_any(text, patterns):
            found.append(canonical)
    return sorted(set(found))


def extract_interest_tags(looking_for: str, highlights: str) -> list[str]:
    text = f"{looking_for} {highlights}".lower()
    tags = []
    domain_patterns = {
        "remote": [r"\bremote\b"],
        "part_time": [r"part[ -]?time"],
        "startup": [r"\bstartup\b"],
        "leadership": [r"\blead\b", r"\bleader\b", r"\bmanager\b", r"\bcto\b"],
        "research": [r"\bresearch\b", r"\bresearcher\b"],
        "blockchain": [r"\bblockchain\b", r"\bweb3\b", r"\bdefi\b", r"\bnft\b"],
        "gaming": [r"\bgaming\b", r"\bgamedev\b", r"\bgame development\b"],
    }
    for tag, patterns in domain_patterns.items():
        if contains_any(text, patterns):
            tags.append(tag)
    return sorted(set(tags))


def load_source_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    if args.input_csv:
        return pd.read_csv(args.input_csv)
    if args.input_jsonl:
        return pd.read_json(args.input_jsonl, lines=True)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install it with `pip install datasets`, "
            "or pass --input-csv/--input-jsonl to use a local file."
        ) from exc

    dataset = load_dataset(args.dataset_name, split=args.split)
    return dataset.to_pandas()


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Keep known columns when present, but do not fail if optional text fields are missing.
    for col in RAW_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[RAW_COLUMNS].copy()


def build_processed_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_columns(df)

    for col in ["id", "Position", "Primary Keyword", "English Level", "CV_lang", "CV", "Moreinfo", "Looking For", "Highlights"]:
        df[col] = df[col].map(clean_text)

    df["experience_years"] = df["Experience Years"].map(normalize_experience)
    english = df["English Level"].map(normalize_english_level)
    df["english_level"] = [item[0] for item in english]
    df["english_score"] = [item[1] for item in english]

    combined_text = (
        df["Position"].fillna("")
        + " "
        + df["Primary Keyword"].fillna("")
        + " "
        + df["CV"].fillna("")
        + " "
        + df["Moreinfo"].fillna("")
        + " "
        + df["Highlights"].fillna("")
    )
    df["combined_text"] = combined_text.map(clean_text)
    df["role_family"] = [
        infer_role_family(pos, keyword, text)
        for pos, keyword, text in zip(df["Position"], df["Primary Keyword"], df["combined_text"])
    ]
    df["skills"] = df["combined_text"].map(extract_skills)
    df["skill_count"] = df["skills"].map(len)
    df["interest_tags"] = [
        extract_interest_tags(looking_for, highlights)
        for looking_for, highlights in zip(df["Looking For"], df["Highlights"])
    ]
    df["experience_bucket"] = df["experience_years"].map(experience_bucket)
    df["cv_char_count"] = df["CV"].map(len)

    return df


def apply_filters(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, int]]:
    counts: dict[str, int] = {"raw_rows": len(df)}

    # Stable IDs are needed by the allocation engine.
    df = df[df["id"].astype(str).str.len() > 0].copy()
    df = df.drop_duplicates(subset=["id"]).copy()
    counts["after_valid_unique_id"] = len(df)

    if args.require_cv_lang.lower() != "any":
        df = df[df["CV_lang"].str.lower() == args.require_cv_lang.lower()].copy()
    counts["after_language_filter"] = len(df)

    df = df[df["cv_char_count"] >= args.min_cv_chars].copy()
    counts["after_min_cv_chars"] = len(df)

    df = df[df["skill_count"] >= args.min_skill_count].copy()
    counts["after_min_skill_count"] = len(df)

    df = df[df["experience_years"].notna()].copy()
    df = df[df["experience_years"] >= args.min_experience].copy()
    if args.max_experience is not None:
        df = df[df["experience_years"] <= args.max_experience].copy()
    counts["after_experience_filter"] = len(df)

    min_english = args.min_english_level.lower().replace("_", "-")
    if min_english != "any":
        threshold = ENGLISH_LEVELS.get(min_english.replace("-", " "), ENGLISH_LEVELS.get(min_english))
        if threshold is None:
            valid = ", ".join(sorted(set(ENGLISH_LEVELS) | {"any"}))
            raise SystemExit(f"Unknown --min-english-level '{args.min_english_level}'. Valid examples: {valid}")
        df = df[df["english_score"].notna() & (df["english_score"] >= threshold)].copy()
    counts["after_english_filter"] = len(df)

    allowed = set(args.allowed_role_families or DEFAULT_ALLOWED_ROLE_FAMILIES)
    if "any" not in allowed:
        df = df[df["role_family"].isin(allowed)].copy()
    counts["after_role_family_filter"] = len(df)

    return df, counts


def stable_balanced_sample(df: pd.DataFrame, sample_size: int | None, seed: int) -> pd.DataFrame:
    if sample_size is None or sample_size <= 0 or len(df) <= sample_size:
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    rng = random.Random(seed)
    groups = {family: group.copy() for family, group in df.groupby("role_family")}
    families = sorted(groups)
    selected_indices: list[int] = []

    # First pass: round-robin by role family to avoid one dominant role family.
    while len(selected_indices) < sample_size and families:
        progressed = False
        rng.shuffle(families)
        for family in list(families):
            group = groups[family]
            remaining = group[~group.index.isin(selected_indices)]
            if remaining.empty:
                families.remove(family)
                continue
            selected_indices.append(rng.choice(list(remaining.index)))
            progressed = True
            if len(selected_indices) >= sample_size:
                break
        if not progressed:
            break

    sampled = df.loc[selected_indices].copy()
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def serialize_list(values: list[str]) -> str:
    return ";".join(values)


def write_outputs(df: pd.DataFrame, args: argparse.Namespace) -> None:
    # output_columns = [
    #     "id",
    #     "position",
    #     "primary_keyword",
    #     "role_family",
    #     "experience_years",
    #     "experience_bucket",
    #     "english_level",
    #     "english_score",
    #     "skills",
    #     "skill_count",
    #     "interest_tags",
    #     "cv_excerpt",
    # ]

    export = pd.DataFrame(
        {
            "id": df["id"],
            "position": df["Position"],
            "primary_keyword": df["Primary Keyword"],
            "role_family": df["role_family"],
            "experience_years": df["experience_years"],
            "experience_bucket": df["experience_bucket"],
            "english_level": df["english_level"],
            "english_score": df["english_score"],
            "skills": df["skills"],
            "skill_count": df["skill_count"],
            "interest_tags": df["interest_tags"],
            "cv_excerpt": df["CV"].map(lambda x: clean_text(x)[:600]),
        }
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    csv_export = export.copy()
    csv_export["skills"] = csv_export["skills"].map(serialize_list)
    csv_export["interest_tags"] = csv_export["interest_tags"].map(serialize_list)
    csv_export.to_csv(output_path, index=False)

    if args.jsonl_output:
        jsonl_path = Path(args.jsonl_output)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        export.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)

    return None


def summarize_skill_counts(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {"min": 0, "mean": 0.0, "median": 0.0, "max": 0}
    counts = df["skill_count"]
    return {
        "min": int(counts.min()),
        "mean": round(float(counts.mean()), 2),
        "median": round(float(counts.median()), 2),
        "max": int(counts.max()),
    }


def write_report(df: pd.DataFrame, counts: dict[str, int], args: argparse.Namespace) -> None:
    report = FilterReport(
        dataset_name=args.dataset_name if not (args.input_csv or args.input_jsonl) else "local_file",
        split=args.split,
        random_seed=args.seed,
        counts={**counts, "final_rows": len(df)},
        role_family_counts=dict(Counter(df["role_family"])),
        english_level_counts=dict(Counter(df["english_level"])),
        skill_count_summary=summarize_skill_counts(df),
        output_columns=[
            "id",
            "position",
            "primary_keyword",
            "role_family",
            "experience_years",
            "experience_bucket",
            "english_level",
            "english_score",
            "skills",
            "skill_count",
            "interest_tags",
            "cv_excerpt",
        ],
        filters={
            "require_cv_lang": args.require_cv_lang,
            "min_cv_chars": args.min_cv_chars,
            "min_skill_count": args.min_skill_count,
            "min_experience": args.min_experience,
            "max_experience": args.max_experience,
            "min_english_level": args.min_english_level,
            "allowed_role_families": sorted(args.allowed_role_families or DEFAULT_ALLOWED_ROLE_FAMILIES),
            "sample_size": args.sample_size,
        },
    )

    report_path = Path(args.report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report.__dict__, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter and normalize candidate profiles for the team-building prototype."
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--input-csv", default=None, help="Use a local CSV instead of downloading from Hugging Face.")
    parser.add_argument("--input-jsonl", default=None, help="Use a local JSONL instead of downloading from Hugging Face.")
    parser.add_argument("--output", default="data/processed/candidates_filtered.csv")
    parser.add_argument("--jsonl-output", default="data/processed/candidates_filtered.jsonl")
    parser.add_argument("--report-output", default="data/processed/candidates_filter_report.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=80, help="Final sample size. Use 0 to keep all filtered rows.")
    parser.add_argument("--require-cv-lang", default="en", help="Required CV_lang value, or 'any'.")
    parser.add_argument("--min-cv-chars", type=int, default=180)
    parser.add_argument("--min-skill-count", type=int, default=2)
    parser.add_argument("--min-experience", type=float, default=0.0)
    parser.add_argument("--max-experience", type=float, default=None)
    parser.add_argument("--min-english-level", default="intermediate", help="Use 'any' to disable.")
    parser.add_argument(
        "--allowed-role-families",
        nargs="*",
        default=sorted(DEFAULT_ALLOWED_ROLE_FAMILIES),
        help="Allowed canonical role families. Use 'any' to disable role filtering.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    if args.input_csv and args.input_jsonl:
        raise SystemExit("Use only one of --input-csv or --input-jsonl.")

    raw = load_source_dataframe(args)
    processed = build_processed_dataframe(raw)
    filtered, counts = apply_filters(processed, args)
    sampled = stable_balanced_sample(filtered, args.sample_size, args.seed)

    if sampled.empty:
        raise SystemExit(
            "No candidates remained after filtering. Try lowering --min-skill-count, "
            "--min-cv-chars, --min-english-level, or broadening --allowed-role-families."
        )

    write_outputs(sampled, args)
    write_report(sampled, counts, args)

    print("Candidate dataset prepared successfully.")
    print(f"Rows after filtering: {len(filtered)}")
    print(f"Rows written: {len(sampled)}")
    print(f"CSV: {args.output}")
    if args.jsonl_output:
        print(f"JSONL: {args.jsonl_output}")
    print(f"Report: {args.report_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
