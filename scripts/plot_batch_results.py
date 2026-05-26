#!/usr/bin/env python3
"""Plot batch evaluation results.

This script reads the aggregate CSV files produced by batch evaluation and
creates simple PNG plots that can be used when comparing prototype runs.

Expected input files:

    batch_runs.csv
    batch_methods.csv

Example:

    python scripts/plot_batch_results.py \
      --batch-dir runs/batches/batch-abc

Optional explicit output directory:

    python scripts/plot_batch_results.py \
      --batch-dir runs/batches/batch-abc \
      --output-dir runs/batches/batch-abc/plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METHOD_DISPLAY_NAMES = {
    "round_based_marginal_contribution_with_local_improvement": "thesis_algorithm",
    "baseline_random_constrained": "random",
    "baseline_greedy_fit": "greedy_fit",
    "baseline_balanced_greedy": "balanced_greedy",
}

DISPLAY_METHOD_ORDER = [
    "thesis_algorithm",
    "random",
    "greedy_fit",
    "balanced_greedy",
]


def add_method_labels(methods: pd.DataFrame) -> pd.DataFrame:
    """Add shorter method labels for plotting."""

    df = methods.copy()
    df["method_label"] = df["method"].map(METHOD_DISPLAY_NAMES).fillna(df["method"])
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create plots from team-builder batch evaluation outputs."
    )
    parser.add_argument(
        "--batch-dir",
        type=Path,
        required=True,
        help="Batch directory containing batch_runs.csv and batch_methods.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where plots should be written. Defaults to <batch-dir>/plots.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Output image resolution.",
    )
    return parser.parse_args()


def read_batch_outputs(batch_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read batch output CSV files."""

    runs_path = batch_dir / "batch_runs.csv"
    methods_path = batch_dir / "batch_methods.csv"

    if not runs_path.exists():
        raise FileNotFoundError(f"Missing batch runs file: {runs_path}")

    if not methods_path.exists():
        raise FileNotFoundError(f"Missing batch methods file: {methods_path}")

    runs = pd.read_csv(runs_path)
    methods = pd.read_csv(methods_path)

    if runs.empty:
        raise ValueError(f"No rows found in {runs_path}")

    if methods.empty:
        raise ValueError(f"No rows found in {methods_path}")

    return runs, methods


def numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric series with invalid values converted to NaN."""

    return pd.to_numeric(df[column], errors="coerce")


def ordered_methods(df: pd.DataFrame) -> list[str]:
    """Return known display method labels first, followed by any additional labels."""

    method_column = "method_label" if "method_label" in df.columns else "method"
    existing = [method for method in DISPLAY_METHOD_ORDER if method in set(df[method_column])]
    additional = sorted(set(df[method_column]) - set(existing))
    return existing + additional


def save_current_plot(path: Path, dpi: int) -> None:
    """Save and close the current matplotlib figure."""

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def plot_objective_by_method(methods: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    """Plot mean objective score by method."""

    df = methods.copy()
    df["objective_score"] = numeric_column(df, "objective_score")
    grouped = (
        df.groupby("method_label", as_index=False)["objective_score"]
        .mean()
        .dropna()
    )

    order = ordered_methods(df)
    grouped["method_label"] = pd.Categorical(grouped["method_label"], categories=order, ordered=True)
    grouped = grouped.sort_values("method_label")

    plt.figure(figsize=(10, 5))
    plt.bar(grouped["method_label"].astype(str), grouped["objective_score"])
    plt.title("Mean objective score by method")
    plt.xlabel("Method")
    plt.ylabel("Mean objective score")
    plt.xticks(rotation=30, ha="right")
    save_current_plot(output_dir / "objective_by_method.png", dpi)


def plot_fairness_by_method(methods: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    """Plot mean fairness deviation by method."""

    df = methods.copy()
    df["fairness_deviation"] = numeric_column(df, "fairness_deviation")
    grouped = (
        df.groupby("method_label", as_index=False)["fairness_deviation"]
        .mean()
        .dropna()
    )

    order = ordered_methods(df)
    grouped["method_label"] = pd.Categorical(grouped["method_label"], categories=order, ordered=True)
    grouped = grouped.sort_values("method_label")

    plt.figure(figsize=(10, 5))
    plt.bar(grouped["method_label"].astype(str), grouped["fairness_deviation"])
    plt.title("Mean fairness deviation by method")
    plt.xlabel("Method")
    plt.ylabel("Mean fairness deviation")
    plt.xticks(rotation=30, ha="right")
    save_current_plot(output_dir / "fairness_by_method.png", dpi)


def plot_min_team_score_by_method(methods: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    """Plot mean minimum team score by method."""

    df = methods.copy()
    df["min_team_score"] = numeric_column(df, "min_team_score")
    grouped = (
        df.groupby("method_label", as_index=False)["min_team_score"]
        .mean()
        .dropna()
    )

    order = ordered_methods(df)
    grouped["method_label"] = pd.Categorical(grouped["method_label"], categories=order, ordered=True)
    grouped = grouped.sort_values("method_label")

    plt.figure(figsize=(10, 5))
    plt.bar(grouped["method_label"].astype(str), grouped["min_team_score"])
    plt.title("Mean minimum team score by method")
    plt.xlabel("Method")
    plt.ylabel("Mean minimum team score")
    plt.xticks(rotation=30, ha="right")
    save_current_plot(output_dir / "min_team_score_by_method.png", dpi)


def plot_runtime_by_project_set(runs: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    """Plot mean total runtime by project set."""

    if "total_runtime_seconds" not in runs.columns:
        return

    df = runs.copy()
    df["total_runtime_seconds"] = numeric_column(df, "total_runtime_seconds")
    grouped = (
        df.groupby("project_set", as_index=False)["total_runtime_seconds"]
        .mean()
        .dropna()
        .sort_values("project_set")
    )

    if grouped.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.bar(grouped["project_set"].astype(str), grouped["total_runtime_seconds"])
    plt.title("Mean runtime by project set")
    plt.xlabel("Project set")
    plt.ylabel("Mean runtime, seconds")
    save_current_plot(output_dir / "runtime_by_project_set.png", dpi)


def plot_runtime_by_participant_count(runs: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    """Plot mean total runtime by participant count."""

    if "total_runtime_seconds" not in runs.columns:
        return

    df = runs.copy()
    df["total_runtime_seconds"] = numeric_column(df, "total_runtime_seconds")
    df["participant_count"] = numeric_column(df, "participant_count")
    grouped = (
        df.groupby("participant_count", as_index=False)["total_runtime_seconds"]
        .mean()
        .dropna()
        .sort_values("participant_count")
    )

    if grouped.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(grouped["participant_count"], grouped["total_runtime_seconds"], marker="o")
    plt.title("Mean runtime by participant count")
    plt.xlabel("Participant count")
    plt.ylabel("Mean runtime, seconds")
    save_current_plot(output_dir / "runtime_by_participant_count.png", dpi)


def plot_local_improvement_gain(runs: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    """Plot mean local-improvement gain by project set."""

    if "local_improvement_gain" not in runs.columns:
        return

    df = runs.copy()
    df["local_improvement_gain"] = numeric_column(df, "local_improvement_gain")
    grouped = (
        df.groupby("project_set", as_index=False)["local_improvement_gain"]
        .mean()
        .dropna()
        .sort_values("project_set")
    )

    if grouped.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.bar(grouped["project_set"].astype(str), grouped["local_improvement_gain"])
    plt.title("Mean local-improvement gain by project set")
    plt.xlabel("Project set")
    plt.ylabel("Mean objective gain")
    save_current_plot(output_dir / "local_improvement_gain_by_project_set.png", dpi)


def plot_objective_by_weight_profile(runs: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    """Plot mean main-method objective score by weight profile."""

    if "main_objective_score" not in runs.columns:
        return

    df = runs.copy()
    df["main_objective_score"] = numeric_column(df, "main_objective_score")
    grouped = (
        df.groupby("weight_profile", as_index=False)["main_objective_score"]
        .mean()
        .dropna()
        .sort_values("weight_profile")
    )

    if grouped.empty:
        return

    plt.figure(figsize=(9, 5))
    plt.bar(grouped["weight_profile"].astype(str), grouped["main_objective_score"])
    plt.title("Mean main-method objective score by weight profile")
    plt.xlabel("Weight profile")
    plt.ylabel("Mean objective score")
    plt.xticks(rotation=30, ha="right")
    save_current_plot(output_dir / "objective_by_weight_profile.png", dpi)


def plot_method_scores_by_project_set(methods: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    """Plot mean objective score by project set and method.

    This plot is useful when the same participant set is run across all project
    sets. It uses a simple grouped bar layout without external plotting
    libraries.
    """

    df = methods.copy()
    df["objective_score"] = numeric_column(df, "objective_score")

    grouped = (
        df.groupby(["project_set", "method_label"], as_index=False)["objective_score"]
        .mean()
        .dropna()
    )

    if grouped.empty:
        return

    project_sets = sorted(grouped["project_set"].unique())
    methods_order = ordered_methods(grouped)

    x_positions = list(range(len(project_sets)))
    bar_width = 0.8 / max(1, len(methods_order))

    plt.figure(figsize=(11, 5))

    for method_index, method in enumerate(methods_order):
        values = []
        for project_set in project_sets:
            match = grouped[
                (grouped["project_set"] == project_set)
                & (grouped["method_label"] == method)
            ]
            values.append(float(match["objective_score"].iloc[0]) if not match.empty else 0.0)

        offsets = [
            x + method_index * bar_width - (bar_width * (len(methods_order) - 1) / 2)
            for x in x_positions
        ]
        plt.bar(offsets, values, width=bar_width, label=method)

    plt.title("Mean objective score by project set and method")
    plt.xlabel("Project set")
    plt.ylabel("Mean objective score")
    plt.xticks(x_positions, project_sets)
    plt.legend(fontsize="small")
    save_current_plot(output_dir / "objective_by_project_set_and_method.png", dpi)


def write_plot_index(output_dir: Path, created_files: list[Path]) -> None:
    """Write a short text index of generated plot files."""

    lines = [
        "Batch result plots",
        "=" * 40,
        "",
    ]

    for path in created_files:
        lines.append(f"- {path.name}")

    (output_dir / "plot_index.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    batch_dir = args.batch_dir
    output_dir = args.output_dir or batch_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    runs, methods = read_batch_outputs(batch_dir)
    methods = add_method_labels(methods)

    before = set(output_dir.glob("*.png"))

    plot_objective_by_method(methods, output_dir, args.dpi)
    plot_fairness_by_method(methods, output_dir, args.dpi)
    plot_min_team_score_by_method(methods, output_dir, args.dpi)
    plot_runtime_by_project_set(runs, output_dir, args.dpi)
    plot_runtime_by_participant_count(runs, output_dir, args.dpi)
    plot_local_improvement_gain(runs, output_dir, args.dpi)
    plot_objective_by_weight_profile(runs, output_dir, args.dpi)
    plot_method_scores_by_project_set(methods, output_dir, args.dpi)

    after = set(output_dir.glob("*.png"))
    created = sorted(after - before)

    if not created:
        created = sorted(after)

    write_plot_index(output_dir, created)

    print(f"Wrote plots to: {output_dir}")
    for path in created:
        print(f"  - {path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
