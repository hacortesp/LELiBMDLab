#!/usr/bin/env python3
"""Generate KMeans cluster visualization figures."""

from __future__ import annotations

import argparse
import contextlib
import io
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

with contextlib.redirect_stderr(io.StringIO()):
    import pandas as pd


def save_pca_scatter_by_cluster(labels: pd.DataFrame, selection: pd.DataFrame | None, output_path: Path) -> None:
    """Save PCA scatter colored by selected KMeans cluster."""

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(labels["pca1"], labels["pca2"], c=labels["cluster_id"], cmap="tab20", s=22, alpha=0.85)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("Descriptor PCA by KMeans Cluster")
    if selection is not None and bool(selection.iloc[0].get("one_cluster_mode", False)):
        ax.text(
            0.02,
            0.98,
            "One-cluster mode: silhouette scores below threshold",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
        )
    fig.colorbar(scatter, ax=ax, label="Cluster ID")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_pca_scatter_by_anion(labels: pd.DataFrame, output_path: Path) -> None:
    """Save PCA scatter colored by anion family."""

    fig, ax = plt.subplots(figsize=(8, 6))
    for anion, group in labels.groupby("anion_name", sort=True):
        ax.scatter(group["pca1"], group["pca2"], s=22, alpha=0.85, label=anion)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("Descriptor PCA by Anion Family")
    ax.legend(title="Anion", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_silhouette_plot(scores: pd.DataFrame, selection: pd.DataFrame | None, output_path: Path) -> None:
    """Save silhouette score versus k plot."""

    fig, ax = plt.subplots(figsize=(7, 5))
    evaluated = scores.dropna(subset=["silhouette_score"])
    ax.plot(scores["k"], scores["silhouette_score"], marker="o", color="#4c78a8")
    ax.set_xlabel("KMeans k")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("KMeans Model Selection")
    ax.set_xticks(scores["k"])

    if selection is not None and not selection.empty:
        selected = selection.iloc[0]
        threshold = float(selected["silhouette_min_threshold"])
        ax.axhline(threshold, color="#d62728", linestyle="--", linewidth=1.2, label=f"threshold = {threshold:g}")
        if bool(selected["one_cluster_mode"]):
            message = "Selected one cluster because all tested scores were too low"
        else:
            message = f"selected k = {int(selected['selected_k'])}"
        ax.text(
            0.02,
            0.03,
            message,
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
        )
        ax.legend(frameon=False)
    elif evaluated.empty:
        ax.text(0.5, 0.5, "No valid k values to evaluate", transform=ax.transAxes, ha="center", va="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_cluster_size_bar(labels: pd.DataFrame, output_path: Path) -> None:
    """Save cluster-size bar chart."""

    counts = labels["cluster_id"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.index.astype(str), counts.values, color="#4c78a8")
    ax.set_xlabel("KMeans Cluster ID")
    ax.set_ylabel("Number of Electrolytes")
    ax.set_title("KMeans Cluster Sizes")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def run(labels_input: Path, scores_input: Path, selection_input: Path, figures_dir: Path) -> None:
    """Generate all KMeans cluster figures."""

    labels = pd.read_csv(labels_input)
    scores = pd.read_csv(scores_input)
    selection = pd.read_csv(selection_input) if selection_input.exists() else None

    figures_dir.mkdir(parents=True, exist_ok=True)
    save_pca_scatter_by_cluster(labels, selection, figures_dir / "pca_by_kmeans_cluster.png")
    save_pca_scatter_by_anion(labels, figures_dir / "pca_by_anion.png")
    save_silhouette_plot(scores, selection, figures_dir / "silhouette_scores.png")
    save_cluster_size_bar(labels, figures_dir / "cluster_sizes.png")
    print(f"Wrote KMeans figures to: {figures_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels_input", type=Path, help="cluster_labels.csv from cluster_electrolytes.py.")
    parser.add_argument(
        "--scores-input",
        type=Path,
        default=Path("electrolyte_outputs/silhouette_scores.csv"),
        help="Silhouette scores CSV from cluster_electrolytes.py.",
    )
    parser.add_argument(
        "--selection-input",
        type=Path,
        default=Path("electrolyte_outputs/cluster_selection.csv"),
        help="Selected-k CSV from cluster_electrolytes.py.",
    )
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"), help="Directory for output PNG files.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args.labels_input, args.scores_input, args.selection_input, args.figures_dir)


if __name__ == "__main__":
    main()
