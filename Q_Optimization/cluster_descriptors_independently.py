#!/usr/bin/env python3
"""Cluster electrolyte descriptors independently, one numeric column at a time.

Each candidate descriptor is treated as a one-dimensional KMeans input. The
script evaluates k values with silhouette scores, applies an optional
low-silhouette one-cluster fallback, and writes per-descriptor outputs plus a
global descriptor ranking table.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import re
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

with contextlib.redirect_stderr(io.StringIO()):
    import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


RANDOM_SEED = 42
DEFAULT_CANDIDATE_DESCRIPTORS = [
    "conductivity",
    "solvent_entropy",
    "weighted_permittivity",
    "weighted_viscosity_mPa_s_25C",
    "salt_conc_x_weighted_permittivity",
    "salt_conc_x_weighted_viscosity",
    "anion_PF6",
    "anion_ClO4",
    "anion_TFSI",
    "anion_FSI",
    "anion_BF4",
    "has_EC",
    "frac_EC",
    "has_PC",
    "frac_PC",
    "has_DME",
    "frac_DME",
    "has_DMC",
    "frac_DMC",
    "has_DEC",
    "frac_DEC",
    "has_EMC",
    "frac_EMC",
]
METADATA_COLUMNS = [
    "original_row_index",
    "doi",
    "anion_name",
    "composition_summary",
]


def safe_folder_name(name: str) -> str:
    """Return a filesystem-friendly folder name for a descriptor."""

    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    return safe_name or "descriptor"


def descriptor_candidates(descriptors: pd.DataFrame, requested: list[str] | None) -> list[str]:
    """Return existing descriptor columns in the requested order."""

    candidates = requested if requested else DEFAULT_CANDIDATE_DESCRIPTORS
    return [col for col in candidates if col in descriptors.columns]


def base_output_frame(descriptors: pd.DataFrame, descriptor_name: str) -> pd.DataFrame:
    """Create the per-row output frame shared by all descriptor analyses."""

    columns = [col for col in METADATA_COLUMNS if col in descriptors.columns]
    labels = descriptors[columns].copy()
    labels.insert(0, "source_row_number", np.arange(len(descriptors), dtype=int))
    labels["descriptor_name"] = descriptor_name
    labels["descriptor_value"] = pd.to_numeric(descriptors[descriptor_name], errors="coerce")
    return labels


def fit_kmeans_1d(values: np.ndarray, k: int, random_seed: int) -> np.ndarray:
    """Fit one-dimensional KMeans and return one-based labels."""

    model = KMeans(n_clusters=k, random_state=random_seed, n_init=50)
    return model.fit_predict(values.reshape(-1, 1)) + 1


def evaluate_k_values(values: np.ndarray, k_min: int, k_max: int, random_seed: int) -> pd.DataFrame:
    """Evaluate silhouette scores for feasible k values on one descriptor."""

    records: list[dict[str, object]] = []
    n_samples = len(values)
    n_unique = int(np.unique(values).size)
    for k in range(k_min, k_max + 1):
        if k < 2:
            records.append({"k": k, "silhouette_score": np.nan, "status": "skipped_k_less_than_2"})
            continue
        if k >= n_samples:
            records.append({"k": k, "silhouette_score": np.nan, "status": "skipped_too_few_samples"})
            continue
        if k > n_unique:
            records.append({"k": k, "silhouette_score": np.nan, "status": "skipped_too_few_unique_values"})
            continue

        labels = fit_kmeans_1d(values, k, random_seed)
        if np.unique(labels).size < 2:
            records.append({"k": k, "silhouette_score": np.nan, "status": "skipped_single_label_result"})
            continue
        score = silhouette_score(values.reshape(-1, 1), labels)
        records.append({"k": k, "silhouette_score": float(score), "status": "evaluated"})
    return pd.DataFrame(records)


def select_k(scores: pd.DataFrame, silhouette_min_threshold: float) -> tuple[int, float, bool]:
    """Select the highest-silhouette k or use the one-cluster fallback."""

    evaluated = scores.dropna(subset=["silhouette_score"])
    if evaluated.empty:
        return 1, float("nan"), True
    best = evaluated.sort_values(["silhouette_score", "k"], ascending=[False, True]).iloc[0]
    best_k = int(best["k"])
    best_score = float(best["silhouette_score"])
    if best_score < silhouette_min_threshold:
        return 1, best_score, True
    return best_k, best_score, False


def representative_position(values: np.ndarray) -> int:
    """Return the position nearest the mean descriptor value."""

    mean_value = float(np.mean(values))
    return int(np.argmin(np.abs(values - mean_value)))


def summarize_clusters(
    labels: pd.DataFrame,
    descriptor_name: str,
    selected_k: int,
    best_silhouette_score: float,
) -> pd.DataFrame:
    """Build a cluster-level summary table for one descriptor."""

    clustered = labels.dropna(subset=["cluster_id"]).copy()
    if clustered.empty:
        return pd.DataFrame(
            columns=[
                "descriptor_name",
                "selected_k",
                "best_silhouette_score",
                "cluster_id",
                "cluster_size",
                "representative_row_index",
                "representative_doi",
                "cluster_mean_descriptor_value",
                "cluster_std_descriptor_value",
            ]
        )

    records: list[dict[str, object]] = []
    for cluster_id, group in clustered.groupby("cluster_id", sort=True):
        values = group["descriptor_value"].to_numpy(dtype=float)
        representative = group.iloc[representative_position(values)]
        row_index = representative.get("original_row_index", representative["source_row_number"])
        records.append(
            {
                "descriptor_name": descriptor_name,
                "selected_k": int(selected_k),
                "best_silhouette_score": best_silhouette_score,
                "cluster_id": int(cluster_id),
                "cluster_size": int(len(group)),
                "representative_row_index": int(row_index),
                "representative_doi": representative.get("doi", ""),
                "cluster_mean_descriptor_value": float(np.mean(values)),
                "cluster_std_descriptor_value": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
        )
    return pd.DataFrame(records)


def save_silhouette_plot(
    scores: pd.DataFrame,
    selected_k: int,
    best_silhouette_score: float,
    fallback_used: bool,
    threshold: float,
    descriptor_name: str,
    output_path: Path,
) -> None:
    """Save silhouette score versus k for one descriptor."""

    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot(scores["k"], scores["silhouette_score"], marker="o", color="#3366aa", linewidth=1.8)
    ax.axhline(threshold, color="#cc3333", linestyle="--", linewidth=1.1, label=f"threshold = {threshold:g}")
    ax.set_xlabel("KMeans k")
    ax.set_ylabel("Silhouette score")
    ax.set_title(f"{descriptor_name}: silhouette by k")
    ax.set_xticks(scores["k"])
    ax.grid(True, axis="y", alpha=0.25)

    if np.isfinite(best_silhouette_score):
        message = "one-cluster fallback" if fallback_used else f"selected k = {selected_k}"
        ax.text(
            0.02,
            0.04,
            f"{message}\nbest score = {best_silhouette_score:.3f}",
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.92},
        )
    else:
        ax.text(0.5, 0.5, "No valid k values", transform=ax.transAxes, ha="center", va="center")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_descriptor_cluster_plot(labels: pd.DataFrame, descriptor_name: str, output_path: Path) -> None:
    """Save a one-dimensional histogram and jitter plot colored by cluster."""

    clustered = labels.dropna(subset=["cluster_id", "descriptor_value"]).copy()
    fig, (hist_ax, strip_ax) = plt.subplots(
        2,
        1,
        figsize=(8, 6.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )

    if clustered.empty:
        hist_ax.text(0.5, 0.5, "No clustered rows", transform=hist_ax.transAxes, ha="center", va="center")
    else:
        clusters = sorted(clustered["cluster_id"].astype(int).unique())
        cmap = plt.get_cmap("tab10")
        for idx, cluster_id in enumerate(clusters):
            group = clustered[clustered["cluster_id"] == cluster_id]
            color = cmap(idx % 10)
            hist_ax.hist(
                group["descriptor_value"],
                bins="auto",
                alpha=0.5,
                color=color,
                edgecolor="white",
                label=f"cluster {cluster_id}",
            )
            rng = np.random.default_rng(RANDOM_SEED + int(cluster_id))
            jitter = rng.normal(loc=0.0, scale=0.035, size=len(group))
            strip_ax.scatter(
                group["descriptor_value"],
                np.full(len(group), cluster_id, dtype=float) + jitter,
                s=20,
                alpha=0.82,
                color=color,
                linewidths=0,
            )

        hist_ax.legend(frameon=False)
        strip_ax.set_yticks(clusters)
        strip_ax.set_yticklabels([str(cluster_id) for cluster_id in clusters])

    hist_ax.set_ylabel("Count")
    hist_ax.set_title(f"{descriptor_name}: one-dimensional clusters")
    strip_ax.set_xlabel(descriptor_name)
    strip_ax.set_ylabel("Cluster")
    strip_ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_skipped_descriptor(
    labels: pd.DataFrame,
    descriptor_dir: Path,
    descriptor_name: str,
    skip_reason: str,
    k_min: int,
    k_max: int,
) -> pd.DataFrame:
    """Write placeholder outputs for a skipped descriptor."""

    descriptor_dir.mkdir(parents=True, exist_ok=True)
    labels["cluster_id"] = pd.Series([pd.NA] * len(labels), dtype="Int64")
    labels["row_status"] = np.where(labels["descriptor_value"].isna(), "missing_descriptor", "skipped_descriptor")
    labels.to_csv(descriptor_dir / "cluster_labels.csv", index=False)
    scores = pd.DataFrame(
        {
            "k": list(range(k_min, k_max + 1)),
            "silhouette_score": [np.nan] * (k_max - k_min + 1),
            "status": [skip_reason] * (k_max - k_min + 1),
        }
    )
    scores.to_csv(descriptor_dir / "silhouette_scores.csv", index=False)
    summarize_clusters(labels, descriptor_name, 1, float("nan")).to_csv(
        descriptor_dir / "cluster_summary.csv",
        index=False,
    )
    save_silhouette_plot(
        scores,
        1,
        float("nan"),
        True,
        float("nan"),
        descriptor_name,
        descriptor_dir / "silhouette_plot.png",
    )
    save_descriptor_cluster_plot(labels, descriptor_name, descriptor_dir / "descriptor_cluster_plot.png")
    return labels


def analyze_descriptor(
    descriptors: pd.DataFrame,
    descriptor_name: str,
    output_dir: Path,
    k_min: int,
    k_max: int,
    silhouette_min_threshold: float,
    min_non_missing: int,
    max_missing_fraction: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Run the complete independent clustering analysis for one descriptor."""

    descriptor_dir = output_dir / "per_descriptor" / safe_folder_name(descriptor_name)
    labels = base_output_frame(descriptors, descriptor_name)
    values_series = labels["descriptor_value"]
    valid_mask = values_series.notna()
    values = values_series[valid_mask].to_numpy(dtype=float)
    n_rows = int(len(labels))
    n_non_missing = int(valid_mask.sum())
    n_missing = n_rows - n_non_missing
    missing_fraction = n_missing / n_rows if n_rows else 1.0
    n_unique = int(np.unique(values).size) if n_non_missing else 0

    skip_reason = ""
    if not pd.api.types.is_numeric_dtype(pd.to_numeric(descriptors[descriptor_name], errors="coerce")):
        skip_reason = "non_numeric_descriptor"
    elif n_non_missing < min_non_missing:
        skip_reason = "too_many_missing_values"
    elif missing_fraction > max_missing_fraction:
        skip_reason = "too_many_missing_values"
    elif n_unique < 2:
        skip_reason = "constant_or_too_few_unique_values"

    if skip_reason:
        write_skipped_descriptor(labels, descriptor_dir, descriptor_name, skip_reason, k_min, k_max)
        summary = summarize_clusters(labels, descriptor_name, 1, float("nan"))
        ranking = {
            "descriptor_name": descriptor_name,
            "number_of_unique_values": n_unique,
            "selected_k": 1,
            "best_silhouette_score": np.nan,
            "one_cluster_fallback_used": True,
            "status": "skipped",
            "skip_reason": skip_reason,
            "n_rows": n_rows,
            "n_non_missing": n_non_missing,
            "n_missing": n_missing,
            "missing_fraction": missing_fraction,
        }
        return labels, summary, ranking

    descriptor_dir.mkdir(parents=True, exist_ok=True)
    scores = evaluate_k_values(values, k_min, k_max, random_seed)
    selected_k, best_score, fallback_used = select_k(scores, silhouette_min_threshold)

    if selected_k == 1:
        clustered_labels = np.ones(n_non_missing, dtype=int)
    else:
        clustered_labels = fit_kmeans_1d(values, selected_k, random_seed)

    labels["cluster_id"] = pd.Series([pd.NA] * len(labels), dtype="Int64")
    labels.loc[valid_mask, "cluster_id"] = clustered_labels
    labels["cluster_id"] = labels["cluster_id"].astype("Int64")
    labels["row_status"] = np.where(valid_mask, "clustered", "missing_descriptor")
    labels["selected_k"] = int(selected_k)
    labels["best_silhouette_score"] = best_score
    labels["one_cluster_fallback_used"] = bool(fallback_used)
    labels.to_csv(descriptor_dir / "cluster_labels.csv", index=False)

    scores.to_csv(descriptor_dir / "silhouette_scores.csv", index=False)
    summary = summarize_clusters(labels, descriptor_name, selected_k, best_score)
    summary.to_csv(descriptor_dir / "cluster_summary.csv", index=False)
    save_silhouette_plot(
        scores,
        selected_k,
        best_score,
        fallback_used,
        silhouette_min_threshold,
        descriptor_name,
        descriptor_dir / "silhouette_plot.png",
    )
    save_descriptor_cluster_plot(labels, descriptor_name, descriptor_dir / "descriptor_cluster_plot.png")

    ranking = {
        "descriptor_name": descriptor_name,
        "number_of_unique_values": n_unique,
        "selected_k": int(selected_k),
        "best_silhouette_score": best_score,
        "one_cluster_fallback_used": bool(fallback_used),
        "status": "tested",
        "skip_reason": "",
        "n_rows": n_rows,
        "n_non_missing": n_non_missing,
        "n_missing": n_missing,
        "missing_fraction": missing_fraction,
    }
    return labels, summary, ranking


def run(
    descriptor_input: Path,
    output_dir: Path,
    candidate_columns: list[str] | None,
    k_min: int,
    k_max: int,
    silhouette_min_threshold: float,
    min_non_missing: int,
    max_missing_fraction: float,
    random_seed: int,
) -> None:
    """Run independent clustering for all candidate descriptors and write outputs."""

    descriptors = pd.read_csv(descriptor_input)
    candidates = descriptor_candidates(descriptors, candidate_columns)
    missing_candidates = [
        col for col in (candidate_columns if candidate_columns else DEFAULT_CANDIDATE_DESCRIPTORS) if col not in descriptors.columns
    ]
    if not candidates:
        raise ValueError("No requested descriptor columns were found in the input CSV.")

    all_labels: list[pd.DataFrame] = []
    all_summaries: list[pd.DataFrame] = []
    rankings: list[dict[str, object]] = []

    for descriptor_name in candidates:
        labels, summary, ranking = analyze_descriptor(
            descriptors,
            descriptor_name,
            output_dir,
            k_min,
            k_max,
            silhouette_min_threshold,
            min_non_missing,
            max_missing_fraction,
            random_seed,
        )
        all_labels.append(labels)
        all_summaries.append(summary)
        rankings.append(ranking)

    for descriptor_name in missing_candidates:
        rankings.append(
            {
                "descriptor_name": descriptor_name,
                "number_of_unique_values": 0,
                "selected_k": 1,
                "best_silhouette_score": np.nan,
                "one_cluster_fallback_used": True,
                "status": "skipped",
                "skip_reason": "descriptor_column_not_found",
                "n_rows": len(descriptors),
                "n_non_missing": 0,
                "n_missing": len(descriptors),
                "missing_fraction": 1.0,
            }
        )

    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    ranking_df = pd.DataFrame(rankings).sort_values(
        ["best_silhouette_score", "descriptor_name"],
        ascending=[False, True],
        na_position="last",
    )
    ranking_df.to_csv(summary_dir / "descriptor_ranking.csv", index=False)
    pd.concat(all_summaries, ignore_index=True).to_csv(
        summary_dir / "all_descriptor_cluster_results.csv",
        index=False,
    )
    pd.concat(all_labels, ignore_index=True).to_csv(summary_dir / "all_descriptor_cluster_labels.csv", index=False)

    tested_count = int((ranking_df["status"] == "tested").sum())
    skipped_count = int((ranking_df["status"] == "skipped").sum())
    print(f"Wrote independent descriptor clustering results to: {output_dir}")
    print(f"Tested descriptors: {tested_count}; skipped descriptors: {skipped_count}")
    print(f"Wrote ranking table: {summary_dir / 'descriptor_ranking.csv'}")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("descriptor_input", type=Path, help="Input descriptors.csv file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Root output directory for per-descriptor and summary outputs.",
    )
    parser.add_argument("--k-min", type=int, default=2, help="Minimum KMeans k to evaluate.")
    parser.add_argument("--k-max", type=int, default=6, help="Maximum KMeans k to evaluate.")
    parser.add_argument(
        "--silhouette-min-threshold",
        type=float,
        default=0.15,
        help="Use one cluster when the best silhouette score is below this threshold.",
    )
    parser.add_argument(
        "--min-non-missing",
        type=int,
        default=10,
        help="Skip a descriptor if fewer non-missing rows are available.",
    )
    parser.add_argument(
        "--max-missing-fraction",
        type=float,
        default=0.50,
        help="Skip a descriptor when this fraction of rows is missing.",
    )
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED, help="Fixed seed for reproducible KMeans.")
    parser.add_argument(
        "--candidate-columns",
        nargs="+",
        default=None,
        help="Optional explicit descriptor columns to test. Defaults to the electrolyte descriptor list.",
    )
    return parser


def main() -> None:
    """Command-line entry point."""

    args = build_parser().parse_args()
    if args.k_min > args.k_max:
        raise ValueError("--k-min must be less than or equal to --k-max.")
    if not 0 <= args.max_missing_fraction <= 1:
        raise ValueError("--max-missing-fraction must be between 0 and 1.")
    run(
        args.descriptor_input,
        args.output_dir,
        args.candidate_columns,
        args.k_min,
        args.k_max,
        args.silhouette_min_threshold,
        args.min_non_missing,
        args.max_missing_fraction,
        args.random_seed,
    )


if __name__ == "__main__":
    main()
