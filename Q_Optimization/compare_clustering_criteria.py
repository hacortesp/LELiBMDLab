#!/usr/bin/env python3
"""Compare KMeans clustering criteria for electrolyte descriptors.

The script evaluates a small set of descriptor criteria independently. Each
criterion uses only its specified feature column or two-column feature space,
standardizes those features, selects KMeans k by silhouette score, applies a
low-silhouette one-cluster fallback, and writes per-criterion outputs plus a
global comparison table.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
from dataclasses import dataclass
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
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


RANDOM_SEED = 42
SUMMARY_VALUE_COLUMNS = [
    "conductivity",
    "weighted_permittivity",
    "weighted_viscosity_mPa_s_25C",
    "salt_conc_x_weighted_permittivity",
    "salt_conc_x_weighted_viscosity",
]
METADATA_COLUMNS = ["original_row_index", "doi", "anion_name", "composition_summary"]


@dataclass(frozen=True)
class CriterionSet:
    """A named descriptor feature set to evaluate independently."""

    name: str
    folder: str
    features: tuple[str, ...]
    display_name: str


DEFAULT_CRITERIA = [
    CriterionSet(
        name="weighted_permittivity",
        folder="criterion_weighted_permittivity",
        features=("weighted_permittivity",),
        display_name="Weighted permittivity",
    ),
    CriterionSet(
        name="weighted_viscosity",
        folder="criterion_weighted_viscosity",
        features=("weighted_viscosity_mPa_s_25C",),
        display_name="Weighted viscosity",
    ),
    CriterionSet(
        name="saltconc_weightedperm_weightedvisc",
        folder="criterion_saltconc_weightedperm_weightedvisc",
        features=("salt_conc_x_weighted_permittivity", "salt_conc_x_weighted_viscosity"),
        display_name="Salt concentration weighted permittivity + viscosity",
    ),
    CriterionSet(
        name="weightedperm_weightedvisc",
        folder="criterion_weightedperm_weightedvisc",
        features=("weighted_permittivity", "weighted_viscosity_mPa_s_25C"),
        display_name="Weighted permittivity + viscosity",
    ),
]


def load_data(descriptor_input: Path) -> pd.DataFrame:
    """Load descriptor data from CSV."""

    return pd.read_csv(descriptor_input)


def prepare_feature_matrix(
    descriptors: pd.DataFrame,
    criterion: CriterionSet,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """Select, clean, and standardize the features for one criterion set."""

    missing_columns = [col for col in criterion.features if col not in descriptors.columns]
    if missing_columns:
        raise ValueError(f"{criterion.name} is missing required columns: {', '.join(missing_columns)}")

    raw_features = descriptors.loc[:, criterion.features].apply(pd.to_numeric, errors="coerce")
    valid_mask = raw_features.notna().all(axis=1).to_numpy()
    feature_frame = raw_features.loc[valid_mask].copy()
    if feature_frame.empty:
        return feature_frame, np.empty((0, len(criterion.features))), valid_mask, list(criterion.features)
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(feature_frame.to_numpy(dtype=float))
    return feature_frame, scaled_matrix, valid_mask, list(criterion.features)


def fit_kmeans(matrix: np.ndarray, k: int, random_seed: int) -> np.ndarray:
    """Fit KMeans and return one-based labels."""

    model = KMeans(n_clusters=k, random_state=random_seed, n_init=50)
    return model.fit_predict(matrix) + 1


def score_labels(matrix: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    """Compute clustering quality metrics for a multi-label solution."""

    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan"), float("nan")
    return (
        float(silhouette_score(matrix, labels)),
        float(calinski_harabasz_score(matrix, labels)),
        float(davies_bouldin_score(matrix, labels)),
    )


def evaluate_kmeans_range(matrix: np.ndarray, k_min: int, k_max: int, random_seed: int) -> pd.DataFrame:
    """Evaluate KMeans quality metrics for a range of k values."""

    n_samples = matrix.shape[0]
    n_unique = np.unique(matrix, axis=0).shape[0]
    records: list[dict[str, object]] = []
    for k in range(k_min, k_max + 1):
        if k < 2:
            records.append(
                {
                    "k": k,
                    "silhouette_score": np.nan,
                    "calinski_harabasz_score": np.nan,
                    "davies_bouldin_score": np.nan,
                    "status": "skipped_k_less_than_2",
                }
            )
            continue
        if k >= n_samples:
            records.append(
                {
                    "k": k,
                    "silhouette_score": np.nan,
                    "calinski_harabasz_score": np.nan,
                    "davies_bouldin_score": np.nan,
                    "status": "skipped_too_few_samples",
                }
            )
            continue
        if k > n_unique:
            records.append(
                {
                    "k": k,
                    "silhouette_score": np.nan,
                    "calinski_harabasz_score": np.nan,
                    "davies_bouldin_score": np.nan,
                    "status": "skipped_too_few_unique_feature_rows",
                }
            )
            continue

        labels = fit_kmeans(matrix, k, random_seed)
        silhouette, calinski, davies = score_labels(matrix, labels)
        records.append(
            {
                "k": k,
                "silhouette_score": silhouette,
                "calinski_harabasz_score": calinski,
                "davies_bouldin_score": davies,
                "status": "evaluated",
            }
        )
    return pd.DataFrame(records)


def select_best_k(scores: pd.DataFrame, silhouette_min_threshold: float) -> tuple[int, int, float, bool]:
    """Choose the best candidate k and final k after the silhouette fallback."""

    evaluated = scores.dropna(subset=["silhouette_score"])
    if evaluated.empty:
        return 1, 1, float("nan"), True

    best = evaluated.sort_values(["silhouette_score", "k"], ascending=[False, True]).iloc[0]
    best_candidate_k = int(best["k"])
    best_score = float(best["silhouette_score"])
    if best_score < silhouette_min_threshold:
        return 1, best_candidate_k, best_score, True
    return best_candidate_k, best_candidate_k, best_score, False


def base_label_frame(descriptors: pd.DataFrame, criterion: CriterionSet) -> pd.DataFrame:
    """Create the per-row label output frame for one criterion."""

    columns = [col for col in METADATA_COLUMNS if col in descriptors.columns]
    labels = descriptors.loc[:, columns].copy()
    labels.insert(0, "source_row_number", np.arange(len(descriptors), dtype=int))
    labels["criterion_name"] = criterion.name
    labels["features_used"] = ", ".join(criterion.features)
    for feature in criterion.features:
        labels[feature] = pd.to_numeric(descriptors[feature], errors="coerce")
    return labels


def representative_position(member_matrix: np.ndarray) -> int:
    """Return the member position nearest the cluster centroid."""

    centroid = member_matrix.mean(axis=0)
    distances = np.linalg.norm(member_matrix - centroid, axis=1)
    return int(np.argmin(distances))


def summarize_clusters(
    descriptors: pd.DataFrame,
    labels: pd.DataFrame,
    matrix: np.ndarray,
) -> pd.DataFrame:
    """Build representative electrolyte summaries for each selected cluster."""

    clustered = labels.dropna(subset=["cluster_id"]).copy()
    records: list[dict[str, object]] = []
    for cluster_id, group in clustered.groupby("cluster_id", sort=True):
        member_positions = group["valid_feature_position"].to_numpy(dtype=int)
        local_matrix = matrix[member_positions]
        representative = group.iloc[representative_position(local_matrix)]
        source_rows = group["source_row_number"].to_numpy(dtype=int)
        source_slice = descriptors.iloc[source_rows]
        row_index = representative.get("original_row_index", representative["source_row_number"])

        record = {
            "criterion_name": representative["criterion_name"],
            "features_used": representative["features_used"],
            "cluster_id": int(cluster_id),
            "cluster_size": int(len(group)),
            "representative_row_index": int(row_index),
            "representative_doi": representative.get("doi", ""),
        }
        for column in SUMMARY_VALUE_COLUMNS:
            if column in source_slice.columns:
                record[f"mean_{column}"] = pd.to_numeric(source_slice[column], errors="coerce").mean()
            else:
                record[f"mean_{column}"] = np.nan
        records.append(record)
    return pd.DataFrame(records)


def metric_row(
    criterion: CriterionSet,
    matrix: np.ndarray,
    labels: np.ndarray,
    number_of_samples: int,
    best_k: int,
    best_candidate_k: int,
    best_silhouette_score: float,
    one_cluster_fallback_used: bool,
    skip_reason: str = "",
) -> dict[str, object]:
    """Create one final comparison-table row for a criterion set."""

    if best_k > 1 and len(np.unique(labels)) > 1:
        _, calinski, davies = score_labels(matrix, labels)
    else:
        calinski = float("nan")
        davies = float("nan")

    return {
        "criterion_name": criterion.name,
        "features_used": ", ".join(criterion.features),
        "number_of_samples": int(number_of_samples),
        "number_of_features": int(len(criterion.features)),
        "best_k": int(best_k),
        "best_candidate_k": int(best_candidate_k),
        "best_silhouette_score": best_silhouette_score,
        "calinski_harabasz_score": calinski,
        "davies_bouldin_score": davies,
        "one_cluster_fallback_used": bool(one_cluster_fallback_used),
        "skip_reason": skip_reason,
    }


def save_silhouette_plot(scores: pd.DataFrame, criterion: CriterionSet, threshold: float, output_path: Path) -> None:
    """Save a silhouette-versus-k line plot."""

    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(scores["k"], scores["silhouette_score"], marker="o", color="#3366aa", linewidth=1.8)
    ax.axhline(threshold, color="#cc3333", linestyle="--", linewidth=1.1, label=f"threshold = {threshold:g}")
    ax.set_title(f"{criterion.display_name}: silhouette by k")
    ax.set_xlabel("KMeans k")
    ax.set_ylabel("Silhouette score")
    ax.set_xticks(scores["k"])
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def title_with_metrics(criterion: CriterionSet, best_k: int, best_score: float) -> str:
    """Return a compact plot title with selected k and silhouette score."""

    if np.isfinite(best_score):
        return f"{criterion.display_name} | selected k = {best_k}, silhouette = {best_score:.3f}"
    return f"{criterion.display_name} | selected k = {best_k}, silhouette = N/A"


def make_plots(
    labels: pd.DataFrame,
    criterion: CriterionSet,
    best_k: int,
    best_silhouette_score: float,
    scores: pd.DataFrame,
    threshold: float,
    output_dir: Path,
    random_seed: int,
) -> None:
    """Create criterion-specific cluster and silhouette plots."""

    output_dir.mkdir(parents=True, exist_ok=True)
    save_silhouette_plot(scores, criterion, threshold, output_dir / "silhouette_plot.png")

    clustered = labels.dropna(subset=["cluster_id"]).copy()
    fig, ax = plt.subplots(figsize=(8, 5.8))
    if clustered.empty:
        ax.text(0.5, 0.5, "No clustered rows", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
    elif len(criterion.features) == 1:
        feature = criterion.features[0]
        rng = np.random.default_rng(random_seed)
        jitter = rng.normal(loc=0.0, scale=0.04, size=len(clustered))
        scatter = ax.scatter(
            clustered[feature],
            clustered["cluster_id"].astype(float) + jitter,
            c=clustered["cluster_id"].astype(int),
            cmap="tab10",
            s=24,
            alpha=0.85,
            linewidths=0,
        )
        ax.set_xlabel(feature)
        ax.set_ylabel("Cluster ID")
        cluster_ids = sorted(clustered["cluster_id"].astype(int).unique())
        ax.set_yticks(cluster_ids)
        ax.set_yticklabels([str(cluster_id) for cluster_id in cluster_ids])
        fig.colorbar(scatter, ax=ax, label="Cluster ID")
        ax.grid(True, axis="x", alpha=0.25)
    else:
        x_feature, y_feature = criterion.features
        scatter = ax.scatter(
            clustered[x_feature],
            clustered[y_feature],
            c=clustered["cluster_id"].astype(int),
            cmap="tab10",
            s=28,
            alpha=0.85,
            linewidths=0,
        )
        ax.set_xlabel(x_feature)
        ax.set_ylabel(y_feature)
        fig.colorbar(scatter, ax=ax, label="Cluster ID")
        ax.grid(True, alpha=0.22)

    ax.set_title(title_with_metrics(criterion, best_k, best_silhouette_score))
    fig.tight_layout()
    fig.savefig(output_dir / "cluster_plot.png", dpi=220)
    plt.close(fig)


def write_skipped_outputs(
    descriptors: pd.DataFrame,
    criterion: CriterionSet,
    criterion_dir: Path,
    k_min: int,
    k_max: int,
    silhouette_min_threshold: float,
    skip_reason: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Write graceful placeholder outputs when a criterion cannot be clustered."""

    labels = base_label_frame(descriptors, criterion)
    labels["cluster_id"] = pd.Series([pd.NA] * len(labels), dtype="Int64")
    labels["row_status"] = "skipped"
    labels["valid_feature_position"] = pd.Series([pd.NA] * len(labels), dtype="Int64")
    labels.to_csv(criterion_dir / "cluster_labels.csv", index=False)

    scores = pd.DataFrame(
        {
            "k": list(range(k_min, k_max + 1)),
            "silhouette_score": [np.nan] * (k_max - k_min + 1),
            "calinski_harabasz_score": [np.nan] * (k_max - k_min + 1),
            "davies_bouldin_score": [np.nan] * (k_max - k_min + 1),
            "status": [skip_reason] * (k_max - k_min + 1),
        }
    )
    scores.to_csv(criterion_dir / "silhouette_scores.csv", index=False)
    summary = summarize_clusters(descriptors, labels, np.empty((0, len(criterion.features))))
    summary.to_csv(criterion_dir / "cluster_summary.csv", index=False)
    make_plots(labels, criterion, 1, float("nan"), scores, silhouette_min_threshold, criterion_dir, RANDOM_SEED)
    metrics = metric_row(criterion, np.empty((0, len(criterion.features))), np.array([], dtype=int), 0, 1, 1, np.nan, True, skip_reason)
    pd.DataFrame([metrics]).to_csv(criterion_dir / "metric_summary.csv", index=False)
    return labels, summary, metrics


def cluster_and_summarize(
    descriptors: pd.DataFrame,
    criterion: CriterionSet,
    criterion_dir: Path,
    k_min: int,
    k_max: int,
    silhouette_min_threshold: float,
    min_non_missing: int,
    max_missing_fraction: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Cluster one criterion set and save all per-criterion outputs."""

    criterion_dir.mkdir(parents=True, exist_ok=True)
    try:
        raw_features, matrix, valid_mask, _ = prepare_feature_matrix(descriptors, criterion)
    except ValueError as exc:
        return write_skipped_outputs(
            descriptors,
            criterion,
            criterion_dir,
            k_min,
            k_max,
            silhouette_min_threshold,
            str(exc),
        )

    n_samples = matrix.shape[0]
    missing_fraction = 1.0 - (n_samples / len(descriptors)) if len(descriptors) else 1.0
    if n_samples < min_non_missing or missing_fraction > max_missing_fraction:
        return write_skipped_outputs(
            descriptors,
            criterion,
            criterion_dir,
            k_min,
            k_max,
            silhouette_min_threshold,
            "too_many_missing_values",
        )

    scores = evaluate_kmeans_range(matrix, k_min, k_max, random_seed)
    best_k, best_candidate_k, best_score, fallback_used = select_best_k(scores, silhouette_min_threshold)
    final_labels = np.ones(n_samples, dtype=int) if best_k == 1 else fit_kmeans(matrix, best_k, random_seed)

    labels = base_label_frame(descriptors, criterion)
    labels["cluster_id"] = pd.Series([pd.NA] * len(labels), dtype="Int64")
    labels.loc[valid_mask, "cluster_id"] = final_labels
    labels["cluster_id"] = labels["cluster_id"].astype("Int64")
    labels["row_status"] = np.where(valid_mask, "clustered", "missing_criterion_features")
    labels["valid_feature_position"] = pd.Series([pd.NA] * len(labels), dtype="Int64")
    labels.loc[valid_mask, "valid_feature_position"] = np.arange(n_samples, dtype=int)
    labels["valid_feature_position"] = labels["valid_feature_position"].astype("Int64")
    labels["best_k"] = int(best_k)
    labels["best_candidate_k"] = int(best_candidate_k)
    labels["best_silhouette_score"] = best_score
    labels["one_cluster_fallback_used"] = bool(fallback_used)
    labels.to_csv(criterion_dir / "cluster_labels.csv", index=False)
    scores.to_csv(criterion_dir / "silhouette_scores.csv", index=False)

    summary = summarize_clusters(descriptors, labels, matrix)
    summary.to_csv(criterion_dir / "cluster_summary.csv", index=False)
    make_plots(labels, criterion, best_k, best_score, scores, silhouette_min_threshold, criterion_dir, random_seed)

    metrics = metric_row(
        criterion,
        matrix,
        final_labels,
        n_samples,
        best_k,
        best_candidate_k,
        best_score,
        fallback_used,
    )
    pd.DataFrame([metrics]).to_csv(criterion_dir / "metric_summary.csv", index=False)
    return labels, summary, metrics


def compare_criteria_sets(
    descriptors: pd.DataFrame,
    criteria: list[CriterionSet],
    output_dir: Path,
    k_min: int,
    k_max: int,
    silhouette_min_threshold: float,
    min_non_missing: int,
    max_missing_fraction: float,
    random_seed: int,
) -> pd.DataFrame:
    """Evaluate all criteria and write the global comparison outputs."""

    all_labels: list[pd.DataFrame] = []
    all_summaries: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []

    for criterion in criteria:
        labels, summary, metrics = cluster_and_summarize(
            descriptors,
            criterion,
            output_dir / criterion.folder,
            k_min,
            k_max,
            silhouette_min_threshold,
            min_non_missing,
            max_missing_fraction,
            random_seed,
        )
        all_labels.append(labels)
        all_summaries.append(summary)
        metric_rows.append(metrics)

    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame(metric_rows).sort_values(
        ["best_silhouette_score", "criterion_name"],
        ascending=[False, True],
        na_position="last",
    )
    comparison.to_csv(summary_dir / "criteria_comparison.csv", index=False)
    pd.concat(all_summaries, ignore_index=True).to_csv(summary_dir / "all_criteria_cluster_results.csv", index=False)
    pd.concat(all_labels, ignore_index=True).to_csv(summary_dir / "all_criteria_cluster_labels.csv", index=False)
    return comparison


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("descriptor_input", type=Path, help="Input descriptors.csv file.")
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="Root output directory.")
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
        help="Skip a criterion set if fewer complete rows are available.",
    )
    parser.add_argument(
        "--max-missing-fraction",
        type=float,
        default=0.50,
        help="Skip a criterion set when more than this fraction of rows is missing.",
    )
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED, help="Fixed seed for reproducible KMeans.")
    return parser


def main() -> None:
    """Command-line entry point."""

    args = build_parser().parse_args()
    if args.k_min > args.k_max:
        raise ValueError("--k-min must be less than or equal to --k-max.")
    if not 0 <= args.max_missing_fraction <= 1:
        raise ValueError("--max-missing-fraction must be between 0 and 1.")

    descriptors = load_data(args.descriptor_input)
    comparison = compare_criteria_sets(
        descriptors,
        DEFAULT_CRITERIA,
        args.output_dir,
        args.k_min,
        args.k_max,
        args.silhouette_min_threshold,
        args.min_non_missing,
        args.max_missing_fraction,
        args.random_seed,
    )
    print(f"Wrote criterion comparison results to: {args.output_dir}")
    print(f"Wrote comparison table: {args.output_dir / 'summary' / 'criteria_comparison.csv'}")
    print(comparison.loc[:, ["criterion_name", "best_k", "best_silhouette_score", "one_cluster_fallback_used"]].to_string(index=False))


if __name__ == "__main__":
    main()
