#!/usr/bin/env python3
"""Cluster electrolyte descriptors with silhouette-selected KMeans."""

from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path

import numpy as np

with contextlib.redirect_stderr(io.StringIO()):
    import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


RANDOM_SEED = 42
NON_FEATURE_COLUMNS = {
    "original_row_index",
    "doi",
    "conductivity",
    "anion_name",
    "composition_summary",
}


def feature_columns(descriptors: pd.DataFrame) -> list[str]:
    """Return numeric descriptor columns used as KMeans features."""

    return [
        col
        for col in descriptors.columns
        if col not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(descriptors[col])
    ]


def standardize_features(descriptors: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Median-impute and standardize numeric descriptor features."""

    cols = feature_columns(descriptors)
    if not cols:
        raise ValueError("No numeric descriptor features available for clustering.")
    matrix = descriptors[cols].to_numpy(dtype=float)
    imputed = SimpleImputer(strategy="median").fit_transform(matrix)
    scaled = StandardScaler().fit_transform(imputed)
    return pd.DataFrame(scaled, columns=cols, index=descriptors.index), cols


def fit_kmeans(scaled: pd.DataFrame, k: int) -> np.ndarray:
    """Fit KMeans and return one-based labels."""

    model = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=50)
    return model.fit_predict(scaled.to_numpy(dtype=float)) + 1


def evaluate_kmeans_range(scaled: pd.DataFrame, k_min: int, k_max: int) -> pd.DataFrame:
    """Evaluate KMeans silhouette scores for a range of k values."""

    n_samples = len(scaled)
    records: list[dict[str, object]] = []
    for k in range(k_min, k_max + 1):
        if k < 2 or k >= n_samples:
            records.append({"k": k, "silhouette_score": np.nan, "status": "skipped"})
            continue
        labels = fit_kmeans(scaled, k)
        score = silhouette_score(scaled.to_numpy(dtype=float), labels)
        records.append({"k": k, "silhouette_score": float(score), "status": "evaluated"})
    return pd.DataFrame(records)


def select_k(scores: pd.DataFrame, silhouette_min_threshold: float) -> tuple[int, float, bool]:
    """Select the best k or fall back to one-cluster mode."""

    evaluated = scores.dropna(subset=["silhouette_score"])
    if evaluated.empty:
        return 1, float("nan"), True
    best = evaluated.sort_values(["silhouette_score", "k"], ascending=[False, True]).iloc[0]
    best_k = int(best["k"])
    best_score = float(best["silhouette_score"])
    if best_score < silhouette_min_threshold:
        return 1, best_score, True
    return best_k, best_score, False


def pca_coordinates(scaled: pd.DataFrame) -> pd.DataFrame:
    """Compute a two-dimensional PCA projection for plotting."""

    if len(scaled) < 2:
        return pd.DataFrame({"pca1": [0.0], "pca2": [0.0]}, index=scaled.index)
    coords = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(scaled.to_numpy(dtype=float))
    return pd.DataFrame(coords, columns=["pca1", "pca2"], index=scaled.index)


def representative_indices(labels: np.ndarray, scaled: pd.DataFrame) -> dict[int, int]:
    """Choose the member nearest each cluster centroid in scaled feature space."""

    representatives: dict[int, int] = {}
    for cluster_id in sorted(set(labels)):
        member_positions = np.flatnonzero(labels == cluster_id)
        vectors = scaled.iloc[member_positions].to_numpy(dtype=float)
        centroid = vectors.mean(axis=0)
        distances = np.linalg.norm(vectors - centroid, axis=1)
        representatives[int(cluster_id)] = int(member_positions[int(np.argmin(distances))])
    return representatives


def metadata_for_outputs(descriptors: pd.DataFrame, cleaned: pd.DataFrame | None) -> pd.DataFrame:
    """Return metadata used in labels and summaries."""

    columns = ["original_row_index", "doi", "conductivity", "anion_name", "composition_summary"]
    metadata = descriptors[columns].copy()
    if cleaned is not None:
        cleaned_metadata = cleaned[
            ["original_row_index", "salt_conc", "temperature", "replicate_count"]
        ].copy()
        metadata = metadata.merge(cleaned_metadata, on="original_row_index", how="left")
    return metadata


def summarize_clusters(metadata: pd.DataFrame, labels: np.ndarray, scaled: pd.DataFrame) -> pd.DataFrame:
    """Create one summary row per selected KMeans cluster."""

    representatives = representative_indices(labels, scaled)
    working = metadata.copy()
    working["cluster_id"] = labels
    summary_records: list[dict[str, object]] = []

    for cluster_id, group in working.groupby("cluster_id", sort=True):
        representative = metadata.iloc[representatives[int(cluster_id)]]
        summary_records.append(
            {
                "cluster_id": int(cluster_id),
                "representative_doi": representative.get("doi"),
                "representative_row_index": int(representative["original_row_index"]),
                "representative_composition": representative.get("composition_summary"),
                "representative_anion": representative.get("anion_name"),
                "mean_conductivity": group["conductivity"].mean(),
                "mean_concentration": group["salt_conc"].mean() if "salt_conc" in group else np.nan,
                "mean_temperature": group["temperature"].mean() if "temperature" in group else np.nan,
                "n_members": int(len(group)),
                "total_replicates": int(group["replicate_count"].sum()) if "replicate_count" in group else int(len(group)),
            }
        )

    return pd.DataFrame(summary_records)


def cluster_electrolytes(
    descriptors: pd.DataFrame,
    cleaned: pd.DataFrame | None,
    k_min: int,
    k_max: int,
    silhouette_min_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run silhouette-selected KMeans and return all output tables."""

    scaled, _ = standardize_features(descriptors)
    scores = evaluate_kmeans_range(scaled, k_min, k_max)
    selected_k, final_score, one_cluster_mode = select_k(scores, silhouette_min_threshold)
    if selected_k == 1:
        labels = np.ones(len(descriptors), dtype=int)
    else:
        labels = fit_kmeans(scaled, selected_k)

    metadata = metadata_for_outputs(descriptors, cleaned)
    labels_df = metadata.copy()
    labels_df["cluster_id"] = labels
    labels_df = pd.concat([labels_df, pca_coordinates(scaled)], axis=1)
    summary = summarize_clusters(metadata, labels, scaled)
    selection = pd.DataFrame(
        [
            {
                "selected_k": selected_k,
                "final_silhouette_score": final_score,
                "silhouette_min_threshold": silhouette_min_threshold,
                "one_cluster_mode": bool(one_cluster_mode),
                "k_min": k_min,
                "k_max": k_max,
            }
        ]
    )
    return labels_df, summary, scores, selection


def run(
    descriptor_input: Path,
    cleaned_input: Path | None,
    labels_output: Path,
    summary_output: Path,
    scores_output: Path,
    selection_output: Path,
    k_min: int,
    k_max: int,
    silhouette_min_threshold: float,
) -> None:
    """Run KMeans model selection and save labels, summaries, and scores."""

    descriptors = pd.read_csv(descriptor_input)
    cleaned = pd.read_csv(cleaned_input) if cleaned_input is not None and cleaned_input.exists() else None
    labels, summary, scores, selection = cluster_electrolytes(
        descriptors,
        cleaned,
        k_min,
        k_max,
        silhouette_min_threshold,
    )
    for path in (labels_output, summary_output, scores_output, selection_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(labels_output, index=False)
    summary.to_csv(summary_output, index=False)
    scores.to_csv(scores_output, index=False)
    selection.to_csv(selection_output, index=False)
    print(f"Wrote cluster labels: {labels_output} ({len(labels)} rows)")
    print(f"Wrote cluster summary: {summary_output} ({len(summary)} clusters)")
    print(f"Wrote silhouette scores: {scores_output}")
    print(f"Wrote cluster selection: {selection_output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("descriptor_input", type=Path, help="descriptors.csv from build_descriptors.py.")
    parser.add_argument(
        "--cleaned-input",
        type=Path,
        default=Path("electrolyte_outputs/cleaned_electrolytes.csv"),
        help="Cleaned electrolyte CSV used for reporting concentration and temperature.",
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
        "--labels-output",
        type=Path,
        default=Path("electrolyte_outputs/cluster_labels.csv"),
        help="Path for cluster labels and PCA coordinates.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("electrolyte_outputs/cluster_summary.csv"),
        help="Path for cluster summary table.",
    )
    parser.add_argument(
        "--scores-output",
        type=Path,
        default=Path("electrolyte_outputs/silhouette_scores.csv"),
        help="Path for tested k values and silhouette scores.",
    )
    parser.add_argument(
        "--selection-output",
        type=Path,
        default=Path("electrolyte_outputs/cluster_selection.csv"),
        help="Path for selected k and final silhouette score.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(
        args.descriptor_input,
        args.cleaned_input,
        args.labels_output,
        args.summary_output,
        args.scores_output,
        args.selection_output,
        args.k_min,
        args.k_max,
        args.silhouette_min_threshold,
    )


if __name__ == "__main__":
    main()
