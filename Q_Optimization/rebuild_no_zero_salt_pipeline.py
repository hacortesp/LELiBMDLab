#!/usr/bin/env python3
"""Rebuild electrolyte outputs after removing salt-free systems."""

from __future__ import annotations

import argparse
import contextlib
import io
import shutil
from pathlib import Path

with contextlib.redirect_stderr(io.StringIO()):
    import pandas as pd

from build_descriptors import run as run_descriptor_stage
from build_sparse_campaign import (
    DEFAULT_Q_SCALE_GRID,
    infer_cluster_allocation,
    run as run_campaign_stage,
)
from compare_clustering_criteria import compare_criteria_sets, load_data
from compare_clustering_criteria import DEFAULT_CRITERIA, RANDOM_SEED
from filter_electrolytes import run as run_filter_stage


WEIGHTED_PERMITTIVITY_DIR = Path("criterion_weighted_permittivity")


def zero_salt_count(path: Path) -> int:
    """Count rows where salt_conc is exactly zero."""

    frame = pd.read_csv(path)
    if "salt_conc" not in frame.columns:
        return 0
    return int((pd.to_numeric(frame["salt_conc"], errors="coerce") == 0).sum())


def removed_index_count(path: Path, removed_indices: set[int]) -> int:
    """Count rows whose original index is one of the removed zero-salt rows."""

    frame = pd.read_csv(path)
    if "original_row_index" not in frame.columns:
        return 0
    indices = pd.to_numeric(frame["original_row_index"], errors="coerce")
    return int(indices.isin(removed_indices).sum())


def write_validation_report(
    report_path: Path,
    rows_removed: int,
    rows_remaining: int,
    cluster_count: int,
    campaign_systems: int,
    campaign_zero_salt_count: int,
    allocation: dict[int, int],
) -> None:
    """Save a compact validation report."""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Zero-salt electrolyte rebuild validation",
        f"rows_removed_because_salt_conc_eq_0: {rows_removed}",
        f"rows_remaining: {rows_remaining}",
        f"updated_weighted_permittivity_clusters: {cluster_count}",
        f"campaign_systems_selected: {campaign_systems}",
        f"campaign_zero_salt_rows: {campaign_zero_salt_count}",
        f"no_zero_salt_systems_remain_in_campaign: {campaign_zero_salt_count == 0}",
        "cluster_allocation: "
        + ", ".join(f"{cluster_id}:{count}" for cluster_id, count in allocation.items()),
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def run(
    raw_input: Path,
    electrolyte_output_dir: Path,
    results_dir: Path,
    sparse_campaign_dir: Path,
    k_min: int,
    k_max: int,
    silhouette_min_threshold: float,
    q_scale_grid: list[float],
    simulation_command: str | None,
) -> None:
    """Run filtering, descriptors, clustering, campaign, and validation."""

    cleaned = electrolyte_output_dir / "cleaned_electrolytes.csv"
    cleaned_no_zero = electrolyte_output_dir / "cleaned_electrolytes_no_zero_salt.csv"
    rejected = electrolyte_output_dir / "rejected_records.csv"
    descriptors = electrolyte_output_dir / "descriptors.csv"

    run_filter_stage(raw_input, cleaned, rejected)
    shutil.copyfile(cleaned, cleaned_no_zero)
    run_descriptor_stage(cleaned, descriptors)

    descriptor_frame = load_data(descriptors)
    compare_criteria_sets(
        descriptor_frame,
        DEFAULT_CRITERIA,
        results_dir,
        k_min,
        k_max,
        silhouette_min_threshold,
        min_non_missing=10,
        max_missing_fraction=0.50,
        random_seed=RANDOM_SEED,
    )

    cluster_dir = results_dir / WEIGHTED_PERMITTIVITY_DIR
    cluster_summary_path = cluster_dir / "cluster_summary.csv"
    cluster_labels_path = cluster_dir / "cluster_labels.csv"
    cluster_summary = pd.read_csv(cluster_summary_path)
    allocation = infer_cluster_allocation(cluster_summary)
    selected, campaign, _ = run_campaign_stage(
        cluster_summary_path,
        cluster_labels_path,
        cleaned,
        descriptors,
        sparse_campaign_dir,
        q_scale_grid,
        allocation,
        RANDOM_SEED,
        Path("launch_sparse_campaign.sh"),
        simulation_command,
    )

    rejected_frame = pd.read_csv(rejected)
    zero_salt_rejections = rejected_frame[
        rejected_frame["rejection_reason"]
        .fillna("")
        .str.contains("zero salt concentration c", regex=False)
    ]
    rows_removed = int(len(zero_salt_rejections))
    removed_indices = set(pd.to_numeric(zero_salt_rejections["original_row_index"], errors="coerce").dropna().astype(int))
    cleaned_frame = pd.read_csv(cleaned)
    campaign_zero_salt = zero_salt_count(sparse_campaign_dir / "campaign_plan.csv")
    write_validation_report(
        sparse_campaign_dir / "validation_report.txt",
        rows_removed,
        len(cleaned_frame),
        int(cluster_summary["cluster_id"].nunique()),
        len(selected),
        campaign_zero_salt,
        allocation,
    )

    if zero_salt_count(cleaned) != 0:
        raise ValueError("Zero-salt rows remain in cleaned_electrolytes.csv.")
    for output_path in [
        cleaned,
        descriptors,
        cluster_summary_path,
        cluster_labels_path,
        sparse_campaign_dir / "selected_representatives.csv",
        sparse_campaign_dir / "campaign_plan.csv",
    ]:
        if removed_index_count(output_path, removed_indices) != 0:
            raise ValueError(f"Removed zero-salt source rows remain in {output_path}.")
    if campaign_zero_salt != 0:
        raise ValueError("Zero-salt rows remain in the sparse campaign.")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-input",
        type=Path,
        default=Path("calisol23_DOI_10.11583DTU.c.6929599.csv"),
        help="Raw electrolyte database.",
    )
    parser.add_argument(
        "--electrolyte-output-dir",
        type=Path,
        default=Path("electrolyte_outputs"),
        help="Directory for cleaned and descriptor outputs.",
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Root clustering output directory.")
    parser.add_argument(
        "--sparse-campaign-dir",
        type=Path,
        default=Path("results/sparse_campaign_weighted_permittivity"),
        help="Directory for regenerated sparse campaign files.",
    )
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=6)
    parser.add_argument("--silhouette-min-threshold", type=float, default=0.15)
    parser.add_argument("--q-scale-grid", type=float, nargs="+", default=DEFAULT_Q_SCALE_GRID)
    parser.add_argument(
        "--simulation-command",
        type=str,
        default=None,
        help="Optional command prefix for executable launcher lines.",
    )
    return parser


def main() -> None:
    """Command-line entry point."""

    args = build_parser().parse_args()
    run(
        args.raw_input,
        args.electrolyte_output_dir,
        args.results_dir,
        args.sparse_campaign_dir,
        args.k_min,
        args.k_max,
        args.silhouette_min_threshold,
        list(args.q_scale_grid),
        args.simulation_command,
    )


if __name__ == "__main__":
    main()
