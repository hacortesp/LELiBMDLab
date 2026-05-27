#!/usr/bin/env python3
"""Convenience wrapper for the modular electrolyte processing workflow.

The main project entry points are:

1. ``filter_electrolytes.py``
2. ``build_descriptors.py``
3. ``cluster_electrolytes.py``
4. ``make_cluster_plots.py``

This wrapper runs those same stages in sequence for users who still want a
single command.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from build_descriptors import run as run_descriptor_stage
from cluster_electrolytes import run as run_cluster_stage
from filter_electrolytes import run as run_filter_stage
from make_cluster_plots import run as run_plot_stage


def run_pipeline(
    input_csv: Path,
    output_dir: Path,
    figures_dir: Path,
    k_min: int,
    k_max: int,
    silhouette_min_threshold: float,
    make_plots: bool,
) -> None:
    """Run filtering, descriptors, clustering, and optional plotting."""

    cleaned = output_dir / "cleaned_electrolytes.csv"
    rejected = output_dir / "rejected_records.csv"
    descriptors = output_dir / "descriptors.csv"
    labels = output_dir / "cluster_labels.csv"
    summary = output_dir / "cluster_summary.csv"
    scores = output_dir / "silhouette_scores.csv"
    selection = output_dir / "cluster_selection.csv"

    run_filter_stage(input_csv, cleaned, rejected)
    run_descriptor_stage(cleaned, descriptors)
    run_cluster_stage(
        descriptors,
        cleaned,
        labels,
        summary,
        scores,
        selection,
        k_min,
        k_max,
        silhouette_min_threshold,
    )
    if make_plots:
        run_plot_stage(labels, scores, selection, figures_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path, help="Raw experimental electrolyte CSV.")
    parser.add_argument("--output-dir", type=Path, default=Path("electrolyte_outputs"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=6)
    parser.add_argument("--silhouette-min-threshold", type=float, default=0.15)
    parser.add_argument("--make-plots", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_pipeline(
        args.input_csv,
        args.output_dir,
        args.figures_dir,
        args.k_min,
        args.k_max,
        args.silhouette_min_threshold,
        args.make_plots,
    )


if __name__ == "__main__":
    main()
