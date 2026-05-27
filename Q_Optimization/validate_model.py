#!/usr/bin/env python3
"""Validate a direct q_scale-optimal model on held-out electrolyte systems.

The trained model predicts q_scale_optimal. For each held-out system, this
script snaps the predicted q_scale to the nearest sampled q_scale in the sparse
campaign and recovers the corresponding simulated conductivity. It then reports
both q_scale prediction error and reconstructed-conductivity error against the
experimental conductivity.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "codex-cache"))
sys.modules.setdefault("pyarrow", None)

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score


SIM_CONDUCTIVITY = "conductivity_ms_cm"
EXP_CONDUCTIVITY = "conductivity"
TARGET = "q_scale_optimal"
SUCCESS_STATUSES = {"parsed", "completed", "complete", "success", "successful", "ok"}


def standardize_column_name(column: str) -> str:
    """Convert a CSV column name to snake_case."""

    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", column.strip()).strip("_").lower()
    return re.sub(r"_+", "_", cleaned)


def read_campaign_csv(path: Path) -> pd.DataFrame:
    """Read and normalize campaign results."""

    df = pd.read_csv(path)
    df.columns = [standardize_column_name(column) for column in df.columns]
    numeric_columns = [
        SIM_CONDUCTIVITY,
        EXP_CONDUCTIVITY,
        "conductivity_uncertainty_ms_cm",
        "q_scale",
        "salt_conc",
        "temperature",
        "cluster_id",
        "original_row_index",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    status = df["collection_status"].astype(str).str.strip().str.lower()
    df = df.loc[status.isin(SUCCESS_STATUSES)].copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[SIM_CONDUCTIVITY, EXP_CONDUCTIVITY, "q_scale"])
    df = df.loc[(df[SIM_CONDUCTIVITY] > 0) & (df[EXP_CONDUCTIVITY] > 0)].copy()
    if "cluster_id" in df.columns:
        df["cluster_id"] = df["cluster_id"].round().astype("Int64")
    return df


def electrolyte_group_key(df: pd.DataFrame) -> pd.Series:
    """Build the same electrolyte-system key used during training."""

    if "duplicate_identity" in df.columns and df["duplicate_identity"].notna().any():
        duplicate_identity = df["duplicate_identity"].astype(str).str.strip()
        fallback = pd.Series("row_" + df.index.astype(str), index=df.index)
        return duplicate_identity.where(duplicate_identity.ne(""), other=np.nan).fillna(fallback)

    key_columns = [
        "original_row_index",
        "doi",
        "cation_name",
        "anion_name",
        "salt_conc",
        "solvents",
        "solvent_fracs",
        "temperature",
        "cluster_id",
    ]
    available = [column for column in key_columns if column in df.columns]
    if not available:
        return pd.Series("row_" + df.index.astype(str), index=df.index)
    return df[available].astype(str).agg("|".join, axis=1)


def derive_qscale_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild q_scale_optimal labels from simulation-vs-experiment errors."""

    work = df.copy()
    work["electrolyte_group"] = electrolyte_group_key(work)
    work["absolute_conductivity_error_ms_cm"] = (
        work[SIM_CONDUCTIVITY] - work[EXP_CONDUCTIVITY]
    ).abs()
    records: list[dict[str, object]] = []
    for group_key, group in work.groupby("electrolyte_group", sort=False):
        best = group.sort_values(["absolute_conductivity_error_ms_cm", "q_scale"]).iloc[0]
        sim_min = float(group[SIM_CONDUCTIVITY].min())
        sim_max = float(group[SIM_CONDUCTIVITY].max())
        exp_value = float(best[EXP_CONDUCTIVITY])
        records.append(
            {
                "electrolyte_group": group_key,
                "original_row_index": best.get("original_row_index", np.nan),
                "doi": best.get("doi", ""),
                "cation_name": best.get("cation_name", ""),
                "anion_name": best.get("anion_name", ""),
                "solvents": best.get("solvents", ""),
                "solvent_fracs": best.get("solvent_fracs", ""),
                "composition_summary": best.get("composition_summary", ""),
                "cluster_id": best.get("cluster_id", np.nan),
                "salt_conc": best.get("salt_conc", np.nan),
                "temperature": best.get("temperature", np.nan),
                "experimental_conductivity_ms_cm": exp_value,
                "mean_conductivity_uncertainty_ms_cm": group.get(
                    "conductivity_uncertainty_ms_cm",
                    pd.Series([np.nan]),
                ).mean(),
                TARGET: float(best["q_scale"]),
                "optimal_simulated_conductivity_ms_cm": float(best[SIM_CONDUCTIVITY]),
                "optimal_abs_error_ms_cm": float(best["absolute_conductivity_error_ms_cm"]),
                "simulated_conductivity_min_ms_cm": sim_min,
                "simulated_conductivity_max_ms_cm": sim_max,
                "experiment_inside_simulated_range": bool(sim_min <= exp_value <= sim_max),
            }
        )
    return pd.DataFrame(records)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict[str, float]:
    """Compute MAE, RMSE, and R2."""

    residual = y_pred - y_true
    return {
        f"{prefix}_mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}_rmse": float(np.sqrt(np.mean(residual**2))),
        f"{prefix}_r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan,
    }


def load_validation_targets(targets: pd.DataFrame, split_csv: Path | None) -> pd.DataFrame:
    """Select held-out systems using the saved training split when available."""

    if split_csv is None or not split_csv.is_file():
        return targets.copy()
    split = pd.read_csv(split_csv)
    test_groups = set(split.loc[split["split"] == "test", "electrolyte_group"])
    return targets.loc[targets["electrolyte_group"].isin(test_groups)].copy()


def reconstruct_from_nearest_q(
    campaign: pd.DataFrame,
    targets: pd.DataFrame,
    predictions: np.ndarray,
    q_min: float,
    q_max: float,
) -> pd.DataFrame:
    """Recover simulated conductivity by snapping predicted q_scale to sampled q."""

    campaign = campaign.copy()
    campaign["electrolyte_group"] = electrolyte_group_key(campaign)
    rows: list[dict[str, object]] = []
    for (_, target), predicted_q in zip(targets.iterrows(), predictions):
        clipped_q = float(np.clip(predicted_q, q_min, q_max))
        group = campaign.loc[campaign["electrolyte_group"] == target["electrolyte_group"]].copy()
        if group.empty:
            continue
        group["q_distance"] = (group["q_scale"] - clipped_q).abs()
        selected = group.sort_values(["q_distance", "q_scale"]).iloc[0]
        rows.append(
            {
                **target.to_dict(),
                "predicted_q_scale": clipped_q,
                "nearest_sampled_q_scale": float(selected["q_scale"]),
                "true_q_scale_optimal": float(target[TARGET]),
                "q_scale_residual": clipped_q - float(target[TARGET]),
                "reconstructed_conductivity_ms_cm": float(selected[SIM_CONDUCTIVITY]),
                "reconstructed_conductivity_error_ms_cm": float(
                    selected[SIM_CONDUCTIVITY] - target["experimental_conductivity_ms_cm"]
                ),
                "reconstructed_abs_error_ms_cm": float(
                    abs(selected[SIM_CONDUCTIVITY] - target["experimental_conductivity_ms_cm"])
                ),
            }
        )
    return pd.DataFrame(rows)


def save_plots(
    predictions: pd.DataFrame,
    campaign: pd.DataFrame,
    output_dir: Path,
    selected_curve_count: int,
) -> None:
    """Generate validation and q_scale coverage diagnostics."""

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.scatter(
        predictions["true_q_scale_optimal"],
        predictions["predicted_q_scale"],
        c=predictions["cluster_id"],
        cmap="viridis",
        edgecolor="black",
        linewidth=0.45,
        s=55,
    )
    lower = min(predictions["true_q_scale_optimal"].min(), predictions["predicted_q_scale"].min())
    upper = max(predictions["true_q_scale_optimal"].max(), predictions["predicted_q_scale"].max())
    ax.plot([lower, upper], [lower, upper], color="#333333", linestyle="--", linewidth=1)
    ax.set_xlabel("True optimal q_scale")
    ax.set_ylabel("Predicted q_scale")
    ax.set_title("Predicted vs true optimal q_scale")
    fig.tight_layout()
    fig.savefig(output_dir / "predicted_vs_true_qscale.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.scatter(
        predictions["experimental_conductivity_ms_cm"],
        predictions["reconstructed_conductivity_ms_cm"],
        c=predictions["cluster_id"],
        cmap="viridis",
        edgecolor="black",
        linewidth=0.45,
        s=55,
    )
    lower = min(
        predictions["experimental_conductivity_ms_cm"].min(),
        predictions["reconstructed_conductivity_ms_cm"].min(),
    )
    upper = max(
        predictions["experimental_conductivity_ms_cm"].max(),
        predictions["reconstructed_conductivity_ms_cm"].max(),
    )
    ax.plot([lower, upper], [lower, upper], color="#333333", linestyle="--", linewidth=1)
    ax.set_xlabel("Experimental conductivity (mS/cm)")
    ax.set_ylabel("Reconstructed simulated conductivity (mS/cm)")
    ax.set_title("Experimental vs reconstructed conductivity")
    fig.tight_layout()
    fig.savefig(output_dir / "experimental_vs_reconstructed_conductivity.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(predictions["q_scale_residual"], bins=8, color="#4c78a8", edgecolor="black")
    axes[0].axvline(0.0, color="#333333", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Predicted q_scale - true q_scale_optimal")
    axes[0].set_ylabel("Count")
    axes[0].set_title("q_scale residuals")
    axes[1].hist(
        predictions["reconstructed_conductivity_error_ms_cm"],
        bins=8,
        color="#72b7b2",
        edgecolor="black",
    )
    axes[1].axvline(0.0, color="#333333", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Reconstructed - experimental conductivity (mS/cm)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Conductivity reconstruction errors")
    fig.tight_layout()
    fig.savefig(output_dir / "residual_error_distributions.png", dpi=300)
    plt.close(fig)

    campaign = campaign.copy()
    campaign["electrolyte_group"] = electrolyte_group_key(campaign)
    curve_groups = predictions.sort_values("reconstructed_abs_error_ms_cm", ascending=False)[
        "electrolyte_group"
    ].head(selected_curve_count)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for group_key in curve_groups:
        group = campaign.loc[campaign["electrolyte_group"] == group_key].sort_values("q_scale")
        label = str(group.iloc[0].get("composition_summary", group_key))
        ax.plot(group["q_scale"], group[SIM_CONDUCTIVITY], marker="o", linewidth=1.6, label=label)
        ax.axhline(group[EXP_CONDUCTIVITY].iloc[0], color=ax.lines[-1].get_color(), linestyle=":", linewidth=1)
    ax.set_xlabel("q_scale")
    ax.set_ylabel("Conductivity (mS/cm)")
    ax.set_title("Conductivity vs q_scale for selected held-out systems")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "selected_system_conductivity_curves.png", dpi=300)
    plt.close(fig)

    all_groups = sorted(campaign["electrolyte_group"].unique())
    group_to_x = {group: i for i, group in enumerate(all_groups)}
    campaign["system_index"] = campaign["electrolyte_group"].map(group_to_x)
    fig, ax = plt.subplots(figsize=(max(8, 0.38 * len(all_groups)), 5.2))
    for group_key, group in campaign.groupby("electrolyte_group", sort=False):
        x = group_to_x[group_key]
        ax.vlines(
            x,
            group[SIM_CONDUCTIVITY].min(),
            group[SIM_CONDUCTIVITY].max(),
            color="#cccccc",
            linewidth=3,
            zorder=0,
        )
        ax.scatter(
            [x],
            [group[EXP_CONDUCTIVITY].iloc[0]],
            marker="D",
            s=42,
            color="#d62728",
            edgecolor="black",
            linewidth=0.4,
            zorder=3,
        )
    scatter = ax.scatter(
        campaign["system_index"],
        campaign[SIM_CONDUCTIVITY],
        c=campaign["q_scale"],
        cmap="viridis",
        s=35,
        edgecolor="black",
        linewidth=0.25,
        zorder=2,
    )
    ax.set_xlabel("Electrolyte system index")
    ax.set_ylabel("Conductivity (mS/cm)")
    ax.set_title("q_scale sampling coverage: simulations and experiment")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("q_scale")
    ax.text(
        0.01,
        0.98,
        "Red diamonds: experiment; gray bars: simulated range",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "qscale_sampling_coverage_diagnostic.png", dpi=300)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    """Run validation."""

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_bundle = joblib.load(args.model)
    pipeline = model_bundle["pipeline"]
    feature_columns = model_bundle["feature_columns"]
    q_min = float(model_bundle["q_scale_min"])
    q_max = float(model_bundle["q_scale_max"])

    campaign = read_campaign_csv(args.input_csv)
    targets = derive_qscale_targets(campaign)
    validation_targets = load_validation_targets(targets, args.split_csv)
    if validation_targets.empty:
        raise ValueError("No validation systems selected.")

    predicted_q = pipeline.predict(validation_targets[feature_columns])
    predictions = reconstruct_from_nearest_q(campaign, validation_targets, predicted_q, q_min, q_max)
    predictions.to_csv(args.output_dir / "validation_predictions.csv", index=False)

    q_metrics = regression_metrics(
        predictions["true_q_scale_optimal"].to_numpy(dtype=float),
        predictions["predicted_q_scale"].to_numpy(dtype=float),
        "q_scale",
    )
    conductivity_metrics = regression_metrics(
        predictions["experimental_conductivity_ms_cm"].to_numpy(dtype=float),
        predictions["reconstructed_conductivity_ms_cm"].to_numpy(dtype=float),
        "conductivity",
    )
    metrics = {
        "n_validation_systems": int(len(predictions)),
        "n_experiment_inside_simulated_range": int(predictions["experiment_inside_simulated_range"].sum()),
        **q_metrics,
        **conductivity_metrics,
    }
    pd.DataFrame([metrics]).to_csv(args.output_dir / "validation_metrics.csv", index=False)
    save_plots(predictions, campaign, args.output_dir, args.selected_curve_count)

    (args.output_dir / "validation_summary.txt").write_text(
        "\n".join(
            [
                "Direct q_scale model validation",
                "==============================",
                f"model: {args.model}",
                f"systems: {metrics['n_validation_systems']}",
                f"q_scale MAE: {metrics['q_scale_mae']:.6g}",
                f"q_scale RMSE: {metrics['q_scale_rmse']:.6g}",
                f"conductivity MAE: {metrics['conductivity_mae']:.6g} mS/cm",
                f"conductivity RMSE: {metrics['conductivity_rmse']:.6g} mS/cm",
            ]
        )
        + "\n"
    )
    metadata = {
        "input_csv": str(args.input_csv),
        "model": str(args.model),
        "split_csv": str(args.split_csv) if args.split_csv else None,
        "output_dir": str(args.output_dir),
        "feature_columns": feature_columns,
    }
    (args.output_dir / "validation_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"Wrote validation outputs to: {args.output_dir}")
    print(f"q_scale MAE/RMSE: {metrics['q_scale_mae']:.4g} / {metrics['q_scale_rmse']:.4g}")
    print(
        "reconstructed conductivity MAE/RMSE: "
        f"{metrics['conductivity_mae']:.4g} / {metrics['conductivity_rmse']:.4g} mS/cm"
    )


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("results/sparse_campaign_weighted_permittivity/conductivity_results.csv"),
        help="Sparse campaign conductivity_results.csv.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("results/qscale_direct_model_weighted_permittivity/qscale_model_train_split.joblib"),
        help="Model bundle trained on training systems only.",
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=Path("results/qscale_direct_model_weighted_permittivity/system_split.csv"),
        help="Saved train/test split from train_model.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/qscale_direct_model_weighted_permittivity/validation"),
        help="Directory for validation outputs.",
    )
    parser.add_argument(
        "--selected-curve-count",
        type=int,
        default=6,
        help="Number of held-out systems to show in q_scale curve plot.",
    )
    return parser


def main() -> None:
    """Command-line entry point."""

    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
