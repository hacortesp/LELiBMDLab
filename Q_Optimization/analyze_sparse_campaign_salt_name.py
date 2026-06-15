#!/usr/bin/env python3
"""Validate and visualize sparse salt campaign conductivity results."""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Work around environments where pandas optional pyarrow import can be noisy/broken.
os.environ.setdefault("PANDAS_NO_IMPORT_PYARROW", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")
sys.modules.setdefault("pyarrow", None)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


SYSTEM_KEY_CANDIDATES: Sequence[str] = (
    "cation_name",
    "anion_name",
    "salt_conc",
    "solvents",
    "solvent_fracs",
    "temperature",
    "original_row_index",
)

PREFERRED_SALT_ORDER: Sequence[str] = ("PF6", "ClO4", "TFSI", "FSI", "BF4")
EXPECTED_Q_SCALES: Sequence[float] = (0.70, 0.75, 0.80, 0.85, 0.90)
SALT_COLORS: Dict[str, str] = {
    "PF6": "#1f77b4",
    "ClO4": "#ff7f0e",
    "TFSI": "#2ca02c",
    "FSI": "#d62728",
    "BF4": "#9467bd",
}


def str_to_bool(value: str) -> bool:
    clean = str(value).strip().lower()
    if clean in ("1", "true", "yes", "y", "on"):
        return True
    if clean in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError("Expected true or false")


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 17,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "legend.title_fontsize": 12,
            "figure.titlesize": 18,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and plot sparse campaign conductivity results"
    )
    parser.add_argument(
        "--input",
        default="results/sparse_campaign_salt_name/conductivity_results.csv",
        help="Input CSV with conductivity results",
    )
    parser.add_argument(
        "--output-dir",
        default="results/sparse_campaign_salt_name/analysis",
        help="Directory where tables and plots will be saved",
    )
    parser.add_argument(
        "--calculated-col",
        default="conductivity_mS_cm",
        help="Column for calculated/simulated conductivity (mS/cm)",
    )
    parser.add_argument(
        "--experimental-col",
        default="conductivity",
        help="Column for experimental/reference conductivity (mS/cm)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI",
    )
    parser.add_argument(
        "--overwrite",
        type=str_to_bool,
        default=True,
        help="Overwrite existing analysis outputs: true or false",
    )
    return parser.parse_args()


def choose_salt_column(df: pd.DataFrame) -> str:
    for col in ("anion_name", "original_salt", "salt_cluster_id"):
        if col in df.columns:
            return col
    raise ValueError("Could not find a salt column among: anion_name, original_salt, salt_cluster_id")


def build_system_key_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in SYSTEM_KEY_CANDIDATES if c in df.columns]
    if not cols:
        raise ValueError("No suitable columns found for system identity")
    return cols


def build_system_identifier(df: pd.DataFrame, key_cols: Sequence[str]) -> pd.Series:
    safe = df.loc[:, key_cols].copy()
    for col in key_cols:
        safe[col] = safe[col].astype(str)
    return safe.agg(" | ".join, axis=1)


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def rounded_q_values(series: pd.Series) -> List[float]:
    q_num = to_numeric(series).dropna()
    return sorted(set(round(float(q), 2) for q in q_num))


def format_q_values(values: Sequence[float]) -> str:
    return ";".join(f"{q:.2f}" for q in values)


def validation_metrics(
    df: pd.DataFrame,
    calc_col: str,
    key_col: str,
    q_col: Optional[str],
) -> Tuple[Dict[str, object], pd.DataFrame, pd.Series]:
    has_calc = df[calc_col].notna()
    expected_q = set(round(float(q), 2) for q in EXPECTED_Q_SCALES)

    planned_jobs = len(df)
    successful_jobs = int(has_calc.sum())
    missing_jobs = int((~has_calc).sum())

    unique_systems = int(df[key_col].nunique(dropna=False))
    systems_with_calc = df.loc[has_calc, key_col].nunique(dropna=False)

    by_system_has_calc = df.groupby(key_col, dropna=False)[calc_col].apply(lambda s: s.notna().any())
    missing_all_systems = by_system_has_calc[~by_system_has_calc].index.tolist()
    complete_systems = 0
    partial_systems = 0
    no_valid_systems = 0

    for _, group in df.groupby(key_col, dropna=False):
        valid_group = group[group[calc_col].notna()]
        if valid_group.empty:
            no_valid_systems += 1
            continue
        if q_col is None:
            available_q = set()
        else:
            available_q = set(rounded_q_values(valid_group[q_col]))
        if expected_q.issubset(available_q):
            complete_systems += 1
        else:
            partial_systems += 1

    metrics = {
        "total_planned_jobs": planned_jobs,
        "jobs_with_valid_conductivity_mS_cm": successful_jobs,
        "jobs_missing_conductivity_mS_cm": missing_jobs,
        "unique_electrolyte_systems": unique_systems,
        "unique_systems_with_at_least_one_conductivity": int(systems_with_calc),
        "systems_with_complete_q_scale_coverage": int(complete_systems),
        "systems_with_partial_q_scale_coverage": int(partial_systems),
        "systems_with_no_valid_conductivity": int(no_valid_systems),
        "any_system_missing_all_conductivity_values": bool(len(missing_all_systems) > 0),
        "systems_missing_all_conductivity_values_count": int(len(missing_all_systems)),
    }

    missing_jobs_df = df.loc[~has_calc].copy()
    return metrics, missing_jobs_df, pd.Index(missing_all_systems)


def make_summary_table(
    df: pd.DataFrame,
    key_col: str,
    salt_col: str,
    calc_col: str,
    exp_col: str,
) -> pd.DataFrame:
    q_col = "q_scale" if "q_scale" in df.columns else None
    expected_q = set(round(float(q), 2) for q in EXPECTED_Q_SCALES)

    grouped = df.groupby(key_col, dropna=False)

    summary = grouped.agg(
        salt_type=(salt_col, "first"),
        experimental_conductivity_mS_cm=(exp_col, "first"),
        mean_calculated_conductivity_mS_cm=(calc_col, "mean"),
        min_calculated_conductivity_mS_cm=(calc_col, "min"),
        max_calculated_conductivity_mS_cm=(calc_col, "max"),
    )

    summary["num_q_scale_points_available"] = grouped[calc_col].apply(lambda s: int(s.notna().sum()))
    if q_col is not None:
        unique_q = (
            df.loc[df[calc_col].notna(), [key_col, q_col]]
            .groupby(key_col, dropna=False)[q_col]
            .nunique(dropna=True)
        )
        summary["num_unique_q_scale_available"] = unique_q.reindex(summary.index).fillna(0).astype(int)

        q_values = (
            df.loc[df[calc_col].notna(), [key_col, q_col]]
            .groupby(key_col, dropna=False)[q_col]
            .apply(rounded_q_values)
        )
        summary["q_scale_values_available"] = q_values.reindex(summary.index).apply(
            lambda values: values if isinstance(values, list) else []
        )
    else:
        summary["num_unique_q_scale_available"] = summary["num_q_scale_points_available"]
        summary["q_scale_values_available"] = [[] for _ in range(len(summary))]

    summary["missing_q_scale_values"] = summary["q_scale_values_available"].apply(
        lambda values: sorted(expected_q.difference(set(values)))
    )
    summary["q_scale_coverage_status"] = np.where(
        summary["num_q_scale_points_available"] == 0,
        "none",
        np.where(summary["missing_q_scale_values"].apply(len) == 0, "complete", "partial"),
    )
    summary["q_scale_values_available"] = summary["q_scale_values_available"].apply(format_q_values)
    summary["missing_q_scale_values"] = summary["missing_q_scale_values"].apply(format_q_values)

    summary = summary.reset_index().rename(columns={key_col: "system_identifier"})

    summary["missing_conductivity_flag"] = summary["mean_calculated_conductivity_mS_cm"].isna()
    salt_order = {salt: i for i, salt in enumerate(PREFERRED_SALT_ORDER)}
    summary["_salt_order"] = summary["salt_type"].astype(str).map(salt_order).fillna(len(salt_order)).astype(int)
    summary = summary.sort_values(
        by=["_salt_order", "salt_type", "experimental_conductivity_mS_cm", "system_identifier"],
        ascending=[True, True, True, True],
        na_position="last",
    ).drop(columns="_salt_order").reset_index(drop=True)
    summary.insert(0, "system_order", np.arange(1, len(summary) + 1))
    return summary


def make_salt_population_summary(
    df: pd.DataFrame,
    key_col: str,
    salt_col: str,
    calc_col: str,
) -> pd.DataFrame:
    rows = []
    for salt, group in df.groupby(salt_col, dropna=False):
        systems = group[key_col].nunique(dropna=False)
        sim_points = int(group[calc_col].notna().sum())
        rows.append(
            {
                "salt_type": str(salt),
                "num_systems": int(systems),
                "num_simulation_points": sim_points,
                "appears_in_combined_parity_plot": bool(sim_points > 0),
                "diagnostic": (
                    "present in plot"
                    if sim_points > 0
                    else "absent from parity plot because no successful simulated conductivity values are available"
                ),
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    order = {salt: i for i, salt in enumerate(PREFERRED_SALT_ORDER)}
    summary["_order"] = summary["salt_type"].map(order).fillna(len(order)).astype(int)
    return summary.sort_values(["_order", "salt_type"]).drop(columns="_order").reset_index(drop=True)


def make_coverage_report(summary: pd.DataFrame) -> pd.DataFrame:
    work = summary.copy()
    work["experiment_within_simulated_range"] = (
        work["experimental_conductivity_mS_cm"].notna()
        & work["min_calculated_conductivity_mS_cm"].notna()
        & work["max_calculated_conductivity_mS_cm"].notna()
        & (work["min_calculated_conductivity_mS_cm"] <= work["experimental_conductivity_mS_cm"])
        & (work["experimental_conductivity_mS_cm"] <= work["max_calculated_conductivity_mS_cm"])
    )
    valid = work[
        work["min_calculated_conductivity_mS_cm"].notna()
        & work["max_calculated_conductivity_mS_cm"].notna()
    ].copy()
    coverage_fraction = (
        float(valid["experiment_within_simulated_range"].mean()) if len(valid) else np.nan
    )
    work["coverage_fraction"] = coverage_fraction
    work["covered_systems"] = int(valid["experiment_within_simulated_range"].sum())
    work["systems_with_at_least_one_simulated_conductivity"] = int(len(valid))
    return work[
        [
            "system_order",
            "system_identifier",
            "salt_type",
            "experimental_conductivity_mS_cm",
            "min_calculated_conductivity_mS_cm",
            "max_calculated_conductivity_mS_cm",
            "experiment_within_simulated_range",
            "coverage_fraction",
            "covered_systems",
            "systems_with_at_least_one_simulated_conductivity",
        ]
    ]


def map_sizes(q_values: pd.Series, min_size: float = 45.0, max_size: float = 220.0) -> pd.Series:
    q_num = to_numeric(q_values)
    q_min = q_num.min()
    q_max = q_num.max()
    if pd.isna(q_min) or pd.isna(q_max) or q_min == q_max:
        return pd.Series(np.full(len(q_num), (min_size + max_size) / 2.0), index=q_num.index)
    scaled = (q_num - q_min) / (q_max - q_min)
    return min_size + scaled * (max_size - min_size)


def salt_color_map(salts: Sequence[str]) -> Dict[str, tuple]:
    unique_salts = sorted(pd.Series(salts).dropna().astype(str).unique())
    cmap = plt.get_cmap("tab10")
    colors = {}
    for i, salt in enumerate(unique_salts):
        colors[salt] = SALT_COLORS.get(salt, cmap(i % 10))
    return colors


def ordered_salts(salts: Sequence[str]) -> List[str]:
    values = pd.Series(salts).dropna().astype(str).drop_duplicates().tolist()
    order = {salt: i for i, salt in enumerate(PREFERRED_SALT_ORDER)}
    return sorted(values, key=lambda salt: (order.get(salt, len(order)), salt))


def read_previous_valid_job_count(out_dir: Path) -> Optional[int]:
    summary_path = out_dir / "validation_summary.csv"
    if not summary_path.exists():
        return None
    try:
        previous = pd.read_csv(summary_path)
    except Exception:
        return None
    for col in (
        "jobs_with_valid_conductivity_mS_cm",
        "jobs_with_successful_calculated_conductivity",
    ):
        if col in previous.columns and len(previous):
            value = pd.to_numeric(previous[col], errors="coerce").iloc[0]
            if pd.notna(value):
                return int(value)
    return None


def ensure_outputs_can_be_written(paths: Sequence[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Analysis outputs already exist and --overwrite=false was supplied: "
            + ", ".join(existing)
        )


def add_qscale_legend(
    ax: plt.Axes,
    q_values: pd.Series,
    size_values: pd.Series,
    anchor=(1.01, 0.5),
    color: str = "0.55",
) -> None:
    q_num = to_numeric(q_values).dropna()
    if q_num.empty:
        return
    q_levels = sorted(q_num.unique())
    if len(q_levels) > 5:
        idx = np.linspace(0, len(q_levels) - 1, 5).round().astype(int)
        q_levels = [q_levels[i] for i in idx]

    handles = []
    for q in q_levels:
        nearest_idx = (to_numeric(q_values) - q).abs().idxmin()
        size = float(size_values.loc[nearest_idx])
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=color,
                markeredgecolor="black",
                alpha=0.7,
                markersize=np.sqrt(size),
                label=f"q_scale={q:.2f}",
            )
        )
    leg = ax.legend(handles=handles, title="Marker size", loc="center left", bbox_to_anchor=anchor)
    ax.add_artist(leg)


def add_salt_legend(
    ax: plt.Axes,
    colors: Dict[str, tuple],
    salts: Sequence[str],
    anchor=(1.01, 1.0),
) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=colors[str(salt)],
            markeredgecolor="black",
            markersize=9,
            label=str(salt),
        )
        for salt in salts
        if str(salt) in colors
    ]
    leg = ax.legend(handles=handles, title="Salt family", loc="upper left", bbox_to_anchor=anchor)
    ax.add_artist(leg)


def plot_system_sorted(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    output_path: Path,
    calc_col: str,
    exp_col: str,
    salt_col: str,
    dpi: int,
) -> None:
    work = df.copy()
    order_map = summary.set_index("system_identifier")["system_order"].to_dict()
    work["system_order"] = work["system_identifier"].map(order_map)
    work = work.sort_values(["system_order", "q_scale"])  # keep each system grouped

    work["q_scale_num"] = to_numeric(work["q_scale"]) if "q_scale" in work.columns else np.nan
    size = map_sizes(work["q_scale_num"])
    work["marker_size"] = size

    colors = salt_color_map(work[salt_col].astype(str).values)
    salts_in_order = summary["salt_type"].dropna().astype(str).drop_duplicates().tolist()

    # Horizontal offset so all q_scale points are visible per system.
    unique_q = sorted(work["q_scale_num"].dropna().unique())
    if unique_q:
        offsets = np.linspace(-0.20, 0.20, len(unique_q))
        offset_map = {q: off for q, off in zip(unique_q, offsets)}
        work["x_plot"] = work["system_order"] + work["q_scale_num"].map(offset_map).fillna(0.0)
    else:
        work["x_plot"] = work["system_order"]

    fig, ax = plt.subplots(figsize=(17, 8.5), constrained_layout=True)

    range_df = summary.dropna(
        subset=["min_calculated_conductivity_mS_cm", "max_calculated_conductivity_mS_cm"]
    ).copy()
    for _, row in range_df.iterrows():
        x = row["system_order"]
        ymin = row["min_calculated_conductivity_mS_cm"]
        ymax = row["max_calculated_conductivity_mS_cm"]
        ax.vlines(x, ymin, ymax, color="0.35", alpha=0.28, linewidth=9, zorder=1)
        ax.scatter(
            [x, x],
            [ymin, ymax],
            s=42,
            color="0.35",
            alpha=0.45,
            marker="_",
            linewidth=1.6,
            zorder=2,
        )

    for salt, group in work.groupby(salt_col):
        ax.scatter(
            group["x_plot"],
            group[calc_col],
            s=group["marker_size"],
            c=[colors[str(salt)]],
            alpha=0.85,
            edgecolor="black",
            linewidth=0.4,
            label=str(salt),
            zorder=3,
        )

    exp_df = summary.dropna(subset=["experimental_conductivity_mS_cm"]).copy()
    ax.scatter(
        exp_df["system_order"],
        exp_df["experimental_conductivity_mS_cm"],
        s=170,
        marker="D",
        color="black",
        edgecolor="white",
        linewidth=0.9,
        label="Experimental conductivity",
        zorder=5,
    )

    for salt in salts_in_order[1:]:
        first_order = int(summary.loc[summary["salt_type"].astype(str) == salt, "system_order"].min())
        ax.axvline(first_order - 0.5, color="0.2", linewidth=1.2, alpha=0.35, zorder=0)

    ylim = ax.get_ylim()
    y_text = ylim[1] - 0.035 * (ylim[1] - ylim[0])
    for salt, group in summary.groupby("salt_type", sort=False):
        center = 0.5 * (group["system_order"].min() + group["system_order"].max())
        ax.text(
            center,
            y_text,
            str(salt),
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
            color=colors.get(str(salt), "black"),
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 2.0},
        )

    add_salt_legend(ax, colors, salts_in_order, anchor=(1.01, 1.0))
    add_qscale_legend(ax, work["q_scale_num"], work["marker_size"], anchor=(1.01, 0.56))
    experiment_handle = Line2D(
        [0],
        [0],
        marker="D",
        linestyle="none",
        markerfacecolor="black",
        markeredgecolor="white",
        markersize=10,
        label="Experimental",
    )
    envelope_handle = Line2D(
        [0],
        [0],
        color="0.35",
        lw=7,
        alpha=0.35,
        label="Simulated min–max",
    )
    ax.legend(
        handles=[experiment_handle, envelope_handle],
        title="Reference/range",
        loc="lower left",
        bbox_to_anchor=(1.01, 0.05),
    )

    total_systems = len(summary)
    total_simulations = int(work[calc_col].notna().sum())
    total_salts = len(salts_in_order)
    ax.set_title(
        f"Conductivity by Electrolyte System "
        f"({total_systems} systems, {total_simulations} simulations, {total_salts} salt families)"
    )
    ax.set_xlabel("Electrolyte systems sorted by salt family, then experimental conductivity")
    ax.set_ylabel("Conductivity (mS cm$^{-1}$)")
    ax.set_xlim(0.4, summary["system_order"].max() + 0.6)
    ax.grid(axis="y", alpha=0.25)

    tick_step = max(1, int(np.ceil(len(summary) / 18)))
    xticks = summary["system_order"][::tick_step]
    ax.set_xticks(xticks)
    ax.tick_params(axis="x", rotation=0)

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_parity(
    df: pd.DataFrame,
    output_path: Path,
    calc_col: str,
    exp_col: str,
    salt_col: str,
    dpi: int,
    title: str = "Parity Plot: Simulated vs Experimental Conductivity",
    fixed_salt: Optional[str] = None,
) -> None:
    source = df[df[salt_col].astype(str) == fixed_salt].copy() if fixed_salt is not None else df.copy()
    work = source.dropna(subset=[calc_col, exp_col]).copy()
    source_salts = ordered_salts(source[salt_col].values)
    system_count = int(source["system_identifier"].nunique(dropna=False)) if "system_identifier" in source.columns else 0
    sim_point_count = int(work[calc_col].notna().sum())
    display_title = title
    if fixed_salt is not None:
        display_title = f"{title} ({system_count} systems, {sim_point_count} simulation points)"

    fig, ax = plt.subplots(figsize=(9, 7.5), constrained_layout=True)

    if work.empty:
        salt_label = fixed_salt or "selected salts"
        ax.text(
            0.5,
            0.5,
            f"No successful simulated conductivity values for {salt_label}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        ax.set_title(display_title)
        ax.set_xlabel("Experimental conductivity (mS cm$^{-1}$)")
        ax.set_ylabel("Simulated conductivity (mS cm$^{-1}$)")
        ax.grid(alpha=0.25)
        fig.savefig(output_path, dpi=dpi)
        plt.close(fig)
        return

    work["q_scale_num"] = to_numeric(work["q_scale"]) if "q_scale" in work.columns else np.nan
    work["marker_size"] = map_sizes(work["q_scale_num"])

    colors = salt_color_map(source[salt_col].astype(str).values)

    for salt, group in work.groupby(salt_col, sort=False):
        ax.scatter(
            group[exp_col],
            group[calc_col],
            s=group["marker_size"],
            c=[colors[str(salt)]],
            alpha=0.68,
            edgecolor="black",
            linewidth=0.4,
            label=str(salt),
            zorder=3,
        )

    min_val = np.nanmin([work[exp_col].min(), work[calc_col].min()])
    max_val = np.nanmax([work[exp_col].max(), work[calc_col].max()])
    pad = 0.05 * (max_val - min_val) if max_val > min_val else 1.0
    lo, hi = min_val - pad, max_val + pad
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="1:1 line", zorder=2)

    if fixed_salt is None:
        add_salt_legend(ax, colors, source_salts, anchor=(0.02, 0.98))
    else:
        salt_handle = Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=colors.get(fixed_salt, "0.5"),
            markeredgecolor="black",
            markersize=9,
            label=fixed_salt,
        )
        salt_leg = ax.legend(
            handles=[salt_handle],
            title="Salt family",
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
        )
        ax.add_artist(salt_leg)
    add_qscale_legend(ax, work["q_scale_num"], work["marker_size"], anchor=(0.02, 0.53))
    line_leg = ax.legend(
        handles=[Line2D([0], [0], color="black", linestyle="--", lw=1.2, label="1:1 reference")],
        loc="lower right",
    )
    ax.add_artist(line_leg)

    ax.set_title(display_title)
    ax.set_xlabel("Experimental conductivity (mS cm$^{-1}$)")
    ax.set_ylabel("Simulated conductivity (mS cm$^{-1}$)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_range_vs_experiment(
    summary: pd.DataFrame,
    output_path: Path,
    dpi: int,
) -> None:
    work = summary.copy()
    colors = salt_color_map(work["salt_type"].astype(str).values)
    salts_in_order = work["salt_type"].dropna().astype(str).drop_duplicates().tolist()

    fig, ax = plt.subplots(figsize=(17, 8.5), constrained_layout=True)

    range_df = work.dropna(
        subset=["min_calculated_conductivity_mS_cm", "max_calculated_conductivity_mS_cm"]
    )
    ax.vlines(
        range_df["system_order"],
        range_df["min_calculated_conductivity_mS_cm"],
        range_df["max_calculated_conductivity_mS_cm"],
        color="0.35",
        alpha=0.45,
        linewidth=4,
        label="Simulated min–max",
        zorder=2,
    )
    ax.scatter(
        range_df["system_order"],
        range_df["min_calculated_conductivity_mS_cm"],
        s=55,
        marker="v",
        color="0.35",
        alpha=0.75,
        label="Minimum simulated",
        zorder=3,
    )
    ax.scatter(
        range_df["system_order"],
        range_df["max_calculated_conductivity_mS_cm"],
        s=55,
        marker="^",
        color="0.35",
        alpha=0.75,
        label="Maximum simulated",
        zorder=3,
    )

    exp_df = work.dropna(subset=["experimental_conductivity_mS_cm"])
    for salt, group in exp_df.groupby("salt_type", sort=False):
        ax.scatter(
            group["system_order"],
            group["experimental_conductivity_mS_cm"],
            s=150,
            marker="D",
            color=colors[str(salt)],
            edgecolor="black",
            linewidth=0.7,
            label=str(salt),
            zorder=4,
        )

    for salt in salts_in_order[1:]:
        first_order = int(work.loc[work["salt_type"].astype(str) == salt, "system_order"].min())
        ax.axvline(first_order - 0.5, color="0.2", linewidth=1.2, alpha=0.35, zorder=0)

    add_salt_legend(ax, colors, salts_in_order, anchor=(1.01, 1.0))
    range_leg = ax.legend(
        handles=[
            Line2D([0], [0], color="0.35", lw=4, alpha=0.5, label="Simulated min–max"),
            Line2D([0], [0], marker="D", linestyle="none", color="black", markersize=9, label="Experimental"),
        ],
        title="Conductivity values",
        loc="center left",
        bbox_to_anchor=(1.01, 0.55),
    )
    ax.add_artist(range_leg)

    tick_step = max(1, int(np.ceil(len(work) / 18)))
    ax.set_xticks(work["system_order"][::tick_step])
    ax.set_xlim(0.4, work["system_order"].max() + 0.6)
    ax.set_title("Simulated Conductivity Range vs Experimental Conductivity")
    ax.set_xlabel("Electrolyte systems sorted by salt family, then experimental conductivity")
    ax.set_ylabel("Conductivity (mS cm$^{-1}$)")
    ax.grid(axis="y", alpha=0.25)

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_plot_style()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    previous_valid_jobs = read_previous_valid_job_count(out_dir)
    df = pd.read_csv(input_path)

    calc_col = args.calculated_col
    exp_col = args.experimental_col

    if calc_col not in df.columns:
        raise ValueError(f"Calculated conductivity column '{calc_col}' not found in input")
    if exp_col not in df.columns:
        raise ValueError(f"Experimental conductivity column '{exp_col}' not found in input")

    df[calc_col] = to_numeric(df[calc_col])
    df[exp_col] = to_numeric(df[exp_col])

    salt_col = choose_salt_column(df)
    key_cols = build_system_key_columns(df)
    df["system_identifier"] = build_system_identifier(df, key_cols)
    q_col = "q_scale" if "q_scale" in df.columns else None

    metrics, missing_jobs_df, systems_missing_all = validation_metrics(
        df=df,
        calc_col=calc_col,
        key_col="system_identifier",
        q_col=q_col,
    )

    summary = make_summary_table(
        df=df,
        key_col="system_identifier",
        salt_col=salt_col,
        calc_col=calc_col,
        exp_col=exp_col,
    )
    salt_population_summary = make_salt_population_summary(
        df=df,
        key_col="system_identifier",
        salt_col=salt_col,
        calc_col=calc_col,
    )
    coverage_report = make_coverage_report(summary)
    present_salts = ordered_salts(df[salt_col].values)

    output_paths = [
        out_dir / "validation_summary.csv",
        out_dir / "missing_conductivity_jobs.csv",
        out_dir / "systems_missing_all_conductivity.csv",
        out_dir / "system_conductivity_summary.csv",
        out_dir / "salt_population_summary.csv",
        out_dir / "coverage_report.csv",
        out_dir / "job_level_sorted_results.csv",
        out_dir / "conductivity_by_system_sorted.png",
        out_dir / "conductivity_parity_plot.png",
        out_dir / "conductivity_range_vs_experiment.png",
        out_dir / "validation_report.txt",
    ]
    output_paths.extend(out_dir / f"parity_{salt}.png" for salt in present_salts)
    ensure_outputs_can_be_written(output_paths, args.overwrite)

    if args.overwrite:
        expected_parity_names = set(f"parity_{salt}.png" for salt in present_salts)
        for old_parity in out_dir.glob("parity_*.png"):
            if old_parity.name not in expected_parity_names:
                old_parity.unlink()

    # Save validation summary.
    validation_summary = pd.DataFrame([metrics])
    validation_summary.to_csv(out_dir / "validation_summary.csv", index=False)

    # Save missing jobs table if any are missing.
    if not missing_jobs_df.empty:
        preferred_cols = [
            "job_id",
            "collection_status",
            salt_col,
            "cation_name",
            "anion_name",
            "salt_conc",
            "solvents",
            "solvent_fracs",
            "temperature",
            "q_scale",
            "system_identifier",
            "folder_name",
            "output_file",
        ]
        cols = []
        seen = set()
        for c in preferred_cols:
            if c in missing_jobs_df.columns and c not in seen:
                cols.append(c)
                seen.add(c)
        missing_jobs_df.loc[:, cols].to_csv(out_dir / "missing_conductivity_jobs.csv", index=False)
    else:
        pd.DataFrame(columns=["message"]).to_csv(out_dir / "missing_conductivity_jobs.csv", index=False)

    if len(systems_missing_all) > 0:
        pd.DataFrame({"system_identifier": list(systems_missing_all)}).to_csv(
            out_dir / "systems_missing_all_conductivity.csv", index=False
        )
    else:
        pd.DataFrame(columns=["system_identifier"]).to_csv(
            out_dir / "systems_missing_all_conductivity.csv", index=False
        )

    summary.to_csv(out_dir / "system_conductivity_summary.csv", index=False)
    salt_population_summary.to_csv(out_dir / "salt_population_summary.csv", index=False)
    coverage_report.to_csv(out_dir / "coverage_report.csv", index=False)

    print("\nSalt population summary")
    print(salt_population_summary.to_string(index=False))
    absent = salt_population_summary.loc[
        ~salt_population_summary["appears_in_combined_parity_plot"], ["salt_type", "diagnostic"]
    ]
    if absent.empty:
        print("\nAll salt families with systems have at least one point in the combined parity plot.")
    else:
        print("\nSalt families absent from the combined parity plot")
        print(absent.to_string(index=False))
    if not coverage_report.empty:
        coverage_fraction = coverage_report["coverage_fraction"].iloc[0]
        if pd.notna(coverage_fraction):
            print(f"\nCoverage fraction: {coverage_fraction:.3f}")

    # Save a system-sorted job-level table for traceability.
    order_map = summary.set_index("system_identifier")["system_order"].to_dict()
    sorted_jobs = df.copy()
    sorted_jobs["system_order"] = sorted_jobs["system_identifier"].map(order_map)
    sorted_jobs = sorted_jobs.sort_values([salt_col, "system_order", "q_scale"])
    sorted_jobs.to_csv(out_dir / "job_level_sorted_results.csv", index=False)

    plot_system_sorted(
        df=sorted_jobs,
        summary=summary,
        output_path=out_dir / "conductivity_by_system_sorted.png",
        calc_col=calc_col,
        exp_col=exp_col,
        salt_col=salt_col,
        dpi=args.dpi,
    )

    plot_parity(
        df=sorted_jobs,
        output_path=out_dir / "conductivity_parity_plot.png",
        calc_col=calc_col,
        exp_col=exp_col,
        salt_col=salt_col,
        dpi=args.dpi,
        title="Combined Parity Plot: Simulated vs Experimental Conductivity",
    )

    for salt in present_salts:
        plot_parity(
            df=sorted_jobs,
            output_path=out_dir / f"parity_{salt}.png",
            calc_col=calc_col,
            exp_col=exp_col,
            salt_col=salt_col,
            dpi=args.dpi,
            title=f"{salt} Parity Plot: Simulated vs Experimental Conductivity",
            fixed_salt=salt,
        )

    plot_range_vs_experiment(
        summary=summary,
        output_path=out_dir / "conductivity_range_vs_experiment.png",
        dpi=args.dpi,
    )

    # Human-readable text report.
    with open(out_dir / "validation_report.txt", "w", encoding="utf-8") as f:
        f.write("Sparse Campaign Conductivity Validation\n")
        f.write("=" * 44 + "\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\nSalt Population Diagnostics\n")
        f.write("-" * 44 + "\n")
        f.write(salt_population_summary.to_string(index=False))
        f.write("\n\nCoverage Diagnostics\n")
        f.write("-" * 44 + "\n")
        if not coverage_report.empty and pd.notna(coverage_report["coverage_fraction"].iloc[0]):
            f.write(f"coverage_fraction: {coverage_report['coverage_fraction'].iloc[0]:.6f}\n")
            f.write(
                "covered_systems: "
                f"{int(coverage_report['covered_systems'].iloc[0])}\n"
            )
            f.write(
                "systems_with_at_least_one_simulated_conductivity: "
                f"{int(coverage_report['systems_with_at_least_one_simulated_conductivity'].iloc[0])}\n"
            )
        else:
            f.write("coverage_fraction: unavailable\n")

    additional_valid_jobs = None
    if previous_valid_jobs is not None:
        additional_valid_jobs = int(metrics["jobs_with_valid_conductivity_mS_cm"]) - previous_valid_jobs

    final_coverage = np.nan
    if not coverage_report.empty:
        final_coverage = coverage_report["coverage_fraction"].iloc[0]

    print("\nValidation summary")
    print(f"Missing conductivity jobs remaining: {metrics['jobs_missing_conductivity_mS_cm']}")
    if additional_valid_jobs is None:
        print("Additional valid jobs compared with previous analysis: unavailable")
    else:
        print(f"Additional valid jobs compared with previous analysis: {additional_valid_jobs}")
    if pd.notna(final_coverage):
        print(f"Updated coverage_fraction: {final_coverage:.6f}")
    else:
        print("Updated coverage_fraction: unavailable")
    print(f"Regenerated figures: {out_dir}")


if __name__ == "__main__":
    main()
