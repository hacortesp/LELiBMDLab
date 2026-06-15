#!/usr/bin/env python3
"""Plot conductivity vs temperature with salt-colored markers and trend lines."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create conductivity-vs-temperature scatter plots using marker size for "
            "concentration and per-salt fitted trend lines."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("electrolyte_outputs/cleaned_electrolytes_no_zero_salt.csv"),
        help="Input electrolyte CSV.",
    )
    parser.add_argument(
        "--output-main",
        type=Path,
        default=Path("figures/conductivity_vs_temperature_by_salt.png"),
        help="Output path for the combined scatter figure.",
    )
    parser.add_argument(
        "--output-faceted",
        type=Path,
        default=Path("figures/conductivity_vs_temperature_by_salt_faceted.png"),
        help="Output path for the per-salt faceted figure.",
    )
    return parser.parse_args()


def read_rows(csv_path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"conductivity", "temperature", "salt_conc"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"Missing required columns: {missing_text}")

        for record in reader:
            try:
                conductivity = float(record["conductivity"])
                temperature = float(record["temperature"])
                salt_conc = float(record["salt_conc"])
            except (TypeError, ValueError):
                continue

            original_salt = (record.get("original_salt") or "").strip()
            if original_salt:
                salt_type = original_salt
            else:
                cation = (record.get("cation_name") or "").strip()
                anion = (record.get("anion_name") or "").strip()
                salt_type = f"{cation}{anion}" if cation or anion else "Unknown salt"

            rows.append(
                {
                    "salt_type": salt_type,
                    "temperature": temperature,
                    "conductivity": conductivity,
                    "salt_conc": salt_conc,
                }
            )
    if not rows:
        raise ValueError("No valid rows available after parsing numeric fields.")
    return rows


def scale_marker_sizes(
    concentrations: np.ndarray,
    size_min: float = 40.0,
    size_max: float = 260.0,
    c_min: float | None = None,
    c_max: float | None = None,
) -> np.ndarray:
    if c_min is None:
        c_min = float(np.min(concentrations))
    if c_max is None:
        c_max = float(np.max(concentrations))
    if np.isclose(c_min, c_max):
        return np.full_like(concentrations, fill_value=(size_min + size_max) / 2.0, dtype=float)
    scaled = (concentrations - c_min) / (c_max - c_min)
    return size_min + scaled * (size_max - size_min)


def build_main_figure(
    rows: list[dict[str, float | str]],
    output_path: Path,
    axis_limits: tuple[float, float, float, float],
) -> None:
    grouped: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["salt_type"])].append(row)

    salt_types = sorted(grouped)
    color_map = plt.get_cmap("tab10")
    colors = {salt: color_map(i % 10) for i, salt in enumerate(salt_types)}

    all_conc = np.array([float(row["salt_conc"]) for row in rows], dtype=float)
    conc_min = float(np.min(all_conc))
    conc_max = float(np.max(all_conc))
    conc_sizes = scale_marker_sizes(all_conc, c_min=conc_min, c_max=conc_max)
    for row, size in zip(rows, conc_sizes):
        row["marker_size"] = float(size)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    legend_handles: list[Line2D] = []

    for salt in salt_types:
        group = grouped[salt]
        x = np.array([float(r["temperature"]) for r in group], dtype=float)
        y = np.array([float(r["conductivity"]) for r in group], dtype=float)
        s = np.array([float(r["marker_size"]) for r in group], dtype=float)
        color = colors[salt]

        ax.scatter(x, y, s=s, alpha=0.7, color=color, edgecolor="black", linewidth=0.4)
        legend_handles.append(Line2D([0], [0], marker="o", color=color, label=salt, linestyle="", markersize=8))

    x_min, x_max, y_min, y_max = axis_limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Conductivity (mS/cm)")
    ax.set_title("Conductivity vs Temperature by Salt Type")
    salt_legend = ax.legend(handles=legend_handles, title="Salt type", loc="upper left")
    ax.add_artist(salt_legend)

    size_levels = np.linspace(np.min(all_conc), np.max(all_conc), 3)
    size_handles = []
    for level in size_levels:
        marker_size = float(scale_marker_sizes(np.array([level]), c_min=conc_min, c_max=conc_max)[0])
        size_handles.append(
            plt.scatter([], [], s=marker_size, color="gray", alpha=0.5, edgecolor="black", linewidth=0.4)
        )
    size_labels = [f"{level:.2f} mol/L" for level in size_levels]
    ax.legend(size_handles, size_labels, title="Salt concentration", loc="lower right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def build_faceted_figure(
    rows: list[dict[str, float | str]],
    output_path: Path,
    axis_limits: tuple[float, float, float, float],
) -> None:
    grouped: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["salt_type"])].append(row)

    salt_types = sorted(grouped)
    color_map = plt.get_cmap("tab10")
    colors = {salt: color_map(i % 10) for i, salt in enumerate(salt_types)}

    n_salts = len(salt_types)
    n_cols = 3
    n_rows = int(np.ceil(n_salts / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes_flat = np.atleast_1d(axes).ravel()

    x_min, x_max, y_min, y_max = axis_limits

    for idx, salt in enumerate(salt_types):
        ax = axes_flat[idx]
        group = grouped[salt]
        x = np.array([float(r["temperature"]) for r in group], dtype=float)
        y = np.array([float(r["conductivity"]) for r in group], dtype=float)
        s = np.array([float(r["marker_size"]) for r in group], dtype=float)
        color = colors[salt]

        ax.scatter(x, y, s=s, alpha=0.75, color=color, edgecolor="black", linewidth=0.4)
        ax.set_title(salt)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    for idx in range(n_salts, len(axes_flat)):
        axes_flat[idx].axis("off")

    for ax in axes_flat:
        if ax.has_data():
            ax.set_xlabel("Temperature (K)")
            ax.set_ylabel("Conductivity (mS/cm)")

    fig.suptitle("Conductivity Trends by Salt Type (Shared Axes)", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input_csv)

    temperatures = np.array([float(row["temperature"]) for row in rows], dtype=float)
    conductivities = np.array([float(row["conductivity"]) for row in rows], dtype=float)

    x_padding = max((temperatures.max() - temperatures.min()) * 0.04, 1.0)
    y_padding = max((conductivities.max() - conductivities.min()) * 0.06, 0.2)
    axis_limits = (
        float(temperatures.min() - x_padding),
        float(temperatures.max() + x_padding),
        max(0.0, float(conductivities.min() - y_padding)),
        float(conductivities.max() + y_padding),
    )

    build_main_figure(rows, args.output_main, axis_limits)
    build_faceted_figure(rows, args.output_faceted, axis_limits)

    print(f"Wrote main figure: {args.output_main}")
    print(f"Wrote faceted figure: {args.output_faceted}")


if __name__ == "__main__":
    main()
