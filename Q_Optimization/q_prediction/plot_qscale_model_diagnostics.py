#!/usr/bin/env python3
"""Generate diagnostic plots for the simplified q_scale model."""

import argparse
import os
import sys

os.environ.setdefault("PANDAS_NO_IMPORT_PYARROW", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")
sys.modules.setdefault("pyarrow", None)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from qscale_common import DEFAULT_INPUT_CSV, DEFAULT_OUTPUT_DIR, build_system_id, load_campaign_csv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot diagnostics for the simplified q_scale prediction workflow."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_CSV,
        help="Filtered conductivity campaign CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory containing q_prediction outputs.",
    )
    parser.add_argument(
        "--max-example-systems",
        type=int,
        default=6,
        help="Number of conductivity curve example systems.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for jitter and example selection.",
    )
    return parser.parse_args()


def q_label(value):
    return "{0:.2f}".format(float(value))


def load_required_outputs(output_dir):
    training = pd.read_csv(os.path.join(output_dir, "qscale_training_table.csv"))
    predictions = pd.read_csv(os.path.join(output_dir, "validation_predictions.csv"))
    feature_importance = pd.read_csv(os.path.join(output_dir, "feature_importance.csv"))
    confusion = pd.read_csv(os.path.join(output_dir, "confusion_matrix.csv"), index_col=0)
    return training, predictions, feature_importance, confusion


def plot_predicted_vs_true(predictions, output_dir, seed):
    rng = np.random.RandomState(seed)
    true_values = predictions["q_scale_optimal"].astype(float).values
    pred_values = predictions["predicted_q_scale"].astype(float).values
    true_jitter = true_values + rng.normal(0, 0.003, len(true_values))
    pred_jitter = pred_values + rng.normal(0, 0.003, len(pred_values))
    q_grid = sorted(set(true_values).union(set(pred_values)))

    plt.figure(figsize=(6.2, 5.2))
    plt.scatter(true_jitter, pred_jitter, alpha=0.8, edgecolor="black", linewidth=0.4)
    axis_min = min(q_grid) - 0.04
    axis_max = max(q_grid) + 0.04
    plt.plot([axis_min, axis_max], [axis_min, axis_max], color="black", linestyle="--", linewidth=1)
    plt.xticks(q_grid, [q_label(v) for v in q_grid])
    plt.yticks(q_grid, [q_label(v) for v in q_grid])
    plt.xlim(axis_min, axis_max)
    plt.ylim(axis_min, axis_max)
    plt.xlabel("True q_scale_optimal")
    plt.ylabel("Predicted q_scale")
    plt.title("Predicted vs true q_scale")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "predicted_vs_true_qscale.png"), dpi=300)
    plt.close()


def plot_confusion_matrix(confusion, output_dir):
    matrix = confusion.values
    labels = [label.replace("true_", "") for label in confusion.index]
    plt.figure(figsize=(6.2, 5.2))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar(label="Count")
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, str(int(matrix[i, j])), ha="center", va="center", color="black")
    plt.xlabel("Predicted q_scale")
    plt.ylabel("True q_scale")
    plt.title("Validation confusion matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.close()


def plot_feature_importance(feature_importance, output_dir):
    top = feature_importance.head(20).iloc[::-1]
    plt.figure(figsize=(8.5, 6.5))
    plt.barh(top["feature"], top["importance"], color="#4C78A8")
    plt.xlabel("Extra Trees feature importance")
    plt.title("Top q_scale model features")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=300)
    plt.close()


def plot_target_distribution(training, output_dir):
    counts = training["q_scale_class"].value_counts().sort_index()
    plt.figure(figsize=(6.2, 4.6))
    plt.bar(counts.index, counts.values, color="#54A24B")
    plt.xlabel("q_scale_optimal")
    plt.ylabel("Number of systems")
    plt.title("q_scale target distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "qscale_target_distribution.png"), dpi=300)
    plt.close()


def build_log_error_table(raw):
    df = raw.copy()
    df["system_id"] = build_system_id(df)
    df["q_scale_numeric"] = pd.to_numeric(df["q_scale"], errors="coerce")
    df["experimental_conductivity"] = pd.to_numeric(df["conductivity"], errors="coerce")
    df["calculated_conductivity"] = pd.to_numeric(df["conductivity_mS_cm"], errors="coerce")
    valid = (
        np.isfinite(df["q_scale_numeric"])
        & np.isfinite(df["experimental_conductivity"])
        & np.isfinite(df["calculated_conductivity"])
        & (df["experimental_conductivity"] > 0)
        & (df["calculated_conductivity"] > 0)
    )
    df = df.loc[valid].copy()
    df["log_error"] = (
        np.log(df["calculated_conductivity"]) - np.log(df["experimental_conductivity"])
    ).abs()
    return df


def plot_min_log_error_vs_qscale(error_table, output_dir):
    q_grid = sorted(error_table["q_scale_numeric"].dropna().unique())
    plt.figure(figsize=(7.5, 5.3))
    for _, group in error_table.groupby("system_id", sort=False):
        ordered = group.sort_values("q_scale_numeric")
        plt.plot(ordered["q_scale_numeric"], ordered["log_error"], color="#4C78A8", alpha=0.16, linewidth=0.8)
        best = ordered.loc[ordered["log_error"].idxmin()]
        plt.scatter([best["q_scale_numeric"]], [best["log_error"]], color="#E45756", s=10, alpha=0.35)
    mean_error = error_table.groupby("q_scale_numeric")["log_error"].mean().reindex(q_grid)
    plt.plot(q_grid, mean_error.values, color="black", linewidth=2.5, label="Mean log error")
    plt.xticks(q_grid, [q_label(v) for v in q_grid])
    plt.xlabel("q_scale")
    plt.ylabel("|log(conductivity_mS_cm) - log(conductivity)|")
    plt.title("Log-conductivity error across q_scale values")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "min_log_error_vs_qscale.png"), dpi=300)
    plt.close()


def build_min_log_error_by_system(error_table):
    records = []
    for system_id, group in error_table.groupby("system_id", sort=False):
        ordered = group.sort_values("q_scale_numeric").copy()
        min_error = float(ordered["log_error"].min())
        tied = ordered.loc[np.isclose(ordered["log_error"], min_error, rtol=1e-12, atol=1e-12)]
        best = tied.sort_values("q_scale_numeric").iloc[0]
        q_values = [q_label(value) for value in ordered["q_scale_numeric"].tolist()]
        calc_values = ["{0:.6g}".format(float(value)) for value in ordered["calculated_conductivity"].tolist()]
        records.append(
            {
                "system_id": system_id,
                "original_row_index": best["original_row_index"],
                "doi": best["doi"],
                "anion_name": best["anion_name"],
                "salt_conc": best["salt_conc"],
                "temperature": best["temperature"],
                "solvents": best["solvents"],
                "solvent_fracs": best["solvent_fracs"],
                "conductivity": best["conductivity"],
                "q_scale_optimal": float(best["q_scale_numeric"]),
                "min_log_error": min_error,
                "number_of_qscale_points": int(ordered["q_scale_numeric"].nunique()),
                "available_qscale_values": ";".join(q_values),
                "available_calculated_conductivities": ";".join(calc_values),
            }
        )
    summary = pd.DataFrame(records)
    summary = summary.sort_values(["anion_name", "min_log_error", "system_id"]).reset_index(drop=True)
    summary["system_plot_index"] = np.arange(1, len(summary) + 1)
    return summary


def marker_size_for_qscale(q_scale, q_min, q_max):
    if q_max <= q_min:
        return 90.0
    normalized = (float(q_scale) - q_min) / (q_max - q_min)
    return 55.0 + 170.0 * normalized


def plot_min_log_error_by_system(system_summary, output_dir):
    path = os.path.join(output_dir, "min_log_error_by_system.png")
    salts = sorted(system_summary["anion_name"].astype(str).unique())
    colors = {
        salt: color
        for salt, color in zip(
            salts,
            ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#9D755D"],
        )
    }
    q_values = sorted(system_summary["q_scale_optimal"].dropna().unique())
    q_min = min(q_values)
    q_max = max(q_values)
    sizes = system_summary["q_scale_optimal"].apply(lambda value: marker_size_for_qscale(value, q_min, q_max))

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for salt in salts:
        block = system_summary.loc[system_summary["anion_name"].astype(str) == salt]
        block_sizes = block["q_scale_optimal"].apply(lambda value: marker_size_for_qscale(value, q_min, q_max))
        ax.scatter(
            block["system_plot_index"],
            block["min_log_error"],
            s=block_sizes,
            color=colors[salt],
            edgecolor="black",
            linewidth=0.35,
            alpha=0.82,
            label=salt,
        )

    salt_blocks = system_summary.groupby("anion_name", sort=False)["system_plot_index"].agg(["min", "max"])
    for _, row in salt_blocks.iloc[:-1].iterrows():
        ax.axvline(float(row["max"]) + 0.5, color="0.82", linestyle="--", linewidth=1)

    tick_step = max(1, int(np.ceil(len(system_summary) / 8.0)))
    ticks = list(range(1, len(system_summary) + 1, tick_step))
    if len(system_summary) not in ticks:
        ticks.append(len(system_summary))
    ax.set_xticks(ticks)
    ax.set_xlabel("Electrolyte system index")
    ax.set_ylabel("min |log(sigma_calc) - log(sigma_exp)|")
    ax.set_title("Minimum log-conductivity error by electrolyte system")

    salt_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=colors[salt],
            markeredgecolor="black",
            markersize=7,
            label=salt,
        )
        for salt in salts
    ]
    salt_legend = ax.legend(handles=salt_handles, title="Salt / anion", loc="upper left", frameon=True)
    ax.add_artist(salt_legend)

    size_handles = [
        plt.scatter(
            [],
            [],
            s=marker_size_for_qscale(value, q_min, q_max),
            color="white",
            edgecolor="black",
            linewidth=0.5,
            label=q_label(value),
        )
        for value in q_values
    ]
    ax.legend(handles=size_handles, title="q_scale_optimal", loc="upper right", frameon=True)

    ax.grid(axis="y", color="0.9", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_conductivity_curve_examples(error_table, output_dir, max_examples, seed):
    systems = (
        error_table.groupby("system_id")["log_error"]
        .min()
        .sort_values()
        .index
        .tolist()
    )
    if len(systems) == 0:
        return
    rng = np.random.RandomState(seed)
    if len(systems) > max_examples:
        easy = systems[: max(1, max_examples // 2)]
        remaining = [s for s in systems if s not in easy]
        random_count = max_examples - len(easy)
        random_pick = list(rng.choice(remaining, size=random_count, replace=False))
        selected = easy + random_pick
    else:
        selected = systems

    n = len(selected)
    ncols = 3
    nrows = int(np.ceil(float(n) / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.8 * nrows), squeeze=False)
    for index, system_id in enumerate(selected):
        ax = axes[index // ncols][index % ncols]
        group = error_table.loc[error_table["system_id"] == system_id].sort_values("q_scale_numeric")
        exp_value = float(group["experimental_conductivity"].iloc[0])
        best = group.loc[group["log_error"].idxmin()]
        ax.plot(group["q_scale_numeric"], group["calculated_conductivity"], marker="o", label="Simulated")
        ax.axhline(exp_value, color="black", linestyle="--", linewidth=1, label="Experiment")
        ax.scatter([best["q_scale_numeric"]], [best["calculated_conductivity"]], color="#E45756", zorder=5)
        title = "{0}, row {1}".format(group["anion_name"].iloc[0], group["original_row_index"].iloc[0])
        ax.set_title(title)
        ax.set_xlabel("q_scale")
        ax.set_ylabel("conductivity_mS_cm")
        ax.set_xticks(sorted(group["q_scale_numeric"].unique()))
        ax.tick_params(axis="x", labelrotation=45)
    for index in range(n, nrows * ncols):
        axes[index // ncols][index % ncols].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(output_dir, "conductivity_curves_examples.png"), dpi=300)
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = os.path.abspath(args.output_dir)
    training, predictions, feature_importance, confusion = load_required_outputs(output_dir)
    raw = load_campaign_csv(args.input)
    error_table = build_log_error_table(raw)
    system_summary = build_min_log_error_by_system(error_table)
    system_summary.to_csv(os.path.join(output_dir, "min_log_error_by_system.csv"), index=False)

    plot_predicted_vs_true(predictions, output_dir, args.random_seed)
    plot_confusion_matrix(confusion, output_dir)
    plot_feature_importance(feature_importance, output_dir)
    plot_target_distribution(training, output_dir)
    plot_min_log_error_vs_qscale(error_table, output_dir)
    plot_min_log_error_by_system(system_summary, output_dir)
    plot_conductivity_curve_examples(error_table, output_dir, args.max_example_systems, args.random_seed)

    print("Saved diagnostic plots in {0}".format(output_dir))
    print("- predicted_vs_true_qscale.png")
    print("- confusion_matrix.png")
    print("- feature_importance.png")
    print("- qscale_target_distribution.png")
    print("- min_log_error_vs_qscale.png")
    print("- min_log_error_by_system.png")
    print("- conductivity_curves_examples.png")
    print("Saved per-system min log error data: min_log_error_by_system.csv")
    print("")
    print("Minimum log-error by system summary:")
    print("- unique systems in input: {0}".format(error_table["system_id"].nunique()))
    print("- systems plotted: {0}".format(len(system_summary)))
    print("- q_scale_optimal distribution:")
    for q_scale, count in system_summary["q_scale_optimal"].value_counts().sort_index().items():
        print("  q_scale {0}: {1}".format(q_label(q_scale), int(count)))
    print("- mean min_log_error: {0:.6f}".format(float(system_summary["min_log_error"].mean())))
    print("- median min_log_error: {0:.6f}".format(float(system_summary["min_log_error"].median())))
    print("- systems with largest min_log_error:")
    largest = system_summary.sort_values("min_log_error", ascending=False).head(5)
    for _, row in largest.iterrows():
        print(
            "  {0} | row {1} | {2} | q={3} | min_log_error={4:.6f}".format(
                row["system_id"],
                row["original_row_index"],
                row["anion_name"],
                q_label(row["q_scale_optimal"]),
                float(row["min_log_error"]),
            )
        )


if __name__ == "__main__":
    main()
