#!/usr/bin/env python3
"""Build a sparse q_scale simulation campaign from clustering results.

Step D selects a compact set of representative electrolyte systems from the
best clustering criterion, then expands each selected system over a sparse
q_scale grid. The selector always includes the cluster representative from the
cluster summary, then adds diverse systems by maximin spread in descriptor
space while avoiding duplicate simulation inputs.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import shlex
from pathlib import Path

import numpy as np

with contextlib.redirect_stderr(io.StringIO()):
    import pandas as pd


RANDOM_SEED = 42
DEFAULT_CLUSTER_ALLOCATION = {1: 4, 4: 3, 3: 2, 2: 2, 6: 1, 5: 1}
DEFAULT_CLUSTER_RANK_WEIGHTS = (4, 3, 2, 2, 1, 1)
DEFAULT_Q_SCALE_GRID = [0.70, 0.75, 0.80, 0.85, 0.90]
SIMULATION_INPUT_COLUMNS = [
    "cation_name",
    "anion_name",
    "salt_conc",
    "solvents",
    "solvent_fracs",
    "temperature",
]
SELECTION_FEATURES = [
    "weighted_permittivity",
    "weighted_viscosity_mPa_s_25C",
    "salt_conc_x_weighted_permittivity",
    "salt_conc_x_weighted_viscosity",
    "conductivity",
]
OUTPUT_COLUMNS = [
    "cluster_id",
    "original_row_index",
    "doi",
    "representative_doi",
    "cation_name",
    "anion_name",
    "salt_conc",
    "solvents",
    "solvent_fracs",
    "temperature",
    "q_scale",
    "cluster_rank_in_sampling",
    "selection_reason",
]


def load_data(
    cluster_summary_path: Path,
    cluster_labels_path: Path,
    cleaned_input_path: Path,
    descriptors_input_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all tables needed to build the campaign."""

    cluster_summary = pd.read_csv(cluster_summary_path)
    cluster_labels = pd.read_csv(cluster_labels_path)
    cleaned = pd.read_csv(cleaned_input_path)
    descriptors = pd.read_csv(descriptors_input_path)
    return cluster_summary, cluster_labels, cleaned, descriptors


def parse_cluster_allocation(value: str | None) -> dict[int, int] | None:
    """Parse a cluster allocation string such as ``1:4,4:3,3:2``."""

    if not value:
        return None
    allocation: dict[int, int] = {}
    for item in value.split(","):
        cluster_text, count_text = item.split(":", maxsplit=1)
        allocation[int(cluster_text.strip())] = int(count_text.strip())
    return allocation


def infer_cluster_allocation(
    cluster_summary: pd.DataFrame,
    total_selected_systems: int = sum(DEFAULT_CLUSTER_ALLOCATION.values()),
) -> dict[int, int]:
    """Allocate representative counts across current clusters by cluster size.

    The historical weighted-permittivity campaign selected 13 systems using
    rank weights ``4,3,2,2,1,1`` over the six clusters sorted by size. This
    keeps that policy but maps it onto whatever cluster IDs the rebuilt
    clustering actually produces.
    """

    require_columns(cluster_summary, ["cluster_id", "cluster_size"], "cluster summary")
    ranked = (
        cluster_summary.loc[:, ["cluster_id", "cluster_size"]]
        .copy()
        .sort_values(["cluster_size", "cluster_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    if ranked.empty:
        raise ValueError("Cluster summary is empty; cannot infer sparse campaign allocation.")

    n_clusters = len(ranked)
    if total_selected_systems < n_clusters:
        raise ValueError(
            f"Cannot select {total_selected_systems} systems while covering {n_clusters} clusters."
        )

    weights = np.array(
        [
            DEFAULT_CLUSTER_RANK_WEIGHTS[index]
            if index < len(DEFAULT_CLUSTER_RANK_WEIGHTS)
            else DEFAULT_CLUSTER_RANK_WEIGHTS[-1]
            for index in range(n_clusters)
        ],
        dtype=float,
    )
    exact = total_selected_systems * weights / weights.sum()
    counts = np.floor(exact).astype(int)
    counts = np.maximum(counts, 1)

    while counts.sum() > total_selected_systems:
        removable = np.where(counts > 1)[0]
        if len(removable) == 0:
            break
        counts[removable[-1]] -= 1

    remainders = exact - np.floor(exact)
    order = np.lexsort((np.arange(n_clusters), -remainders))
    for index in order:
        if counts.sum() >= total_selected_systems:
            break
        counts[index] += 1

    return {
        int(row["cluster_id"]): int(count)
        for (_, row), count in zip(ranked.iterrows(), counts)
    }


def normalize_json_text(value: object) -> str:
    """Return stable compact JSON text when possible, otherwise a clean string."""

    if pd.isna(value):
        return ""
    text = str(value)
    try:
        return json.dumps(json.loads(text), separators=(",", ":"))
    except json.JSONDecodeError:
        return text


def simulation_identity(row: pd.Series) -> tuple[object, ...]:
    """Return the exact simulation-input identity used to avoid duplicates."""

    return (
        row.get("cation_name", ""),
        row.get("anion_name", ""),
        round(float(row.get("salt_conc", np.nan)), 12),
        normalize_json_text(row.get("solvents", "")),
        normalize_json_text(row.get("solvent_fracs", "")),
        round(float(row.get("temperature", np.nan)), 8),
    )


def require_columns(frame: pd.DataFrame, columns: list[str], table_name: str) -> None:
    """Raise a clear error if a required table column is absent."""

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {', '.join(missing)}")


def build_analysis_table(
    cluster_labels: pd.DataFrame,
    cleaned: pd.DataFrame,
    descriptors: pd.DataFrame,
) -> pd.DataFrame:
    """Combine cluster labels, descriptors, and simulation-ready input fields."""

    require_columns(cluster_labels, ["original_row_index", "cluster_id"], "cluster labels")
    require_columns(cleaned, ["original_row_index", "doi", *SIMULATION_INPUT_COLUMNS], "cleaned electrolyte table")
    require_columns(descriptors, ["original_row_index", *SELECTION_FEATURES], "descriptor table")

    label_columns = ["source_row_number", "original_row_index", "cluster_id"]
    label_columns = [column for column in label_columns if column in cluster_labels.columns]
    labels = cluster_labels.loc[:, label_columns].copy()
    labels["cluster_id"] = pd.to_numeric(labels["cluster_id"], errors="coerce").astype("Int64")

    descriptor_columns = ["original_row_index", *SELECTION_FEATURES]
    descriptor_values = descriptors.loc[:, descriptor_columns].copy()
    for column in SELECTION_FEATURES:
        descriptor_values[column] = pd.to_numeric(descriptor_values[column], errors="coerce")

    cleaned_columns = [
        "original_row_index",
        "doi",
        "composition_summary",
        *SIMULATION_INPUT_COLUMNS,
    ]
    cleaned_columns = [column for column in cleaned_columns if column in cleaned.columns]
    merged = labels.merge(descriptor_values, on="original_row_index", how="left")
    merged = merged.merge(cleaned.loc[:, cleaned_columns], on="original_row_index", how="left")
    merged["duplicate_identity"] = merged.apply(simulation_identity, axis=1)
    return merged


def scaled_feature_matrix(rows: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    """Return a within-cluster standardized descriptor matrix for selection."""

    matrix = rows.loc[:, feature_columns].to_numpy(dtype=float)
    means = np.nanmean(matrix, axis=0)
    matrix = np.where(np.isnan(matrix), means, matrix)
    stds = np.nanstd(matrix, axis=0)
    stds = np.where(stds < 1e-12, 1.0, stds)
    return (matrix - means) / stds


def representative_fallback(cluster_rows: pd.DataFrame, matrix: np.ndarray) -> int:
    """Choose the row nearest the cluster centroid if summary representative is unavailable."""

    centroid = matrix.mean(axis=0)
    distances = np.linalg.norm(matrix - centroid, axis=1)
    return int(np.argmin(distances))


def add_selection_metadata(row: pd.Series, rank: int, reason: str, cluster_size: int) -> dict[str, object]:
    """Convert a selected row into a serializable selected-system record."""

    record = row.to_dict()
    record["cluster_rank_in_sampling"] = int(rank)
    record["selection_reason"] = reason
    record["cluster_size"] = int(cluster_size)
    record["representative_doi"] = row.get("doi", "")
    return record


def select_cluster_systems(
    cluster_rows: pd.DataFrame,
    cluster_summary_row: pd.Series,
    n_select: int,
    feature_columns: list[str],
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    """Select representative and diverse non-duplicate systems from one cluster."""

    cluster_id = int(cluster_summary_row["cluster_id"])
    cluster_rows = cluster_rows.dropna(subset=["cluster_id"]).copy()
    cluster_rows = cluster_rows.drop_duplicates(subset=["duplicate_identity"], keep="first").reset_index(drop=True)
    cluster_size = int(cluster_summary_row.get("cluster_size", len(cluster_rows)))
    if cluster_rows.empty:
        raise ValueError(f"Cluster {cluster_id} has no candidate rows.")

    matrix = scaled_feature_matrix(cluster_rows, feature_columns)
    representative_row_index = int(cluster_summary_row["representative_row_index"])
    representative_matches = cluster_rows.index[cluster_rows["original_row_index"] == representative_row_index].to_list()
    if representative_matches:
        representative_pos = representative_matches[0]
        representative_reason = "representative_row_index from cluster_summary.csv"
    else:
        representative_pos = representative_fallback(cluster_rows, matrix)
        representative_reason = "centroid-nearest fallback because representative_row_index was not in cluster labels"

    selected_positions = [representative_pos]
    selected_records = [
        add_selection_metadata(
            cluster_rows.iloc[representative_pos],
            1,
            representative_reason,
            cluster_size,
        )
    ]

    while len(selected_records) < n_select and len(selected_positions) < len(cluster_rows):
        selected_matrix = matrix[selected_positions]
        candidate_positions = [pos for pos in range(len(cluster_rows)) if pos not in selected_positions]
        best_tuple: tuple[float, float, float, float] | None = None
        best_position: int | None = None
        selected_compositions = set(cluster_rows.iloc[selected_positions]["composition_summary"].fillna("").astype(str))
        selected_permittivities = set(
            round(float(value), 8) for value in cluster_rows.iloc[selected_positions]["weighted_permittivity"]
        )

        for pos in candidate_positions:
            candidate = cluster_rows.iloc[pos]
            distances = np.linalg.norm(selected_matrix - matrix[pos], axis=1)
            min_distance = float(np.min(distances))
            distinct_composition = float(str(candidate.get("composition_summary", "")) not in selected_compositions)
            distinct_permittivity = float(round(float(candidate["weighted_permittivity"]), 8) not in selected_permittivities)
            tie_jitter = float(rng.random() * 1e-9)
            score_tuple = (min_distance, distinct_composition, distinct_permittivity, tie_jitter)
            if best_tuple is None or score_tuple > best_tuple:
                best_tuple = score_tuple
                best_position = pos

        assert best_position is not None
        selected_positions.append(best_position)
        reason = (
            "maximin descriptor spread from existing selections using "
            + ", ".join(feature_columns)
            + "; duplicate simulation inputs excluded"
        )
        selected_records.append(
            add_selection_metadata(
                cluster_rows.iloc[best_position],
                len(selected_records) + 1,
                reason,
                cluster_size,
            )
        )

    if len(selected_records) < n_select:
        raise ValueError(
            f"Cluster {cluster_id} requested {n_select} systems, but only "
            f"{len(selected_records)} unique simulation inputs were available."
        )
    return selected_records


def select_representatives(
    analysis_table: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    allocation: dict[int, int] | None,
    random_seed: int,
) -> pd.DataFrame:
    """Select the sparse representative systems according to cluster allocation."""

    rng = np.random.default_rng(random_seed)
    selected: list[dict[str, object]] = []
    allocation_order = {cluster_id: order for order, cluster_id in enumerate(allocation)}
    summary_by_cluster = cluster_summary.set_index("cluster_id", drop=False)
    for cluster_id, n_select in allocation.items():
        if cluster_id not in summary_by_cluster.index:
            raise ValueError(f"Cluster {cluster_id} is not present in cluster summary.")
        cluster_rows = analysis_table[analysis_table["cluster_id"] == cluster_id].copy()
        selected.extend(
            select_cluster_systems(
                cluster_rows,
                summary_by_cluster.loc[cluster_id],
                n_select,
                SELECTION_FEATURES,
                rng,
            )
        )

    selected_df = pd.DataFrame(selected)
    selected_df["cluster_id"] = selected_df["cluster_id"].astype(int)
    selected_df["original_row_index"] = selected_df["original_row_index"].astype(int)
    selected_df["cluster_sampling_order"] = selected_df["cluster_id"].map(allocation_order).astype(int)
    selected_df = selected_df.sort_values(["cluster_sampling_order", "cluster_rank_in_sampling"]).reset_index(drop=True)
    return selected_df


def build_campaign_plan(selected: pd.DataFrame, q_scale_grid: list[float]) -> pd.DataFrame:
    """Expand selected systems into one planned simulation row per q_scale value."""

    jobs: list[dict[str, object]] = []
    for _, row in selected.iterrows():
        for q_scale in q_scale_grid:
            job = row.to_dict()
            job["q_scale"] = float(q_scale)
            job["job_id"] = (
                f"cluster{int(row['cluster_id']):02d}_"
                f"rank{int(row['cluster_rank_in_sampling']):02d}_"
                f"row{int(row['original_row_index'])}_"
                f"q{q_scale:.2f}".replace(".", "p")
            )
            jobs.append(job)
    campaign = pd.DataFrame(jobs)
    front_columns = ["job_id", *OUTPUT_COLUMNS]
    remaining_columns = [column for column in campaign.columns if column not in front_columns]
    return campaign.loc[:, front_columns + remaining_columns]


def build_campaign_summary(
    selected: pd.DataFrame,
    campaign: pd.DataFrame,
    q_scale_grid: list[float],
    allocation: dict[int, int],
) -> pd.DataFrame:
    """Build counts by cluster and q_scale plus campaign-wide coverage stats."""

    total_planned = int(len(campaign))
    total_unique_systems = int(len(selected))
    records: list[dict[str, object]] = []
    for cluster_id in allocation:
        cluster_selected = selected[selected["cluster_id"] == cluster_id]
        cluster_size = int(cluster_selected["cluster_size"].iloc[0]) if not cluster_selected.empty else 0
        coverage = len(cluster_selected) / cluster_size if cluster_size else np.nan
        for q_scale in q_scale_grid:
            planned_at_q = campaign[
                (campaign["cluster_id"] == cluster_id)
                & np.isclose(campaign["q_scale"].astype(float), float(q_scale))
            ]
            records.append(
                {
                    "cluster_id": int(cluster_id),
                    "q_scale": float(q_scale),
                    "selected_systems_in_cluster": int(len(cluster_selected)),
                    "planned_simulations_at_q_scale": int(len(planned_at_q)),
                    "q_scale_values_per_system": int(len(q_scale_grid)),
                    "planned_simulations_in_cluster": int(len(cluster_selected) * len(q_scale_grid)),
                    "original_cluster_size": cluster_size,
                    "cluster_coverage_fraction": coverage,
                    "total_planned_simulations": total_planned,
                    "total_unique_systems": total_unique_systems,
                    "clusters_covered": int(selected["cluster_id"].nunique()),
                    "cluster_allocation_target": int(allocation[cluster_id]),
                }
            )
    return pd.DataFrame(records)


def write_launcher_script(campaign: pd.DataFrame, output_path: Path, simulation_command: str | None) -> None:
    """Write a simple shell launcher or dry-run template for campaign jobs."""

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated sparse q_scale campaign launcher.",
    ]
    if not simulation_command:
        lines.extend(
            [
                "# Pass --simulation-command to build_sparse_campaign.py to emit executable commands.",
                "# This dry-run template prints the planned simulation inputs.",
                "",
            ]
        )
    else:
        lines.append("")

    for _, row in campaign.iterrows():
        prefix = "echo" if not simulation_command else simulation_command
        lines.append(
            " ".join(
                [
                    prefix,
                    f"--job-id {shlex.quote(str(row['job_id']))}",
                    f"--cation-name {shlex.quote(str(row['cation_name']))}",
                    f"--anion-name {shlex.quote(str(row['anion_name']))}",
                    f"--salt-conc {shlex.quote(str(row['salt_conc']))}",
                    f"--solvents {shlex.quote(str(row['solvents']))}",
                    f"--solvent-fracs {shlex.quote(str(row['solvent_fracs']))}",
                    f"--temperature {shlex.quote(str(row['temperature']))}",
                    f"--q-scale {row['q_scale']:.2f}",
                ]
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output_path.chmod(0o755)


def write_outputs(
    selected: pd.DataFrame,
    campaign: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    launcher_output: Path | None,
    simulation_command: str | None,
) -> None:
    """Write selected systems, job campaign, summary, and optional launcher."""

    output_dir.mkdir(parents=True, exist_ok=True)
    selected_front_columns = [
        "cluster_sampling_order",
        "cluster_id",
        "cluster_rank_in_sampling",
        "original_row_index",
        "doi",
        "representative_doi",
        *SIMULATION_INPUT_COLUMNS,
        "selection_reason",
        "cluster_size",
        *SELECTION_FEATURES,
    ]
    selected_remaining = [column for column in selected.columns if column not in selected_front_columns]
    selected.loc[:, selected_front_columns + selected_remaining].to_csv(
        output_dir / "selected_representatives.csv",
        index=False,
    )
    campaign.to_csv(output_dir / "campaign_plan.csv", index=False)
    summary.to_csv(output_dir / "campaign_summary.csv", index=False)
    if launcher_output is not None:
        launcher_path = launcher_output if launcher_output.is_absolute() else output_dir / launcher_output
        write_launcher_script(campaign, launcher_path, simulation_command)


def run(
    cluster_summary_path: Path,
    cluster_labels_path: Path,
    cleaned_input_path: Path,
    descriptors_input_path: Path,
    output_dir: Path,
    q_scale_grid: list[float],
    allocation: dict[int, int],
    random_seed: int,
    launcher_output: Path | None,
    simulation_command: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build and write the sparse campaign."""

    cluster_summary, cluster_labels, cleaned, descriptors = load_data(
        cluster_summary_path,
        cluster_labels_path,
        cleaned_input_path,
        descriptors_input_path,
    )
    if allocation is None:
        allocation = infer_cluster_allocation(cluster_summary)
    analysis_table = build_analysis_table(cluster_labels, cleaned, descriptors)
    selected = select_representatives(analysis_table, cluster_summary, allocation, random_seed)
    campaign = build_campaign_plan(selected, q_scale_grid)
    summary = build_campaign_summary(selected, campaign, q_scale_grid, allocation)
    write_outputs(selected, campaign, summary, output_dir, launcher_output, simulation_command)
    return selected, campaign, summary


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cluster-summary",
        type=Path,
        default=Path("results/criterion_weighted_permittivity/cluster_summary.csv"),
        help="Weighted-permittivity cluster summary CSV.",
    )
    parser.add_argument(
        "--cluster-labels",
        type=Path,
        default=Path("results/criterion_weighted_permittivity/cluster_labels.csv"),
        help="Weighted-permittivity cluster labels CSV.",
    )
    parser.add_argument(
        "--cleaned-input",
        type=Path,
        default=Path("electrolyte_outputs/cleaned_electrolytes.csv"),
        help="Cleaned simulation-ready electrolyte table.",
    )
    parser.add_argument(
        "--descriptors-input",
        type=Path,
        default=Path("electrolyte_outputs/descriptors.csv"),
        help="Descriptor table used for diversity selection.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/sparse_campaign_weighted_permittivity"),
        help="Output directory for campaign files.",
    )
    parser.add_argument(
        "--q-scale-grid",
        type=float,
        nargs="+",
        default=DEFAULT_Q_SCALE_GRID,
        help="q_scale values for each selected system.",
    )
    parser.add_argument(
        "--cluster-allocation",
        type=str,
        default=None,
        help="Optional cluster allocation like '1:4,4:3,3:2,2:2,6:1,5:1'.",
    )
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED, help="Fixed seed for tie-breaking.")
    parser.add_argument(
        "--launcher-output",
        type=Path,
        default=Path("launch_sparse_campaign.sh"),
        help="Optional shell launcher path. Use an absolute path or a path relative to output-dir.",
    )
    parser.add_argument(
        "--simulation-command",
        type=str,
        default=None,
        help="Command prefix for launcher lines. If omitted, launcher is a dry-run echo template.",
    )
    return parser


def main() -> None:
    """Command-line entry point."""

    args = build_parser().parse_args()
    if any(q_scale < 0.70 for q_scale in args.q_scale_grid):
        raise ValueError("q_scale values below 0.70 are not allowed for this campaign.")
    allocation = parse_cluster_allocation(args.cluster_allocation)
    selected, campaign, _ = run(
        args.cluster_summary,
        args.cluster_labels,
        args.cleaned_input,
        args.descriptors_input,
        args.output_dir,
        list(args.q_scale_grid),
        allocation,
        args.random_seed,
        args.launcher_output,
        args.simulation_command,
    )
    print(f"Wrote sparse campaign outputs to: {args.output_dir}")
    print(f"Selected systems: {len(selected)}")
    print(f"Planned simulations: {len(campaign)}")
    print(f"q_scale grid: {', '.join(f'{value:.2f}' for value in args.q_scale_grid)}")


if __name__ == "__main__":
    main()
