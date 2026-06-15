#!/usr/bin/env python3
"""Build a salt-family sparse q_scale simulation campaign.

The campaign clusters electrolyte systems by canonical salt anion family
(PF6, ClO4, TFSI, FSI, BF4), selects a larger chemically diverse subset from
``cleaned_electrolytes_no_zero_salt.csv``, and expands each selected system
over the fixed q_scale grid used by the existing sparse-campaign workflow.
"""

import argparse
import csv
import json
import math
import os
import shutil
import shlex
from pathlib import Path


CANONICAL_SALTS = ("PF6", "ClO4", "TFSI", "FSI", "BF4")
DEFAULT_Q_SCALE_GRID = (0.70, 0.75, 0.80, 0.85, 0.90)
DEFAULT_TARGET_SYSTEMS = 75
DEFAULT_MIN_REPRESENTATIVES = 5

SIMULATION_INPUT_COLUMNS = (
    "cation_name",
    "anion_name",
    "salt_conc",
    "solvents",
    "solvent_fracs",
    "temperature",
)

SELECTION_FEATURES = (
    "salt_conc",
    "temperature",
    "weighted_permittivity",
    "weighted_viscosity_mPa_s_25C",
    "solvent_entropy",
    "salt_conc_x_weighted_permittivity",
    "salt_conc_x_weighted_viscosity",
    "frac_EC",
    "frac_PC",
    "frac_DME",
    "frac_DMC",
    "frac_DEC",
    "frac_EMC",
    "conductivity",
)

SELECTED_FRONT_COLUMNS = (
    "sampling_order",
    "salt_cluster_id",
    "cluster_id",
    "cluster_rank_in_sampling",
    "original_row_index",
    "doi",
    "representative_doi",
    "cation_name",
    "anion_name",
    "salt_conc",
    "solvents",
    "solvent_fracs",
    "temperature",
    "selection_reason",
    "cluster_size",
    "weighted_permittivity",
    "weighted_viscosity_mPa_s_25C",
    "solvent_entropy",
    "salt_conc_x_weighted_permittivity",
    "salt_conc_x_weighted_viscosity",
    "conductivity",
    "composition_summary",
    "duplicate_identity",
)

CAMPAIGN_FRONT_COLUMNS = (
    "job_id",
    "original_row_index",
    "doi",
    "cation_name",
    "anion_name",
    "salt_conc",
    "solvents",
    "solvent_fracs",
    "temperature",
    "salt_cluster_id",
    "q_scale",
    "selection_reason",
    "sampling_order",
    "cluster_id",
    "cluster_rank_in_sampling",
    "representative_doi",
)


COLLECT_CONDUCTIVITY_RESULTS = r'''#!/usr/bin/env python3
"""Collect ionic conductivity values from completed sparse campaign folders.

Each simulation folder is expected to contain ``output_LELiB_MDLab.txt`` with a
line like:

    Ionic conductivity = 8.6627 +/- 1.4598 mS/cm

The script reads ``campaign_plan.csv``, derives the same folder names used by
``submit_sparse_campaign.sh``, parses each output file, and writes one CSV row
per planned simulation.
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from parse_sparse_campaign import make_unique_folders, normalize_key, validate_row


OUTPUT_FILE_NAME = "output_LELiB_MDLab.txt"
FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
CONDUCTIVITY_PATTERN = re.compile(
    r"Ionic\s+conductivity\s*=\s*({0})(?:\s*(?:±|\+/-|\+-)\s*({0}))?\s*mS\s*/\s*cm".format(
        FLOAT_PATTERN
    ),
    re.IGNORECASE,
)


def read_campaign_rows(campaign_file):
    # type: (Path) -> List[Dict[str, str]]
    """Read campaign rows while retaining all campaign metadata columns."""

    rows = []
    with campaign_file.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for line_number, raw_row in enumerate(reader, start=2):
            row = {
                normalize_key(key): (value or "").strip()
                for key, value in raw_row.items()
                if key is not None
            }
            validate_row(row, campaign_file, line_number)
            rows.append(row)
    return rows


def parse_conductivity(output_file):
    # type: (Path) -> Optional[Tuple[str, str, str]]
    """Return conductivity, uncertainty, and raw line from an output file."""

    latest_match = None
    for line in output_file.read_text(errors="replace").splitlines():
        match = CONDUCTIVITY_PATTERN.search(line)
        if match:
            latest_match = (
                match.group(1),
                match.group(2) or "",
                line.strip(),
            )
    return latest_match


def collect_results(campaign_file, output_root, output_file_name):
    # type: (Path, Path, str) -> List[Dict[str, str]]
    """Collect parsed conductivity values for every planned campaign row."""

    rows = read_campaign_rows(campaign_file)
    folders = make_unique_folders(rows)
    collected = []

    for campaign_row, folder_pair in zip(rows, folders):
        folder_name = folder_pair[1]
        folder = output_root / folder_name
        output_file = folder / output_file_name
        result = dict(campaign_row)
        result["folder_name"] = folder_name
        result["folder_path"] = str(folder)
        result["output_file"] = str(output_file)
        result["conductivity_mS_cm"] = ""
        result["conductivity_uncertainty_mS_cm"] = ""
        result["conductivity_raw_line"] = ""

        if not folder.is_dir():
            result["collection_status"] = "missing_folder"
        elif not output_file.is_file():
            result["collection_status"] = "missing_output_file"
        else:
            parsed = parse_conductivity(output_file)
            if parsed is None:
                result["collection_status"] = "conductivity_line_not_found"
            else:
                conductivity, uncertainty, raw_line = parsed
                result["conductivity_mS_cm"] = conductivity
                result["conductivity_uncertainty_mS_cm"] = uncertainty
                result["conductivity_raw_line"] = raw_line
                result["collection_status"] = "parsed"

        collected.append(result)

    return collected


def output_columns(rows):
    # type: (Iterable[Dict[str, str]]) -> List[str]
    """Return stable output columns with parsed fields up front."""

    front = [
        "job_id",
        "folder_name",
        "collection_status",
        "conductivity_mS_cm",
        "conductivity_uncertainty_mS_cm",
        "conductivity_raw_line",
        "folder_path",
        "output_file",
    ]
    seen = set(front)
    columns = list(front)
    for row in rows:
        for column in row:
            if column not in seen:
                columns.append(column)
                seen.add(column)
    return columns


def write_results(rows, output_csv):
    # type: (List[Dict[str, str]], Path) -> None
    """Write collected results to CSV."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    columns = output_columns(rows)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows, output_csv):
    # type: (List[Dict[str, str]], Path) -> None
    """Print a compact collection summary."""

    parsed = sum(row["collection_status"] == "parsed" for row in rows)
    missing_folder = sum(row["collection_status"] == "missing_folder" for row in rows)
    missing_output = sum(row["collection_status"] == "missing_output_file" for row in rows)
    missing_line = sum(row["collection_status"] == "conductivity_line_not_found" for row in rows)
    print("Wrote conductivity collection: {0}".format(output_csv))
    print("planned_simulations: {0}".format(len(rows)))
    print("parsed_conductivity_rows: {0}".format(parsed))
    print("missing_folders: {0}".format(missing_folder))
    print("missing_output_files: {0}".format(missing_output))
    print("outputs_without_conductivity_line: {0}".format(missing_line))


def build_parser():
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-file",
        type=Path,
        default=SCRIPT_DIR / "campaign_plan.csv",
        help="Campaign plan CSV. Default: campaign_plan.csv next to this script.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=SCRIPT_DIR,
        help="Directory containing the per-simulation folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "conductivity_results.csv",
        help="CSV file to write with collected conductivity values.",
    )
    parser.add_argument(
        "--output-file-name",
        default=OUTPUT_FILE_NAME,
        help="Simulation output filename inside each folder. Default: {0}".format(OUTPUT_FILE_NAME),
    )
    return parser


def main():
    """Command-line entry point."""

    args = build_parser().parse_args()
    if not args.campaign_file.is_file():
        raise SystemExit("Campaign file not found: {0}".format(args.campaign_file))

    rows = collect_results(args.campaign_file, args.output_root, args.output_file_name)
    write_results(rows, args.output)
    print_summary(rows, args.output)


if __name__ == "__main__":
    main()
'''


def read_csv_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path, rows, front_columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    all_columns = []
    seen = set()
    for column in front_columns:
        if column not in seen:
            all_columns.append(column)
            seen.add(column)
    for row in rows:
        for column in row:
            if column not in seen:
                all_columns.append(column)
                seen.add(column)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def to_float(value, default=float("nan")):
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value, default=0):
    try:
        if value is None or str(value).strip() == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def canonical_json_text(value):
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        return json.dumps(json.loads(text), separators=(",", ":"))
    except (TypeError, ValueError):
        return text


def simulation_identity(row):
    return (
        row.get("cation_name", ""),
        row.get("anion_name", ""),
        round(to_float(row.get("salt_conc")), 12),
        canonical_json_text(row.get("solvents", "")),
        canonical_json_text(row.get("solvent_fracs", "")),
        round(to_float(row.get("temperature")), 8),
    )


def stable_identity_text(identity):
    return repr(identity)


def require_columns(rows, columns, table_name):
    if not rows:
        raise ValueError("{0} is empty.".format(table_name))
    missing = [column for column in columns if column not in rows[0]]
    if missing:
        raise ValueError(
            "{0} is missing required columns: {1}".format(table_name, ", ".join(missing))
        )


def load_analysis_rows(cleaned_input, descriptors_input):
    cleaned = read_csv_rows(cleaned_input)
    descriptors = read_csv_rows(descriptors_input)
    require_columns(
        cleaned,
        ("original_row_index", "doi") + SIMULATION_INPUT_COLUMNS,
        "cleaned electrolyte table",
    )
    descriptor_feature_columns = tuple(
        column for column in SELECTION_FEATURES if column not in ("salt_conc", "temperature")
    )
    require_columns(descriptors, ("original_row_index",) + descriptor_feature_columns, "descriptor table")

    descriptors_by_index = {
        row["original_row_index"]: row
        for row in descriptors
        if row.get("original_row_index", "").strip()
    }

    merged = []
    seen_identities = set()
    for row in cleaned:
        salt = row.get("anion_name", "").strip()
        if salt not in CANONICAL_SALTS:
            continue

        merged_row = dict(row)
        descriptor_row = descriptors_by_index.get(row.get("original_row_index", ""), {})
        for key, value in descriptor_row.items():
            if key not in merged_row or merged_row.get(key, "") == "":
                merged_row[key] = value

        identity = simulation_identity(merged_row)
        if identity in seen_identities:
            continue
        seen_identities.add(identity)
        merged_row["duplicate_identity"] = stable_identity_text(identity)
        merged_row["salt_cluster_id"] = salt
        merged.append(merged_row)

    return merged


def cluster_rows_by_salt(rows):
    clusters = {salt: [] for salt in CANONICAL_SALTS}
    for row in rows:
        clusters[row["salt_cluster_id"]].append(row)
    return clusters


def allocate_sampling_quotas(clusters, target_systems, minimum_representatives):
    if target_systems < len(CANONICAL_SALTS) * minimum_representatives:
        raise ValueError("Target systems is too small for the requested per-family minimum.")

    total = sum(len(clusters[salt]) for salt in CANONICAL_SALTS)
    if total == 0:
        raise ValueError("No canonical salt-family rows were found.")

    exact = {}
    counts = {}
    for salt in CANONICAL_SALTS:
        exact_count = target_systems * (float(len(clusters[salt])) / float(total))
        exact[salt] = exact_count
        counts[salt] = max(minimum_representatives, int(math.floor(exact_count)))
        if counts[salt] > len(clusters[salt]):
            counts[salt] = len(clusters[salt])

    while sum(counts.values()) < target_systems:
        candidates = [
            salt for salt in CANONICAL_SALTS
            if counts[salt] < len(clusters[salt])
        ]
        if not candidates:
            break
        candidates.sort(
            key=lambda salt: (
                exact[salt] - math.floor(exact[salt]),
                len(clusters[salt]),
                -CANONICAL_SALTS.index(salt),
            ),
            reverse=True,
        )
        counts[candidates[0]] += 1

    while sum(counts.values()) > target_systems:
        candidates = [
            salt for salt in CANONICAL_SALTS
            if counts[salt] > minimum_representatives
        ]
        if not candidates:
            break
        candidates.sort(
            key=lambda salt: (
                counts[salt] - exact[salt],
                -len(clusters[salt]),
                CANONICAL_SALTS.index(salt),
            ),
            reverse=True,
        )
        counts[candidates[0]] -= 1

    return counts, exact


def feature_matrix(rows):
    raw_matrix = []
    for row in rows:
        raw_matrix.append([to_float(row.get(column)) for column in SELECTION_FEATURES])

    if not raw_matrix:
        return []

    column_count = len(SELECTION_FEATURES)
    means = []
    stds = []
    for column_index in range(column_count):
        values = [
            values[column_index]
            for values in raw_matrix
            if not math.isnan(values[column_index])
        ]
        if values:
            mean = sum(values) / float(len(values))
            variance = sum((value - mean) ** 2 for value in values) / float(len(values))
            std = math.sqrt(variance)
        else:
            mean = 0.0
            std = 1.0
        means.append(mean)
        stds.append(std if std > 1e-12 else 1.0)

    scaled = []
    for values in raw_matrix:
        scaled.append([
            ((means[index] if math.isnan(value) else value) - means[index]) / stds[index]
            for index, value in enumerate(values)
        ])
    return scaled


def euclidean(left, right):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def centroid_position(rows, matrix):
    centroid = []
    for column_index in range(len(SELECTION_FEATURES)):
        centroid.append(sum(values[column_index] for values in matrix) / float(len(matrix)))
    ordered = sorted(
        range(len(rows)),
        key=lambda index: (
            euclidean(matrix[index], centroid),
            to_int(rows[index].get("original_row_index")),
        ),
    )
    return ordered[0]


def selection_record(row, salt, numeric_cluster_id, rank, sampling_order, reason, cluster_size):
    record = dict(row)
    record["cluster_id"] = str(numeric_cluster_id)
    record["salt_cluster_id"] = salt
    record["cluster_rank_in_sampling"] = str(rank)
    record["sampling_order"] = str(sampling_order)
    record["selection_reason"] = reason
    record["cluster_size"] = str(cluster_size)
    record["representative_doi"] = row.get("doi", "")
    return record


def select_from_cluster(salt, rows, quota, numeric_cluster_id, starting_order):
    if quota > len(rows):
        raise ValueError(
            "Salt cluster {0} requested {1} systems, but only {2} unique systems are available.".format(
                salt, quota, len(rows)
            )
        )

    ordered_rows = sorted(rows, key=lambda row: to_int(row.get("original_row_index")))
    matrix = feature_matrix(ordered_rows)
    representative_pos = centroid_position(ordered_rows, matrix)
    selected_positions = [representative_pos]
    selected = [
        selection_record(
            ordered_rows[representative_pos],
            salt,
            numeric_cluster_id,
            1,
            starting_order,
            "salt-family centroid representative selected from {0} descriptor space".format(
                ", ".join(SELECTION_FEATURES)
            ),
            len(ordered_rows),
        )
    ]

    while len(selected) < quota:
        best_tuple = None
        best_position = None
        selected_matrix = [matrix[index] for index in selected_positions]
        selected_compositions = set(
            str(ordered_rows[index].get("composition_summary", "")) for index in selected_positions
        )
        selected_concentrations = set(
            round(to_float(ordered_rows[index].get("salt_conc")), 8) for index in selected_positions
        )
        selected_temperatures = set(
            round(to_float(ordered_rows[index].get("temperature")), 8) for index in selected_positions
        )

        for position, candidate in enumerate(ordered_rows):
            if position in selected_positions:
                continue
            distances = [euclidean(matrix[position], selected) for selected in selected_matrix]
            min_distance = min(distances)
            distinct_composition = (
                1.0 if str(candidate.get("composition_summary", "")) not in selected_compositions else 0.0
            )
            distinct_concentration = (
                1.0 if round(to_float(candidate.get("salt_conc")), 8) not in selected_concentrations else 0.0
            )
            distinct_temperature = (
                1.0 if round(to_float(candidate.get("temperature")), 8) not in selected_temperatures else 0.0
            )
            # Prefer descriptor spread first; ties favor composition, concentration,
            # and temperature diversity, then deterministic original row order.
            score = (
                min_distance,
                distinct_composition,
                distinct_concentration,
                distinct_temperature,
                -to_int(candidate.get("original_row_index")),
            )
            if best_tuple is None or score > best_tuple:
                best_tuple = score
                best_position = position

        if best_position is None:
            break

        selected_positions.append(best_position)
        selected.append(
            selection_record(
                ordered_rows[best_position],
                salt,
                numeric_cluster_id,
                len(selected) + 1,
                starting_order + len(selected),
                "maximin descriptor spread within salt family; exact duplicate simulation inputs excluded",
                len(ordered_rows),
            )
        )

    return selected


def select_representatives(clusters, quotas):
    selected = []
    sampling_order = 1
    for numeric_cluster_id, salt in enumerate(CANONICAL_SALTS, start=1):
        cluster_selected = select_from_cluster(
            salt,
            clusters[salt],
            quotas[salt],
            numeric_cluster_id,
            sampling_order,
        )
        selected.extend(cluster_selected)
        sampling_order += len(cluster_selected)
    return selected


def q_scale_label(value):
    return ("{0:.2f}".format(value)).replace(".", "p")


def build_campaign_plan(selected, q_scale_grid):
    jobs = []
    for row in selected:
        for q_scale in q_scale_grid:
            job = dict(row)
            job["q_scale"] = "{0:.2f}".format(q_scale)
            job["job_id"] = "salt{0}_rank{1:02d}_row{2}_q{3}".format(
                row["salt_cluster_id"],
                to_int(row["cluster_rank_in_sampling"]),
                to_int(row["original_row_index"]),
                q_scale_label(q_scale),
            )
            jobs.append(job)
    return jobs


def cluster_summary_records(clusters, quotas, exact, selected, q_scale_grid):
    selected_by_salt = {}
    for row in selected:
        selected_by_salt.setdefault(row["salt_cluster_id"], []).append(row)

    total = sum(len(clusters[salt]) for salt in CANONICAL_SALTS)
    records = []
    for numeric_cluster_id, salt in enumerate(CANONICAL_SALTS, start=1):
        cluster_selected = selected_by_salt.get(salt, [])
        representative = cluster_selected[0] if cluster_selected else {}
        cluster_size = len(clusters[salt])
        records.append(
            {
                "cluster_id": str(numeric_cluster_id),
                "salt_cluster_id": salt,
                "cluster_size": str(cluster_size),
                "cluster_fraction": "{0:.12g}".format(cluster_size / float(total) if total else 0.0),
                "proportional_quota_exact": "{0:.12g}".format(exact.get(salt, 0.0)),
                "sampling_quota": str(quotas[salt]),
                "selected_representatives": str(len(cluster_selected)),
                "planned_simulations": str(len(cluster_selected) * len(q_scale_grid)),
                "q_scale_values_per_system": str(len(q_scale_grid)),
                "representative_row_index": representative.get("original_row_index", ""),
                "representative_doi": representative.get("doi", ""),
                "representative_salt_conc": representative.get("salt_conc", ""),
                "representative_temperature": representative.get("temperature", ""),
                "representative_weighted_permittivity": representative.get("weighted_permittivity", ""),
                "representative_weighted_viscosity_mPa_s_25C": representative.get(
                    "weighted_viscosity_mPa_s_25C", ""
                ),
            }
        )
    return records


def campaign_summary_records(clusters, selected, campaign, quotas, q_scale_grid):
    selected_by_salt = {}
    for row in selected:
        selected_by_salt.setdefault(row["salt_cluster_id"], []).append(row)

    records = []
    for numeric_cluster_id, salt in enumerate(CANONICAL_SALTS, start=1):
        cluster_selected = selected_by_salt.get(salt, [])
        cluster_size = len(clusters[salt])
        coverage = len(cluster_selected) / float(cluster_size) if cluster_size else 0.0
        for q_scale in q_scale_grid:
            records.append(
                {
                    "cluster_id": str(numeric_cluster_id),
                    "salt_cluster_id": salt,
                    "q_scale": "{0:.2f}".format(q_scale),
                    "selected_systems_in_cluster": str(len(cluster_selected)),
                    "planned_simulations_at_q_scale": str(len(cluster_selected)),
                    "q_scale_values_per_system": str(len(q_scale_grid)),
                    "planned_simulations_in_cluster": str(len(cluster_selected) * len(q_scale_grid)),
                    "original_cluster_size": str(cluster_size),
                    "cluster_coverage_fraction": "{0:.12g}".format(coverage),
                    "total_planned_simulations": str(len(campaign)),
                    "total_unique_systems": str(len(selected)),
                    "clusters_covered": str(len([s for s in CANONICAL_SALTS if selected_by_salt.get(s)])),
                    "cluster_allocation_target": str(quotas[salt]),
                }
            )
    return records


def write_launcher_script(campaign, output_path, simulation_command):
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated salt-family sparse q_scale campaign launcher.",
    ]
    if not simulation_command:
        lines.extend(
            [
                "# Pass --simulation-command to build_sparse_campaign_salt_name.py to emit executable commands.",
                "# This dry-run template prints the planned simulation inputs.",
                "",
            ]
        )
    else:
        lines.append("")

    for row in campaign:
        prefix = "echo" if not simulation_command else simulation_command
        lines.append(
            " ".join(
                [
                    prefix,
                    "--job-id {0}".format(shlex.quote(str(row["job_id"]))),
                    "--cation-name {0}".format(shlex.quote(str(row["cation_name"]))),
                    "--anion-name {0}".format(shlex.quote(str(row["anion_name"]))),
                    "--salt-conc {0}".format(shlex.quote(str(row["salt_conc"]))),
                    "--solvents {0}".format(shlex.quote(str(row["solvents"]))),
                    "--solvent-fracs {0}".format(shlex.quote(str(row["solvent_fracs"]))),
                    "--temperature {0}".format(shlex.quote(str(row["temperature"]))),
                    "--q-scale {0:.2f}".format(to_float(row["q_scale"])),
                ]
            )
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output_path.chmod(0o755)


def write_validation_report(output_path, rows, clusters, selected, campaign):
    selected_salts = set(row["salt_cluster_id"] for row in selected)
    lines = [
        "Salt-family sparse campaign validation",
        "total_electrolytes_available: {0}".format(len(rows)),
        "salt_cluster_sizes:",
    ]
    for salt in CANONICAL_SALTS:
        lines.append("  {0}: {1}".format(salt, len(clusters[salt])))
    lines.append("selected_representatives_by_salt_cluster:")
    for salt in CANONICAL_SALTS:
        count = sum(row["salt_cluster_id"] == salt for row in selected)
        lines.append("  {0}: {1}".format(salt, count))
    lines.extend(
        [
            "total_selected_systems: {0}".format(len(selected)),
            "total_planned_simulations: {0}".format(len(campaign)),
            "total_simulations_exceed_300: {0}".format(str(len(campaign) > 300)),
            "every_salt_family_represented: {0}".format(
                str(all(salt in selected_salts for salt in CANONICAL_SALTS))
            ),
            "q_scale_grid: {0}".format(", ".join("{0:.2f}".format(value) for value in DEFAULT_Q_SCALE_GRID)),
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_compatible_helpers(output_dir, previous_campaign_dir):
    for file_name in ("parse_sparse_campaign.py", "submit_sparse_campaign.sh"):
        source = previous_campaign_dir / file_name
        if source.is_file():
            destination = output_dir / file_name
            shutil.copy2(str(source), str(destination))
            if file_name.endswith(".sh") or file_name.endswith(".py"):
                destination.chmod(0o755)

    collect_path = output_dir / "collect_conductivity_results.py"
    collect_path.write_text(COLLECT_CONDUCTIVITY_RESULTS, encoding="utf-8")
    collect_path.chmod(0o755)


def run(args):
    rows = load_analysis_rows(args.cleaned_input, args.descriptors_input)
    clusters = cluster_rows_by_salt(rows)
    quotas, exact = allocate_sampling_quotas(clusters, args.target_systems, args.minimum_representatives)
    selected = select_representatives(clusters, quotas)
    q_scale_grid = tuple(args.q_scale_grid)
    campaign = build_campaign_plan(selected, q_scale_grid)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(
        args.output_dir / "cluster_summary.csv",
        cluster_summary_records(clusters, quotas, exact, selected, q_scale_grid),
        (
            "cluster_id",
            "salt_cluster_id",
            "cluster_size",
            "cluster_fraction",
            "proportional_quota_exact",
            "sampling_quota",
            "selected_representatives",
            "planned_simulations",
            "q_scale_values_per_system",
            "representative_row_index",
            "representative_doi",
        ),
    )
    write_csv_rows(args.output_dir / "selected_representatives.csv", selected, SELECTED_FRONT_COLUMNS)
    write_csv_rows(args.output_dir / "campaign_plan.csv", campaign, CAMPAIGN_FRONT_COLUMNS)
    write_csv_rows(
        args.output_dir / "campaign_summary.csv",
        campaign_summary_records(clusters, selected, campaign, quotas, q_scale_grid),
        (
            "cluster_id",
            "salt_cluster_id",
            "q_scale",
            "selected_systems_in_cluster",
            "planned_simulations_at_q_scale",
            "q_scale_values_per_system",
            "planned_simulations_in_cluster",
            "original_cluster_size",
            "cluster_coverage_fraction",
            "total_planned_simulations",
            "total_unique_systems",
            "clusters_covered",
            "cluster_allocation_target",
        ),
    )
    write_launcher_script(campaign, args.output_dir / "launch_sparse_campaign.sh", args.simulation_command)
    write_validation_report(args.output_dir / "validation_report.txt", rows, clusters, selected, campaign)
    copy_compatible_helpers(args.output_dir, args.previous_campaign_dir)
    return rows, clusters, selected, campaign


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cleaned-input",
        type=Path,
        default=Path("electrolyte_outputs/cleaned_electrolytes_no_zero_salt.csv"),
        help="Cleaned no-zero-salt electrolyte table.",
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
        default=Path("results/sparse_campaign_salt_name"),
        help="Output directory for the salt-family campaign.",
    )
    parser.add_argument(
        "--previous-campaign-dir",
        type=Path,
        default=Path("results/sparse_campaign_weighted_permittivity"),
        help="Previous campaign directory containing compatible helper scripts.",
    )
    parser.add_argument(
        "--target-systems",
        type=int,
        default=DEFAULT_TARGET_SYSTEMS,
        help="Approximate number of unique electrolyte systems to select.",
    )
    parser.add_argument(
        "--minimum-representatives",
        type=int,
        default=DEFAULT_MIN_REPRESENTATIVES,
        help="Minimum representatives per canonical salt family.",
    )
    parser.add_argument(
        "--q-scale-grid",
        type=float,
        nargs="+",
        default=list(DEFAULT_Q_SCALE_GRID),
        help="q_scale values for each selected system.",
    )
    parser.add_argument(
        "--simulation-command",
        type=str,
        default=None,
        help="Command prefix for launcher lines. If omitted, launcher is a dry-run echo template.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    expected_grid = list(DEFAULT_Q_SCALE_GRID)
    provided_grid = [round(value, 8) for value in args.q_scale_grid]
    if provided_grid != [round(value, 8) for value in expected_grid]:
        raise ValueError("This campaign must use exactly q_scale values: 0.70 0.75 0.80 0.85 0.90")
    rows, clusters, selected, campaign = run(args)
    print("Wrote salt-family sparse campaign outputs to: {0}".format(args.output_dir))
    print("Total electrolytes available: {0}".format(len(rows)))
    for salt in CANONICAL_SALTS:
        count = sum(row["salt_cluster_id"] == salt for row in selected)
        print("{0}: available={1}, selected={2}".format(salt, len(clusters[salt]), count))
    print("Selected systems: {0}".format(len(selected)))
    print("Planned simulations: {0}".format(len(campaign)))
    print("Total simulations exceed 300: {0}".format(len(campaign) > 300))


if __name__ == "__main__":
    main()
