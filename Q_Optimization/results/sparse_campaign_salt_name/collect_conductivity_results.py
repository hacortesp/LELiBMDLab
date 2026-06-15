#!/usr/bin/env python3
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
