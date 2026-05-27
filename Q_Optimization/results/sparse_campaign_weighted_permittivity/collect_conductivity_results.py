#!/usr/bin/env python3
"""Collect ionic conductivity values from completed sparse campaign folders.

Each simulation folder is expected to contain ``output_LELiB_MDLab.txt`` with a
line like:

    Ionic conductivity = 8.6627 +/- 1.4598 mS/cm

The script reads ``campaign_plan.csv``, derives the same folder names used by
``submit_sparse_campaign.sh``, parses each output file, and writes one CSV row
per planned simulation.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from parse_sparse_campaign import make_unique_folders, normalize_key, validate_row


OUTPUT_FILE_NAME = "output_LELiB_MDLab.txt"
FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
CONDUCTIVITY_PATTERN = re.compile(
    rf"Ionic\s+conductivity\s*=\s*({FLOAT_PATTERN})"
    rf"(?:\s*(?:±|\+/-|\+-)\s*({FLOAT_PATTERN}))?"
    r"\s*mS\s*/\s*cm",
    re.IGNORECASE,
)


def read_campaign_rows(campaign_file: Path) -> list[dict[str, str]]:
    """Read campaign rows while retaining all campaign metadata columns."""

    rows: list[dict[str, str]] = []
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


def parse_conductivity(output_file: Path) -> tuple[str, str, str] | None:
    """Return conductivity, uncertainty, and raw line from an output file."""

    latest_match: tuple[str, str, str] | None = None
    for line in output_file.read_text(errors="replace").splitlines():
        match = CONDUCTIVITY_PATTERN.search(line)
        if match:
            latest_match = (
                match.group(1),
                match.group(2) or "",
                line.strip(),
            )
    return latest_match


def collect_results(
    campaign_file: Path,
    output_root: Path,
    output_file_name: str,
) -> list[dict[str, str]]:
    """Collect parsed conductivity values for every planned campaign row."""

    rows = read_campaign_rows(campaign_file)
    folders = make_unique_folders(rows)
    collected: list[dict[str, str]] = []

    for campaign_row, (_, folder_name) in zip(rows, folders):
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


def output_columns(rows: Iterable[dict[str, str]]) -> list[str]:
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


def write_results(rows: list[dict[str, str]], output_csv: Path) -> None:
    """Write collected results to CSV."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    columns = output_columns(rows)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, str]], output_csv: Path) -> None:
    """Print a compact collection summary."""

    parsed = sum(row["collection_status"] == "parsed" for row in rows)
    missing_folder = sum(row["collection_status"] == "missing_folder" for row in rows)
    missing_output = sum(row["collection_status"] == "missing_output_file" for row in rows)
    missing_line = sum(row["collection_status"] == "conductivity_line_not_found" for row in rows)
    print(f"Wrote conductivity collection: {output_csv}")
    print(f"planned_simulations: {len(rows)}")
    print(f"parsed_conductivity_rows: {parsed}")
    print(f"missing_folders: {missing_folder}")
    print(f"missing_output_files: {missing_output}")
    print(f"outputs_without_conductivity_line: {missing_line}")


def build_parser() -> argparse.ArgumentParser:
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
        help=f"Simulation output filename inside each folder. Default: {OUTPUT_FILE_NAME}",
    )
    return parser


def main() -> None:
    """Command-line entry point."""

    args = build_parser().parse_args()
    if not args.campaign_file.is_file():
        raise SystemExit(f"Campaign file not found: {args.campaign_file}")

    rows = collect_results(args.campaign_file, args.output_root, args.output_file_name)
    write_results(rows, args.output)
    print_summary(rows, args.output)


if __name__ == "__main__":
    main()
