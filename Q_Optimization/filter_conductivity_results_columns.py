#!/usr/bin/env python3
"""Filter conductivity results to the key reporting columns."""

import argparse
import os
import sys
from pathlib import Path

# Avoid noisy optional pyarrow imports in environments with mixed NumPy builds.
os.environ.setdefault("PANDAS_NO_IMPORT_PYARROW", "1")
sys.modules.setdefault("pyarrow", None)

import pandas as pd


DEFAULT_INPUT = (
    "/home/hacortes/Li-EMDLab/Q_Optimization/"
    "results/sparse_campaign_salt_name/conductivity_results.csv"
)
DEFAULT_OUTPUT = (
    "/home/hacortes/Li-EMDLab/Q_Optimization/"
    "results/sparse_campaign_salt_name/conductivity_results_filtered.csv"
)

COLUMN_CANDIDATES = (
    ("original row index", ("original_row_index", "source_row_index", "row_index")),
    ("DOI", ("doi", "DOI")),
    ("q_scale", ("q_scale", "q_sca", "charge_scale")),
    ("temperature", ("temperature", "temperature_K", "temp")),
    ("salt identity", ("anion_name", "original_salt", "salt_cluster_id", "salt")),
    ("salt concentration", ("salt_conc", "concentration", "original_concentration")),
    ("solvent names", ("solvents", "solvent_names")),
    ("solvent fractions", ("solvent_fracs", "solvent_fractions")),
    ("experimental conductivity", ("conductivity", "experimental_conductivity")),
    ("calculated conductivity", ("conductivity_mS_cm", "calculated_conductivity")),
    (
        "calculated conductivity uncertainty",
        ("conductivity_uncertainty_mS_cm", "calculated_conductivity_uncertainty"),
    ),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a column-filtered conductivity results CSV."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Input conductivity results CSV.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output filtered CSV path.",
    )
    return parser.parse_args()


def choose_column(columns, label, candidates):
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    raise ValueError(
        "Could not identify column for {0}. Tried: {1}".format(
            label, ", ".join(candidates)
        )
    )


def identify_columns(columns):
    identified = []
    for label, candidates in COLUMN_CANDIDATES:
        identified.append((label, choose_column(columns, label, candidates)))
    return identified


def print_columns(title, columns):
    print(title)
    for column in columns:
        print("  - {0}".format(column))


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    header = pd.read_csv(input_path, nrows=0)
    all_columns = list(header.columns)
    print_columns("Available column names:", all_columns)

    identified = identify_columns(all_columns)
    print("Identified column mapping:")
    for label, column in identified:
        print("  - {0}: {1}".format(label, column))

    retained_columns = [column for _, column in identified]

    df = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    filtered = df.loc[:, retained_columns]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_path, index=False)

    print_columns("Retained column names:", retained_columns)
    print("Original dataframe shape: {0}".format(df.shape))
    print("Filtered dataframe shape: {0}".format(filtered.shape))
    print("Saved output path: {0}".format(output_path))


if __name__ == "__main__":
    main()
