#!/usr/bin/env python3
"""
Extract relevant electrolyte systems from CALiSol-23 database
for compatibility with MD workflow.

Filters:
- Supported salts
- Supported solvents only
- mol/kg concentration units
- Valid solvent compositions

Outputs:
    calisol23_filtered_for_md.csv
"""

import pandas as pd
import numpy as np

INPUT_FILE = "calisol23_DOI_10.11583DTU.c.6929599.csv"
OUTPUT_FILE = "calisol23_filtered_for_md.csv"

SUPPORTED_SALTS = {
    "LiPF6": "PF6",
    "LiBF4": "BF4",
    "LiTFSI": "TFSI",
    "LiFSI": "FSI",
    "LiClO4": "ClO4",
}

SUPPORTED_SOLVENTS = [
    "EC",
    "PC",
    "DMC",
    "EMC",
    "DEC",
    "DME",
]


def load_dataset(filepath: str) -> pd.DataFrame:
    """Load CALiSol CSV file."""
    return pd.read_csv(filepath)


def filter_supported_salts(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only salts supported by MD code."""
    return df[df["salt"].isin(SUPPORTED_SALTS.keys())].copy()


def filter_supported_solvents(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep rows where ONLY supported solvents are present.
    Any non-zero fraction in unsupported solvent → drop row.
    """
    solvent_columns = df.columns[8:]  # solvent columns start after metadata

    unsupported_solvents = [
        col for col in solvent_columns if col not in SUPPORTED_SOLVENTS
    ]

    mask = (df[unsupported_solvents] == 0).all(axis=1)
    return df[mask].copy()


def filter_molality_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only mol/kg concentration units."""
    return df[df["c units"] == "mol/kg"].copy()


def normalize_solvent_fractions(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure solvent fractions sum to 1 (robust against rounding)."""
    solvent_data = df[SUPPORTED_SOLVENTS]
    sums = solvent_data.sum(axis=1)

    df[SUPPORTED_SOLVENTS] = solvent_data.div(sums, axis=0)
    return df


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only relevant ML features."""
    selected_columns = (
        ["doi", "k", "T", "c", "salt"]
        + SUPPORTED_SOLVENTS
    )

    df = df[selected_columns].copy()

    df.rename(
        columns={
            "k": "conductivity_mScm",
            "T": "temperature_K",
            "c": "concentration_molkg",
        },
        inplace=True,
    )

    df["salt"] = df["salt"].map(SUPPORTED_SALTS)

    return df

df = load_dataset(INPUT_FILE)

df = filter_supported_salts(df)
df = filter_supported_solvents(df)
df = filter_molality_only(df)
df = normalize_solvent_fractions(df)
df = clean_columns(df)

df.to_csv(OUTPUT_FILE, index=False)

print(f"Filtered dataset written to: {OUTPUT_FILE}")
print(f"Remaining data points: {len(df)}")
print(df.head())

