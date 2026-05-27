#!/usr/bin/env python3
"""Build interpretable chemistry descriptors from cleaned electrolyte rows."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np

with contextlib.redirect_stderr(io.StringIO()):
    import pandas as pd


ALLOWED_ANIONS = ("PF6", "ClO4", "TFSI", "FSI", "BF4")
ALLOWED_SOLVENTS = ("EC", "PC", "DME", "DMC", "DEC", "EMC")
CARBONATE_SOLVENTS = {"EC", "PC", "DMC", "DEC", "EMC"}
ETHER_SOLVENTS = {"DME"}
REMOVED_MODEL_FEATURES = {
    "salt_conc",
    "n_solvents",
    "temperature",
    "carbonate_fraction",
    "ether_fraction",
}

PERMITTIVITY = {
    "DEC": 2.80,
    "DMC": 3.10,
    "DME": 7.20,
    "EC": 89.0,
    "EMC": 2.90,
    "PC": 64.9,
}

VISCOSITY_MPA_S_25C = {
    "DEC": 0.61,
    "DMC": 0.59,
    "DME": 0.39,
    "EC": 1.90,
    "EMC": 0.65,
    "PC": 2.53,
}


def parse_json_list(value: object, row_id: object, column: str) -> list:
    """Parse a JSON list from the cleaned CSV."""

    try:
        parsed = json.loads(value)
    except TypeError as exc:
        raise ValueError(f"Row {row_id}: {column} is missing") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Row {row_id}: {column} is not valid JSON: {value}") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"Row {row_id}: {column} must be a JSON list")
    return parsed


def solvent_fraction_map(row: pd.Series) -> dict[str, float]:
    """Return validated solvent fractions keyed by canonical solvent name."""

    row_id = row.get("original_row_index", row.name)
    solvents = parse_json_list(row["solvents"], row_id, "solvents")
    fractions = parse_json_list(row["solvent_fracs"], row_id, "solvent_fracs")
    if len(solvents) != len(fractions):
        raise ValueError(f"Row {row_id}: solvents and solvent_fracs lengths differ")

    fraction_map = {solvent: 0.0 for solvent in ALLOWED_SOLVENTS}
    for solvent, fraction in zip(solvents, fractions):
        if solvent not in ALLOWED_SOLVENTS:
            raise ValueError(f"Row {row_id}: unsupported cleaned solvent {solvent}")
        fraction_map[solvent] += float(fraction)

    total = sum(fraction_map.values())
    if not np.isclose(total, 1.0, rtol=1e-8, atol=1e-8):
        raise ValueError(f"Row {row_id}: solvent fractions sum to {total}, not 1")
    return fraction_map


def mixing_entropy(fractions: list[float]) -> float:
    """Shannon entropy of solvent composition."""

    arr = np.array([x for x in fractions if x > 1e-12], dtype=float)
    if len(arr) == 0:
        return 0.0
    return float(-np.sum(arr * np.log(arr)))


def weighted_average(fraction_map: dict[str, float], property_values: dict[str, float]) -> float:
    """Weighted average solvent property."""

    return float(sum(fraction_map[solvent] * property_values[solvent] for solvent in ALLOWED_SOLVENTS))


def build_descriptors(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """Create a descriptor matrix from simulation-ready electrolyte rows."""

    records: list[dict[str, object]] = []
    for _, row in cleaned_df.iterrows():
        row_id = int(row["original_row_index"])
        anion = row["anion_name"]
        if anion not in ALLOWED_ANIONS:
            raise ValueError(f"Row {row_id}: unsupported anion in cleaned data: {anion}")

        fraction_map = solvent_fraction_map(row)
        salt_conc = float(row["salt_conc"])
        temperature = float(row["temperature"])
        weighted_permittivity = weighted_average(fraction_map, PERMITTIVITY)
        weighted_viscosity = weighted_average(fraction_map, VISCOSITY_MPA_S_25C)

        record: dict[str, object] = {
            "original_row_index": row_id,
            "doi": row.get("doi"),
            "conductivity": float(row["conductivity"]),
            "anion_name": anion,
            "composition_summary": row.get("composition_summary"),
            "solvent_entropy": mixing_entropy(list(fraction_map.values())),
            "weighted_permittivity": weighted_permittivity,
            "weighted_viscosity_mPa_s_25C": weighted_viscosity,
            "salt_conc_x_weighted_permittivity": salt_conc * weighted_permittivity,
            "salt_conc_x_weighted_viscosity": salt_conc * weighted_viscosity,
        }

        for allowed_anion in ALLOWED_ANIONS:
            record[f"anion_{allowed_anion}"] = int(anion == allowed_anion)
        for solvent in ALLOWED_SOLVENTS:
            record[f"has_{solvent}"] = int(fraction_map[solvent] > 1e-12)
            record[f"frac_{solvent}"] = fraction_map[solvent]

        records.append(record)

    descriptors = pd.DataFrame(records)
    return descriptors.drop(columns=[col for col in REMOVED_MODEL_FEATURES if col in descriptors.columns])


def run(cleaned_input: Path, descriptor_output: Path) -> None:
    """Read cleaned rows, build descriptors, and save them."""

    cleaned_df = pd.read_csv(cleaned_input)
    descriptors = build_descriptors(cleaned_df)
    descriptor_output.parent.mkdir(parents=True, exist_ok=True)
    descriptors.to_csv(descriptor_output, index=False)
    print(f"Wrote descriptors: {descriptor_output} ({len(descriptors)} rows)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "cleaned_input",
        type=Path,
        help="Simulation-ready cleaned_electrolytes.csv from filter_electrolytes.py.",
    )
    parser.add_argument(
        "--descriptor-output",
        type=Path,
        default=Path("electrolyte_outputs/descriptors.csv"),
        help="Path for descriptor matrix.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args.cleaned_input, args.descriptor_output)


if __name__ == "__main__":
    main()
