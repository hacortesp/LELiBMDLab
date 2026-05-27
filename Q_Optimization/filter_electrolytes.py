#!/usr/bin/env python3
"""Filter raw electrolyte data into simulation-ready inputs.

The output rows map directly to the MD parser inputs:

* ``cation_name`` -> ``-cation_name``
* ``anion_name`` -> ``-anion_name``
* ``salt_conc`` -> ``-salt-conc`` in mol/L
* ``solvents`` -> ``-solvents`` as a JSON list
* ``solvent_fracs`` -> ``-solvent-fracs`` as a JSON list
* ``temperature`` -> ``-temperature`` in Kelvin

Rows are rejected, with an explicit reason, if the salt, solvent set,
composition, concentration units, conductivity, or temperature cannot be used
unambiguously by the simulation code.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np

with contextlib.redirect_stderr(io.StringIO()):
    import pandas as pd


ALLOWED_ANIONS = ("PF6", "ClO4", "TFSI", "FSI", "BF4")
ALLOWED_SOLVENTS = ("EC", "PC", "DME", "DMC", "DEC", "EMC")
SIMULATION_INPUT_COLUMNS = (
    "cation_name",
    "anion_name",
    "salt_conc",
    "solvents",
    "solvent_fracs",
    "temperature",
)

METADATA_COLUMNS = {
    "original_row_index",
    "doi",
    "k",
    "T",
    "c",
    "salt",
    "c units",
    "solvent ratio type",
}

SALT_ALIASES = {
    "lipf6": "PF6",
    "pf6": "PF6",
    "liclo4": "ClO4",
    "clo4": "ClO4",
    "litfsi": "TFSI",
    "tfsi": "TFSI",
    "lifsi": "FSI",
    "fsi": "FSI",
    "libf4": "BF4",
    "bf4": "BF4",
}

SOLVENT_ALIASES = {
    "ec": "EC",
    "ethylene carbonate": "EC",
    "ethylenecarbonate": "EC",
    "pc": "PC",
    "propylene carbonate": "PC",
    "propylenecarbonate": "PC",
    "dme": "DME",
    "1,2-dimethoxyethane": "DME",
    "12dimethoxyethane": "DME",
    "dimethoxyethane": "DME",
    "monoglyme": "DME",
    "glyme": "DME",
    "dmc": "DMC",
    "dimethyl carbonate": "DMC",
    "dimethylcarbonate": "DMC",
    "dec": "DEC",
    "diethyl carbonate": "DEC",
    "diethylcarbonate": "DEC",
    "emc": "EMC",
    "ethyl methyl carbonate": "EMC",
    "ethylmethylcarbonate": "EMC",
}

RATIO_TYPES = {
    "w": "weight",
    "wt": "weight",
    "weight": "weight",
    "weightfraction": "weight",
    "mass": "weight",
    "massfraction": "weight",
    "v": "volume",
    "vol": "volume",
    "volume": "volume",
    "volumefraction": "volume",
    "mol": "molar",
    "mole": "molar",
    "molar": "molar",
    "molfraction": "molar",
    "molefraction": "molar",
    "x": "molar",
}

MOLAR_UNITS = {
    "mol/l",
    "mol/liter",
    "mol/litre",
    "moll-1",
    "moldm-3",
    "mol/dm3",
    "mol/dm^3",
    "m",
}


def compact_name(value: object) -> str:
    """Normalize whitespace, case, and dash variants for comparisons."""

    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text)


def alias_key(value: object) -> str:
    """Return a compact key for exact alias lookups."""

    return re.sub(r"[\s_\-·\.]", "", compact_name(value))


def standardize_salt(value: object) -> tuple[str | None, str | None]:
    """Map only exact supported lithium-salt or anion labels to anion names."""

    if pd.isna(value) or str(value).strip() == "":
        return None, "missing salt"
    raw = str(value).strip()
    key = alias_key(raw)
    anion = SALT_ALIASES.get(key)
    if anion is None:
        return None, f"unsupported or ambiguous salt: {raw}"
    return anion, None


def standardize_solvent_name(name: object) -> str:
    """Map a solvent column name to a canonical solvent name when known."""

    text = compact_name(name)
    return SOLVENT_ALIASES.get(text, SOLVENT_ALIASES.get(alias_key(text), str(name).strip()))


def standardize_ratio_type(value: object) -> tuple[str | None, str | None]:
    """Interpret the solvent composition basis."""

    if pd.isna(value) or str(value).strip() == "":
        return None, "missing solvent ratio type"
    basis = RATIO_TYPES.get(alias_key(value))
    if basis is None:
        return None, f"unsupported solvent ratio type: {value}"
    return basis, None


def detect_solvent_columns(columns: Iterable[str]) -> dict[str, str]:
    """Detect wide solvent columns and map each original name to a canonical name."""

    solvent_columns: dict[str, str] = {}
    for col in columns:
        if col in METADATA_COLUMNS or str(col).startswith("Unnamed:"):
            continue
        solvent_columns[col] = standardize_solvent_name(col)
    return solvent_columns


def parse_composition(
    row: pd.Series,
    solvent_columns: dict[str, str],
    tolerance: float = 1e-10,
) -> tuple[list[str] | None, list[float] | None, str | None, str, list[str]]:
    """Return normalized solvent names/fractions or rejection reasons."""

    reasons: list[str] = []
    basis, ratio_reason = standardize_ratio_type(row.get("solvent ratio type"))
    if ratio_reason:
        reasons.append(ratio_reason)

    values_by_solvent: dict[str, float] = {}
    unsupported_nonzero: list[str] = []

    for original_col, solvent in solvent_columns.items():
        raw_value = row.get(original_col, 0.0)
        value = pd.to_numeric(raw_value, errors="coerce")
        if pd.isna(value):
            if pd.isna(raw_value) or str(raw_value).strip() == "":
                value = 0.0
            else:
                reasons.append(f"non-numeric solvent fraction in {original_col}: {raw_value}")
                continue
        value = float(value)
        if value < -tolerance:
            reasons.append(f"negative solvent fraction in {original_col}: {value}")
            continue
        if abs(value) <= tolerance:
            continue

        if solvent in ALLOWED_SOLVENTS:
            values_by_solvent[solvent] = values_by_solvent.get(solvent, 0.0) + value
        else:
            unsupported_nonzero.append(f"{original_col}={value:g}")

    if unsupported_nonzero:
        reasons.append("unsupported nonzero solvent(s): " + "; ".join(unsupported_nonzero))

    total = sum(values_by_solvent.values())
    if total <= tolerance:
        reasons.append("no nonzero supported solvent composition")

    if reasons:
        return None, None, basis, "", reasons

    warning = ""
    if math.isclose(total, 1.0, rel_tol=1e-4, abs_tol=1e-4):
        scale = total
    elif math.isclose(total, 100.0, rel_tol=1e-4, abs_tol=1e-4):
        scale = total
        warning = "solvent fractions summed to 100; interpreted as percent and normalized"
    else:
        scale = total
        warning = f"solvent fractions summed to {total:.6g}; normalized to 1"

    solvents = [solvent for solvent in ALLOWED_SOLVENTS if values_by_solvent.get(solvent, 0.0) > tolerance]
    fractions = [values_by_solvent[solvent] / scale for solvent in solvents]
    total_fraction = sum(fractions)
    if not math.isclose(total_fraction, 1.0, rel_tol=1e-8, abs_tol=1e-8):
        fractions = [fraction / total_fraction for fraction in fractions]

    return solvents, fractions, basis, warning, []


def convert_concentration_to_molar(value: object, units: object) -> tuple[float | None, str | None, str]:
    """Convert concentration to mol/L when possible.

    ``mol/kg`` values are rejected because conversion requires density or a
    solvent-mass model that is not present in the raw CSV.
    """

    concentration = pd.to_numeric(value, errors="coerce")
    if pd.isna(concentration):
        return None, "missing or non-numeric concentration c", ""
    concentration = float(concentration)
    if concentration < 0:
        return None, f"negative concentration c: {concentration}", ""
    if concentration == 0:
        return None, "zero salt concentration c", ""

    unit_text = compact_name(units)
    unit_key = alias_key(unit_text)
    if unit_key in MOLAR_UNITS:
        return concentration, None, "mol/L"
    if unit_key in {"mol/kg", "molkg-1", "molkg"}:
        return None, "concentration unit mol/kg cannot be converted to mol/L without density", unit_text
    if unit_text == "":
        return None, "missing concentration units", ""
    return None, f"unsupported concentration units: {units}", unit_text


def numeric_temperature_kelvin(value: object) -> tuple[float | None, str | None]:
    """Parse temperature as Kelvin."""

    temperature = pd.to_numeric(value, errors="coerce")
    if pd.isna(temperature):
        return None, "missing or non-numeric temperature T"
    temperature = float(temperature)
    if temperature <= 0:
        return None, f"temperature must be positive Kelvin: {temperature}"
    return temperature, None


def composition_summary(solvents: list[str], fractions: list[float]) -> str:
    """Compact composition label for inspection and summaries."""

    return "+".join(f"{solvent}:{fraction:.3f}" for solvent, fraction in zip(solvents, fractions))


def reject_record(row: pd.Series, reasons: list[str]) -> dict[str, object]:
    """Build the required rejected-record row."""

    return {
        "original_row_index": int(row["original_row_index"]),
        "doi": row.get("doi"),
        "original_salt": row.get("salt"),
        "rejection_reason": " | ".join(reasons),
    }


def filter_electrolytes(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter raw records and emit simulation-ready electrolyte rows."""

    raw_df = raw_df.copy()
    if "original_row_index" not in raw_df.columns:
        raw_df.insert(0, "original_row_index", raw_df.index)

    solvent_columns = detect_solvent_columns(raw_df.columns)
    clean_records: list[dict[str, object]] = []
    rejected_records: list[dict[str, object]] = []

    for _, row in raw_df.iterrows():
        reasons: list[str] = []

        anion, salt_reason = standardize_salt(row.get("salt"))
        if salt_reason:
            reasons.append(salt_reason)

        conductivity = pd.to_numeric(row.get("k"), errors="coerce")
        if pd.isna(conductivity):
            reasons.append("missing or non-numeric conductivity k")

        salt_conc, concentration_reason, concentration_units = convert_concentration_to_molar(
            row.get("c"), row.get("c units")
        )
        if concentration_reason:
            reasons.append(concentration_reason)

        temperature, temperature_reason = numeric_temperature_kelvin(row.get("T"))
        if temperature_reason:
            reasons.append(temperature_reason)

        solvents, fractions, basis, composition_warning, composition_reasons = parse_composition(
            row, solvent_columns
        )
        reasons.extend(composition_reasons)

        if reasons:
            rejected_records.append(reject_record(row, reasons))
            continue

        assert anion is not None
        assert salt_conc is not None
        assert temperature is not None
        assert solvents is not None
        assert fractions is not None

        clean_records.append(
            {
                "original_row_index": int(row["original_row_index"]),
                "doi": row.get("doi"),
                "conductivity": float(conductivity),
                "cation_name": "Li",
                "anion_name": anion,
                "salt_conc": salt_conc,
                "temperature": temperature,
                "solvents": json.dumps(solvents, separators=(",", ":")),
                "solvent_fracs": json.dumps([round(float(x), 12) for x in fractions], separators=(",", ":")),
                "original_salt": row.get("salt"),
                "original_concentration": row.get("c"),
                "original_concentration_units": row.get("c units"),
                "salt_conc_units": "mol/L",
                "solvent_ratio_basis": basis,
                "composition_warning": composition_warning,
                "composition_summary": composition_summary(solvents, fractions),
            }
        )

    cleaned = collapse_duplicate_electrolytes(pd.DataFrame(clean_records))
    return cleaned, pd.DataFrame(rejected_records)


def unique_nonempty(values: pd.Series) -> list[str]:
    """Return sorted unique nonempty string values."""

    return sorted({str(value) for value in values if not pd.isna(value) and str(value) != ""})


def collapse_duplicate_electrolytes(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Average exact replicate measurements with identical simulation inputs.

    Duplicate identity is defined only by canonical simulation inputs:
    ``cation_name``, ``anion_name``, ``salt_conc``, ``solvents``,
    ``solvent_fracs``, and ``temperature``.  Conductivity is averaged within
    each duplicate group; no other chemistry or condition fields are averaged.
    """

    if cleaned.empty:
        return cleaned

    collapsed_records: list[dict[str, object]] = []
    for _, group in cleaned.groupby(list(SIMULATION_INPUT_COLUMNS), sort=False, dropna=False):
        first = group.iloc[0].to_dict()
        first["conductivity"] = float(group["conductivity"].mean())
        first["conductivity_std"] = (
            float(group["conductivity"].std(ddof=1)) if len(group) > 1 else 0.0
        )
        first["replicate_count"] = int(len(group))
        first["original_row_index"] = int(group["original_row_index"].iloc[0])
        first["source_row_indices"] = json.dumps(
            [int(value) for value in group["original_row_index"].tolist()],
            separators=(",", ":"),
        )
        first["doi"] = ";".join(unique_nonempty(group["doi"]))
        first["source_dois"] = json.dumps(unique_nonempty(group["doi"]), separators=(",", ":"))
        collapsed_records.append(first)

    output = pd.DataFrame(collapsed_records)
    front_columns = [
        "original_row_index",
        "source_row_indices",
        "doi",
        "source_dois",
        "conductivity",
        "conductivity_std",
        "replicate_count",
        *SIMULATION_INPUT_COLUMNS,
    ]
    ordered_columns = [col for col in front_columns if col in output.columns]
    ordered_columns.extend(col for col in output.columns if col not in ordered_columns)
    return output[ordered_columns]


def run(input_csv: Path, cleaned_output: Path, rejected_output: Path) -> None:
    """Run filtering and save outputs."""

    raw_df = pd.read_csv(input_csv)
    cleaned, rejected = filter_electrolytes(raw_df)
    cleaned_output.parent.mkdir(parents=True, exist_ok=True)
    rejected_output.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(cleaned_output, index=False)
    rejected.to_csv(rejected_output, index=False)
    print(f"Wrote cleaned electrolytes: {cleaned_output} ({len(cleaned)} rows)")
    print(f"Wrote rejected records: {rejected_output} ({len(rejected)} rows)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path, help="Raw experimental electrolyte CSV.")
    parser.add_argument(
        "--cleaned-output",
        type=Path,
        default=Path("electrolyte_outputs/cleaned_electrolytes.csv"),
        help="Path for simulation-ready cleaned rows.",
    )
    parser.add_argument(
        "--rejected-output",
        type=Path,
        default=Path("electrolyte_outputs/rejected_records.csv"),
        help="Path for rejected rows and reasons.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args.input_csv, args.cleaned_output, args.rejected_output)


if __name__ == "__main__":
    main()
