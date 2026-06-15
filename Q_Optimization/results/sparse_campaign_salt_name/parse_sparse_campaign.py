#!/usr/bin/env python3
"""Parse a sparse electrolyte campaign into tab-separated submission rows.

Expected input formats:
- Shell-friendly launcher lines like:
  echo --job-id ... --cation-name Li --anion-name PF6 --salt-conc 1.0 \
       --solvents '["EC","DMC"]' --solvent-fracs '[0.5,0.5]' \
       --temperature 298.15 --q-scale 0.85
- CSV tables with columns matching either dashed or underscored names, e.g.
  job_id,cation_name,anion_name,salt_conc,solvents,solvent_fracs,temperature,q_scale

The output is one TSV row per planned simulation:
job_id, folder_name, job_name, cation_name, anion_name, salt_conc,
temperature, q_scale, space-separated solvents, space-separated solvent_fracs.
"""

import argparse
import csv
import json
import re
import shlex
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


REQUIRED_FIELDS = (
    "job_id",
    "cation_name",
    "anion_name",
    "salt_conc",
    "solvents",
    "solvent_fracs",
    "temperature",
    "q_scale",
)


def normalize_key(key: str) -> str:
    return key.strip().lower().lstrip("-").replace("-", "_")


def parse_list(value):
    value = str(value).strip()
    if not value:
        return []

    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        decoded = None

    if isinstance(decoded, list):
        return [str(item).strip() for item in decoded]

    cleaned = value.strip("[]()")
    cleaned = cleaned.replace(",", " ")
    return [item.strip().strip("'\"") for item in cleaned.split() if item.strip()]


def sanitize(value):
    value = str(value).strip()
    value = value.replace(".", "p")
    value = re.sub(r"[^A-Za-z0-9_+.-]+", "-", value)
    return value.strip("-") or "NA"


def format_q_label(q_scale):
    try:
        q_value = Decimal(str(q_scale))
    except InvalidOperation:
        return "q" + sanitize(q_scale)

    hundred = q_value * Decimal("100")
    if hundred == hundred.to_integral_value():
        return f"q{int(hundred):03d}"

    return "q" + sanitize(str(q_scale))


def solvent_label_for(row):
    solvents = parse_list(row["solvents"])
    solvent_fracs = parse_list(row["solvent_fracs"])
    return "".join(
        f"{sanitize(solvent)}{sanitize(frac)}"
        for solvent, frac in zip(solvents, solvent_fracs)
    )


def folder_for(row):
    parts = (
        sanitize(row["cation_name"]),
        sanitize(row["anion_name"]),
        f"{sanitize(row['salt_conc'])}M",
        solvent_label_for(row),
        f"T{sanitize(row['temperature'])}",
        format_q_label(row["q_scale"]),
    )
    return "_".join(part for part in parts if part)


def job_name_for(row):
    base = "_".join(
        (
            sanitize(row["cation_name"]),
            sanitize(row["anion_name"]),
            sanitize(row["salt_conc"]) + "M",
            solvent_label_for(row),
            format_q_label(row["q_scale"]),
        )
    )
    return base[:120]


def rows_from_shell(path):
    rows = []  # type: List[Dict[str, str]]
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("set "):
            continue

        try:
            tokens = shlex.split(line, comments=False, posix=True)
        except ValueError as exc:
            raise SystemExit(f"{path}:{line_number}: cannot parse shell line: {exc}") from exc

        if not tokens:
            continue
        if tokens[0] == "echo":
            tokens = tokens[1:]
        if "--job-id" not in tokens:
            continue

        parsed = {}  # type: Dict[str, str]
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if not token.startswith("--"):
                index += 1
                continue
            key = normalize_key(token)
            if index + 1 >= len(tokens) or tokens[index + 1].startswith("--"):
                raise SystemExit(f"{path}:{line_number}: missing value for {token}")
            parsed[key] = tokens[index + 1]
            index += 2

        rows.append(validate_row(parsed, path, line_number))
    return rows


def rows_from_csv(path):
    rows = []  # type: List[Dict[str, str]]
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for line_number, raw_row in enumerate(reader, start=2):
            row = {
                normalize_key(key): (value or "").strip()
                for key, value in raw_row.items()
                if key is not None
            }
            rows.append(validate_row(row, path, line_number))
    return rows


def validate_row(row, path, line_number):
    missing = [field for field in REQUIRED_FIELDS if not row.get(field)]
    if missing:
        missing_text = ", ".join(missing)
        raise SystemExit(f"{path}:{line_number}: missing required fields: {missing_text}")

    solvents = parse_list(row["solvents"])
    solvent_fracs = parse_list(row["solvent_fracs"])
    if not solvents:
        raise SystemExit(f"{path}:{line_number}: no solvents parsed from {row['solvents']!r}")
    if len(solvents) != len(solvent_fracs):
        raise SystemExit(
            f"{path}:{line_number}: solvents/fracs length mismatch "
            f"({len(solvents)} solvents, {len(solvent_fracs)} fractions)"
        )

    return {field: str(row[field]).strip() for field in REQUIRED_FIELDS}


def detect_format(path):
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#!") or line.startswith("#") or line.startswith("set "):
            continue
        if line.startswith("echo ") or "--job-id" in line:
            return "shell"
        if "," in line:
            return "csv"
    raise SystemExit(f"{path}: no campaign rows found")


def make_unique_folders(rows):
    base_names = [folder_for(row) for row in rows]
    counts = Counter(base_names)
    seen = Counter()
    unique = []  # type: List[Tuple[Dict[str, str], str]]

    for row, base_name in zip(rows, base_names):
        if counts[base_name] == 1:
            folder_name = base_name
        else:
            seen[base_name] += 1
            folder_name = f"{base_name}_{sanitize(row['job_id'])}"
        unique.append((row, folder_name))

    return unique


def print_rows(rows):
    for row, folder_name in make_unique_folders(rows):
        solvents = " ".join(parse_list(row["solvents"]))
        solvent_fracs = " ".join(parse_list(row["solvent_fracs"]))
        fields = (
            row["job_id"],
            folder_name,
            job_name_for(row),
            row["cation_name"],
            row["anion_name"],
            row["salt_conc"],
            row["temperature"],
            row["q_scale"],
            solvents,
            solvent_fracs,
        )
        print("\t".join(fields))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_file", type=Path)
    args = parser.parse_args()

    if not args.campaign_file.is_file():
        raise SystemExit(f"Campaign file not found: {args.campaign_file}")

    campaign_format = detect_format(args.campaign_file)
    rows = rows_from_shell(args.campaign_file) if campaign_format == "shell" else rows_from_csv(args.campaign_file)
    if not rows:
        raise SystemExit(f"No campaign rows found in {args.campaign_file}")
    print_rows(rows)


if __name__ == "__main__":
    main()
