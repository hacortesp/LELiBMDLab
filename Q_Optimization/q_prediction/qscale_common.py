#!/usr/bin/env python3
"""Shared helpers for the simplified q_scale prediction workflow."""

import ast
import json
import math
import os
import sys

os.environ.setdefault("PANDAS_NO_IMPORT_PYARROW", "1")
sys.modules.setdefault("pyarrow", None)

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
DEFAULT_INPUT_CSV = os.path.join(
    REPO_ROOT,
    "results",
    "sparse_campaign_salt_name",
    "conductivity_results_filtered.csv",
)
DEFAULT_OUTPUT_DIR = THIS_DIR

ALLOWED_SOLVENTS = ["EC", "PC", "DME", "DMC", "DEC", "EMC"]
ALLOWED_QSCALES = [0.70, 0.75, 0.80, 0.85, 0.90]

NUMERIC_FEATURES = [
    "salt_conc_numeric",
    "temperature_numeric",
    "has_ec",
    "frac_ec",
    "has_pc",
    "frac_pc",
    "has_dme",
    "frac_dme",
    "has_dmc",
    "frac_dmc",
    "has_dec",
    "frac_dec",
    "has_emc",
    "frac_emc",
]
CATEGORICAL_FEATURES = ["anion_name"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def q_label(value):
    return "{0:.2f}".format(float(value))


def parse_list_like(value):
    if pd.isna(value):
        return []
    text = str(value).strip()
    if text == "":
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
        return [parsed]
    except Exception:
        return [part.strip() for part in text.split(",") if part.strip()]


def parse_solvents(value):
    solvents = []
    for item in parse_list_like(value):
        solvent = str(item).strip().strip("'\"").upper()
        if solvent:
            solvents.append(solvent)
    return solvents


def parse_fractions(value):
    fractions = []
    for item in parse_list_like(value):
        try:
            fractions.append(float(item))
        except Exception:
            continue
    return fractions


def normalize_fractions(fractions):
    values = [float(v) for v in fractions]
    if len(values) == 0:
        raise ValueError("At least one solvent fraction is required.")
    if any(v < 0 for v in values):
        raise ValueError("Solvent fractions must be non-negative.")
    total = sum(values)
    if total <= 0:
        raise ValueError("At least one solvent fraction must be positive.")
    return [v / total for v in values]


def validate_solvent_inputs(solvents, fractions):
    normalized_solvents = [str(s).strip().upper() for s in solvents]
    if len(normalized_solvents) == 0:
        raise ValueError("At least one solvent is required.")
    unknown = [s for s in normalized_solvents if s not in ALLOWED_SOLVENTS]
    if unknown:
        raise ValueError(
            "Unknown solvent(s): {0}. Allowed solvents: {1}".format(
                ", ".join(unknown),
                ", ".join(ALLOWED_SOLVENTS),
            )
        )
    if len(normalized_solvents) != len(fractions):
        raise ValueError("The number of solvents must match the number of solvent fractions.")
    return normalized_solvents, normalize_fractions(fractions)


def solvent_feature_values(solvents, fractions):
    solvents, fractions = validate_solvent_inputs(solvents, fractions)
    features = {}
    for solvent in ALLOWED_SOLVENTS:
        lower = solvent.lower()
        features["has_" + lower] = 1 if solvent in solvents else 0
        total_fraction = 0.0
        for index, name in enumerate(solvents):
            if name == solvent:
                total_fraction += fractions[index]
        features["frac_" + lower] = total_fraction
    return features


def add_solvent_features(df):
    rows = []
    for _, row in df.iterrows():
        solvents = parse_solvents(row["solvents"])
        fractions = parse_fractions(row["solvent_fracs"])
        try:
            rows.append(solvent_feature_values(solvents, fractions))
        except ValueError:
            empty = {}
            for solvent in ALLOWED_SOLVENTS:
                empty["has_" + solvent.lower()] = 0
                empty["frac_" + solvent.lower()] = np.nan
            rows.append(empty)
    features = pd.DataFrame(rows, index=df.index)
    out = pd.concat([df.copy(), features], axis=1)
    return out


def load_campaign_csv(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = [
        "original_row_index",
        "doi",
        "q_scale",
        "temperature",
        "anion_name",
        "salt_conc",
        "solvents",
        "solvent_fracs",
        "conductivity",
        "conductivity_mS_cm",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError("Input CSV is missing required columns: {0}".format(", ".join(missing)))
    return df


def build_system_id(df):
    identity_cols = ["doi", "anion_name", "salt_conc", "temperature", "solvents", "solvent_fracs"]
    checks = df.groupby("original_row_index", dropna=False)[identity_cols].nunique(dropna=False)
    if len(checks) > 0 and (checks <= 1).all().all():
        return "row_" + df["original_row_index"].astype(str)
    return df[["anion_name", "salt_conc", "solvents", "solvent_fracs", "temperature", "doi"]].astype(str).agg("|".join, axis=1)


def build_training_table(raw_df):
    df = raw_df.copy()
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
    df["valid_for_target"] = valid

    records = []
    for system_id, group in df.groupby("system_id", sort=False, dropna=False):
        valid_group = group.loc[group["valid_for_target"]].copy()
        if valid_group.empty:
            continue
        valid_group["log_error"] = (
            np.log(valid_group["calculated_conductivity"])
            - np.log(valid_group["experimental_conductivity"])
        ).abs()
        min_error = float(valid_group["log_error"].min())
        tied = valid_group.loc[np.isclose(valid_group["log_error"], min_error, rtol=1e-12, atol=1e-12)]
        best = tied.sort_values("q_scale_numeric").iloc[0]
        q_values = sorted(set(float(v) for v in valid_group["q_scale_numeric"].dropna()))

        record = {
            "system_id": system_id,
            "original_row_index": best["original_row_index"],
            "doi": best["doi"],
            "anion_name": best["anion_name"],
            "salt_conc": best["salt_conc"],
            "temperature": best["temperature"],
            "solvents": best["solvents"],
            "solvent_fracs": best["solvent_fracs"],
            "experimental_conductivity": float(best["experimental_conductivity"]),
            "q_scale_optimal": float(best["q_scale_numeric"]),
            "q_scale_class": q_label(best["q_scale_numeric"]),
            "min_log_conductivity_error": min_error,
            "available_q_scales": ";".join(q_label(v) for v in q_values),
            "number_of_qscale_points": int(len(q_values)),
            "tie_flag": bool(len(tied) > 1),
            "raw_rows_for_system": int(len(group)),
            "valid_rows_for_system": int(len(valid_group)),
        }
        records.append(record)

    table = pd.DataFrame(records)
    if table.empty:
        raise ValueError("No valid electrolyte systems were found.")
    table["salt_conc_numeric"] = pd.to_numeric(table["salt_conc"], errors="coerce")
    table["temperature_numeric"] = pd.to_numeric(table["temperature"], errors="coerce")
    table = add_solvent_features(table)
    return table


def one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor():
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def make_model(random_seed):
    return ExtraTreesClassifier(
        n_estimators=500,
        random_state=random_seed,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
    )


def build_input_dataframe(anion_name, salt_conc, solvents, solvent_fracs, temperature):
    solvents, fractions = validate_solvent_inputs(solvents, solvent_fracs)
    row = {
        "anion_name": str(anion_name).strip().upper(),
        "salt_conc_numeric": float(salt_conc),
        "temperature_numeric": float(temperature),
    }
    row.update(solvent_feature_values(solvents, fractions))
    return pd.DataFrame([row])[FEATURE_COLUMNS]


def transformed_feature_names(preprocessor):
    names = list(NUMERIC_FEATURES)
    cat_pipeline = preprocessor.named_transformers_["categorical"]
    onehot = cat_pipeline.named_steps["onehot"]
    try:
        cat_names = list(onehot.get_feature_names_out(CATEGORICAL_FEATURES))
    except AttributeError:
        cat_names = list(onehot.get_feature_names(CATEGORICAL_FEATURES))
    names.extend(cat_names)
    return names


def save_metadata(path, random_seed):
    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "allowed_solvents": ALLOWED_SOLVENTS,
        "allowed_q_scales": [q_label(v) for v in ALLOWED_QSCALES],
        "random_seed": random_seed,
        "target_definition": "argmin_q |log(conductivity_mS_cm) - log(conductivity)|",
    }
    with open(path, "w") as handle:
        json.dump(metadata, handle, indent=2)


def qscale_step(classes):
    values = sorted(float(c) for c in classes)
    if len(values) < 2:
        return 0.0
    return float(np.min(np.diff(values)))


def rmse(y_true, y_pred):
    y_true = np.array([float(v) for v in y_true])
    y_pred = np.array([float(v) for v in y_pred])
    return math.sqrt(float(np.mean((y_pred - y_true) ** 2)))
