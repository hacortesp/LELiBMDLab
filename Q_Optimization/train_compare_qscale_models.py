#!/usr/bin/env python3
"""Train and compare models for predicting system-level optimal q_scale."""

import argparse
import ast
import json
import math
import os
import re
import shutil
import sys
import tempfile
import warnings
from collections import Counter
from pathlib import Path

os.environ.setdefault("PANDAS_NO_IMPORT_PYARROW", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "codex-cache"))
sys.modules.setdefault("pyarrow", None)
warnings.filterwarnings(
    "ignore",
    message="'multi_class' was deprecated.*",
    category=FutureWarning,
)

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:
    StratifiedGroupKFold = None


DEFAULT_INPUT = (
    "/home/hacortes/Li-EMDLab/Q_Optimization/"
    "results/sparse_campaign_salt_name/conductivity_results_filtered.csv"
)
DEFAULT_OUTPUT_DIR = (
    "/home/hacortes/Li-EMDLab/Q_Optimization/"
    "results/sparse_campaign_salt_name/qscale_model_comparison"
)
EXPECTED_Q_GRID = [0.70, 0.75, 0.80, 0.85, 0.90]

COLUMN_CANDIDATES = {
    "original_row_index": ["original_row_index", "source_row_index", "row_index"],
    "doi": ["doi", "DOI"],
    "q_scale": ["q_scale", "q_sca", "charge_scale"],
    "temperature": ["temperature", "temperature_K", "temp"],
    "salt": ["anion_name", "original_salt", "salt_cluster_id", "salt"],
    "concentration": ["salt_conc", "concentration", "original_concentration"],
    "solvents": ["solvents", "solvent_names"],
    "solvent_fracs": ["solvent_fracs", "solvent_fractions"],
    "experimental_conductivity": ["conductivity", "experimental_conductivity"],
    "calculated_conductivity": ["conductivity_mS_cm", "calculated_conductivity"],
    "calculated_uncertainty": [
        "conductivity_uncertainty_mS_cm",
        "calculated_conductivity_uncertainty",
    ],
}


def str_to_bool(value):
    clean = str(value).strip().lower()
    if clean in ("1", "true", "yes", "y", "on"):
        return True
    if clean in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError("Expected true or false")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and compare q_scale prediction models."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input filtered CSV path.")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for all model comparison outputs.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Requested number of grouped cross-validation folds.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Reproducible random seed.",
    )
    parser.add_argument(
        "--overwrite",
        type=str_to_bool,
        default=True,
        help="Overwrite the output directory if it already exists: true or false.",
    )
    return parser.parse_args()


def slugify(value):
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", value.strip().lower()).strip("_")
    return re.sub(r"_+", "_", cleaned)


def q_label(value):
    if pd.isna(value):
        return ""
    return "{0:.2f}".format(float(value))


def choose_column(columns, label):
    available = set(columns)
    for candidate in COLUMN_CANDIDATES[label]:
        if candidate in available:
            return candidate
    raise ValueError(
        "Could not identify {0} column. Tried: {1}".format(
            label, ", ".join(COLUMN_CANDIDATES[label])
        )
    )


def identify_columns(df):
    return {
        label: choose_column(df.columns, label)
        for label in sorted(COLUMN_CANDIDATES.keys())
    }


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


def parse_solvent_names(value):
    names = []
    for item in parse_list_like(value):
        name = str(item).strip().strip("'\"")
        if name:
            names.append(name.upper())
    return names


def parse_fraction_values(value):
    fractions = []
    for item in parse_list_like(value):
        try:
            fractions.append(float(item))
        except Exception:
            continue
    return fractions


def stable_system_id(df, columns):
    original_col = columns["original_row_index"]
    group_cols = [
        columns["doi"],
        columns["salt"],
        columns["concentration"],
        columns["temperature"],
        columns["solvents"],
        columns["solvent_fracs"],
    ]
    check_cols = [col for col in group_cols if col in df.columns]
    original_to_systems = df.groupby(original_col, dropna=False)[check_cols].nunique(dropna=False)
    if len(original_to_systems) > 0 and (original_to_systems <= 1).all().all():
        return "row_" + df[original_col].astype(str)

    identity_cols = [
        columns["salt"],
        columns["concentration"],
        columns["temperature"],
        columns["solvents"],
        columns["solvent_fracs"],
        columns["doi"],
    ]
    return df[identity_cols].astype(str).agg("|".join, axis=1)


def finite_positive(series):
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric[np.isfinite(numeric) & (numeric > 0)]


def build_target_table(raw_df, columns, available_q_grid):
    df = raw_df.copy()
    q_col = columns["q_scale"]
    exp_col = columns["experimental_conductivity"]
    calc_col = columns["calculated_conductivity"]

    df["_system_id"] = stable_system_id(df, columns)
    df["_q_scale_num"] = pd.to_numeric(df[q_col], errors="coerce")
    df["_exp_num"] = pd.to_numeric(df[exp_col], errors="coerce")
    df["_calc_num"] = pd.to_numeric(df[calc_col], errors="coerce")
    valid_mask = (
        np.isfinite(df["_q_scale_num"])
        & np.isfinite(df["_exp_num"])
        & np.isfinite(df["_calc_num"])
        & (df["_exp_num"] > 0)
        & (df["_calc_num"] > 0)
    )
    df["_valid_for_target"] = valid_mask

    records = []
    excluded_systems = 0
    expected_grid = set([round(float(value), 8) for value in available_q_grid])

    for system_id, group in df.groupby("_system_id", sort=False, dropna=False):
        valid = group.loc[group["_valid_for_target"]].copy()
        if valid.empty:
            excluded_systems += 1
            continue

        valid["_log_error"] = (
            np.log(valid["_calc_num"].astype(float)) - np.log(valid["_exp_num"].astype(float))
        ).abs()
        min_error = float(valid["_log_error"].min())
        tied = valid.loc[np.isclose(valid["_log_error"], min_error, rtol=1e-12, atol=1e-12)].copy()
        tied = tied.sort_values("_q_scale_num")
        best = tied.iloc[0]

        q_values = sorted(set([round(float(v), 8) for v in valid["_q_scale_num"].dropna()]))
        complete_coverage = set(q_values) == expected_grid
        min_calc = float(valid["_calc_num"].min())
        max_calc = float(valid["_calc_num"].max())
        exp_value = float(best["_exp_num"])
        coverage_flag = bool(exp_value >= min_calc and exp_value <= max_calc)

        record = {
            "system_id": system_id,
            "original_row_index": best[columns["original_row_index"]],
            "doi": best[columns["doi"]],
            columns["salt"]: best[columns["salt"]],
            columns["concentration"]: best[columns["concentration"]],
            columns["temperature"]: best[columns["temperature"]],
            columns["solvents"]: best[columns["solvents"]],
            columns["solvent_fracs"]: best[columns["solvent_fracs"]],
            columns["experimental_conductivity"]: best[columns["experimental_conductivity"]],
            "q_scale_optimal": float(best["_q_scale_num"]),
            "min_log_conductivity_error": min_error,
            "all_available_q_scale_values": ";".join([q_label(v) for v in q_values]),
            "all_available_calculated_conductivities": ";".join(
                [str(value) for value in valid[columns["calculated_conductivity"]].tolist()]
            ),
            "number_of_qscale_points": int(len(q_values)),
            "tie_flag": bool(len(tied) > 1),
            "coverage_flag": coverage_flag,
            "complete_qscale_coverage_flag": bool(complete_coverage),
            "min_valid_calculated_conductivity": min_calc,
            "max_valid_calculated_conductivity": max_calc,
            "valid_simulation_rows": int(len(valid)),
            "raw_simulation_rows": int(len(group)),
        }
        records.append(record)

    targets = pd.DataFrame(records)
    if targets.empty:
        raise ValueError("No systems have valid positive experimental and calculated conductivity.")

    diagnostics = {
        "number_of_raw_simulation_rows": int(len(df)),
        "number_of_valid_conductivity_rows": int(valid_mask.sum()),
        "number_of_unique_electrolyte_systems": int(df["_system_id"].nunique(dropna=False)),
        "number_of_systems_with_complete_q_scale_coverage": int(
            targets["complete_qscale_coverage_flag"].sum()
        ),
        "number_of_systems_with_partial_q_scale_coverage": int(
            (~targets["complete_qscale_coverage_flag"]).sum()
        ),
        "number_of_systems_excluded_invalid_experimental_or_calculated": int(excluded_systems),
        "fraction_of_systems_experiment_in_simulated_range": float(targets["coverage_flag"].mean()),
    }
    for value, count in targets["q_scale_optimal"].value_counts().sort_index().items():
        diagnostics["q_scale_optimal_count_{0}".format(q_label(value))] = int(count)

    return targets, diagnostics


def add_solvent_features(targets, columns):
    table = targets.copy()
    solvent_col = columns["solvents"]
    frac_col = columns["solvent_fracs"]

    parsed_names = table[solvent_col].apply(parse_solvent_names)
    parsed_fracs = table[frac_col].apply(parse_fraction_values)
    all_solvents = sorted(set([name for names in parsed_names for name in names]))

    for solvent in all_solvents:
        clean = slugify(solvent)
        presence_col = "has_solvent_{0}".format(clean)
        fraction_col = "frac_solvent_{0}".format(clean)
        presence_values = []
        fraction_values = []
        for names, fracs in zip(parsed_names, parsed_fracs):
            presence_values.append(int(solvent in names))
            total_fraction = 0.0
            for index, name in enumerate(names):
                if name == solvent and index < len(fracs):
                    total_fraction += float(fracs[index])
            fraction_values.append(total_fraction)
        table[presence_col] = presence_values
        table[fraction_col] = fraction_values

    table["solvent_count"] = parsed_names.apply(len)
    table["solvent_set"] = parsed_names.apply(lambda names: "+".join(sorted(names)))
    return table, all_solvents


def build_training_table(targets, columns):
    table, all_solvents = add_solvent_features(targets, columns)
    table["salt_identity"] = table[columns["salt"]].astype(str)
    table["concentration_numeric"] = pd.to_numeric(table[columns["concentration"]], errors="coerce")
    table["temperature_numeric"] = pd.to_numeric(table[columns["temperature"]], errors="coerce")
    table["experimental_conductivity_numeric"] = pd.to_numeric(
        table[columns["experimental_conductivity"]], errors="coerce"
    )
    table["log_experimental_conductivity"] = np.log(table["experimental_conductivity_numeric"])

    solvent_feature_cols = []
    for solvent in all_solvents:
        clean = slugify(solvent)
        solvent_feature_cols.append("has_solvent_{0}".format(clean))
        solvent_feature_cols.append("frac_solvent_{0}".format(clean))

    numeric_features = [
        "concentration_numeric",
        "temperature_numeric",
        "experimental_conductivity_numeric",
        "log_experimental_conductivity",
        "solvent_count",
    ] + solvent_feature_cols
    categorical_features = ["salt_identity", "solvent_set"]
    feature_columns = numeric_features + categorical_features
    return table, numeric_features, categorical_features, feature_columns


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(numeric_features, categorical_features, scale_numeric):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_one_hot_encoder()),
    ]
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), numeric_features),
            ("categorical", Pipeline(categorical_steps), categorical_features),
        ]
    )


def build_models(seed, numeric_features, categorical_features):
    models = []
    regression_specs = [
        ("Ridge Regression", Ridge(alpha=1.0, random_state=seed), True),
        (
            "Elastic Net Regression",
            ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000, random_state=seed),
            True,
        ),
        (
            "Random Forest Regressor",
            RandomForestRegressor(
                n_estimators=500,
                random_state=seed,
                min_samples_leaf=2,
                n_jobs=-1,
            ),
            False,
        ),
        (
            "Extra Trees Regressor",
            ExtraTreesRegressor(
                n_estimators=500,
                random_state=seed,
                min_samples_leaf=2,
                n_jobs=-1,
            ),
            False,
        ),
        (
            "Gradient Boosting Regressor",
            GradientBoostingRegressor(random_state=seed),
            False,
        ),
    ]
    classification_specs = [
        (
            "Multinomial Logistic Regression",
            LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
                multi_class="multinomial",
                class_weight="balanced",
                random_state=seed,
            ),
            True,
        ),
        (
            "Random Forest Classifier",
            RandomForestClassifier(
                n_estimators=500,
                random_state=seed,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
            ),
            False,
        ),
        (
            "Extra Trees Classifier",
            ExtraTreesClassifier(
                n_estimators=500,
                random_state=seed,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
            ),
            False,
        ),
        (
            "Gradient Boosting Classifier",
            GradientBoostingClassifier(random_state=seed),
            False,
        ),
    ]

    for name, estimator, scale_numeric in regression_specs:
        preprocessor = make_preprocessor(numeric_features, categorical_features, scale_numeric)
        models.append(
            {
                "method_name": name,
                "model_type": "regression",
                "pipeline": Pipeline([("preprocess", preprocessor), ("model", estimator)]),
            }
        )
    for name, estimator, scale_numeric in classification_specs:
        preprocessor = make_preprocessor(numeric_features, categorical_features, scale_numeric)
        models.append(
            {
                "method_name": name,
                "model_type": "classification",
                "pipeline": Pipeline([("preprocess", preprocessor), ("model", estimator)]),
            }
        )
    return models


def nearest_q_grid(values, q_grid):
    grid = np.array(q_grid, dtype=float)
    values = np.array(values, dtype=float)
    indices = np.abs(values.reshape(-1, 1) - grid.reshape(1, -1)).argmin(axis=1)
    return grid[indices]


def rmse(y_true, y_pred):
    try:
        return math.sqrt(mean_squared_error(y_true, y_pred))
    except TypeError:
        return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def one_step_accuracy(y_true, y_pred, q_grid):
    if len(q_grid) < 2:
        step = 0.0
    else:
        step = float(np.min(np.diff(np.array(sorted(q_grid), dtype=float))))
    errors = np.abs(np.array(y_pred, dtype=float) - np.array(y_true, dtype=float))
    return float(np.mean(errors <= step + 1e-9))


def evaluate_predictions(y_true, y_pred, q_grid):
    y_true_arr = np.array(y_true, dtype=float)
    y_pred_arr = np.array(y_pred, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "RMSE": float(rmse(y_true_arr, y_pred_arr)),
        "median_absolute_error": float(np.median(np.abs(y_pred_arr - y_true_arr))),
        "exact_qscale_accuracy": float(np.mean(np.isclose(y_pred_arr, y_true_arr, atol=1e-9))),
        "within_one_step_accuracy": one_step_accuracy(y_true_arr, y_pred_arr, q_grid),
    }


def determine_cv(y_class, groups, requested_folds, warnings):
    class_counts = Counter(y_class)
    n_systems = len(y_class)
    min_class_count = min(class_counts.values())
    if min_class_count >= 2 and StratifiedGroupKFold is not None:
        n_splits = max(2, min(int(requested_folds), int(min_class_count), int(n_systems)))
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
        splits = list(splitter.split(np.zeros(n_systems), y_class, groups))
        cv_name = "StratifiedGroupKFold"
    else:
        n_splits = max(2, min(int(requested_folds), int(n_systems)))
        splitter = GroupKFold(n_splits=n_splits)
        splits = list(splitter.split(np.zeros(n_systems), y_class, groups))
        cv_name = "GroupKFold"
        if StratifiedGroupKFold is None:
            warnings.append("StratifiedGroupKFold is not available; used GroupKFold.")
        if min_class_count < 2:
            warnings.append(
                "At least one q_scale_optimal class has fewer than 2 systems; stratified validation is unreliable."
            )
    return splits, cv_name


def fit_predict_oof(model_spec, X, y_numeric, y_class, groups, splits, q_grid, seed):
    pipeline = model_spec["pipeline"]
    model_type = model_spec["model_type"]
    n = len(X)
    raw_pred = np.full(n, np.nan, dtype=float)
    snapped_pred = np.full(n, np.nan, dtype=float)
    fold_ids = np.full(n, -1, dtype=int)

    for fold_index, (train_idx, val_idx) in enumerate(splits, start=1):
        fold_pipeline = clone(pipeline)
        if model_type == "classification":
            fold_pipeline.fit(X.iloc[train_idx], y_class[train_idx])
            pred = fold_pipeline.predict(X.iloc[val_idx])
            pred_numeric = np.array([float(value) for value in pred], dtype=float)
            raw_pred[val_idx] = pred_numeric
            snapped_pred[val_idx] = pred_numeric
        else:
            fold_pipeline.fit(X.iloc[train_idx], y_numeric[train_idx])
            pred = fold_pipeline.predict(X.iloc[val_idx])
            raw_pred[val_idx] = np.array(pred, dtype=float)
            snapped_pred[val_idx] = nearest_q_grid(pred, q_grid)
        fold_ids[val_idx] = fold_index

    final_pipeline = clone(pipeline)
    if model_type == "classification":
        final_pipeline.fit(X, y_class)
    else:
        final_pipeline.fit(X, y_numeric)

    return raw_pred, snapped_pred, fold_ids, final_pipeline


def save_predicted_vs_true(path, true_values, predicted_values, title, seed, q_grid):
    rng = np.random.RandomState(seed)
    true_jitter = np.array(true_values, dtype=float) + rng.normal(0, 0.003, size=len(true_values))
    pred_jitter = np.array(predicted_values, dtype=float) + rng.normal(0, 0.003, size=len(predicted_values))
    plt.figure(figsize=(6.5, 5.5))
    plt.scatter(true_jitter, pred_jitter, alpha=0.75, edgecolor="black", linewidth=0.4)
    axis_min = min(min(q_grid), np.nanmin(predicted_values)) - 0.035
    axis_max = max(max(q_grid), np.nanmax(predicted_values)) + 0.035
    plt.plot([axis_min, axis_max], [axis_min, axis_max], color="black", linestyle="--", linewidth=1)
    plt.xticks(q_grid, [q_label(v) for v in q_grid])
    plt.yticks(q_grid, [q_label(v) for v in q_grid])
    plt.xlim(axis_min, axis_max)
    plt.ylim(axis_min, axis_max)
    plt.xlabel("True q_scale")
    plt.ylabel("Predicted q_scale")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_residual_distribution(path, true_values, predicted_values, q_grid, title):
    residuals = np.array(predicted_values, dtype=float) - np.array(true_values, dtype=float)
    step = 0.05 if len(q_grid) < 2 else float(np.min(np.diff(np.array(q_grid, dtype=float))))
    bins = np.arange(residuals.min() - step, residuals.max() + 2 * step, step)
    plt.figure(figsize=(6.5, 4.8))
    plt.hist(residuals, bins=bins, color="#4C78A8", edgecolor="white")
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Predicted q_scale - true q_scale")
    plt.ylabel("Number of systems")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_confusion_matrix(path, true_values, predicted_values, q_grid, title):
    labels = [q_label(value) for value in q_grid]
    true_labels = [q_label(value) for value in true_values]
    predicted_labels = [q_label(value) for value in predicted_values]
    matrix = confusion_matrix(true_labels, predicted_labels, labels=labels)
    plt.figure(figsize=(6.2, 5.3))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar(label="Count")
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
    plt.xlabel("Predicted q_scale")
    plt.ylabel("True q_scale")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_error_by_true_qscale(path, predictions, q_grid, title):
    data = predictions.copy()
    data["absolute_error"] = (data["predicted_q_scale"] - data["true_q_scale"]).abs()
    summary = data.groupby("true_q_scale", sort=True)["absolute_error"].mean().reindex(q_grid)
    plt.figure(figsize=(6.5, 4.8))
    plt.bar([q_label(v) for v in summary.index], summary.values, color="#F58518")
    plt.xlabel("True q_scale")
    plt.ylabel("Mean absolute q_scale error")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_global_metric_plots(output_dir, summary):
    ordered = summary.sort_values(["MAE", "RMSE", "exact_qscale_accuracy"], ascending=[True, True, False])
    x = np.arange(len(ordered))
    width = 0.38

    plt.figure(figsize=(max(9, len(ordered) * 1.1), 5.2))
    plt.bar(x - width / 2.0, ordered["MAE"], width, label="MAE")
    plt.bar(x + width / 2.0, ordered["RMSE"], width, label="RMSE")
    plt.xticks(x, ordered["method_name"], rotation=45, ha="right")
    plt.ylabel("q_scale units")
    plt.title("Cross-validated MAE and RMSE by method")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison_mae_rmse.png", dpi=300)
    plt.close()

    plt.figure(figsize=(max(9, len(ordered) * 1.1), 5.2))
    plt.bar(x - width / 2.0, ordered["exact_qscale_accuracy"], width, label="Exact accuracy")
    plt.bar(x + width / 2.0, ordered["within_one_step_accuracy"], width, label="Within one step")
    plt.xticks(x, ordered["method_name"], rotation=45, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.title("Cross-validated q_scale accuracy by method")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison_accuracy.png", dpi=300)
    plt.close()


def save_global_predicted_vs_true(output_dir, all_predictions, q_grid, seed):
    method_names = sorted(all_predictions["method_name"].unique())
    n_methods = len(method_names)
    ncols = 3
    nrows = int(math.ceil(float(n_methods) / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.2 * nrows), squeeze=False)
    rng = np.random.RandomState(seed)
    axis_min = min(q_grid) - 0.035
    axis_max = max(q_grid) + 0.035
    for index, method_name in enumerate(method_names):
        ax = axes[index // ncols][index % ncols]
        data = all_predictions.loc[all_predictions["method_name"] == method_name]
        true_jitter = data["true_q_scale"].values + rng.normal(0, 0.003, size=len(data))
        pred_jitter = data["predicted_q_scale"].values + rng.normal(0, 0.003, size=len(data))
        ax.scatter(true_jitter, pred_jitter, alpha=0.7, edgecolor="black", linewidth=0.3)
        ax.plot([axis_min, axis_max], [axis_min, axis_max], color="black", linestyle="--", linewidth=1)
        ax.set_xlim(axis_min, axis_max)
        ax.set_ylim(axis_min, axis_max)
        ax.set_xticks(q_grid)
        ax.set_yticks(q_grid)
        ax.set_xticklabels([q_label(v) for v in q_grid], rotation=45)
        ax.set_yticklabels([q_label(v) for v in q_grid])
        ax.set_title(method_name)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
    for index in range(n_methods, nrows * ncols):
        axes[index // ncols][index % ncols].axis("off")
    fig.suptitle("Out-of-fold predicted versus true q_scale", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison_global_predicted_vs_true.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_target_distribution(output_dir, targets, q_grid):
    counts = targets["q_scale_optimal"].value_counts().sort_index().reindex(q_grid).fillna(0)
    plt.figure(figsize=(6.5, 4.8))
    plt.bar([q_label(v) for v in counts.index], counts.values, color="#54A24B")
    plt.xlabel("Optimal q_scale")
    plt.ylabel("Number of systems")
    plt.title("True optimal q_scale distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "true_optimal_qscale_distribution.png", dpi=300)
    plt.close()


def save_error_by_salt(output_dir, all_predictions, salt_col, warnings):
    counts = all_predictions.drop_duplicates("system_id")[salt_col].value_counts()
    salts = counts[counts >= 2].index.tolist()
    path = output_dir / "qscale_error_by_salt.png"
    if not salts:
        warnings.append("Not enough systems per salt to make a reliable qscale_error_by_salt plot.")
        plt.figure(figsize=(7, 4))
        plt.text(0.5, 0.5, "Not enough systems per salt", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()
        return
    data = all_predictions.loc[all_predictions[salt_col].isin(salts)].copy()
    data["absolute_error"] = (data["predicted_q_scale"] - data["true_q_scale"]).abs()
    pivot = data.pivot_table(
        index=salt_col,
        columns="method_name",
        values="absolute_error",
        aggfunc="mean",
    )
    pivot = pivot.loc[sorted(pivot.index)]
    plt.figure(figsize=(max(9, len(pivot.columns) * 1.2), 5.2))
    x = np.arange(len(pivot.index))
    width = 0.8 / max(1, len(pivot.columns))
    for i, method in enumerate(pivot.columns):
        plt.bar(x - 0.4 + width / 2.0 + i * width, pivot[method].values, width, label=method)
    plt.xticks(x, pivot.index)
    plt.ylabel("Mean absolute q_scale error")
    plt.xlabel("Salt identity")
    plt.title("q_scale prediction error by salt")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def write_diagnostics(output_dir, diagnostics, warnings, available_q_grid, cv_name, n_folds):
    rows = []
    for key, value in diagnostics.items():
        rows.append({"metric": key, "value": value})
    rows.append({"metric": "actual_available_q_scale_values", "value": ";".join([q_label(v) for v in available_q_grid])})
    rows.append({"metric": "cross_validation_method", "value": cv_name})
    rows.append({"metric": "number_of_folds_used", "value": n_folds})
    pd.DataFrame(rows).to_csv(output_dir / "qscale_target_diagnostics.csv", index=False)
    with open(output_dir / "training_warnings.txt", "w") as handle:
        if warnings:
            for warning in warnings:
                handle.write(warning + "\n")
        else:
            handle.write("No major warnings.\n")


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    warnings = []

    if output_dir.exists() and not args.overwrite:
        raise ValueError("Output directory already exists and --overwrite is false: {0}".format(output_dir))
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(str(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    columns = identify_columns(raw_df)
    print("Available column names:")
    for column in raw_df.columns:
        print("  - {0}".format(column))
    print("Identified column mapping:")
    for key in sorted(columns.keys()):
        print("  - {0}: {1}".format(key, columns[key]))

    q_values = pd.to_numeric(raw_df[columns["q_scale"]], errors="coerce")
    available_q_grid = sorted([float(value) for value in q_values.dropna().unique()])
    print("Actual available q_scale values: {0}".format(", ".join([q_label(v) for v in available_q_grid])))
    expected_missing = sorted(set([round(v, 8) for v in EXPECTED_Q_GRID]) - set([round(v, 8) for v in available_q_grid]))
    if expected_missing:
        warnings.append("Some expected q_scale values are missing: {0}".format(", ".join([q_label(v) for v in expected_missing])))

    targets, diagnostics = build_target_table(raw_df, columns, available_q_grid)
    targets.to_csv(output_dir / "optimal_qscale_targets.csv", index=False)

    training_table, numeric_features, categorical_features, feature_columns = build_training_table(targets, columns)
    training_table.to_csv(output_dir / "system_level_training_table.csv", index=False)

    y_numeric = training_table["q_scale_optimal"].astype(float).values
    y_class = np.array([q_label(value) for value in y_numeric])
    groups = training_table["system_id"].astype(str).values
    class_counts = Counter(y_class)
    largest_fraction = float(max(class_counts.values())) / float(len(y_class))
    if largest_fraction >= 0.70:
        warnings.append(
            "Targets are strongly imbalanced: largest q_scale_optimal class contains {0:.1%} of systems.".format(
                largest_fraction
            )
        )
    for label, count in sorted(class_counts.items()):
        if count < 5:
            warnings.append(
                "q_scale_optimal class {0} has only {1} systems; validation reliability is limited.".format(
                    label, count
                )
            )

    print("q_scale_optimal class counts:")
    for label, count in sorted(class_counts.items()):
        print("  - {0}: {1}".format(label, count))

    splits, cv_name = determine_cv(y_class, groups, args.folds, warnings)
    n_folds = len(splits)
    X = training_table[feature_columns].copy()

    models = build_models(args.random_seed, numeric_features, categorical_features)
    summary_rows = []
    all_predictions = []

    for model_spec in models:
        method_name = model_spec["method_name"]
        method_dir = output_dir / slugify(method_name)
        method_dir.mkdir(parents=True, exist_ok=True)
        raw_pred, snapped_pred, fold_ids, fitted_pipeline = fit_predict_oof(
            model_spec,
            X,
            y_numeric,
            y_class,
            groups,
            splits,
            available_q_grid,
            args.random_seed,
        )
        if model_spec["model_type"] == "regression":
            final_pred = snapped_pred
            raw_metrics = evaluate_predictions(y_numeric, raw_pred, available_q_grid)
            raw_metric_rows = [
                {"metric": "raw_" + key, "value": value}
                for key, value in raw_metrics.items()
            ]
        else:
            final_pred = snapped_pred
            raw_metric_rows = []

        metrics = evaluate_predictions(y_numeric, final_pred, available_q_grid)
        if model_spec["model_type"] == "classification":
            true_labels = np.array([q_label(value) for value in y_numeric])
            pred_labels = np.array([q_label(value) for value in final_pred])
            metric_labels = [q_label(value) for value in available_q_grid]
            metrics["macro_F1"] = float(
                f1_score(
                    true_labels,
                    pred_labels,
                    labels=metric_labels,
                    average="macro",
                    zero_division=0,
                )
            )
            metrics["balanced_accuracy"] = float(balanced_accuracy_score(true_labels, pred_labels))
            snapping_note = "classification predicts discrete q_scale classes; no snapping needed"
        else:
            metrics["macro_F1"] = np.nan
            metrics["balanced_accuracy"] = np.nan
            snapping_note = "continuous predictions snapped to nearest available q_scale for final comparison"

        predictions = training_table[
            [
                "system_id",
                "original_row_index",
                "doi",
                columns["salt"],
                columns["concentration"],
                columns["temperature"],
                columns["solvents"],
                columns["solvent_fracs"],
            ]
        ].copy()
        predictions["fold"] = fold_ids
        predictions["true_q_scale"] = y_numeric
        predictions["raw_predicted_q_scale"] = raw_pred
        predictions["predicted_q_scale"] = final_pred
        predictions["q_scale_residual"] = predictions["predicted_q_scale"] - predictions["true_q_scale"]
        predictions["absolute_q_scale_error"] = predictions["q_scale_residual"].abs()
        predictions["method_name"] = method_name
        predictions["model_type"] = model_spec["model_type"]
        predictions.to_csv(method_dir / "out_of_fold_predictions.csv", index=False)

        metric_rows = [{"metric": key, "value": value} for key, value in metrics.items()]
        metric_rows.extend(raw_metric_rows)
        pd.DataFrame(metric_rows).to_csv(method_dir / "metrics.csv", index=False)
        joblib.dump(fitted_pipeline, method_dir / "fitted_model.joblib")
        joblib.dump(fitted_pipeline.named_steps["preprocess"], method_dir / "preprocessing_pipeline.joblib")

        title = "{0}: MAE={1:.3f}, RMSE={2:.3f}".format(method_name, metrics["MAE"], metrics["RMSE"])
        save_predicted_vs_true(
            method_dir / "predicted_vs_true_qscale.png",
            y_numeric,
            final_pred,
            title,
            args.random_seed,
            available_q_grid,
        )
        save_residual_distribution(
            method_dir / "residual_distribution.png",
            y_numeric,
            final_pred,
            available_q_grid,
            "{0} residual distribution".format(method_name),
        )
        save_error_by_true_qscale(
            method_dir / "qscale_error_by_true_qscale.png",
            predictions,
            available_q_grid,
            "{0} error by true q_scale".format(method_name),
        )
        if model_spec["model_type"] == "classification":
            save_confusion_matrix(
                method_dir / "confusion_matrix.png",
                y_numeric,
                final_pred,
                available_q_grid,
                "{0} confusion matrix".format(method_name),
            )

        summary_row = {
            "method_name": method_name,
            "model_type": model_spec["model_type"],
            "number_of_systems": int(len(training_table)),
            "number_of_folds": int(n_folds),
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "median_absolute_error": metrics["median_absolute_error"],
            "exact_qscale_accuracy": metrics["exact_qscale_accuracy"],
            "within_one_step_accuracy": metrics["within_one_step_accuracy"],
            "macro_F1": metrics["macro_F1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "notes": snapping_note,
        }
        summary_rows.append(summary_row)
        all_predictions.append(predictions)

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values(
        ["MAE", "RMSE", "exact_qscale_accuracy"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    summary.to_csv(output_dir / "model_comparison_summary.csv", index=False)

    all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    all_predictions_df.to_csv(output_dir / "all_methods_out_of_fold_predictions.csv", index=False)
    save_global_metric_plots(output_dir, summary)
    save_global_predicted_vs_true(output_dir, all_predictions_df, available_q_grid, args.random_seed)
    save_target_distribution(output_dir, targets, available_q_grid)
    save_error_by_salt(output_dir, all_predictions_df, columns["salt"], warnings)

    metadata = {
        "input_csv": str(input_path),
        "output_dir": str(output_dir),
        "column_mapping": columns,
        "available_q_scale_values": available_q_grid,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "feature_columns": feature_columns,
        "cross_validation_method": cv_name,
        "number_of_folds": n_folds,
        "random_seed": args.random_seed,
    }
    with open(output_dir / "training_metadata.json", "w") as handle:
        json.dump(metadata, handle, indent=2)
    write_diagnostics(output_dir, diagnostics, warnings, available_q_grid, cv_name, n_folds)

    best = summary.iloc[0]
    print("Best-performing method: {0}".format(best["method_name"]))
    print("Cross-validated MAE: {0:.4f}".format(best["MAE"]))
    print("Cross-validated RMSE: {0:.4f}".format(best["RMSE"]))
    print("Exact q_scale accuracy: {0:.4f}".format(best["exact_qscale_accuracy"]))
    print("Output directory: {0}".format(output_dir))
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print("  - {0}".format(warning))
    else:
        print("Warnings: none")


if __name__ == "__main__":
    main()
