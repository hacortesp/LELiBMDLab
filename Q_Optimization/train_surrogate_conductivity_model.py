#!/usr/bin/env python3
"""Train a q-scale-aware surrogate model for simulated ionic conductivity.

Step E learns the sparse MD campaign response surface

    conductivity_mS_cm = f(electrolyte descriptors, q_scale)

while keeping all q_scale values for the same electrolyte system in the same
train/validation/test group. The saved model predicts simulated conductivity,
not experimental conductivity.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "codex-cache"))

# The local environment may contain an optional pyarrow build compiled against
# an incompatible NumPy. Pandas works without pyarrow, so mark it unavailable.
sys.modules.setdefault("pyarrow", None)

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_SEED = 42
DEFAULT_SUCCESS_STATUSES = ("parsed", "completed", "complete", "success", "successful", "ok")
TARGET_COLUMN = "conductivity_ms_cm"
UNCERTAINTY_COLUMN = "conductivity_uncertainty_ms_cm"

IDENTIFIER_AND_METADATA_COLUMNS = {
    "job_id",
    "folder_name",
    "collection_status",
    "conductivity_raw_line",
    "folder_path",
    "output_file",
    "cluster_id",
    "original_row_index",
    "doi",
    "representative_doi",
    "cluster_rank_in_sampling",
    "selection_reason",
    "source_row_number",
    "composition_summary",
    "duplicate_identity",
    "cluster_size",
    "cluster_sampling_order",
    "electrolyte_group",
    "simulation_key",
    "n_collapsed_simulations",
}

TARGET_AND_LEAKAGE_COLUMNS = {
    TARGET_COLUMN,
    UNCERTAINTY_COLUMN,
    "conductivity",  # Experimental conductivity from descriptor generation.
    "conductivity_descriptor",
}


@dataclass(frozen=True)
class TargetTransform:
    """Forward and inverse transforms for the regression target."""

    name: str
    forward: Callable[[np.ndarray], np.ndarray]
    inverse: Callable[[np.ndarray], np.ndarray]


def standardize_column_name(column: str) -> str:
    """Return a stable snake_case column name."""

    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", column.strip()).strip("_").lower()
    return re.sub(r"_+", "_", cleaned)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize DataFrame column names."""

    renamed = df.copy()
    renamed.columns = [standardize_column_name(column) for column in renamed.columns]
    return renamed


def read_csv_standardized(path: Path) -> pd.DataFrame:
    """Read a CSV and normalize column names."""

    return standardize_columns(pd.read_csv(path))


def coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Coerce selected columns to numeric values, setting invalid entries to NaN."""

    converted = df.copy()
    for column in columns:
        if column in converted.columns:
            converted[column] = pd.to_numeric(converted[column], errors="coerce")
    return converted


def load_campaign_results(input_csv: Path, success_statuses: tuple[str, ...]) -> pd.DataFrame:
    """Load campaign results and keep rows with successful collection status."""

    results = read_csv_standardized(input_csv)
    if "collection_status" not in results.columns:
        raise ValueError("Input CSV must contain collection_status.")
    if TARGET_COLUMN not in results.columns:
        raise ValueError(f"Input CSV must contain {TARGET_COLUMN}.")
    if "q_scale" not in results.columns:
        raise ValueError("Input CSV must contain q_scale.")

    normalized_status = results["collection_status"].astype(str).str.strip().str.lower()
    keep_statuses = {status.lower() for status in success_statuses}
    results = results.loc[normalized_status.isin(keep_statuses)].copy()

    numeric_columns = [
        TARGET_COLUMN,
        UNCERTAINTY_COLUMN,
        "q_scale",
        "original_row_index",
        "salt_conc",
        "temperature",
    ]
    results = coerce_numeric(results, numeric_columns)
    results = results.replace([np.inf, -np.inf], np.nan)
    results = results.dropna(subset=[TARGET_COLUMN, "q_scale"])
    results = results.loc[results[TARGET_COLUMN] > 0].copy()

    if results.empty:
        raise ValueError("No successful rows with positive conductivity and numeric q_scale were found.")
    return results


def merge_descriptor_data(results: pd.DataFrame, descriptor_csv: Path | None) -> pd.DataFrame:
    """Merge descriptor columns by original_row_index when a descriptor table is available."""

    if descriptor_csv is None:
        return results
    if not descriptor_csv.is_file():
        print(f"Descriptor file not found, using campaign columns only: {descriptor_csv}")
        return results
    if "original_row_index" not in results.columns:
        print("original_row_index missing from campaign results; using campaign columns only.")
        return results

    descriptors = read_csv_standardized(descriptor_csv)
    if "original_row_index" not in descriptors.columns:
        print(f"original_row_index missing from descriptor file; using campaign columns only: {descriptor_csv}")
        return results

    descriptors = descriptors.drop_duplicates(subset=["original_row_index"]).copy()
    merged = results.merge(descriptors, on="original_row_index", how="left", suffixes=("", "_descriptor"))

    # Prefer campaign values for overlapping columns, but fill gaps from the
    # descriptor matrix and expose descriptor-only columns under their base name.
    for column in list(merged.columns):
        if not column.endswith("_descriptor"):
            continue
        base = column.removesuffix("_descriptor")
        if base in merged.columns:
            merged[base] = merged[base].combine_first(merged[column])
            merged = merged.drop(columns=[column])
        else:
            merged = merged.rename(columns={column: base})
    return merged


def build_group_key(df: pd.DataFrame) -> pd.Series:
    """Build an electrolyte-system group key for leakage-safe splitting."""

    if "duplicate_identity" in df.columns and df["duplicate_identity"].notna().any():
        duplicate_identity = df["duplicate_identity"].astype(str).str.strip()
        if duplicate_identity.ne("").any():
            fallback = pd.Series("row_" + df.index.astype(str), index=df.index)
            return duplicate_identity.where(duplicate_identity.ne(""), other=np.nan).fillna(fallback)

    fallback_columns = [
        "original_row_index",
        "doi",
        "cation_name",
        "anion_name",
        "salt_conc",
        "solvents",
        "solvent_fracs",
        "temperature",
        "cluster_id",
    ]
    available = [column for column in fallback_columns if column in df.columns]
    if not available:
        return pd.Series(["row_" + str(index) for index in df.index], index=df.index)
    return df[available].astype(str).agg("|".join, axis=1)


def collapse_duplicate_simulations(df: pd.DataFrame) -> pd.DataFrame:
    """Average duplicate simulations for the same electrolyte group and q_scale."""

    collapsed = df.copy()
    collapsed["electrolyte_group"] = build_group_key(collapsed)
    collapsed["simulation_key"] = (
        collapsed["electrolyte_group"].astype(str)
        + "|q="
        + collapsed["q_scale"].round(8).astype(str)
    )
    sizes = collapsed.groupby("simulation_key", sort=False).size()

    numeric_columns = collapsed.select_dtypes(include=[np.number]).columns.tolist()
    aggregations: dict[str, str] = {column: "mean" for column in numeric_columns}
    for column in collapsed.columns:
        if column not in aggregations and column != "simulation_key":
            aggregations[column] = "first"

    collapsed = collapsed.groupby("simulation_key", as_index=False, sort=False).agg(aggregations)
    collapsed["n_collapsed_simulations"] = collapsed["simulation_key"].map(sizes).astype(int)
    return collapsed


def feature_columns(df: pd.DataFrame, include_categorical: bool) -> tuple[list[str], list[str]]:
    """Choose model feature columns while excluding IDs, metadata, and leakage columns."""

    forbidden = IDENTIFIER_AND_METADATA_COLUMNS | TARGET_AND_LEAKAGE_COLUMNS
    numeric_features = [
        column
        for column in df.select_dtypes(include=[np.number]).columns
        if column not in forbidden
    ]
    if "q_scale" not in numeric_features:
        numeric_features.append("q_scale")

    categorical_features: list[str] = []
    if include_categorical:
        candidates = ["cation_name", "anion_name", "solvents", "solvent_fracs"]
        categorical_features = [
            column
            for column in candidates
            if column in df.columns and column not in forbidden
        ]

    return sorted(dict.fromkeys(numeric_features)), categorical_features


def make_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """Create the preprocessing pipeline."""

    transformers: list[tuple[str, Pipeline, list[str]]] = [
        (
            "numeric",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric_features,
        )
    ]
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def candidate_models(random_seed: int, n_rows: int, gpr_threshold: int) -> dict[str, BaseEstimator]:
    """Return candidate regressors for the surrogate comparison."""

    models: dict[str, BaseEstimator] = {
        "ridge": Ridge(alpha=1.0, random_state=random_seed),
        "random_forest": RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            random_state=random_seed,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            random_state=random_seed,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(random_state=random_seed),
    }
    if n_rows <= gpr_threshold:
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0) + WhiteKernel(
            noise_level=1e-3
        )
        models["gaussian_process"] = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=random_seed,
        )
    return models


def target_transforms(mode: str) -> list[TargetTransform]:
    """Return target transforms to evaluate."""

    transforms = {
        "raw": TargetTransform("raw", lambda y: y, lambda y: y),
        "log1p": TargetTransform("log1p", np.log1p, np.expm1),
    }
    if mode == "auto":
        return [transforms["raw"], transforms["log1p"]]
    return [transforms[mode]]


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> dict[str, float]:
    """Compute MAE, RMSE, and R2 metrics."""

    residual = y_pred - y_true
    metrics = {
        f"{prefix}mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}rmse": float(np.sqrt(np.mean(residual**2))),
    }
    metrics[f"{prefix}r2"] = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan
    return metrics


def evaluate_candidates(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: pd.Series,
    preprocessor: ColumnTransformer,
    models: dict[str, BaseEstimator],
    transforms: list[TargetTransform],
    cv_splits: int,
) -> pd.DataFrame:
    """Run grouped cross-validation for every model/target-transform pair."""

    unique_groups = groups.nunique()
    if unique_groups < 2:
        raise ValueError("At least two electrolyte groups are required for grouped cross-validation.")
    n_splits = min(cv_splits, unique_groups)
    splitter = GroupKFold(n_splits=n_splits)
    records: list[dict[str, object]] = []

    for transform in transforms:
        y_model = transform.forward(y)
        for model_name, model in models.items():
            fold_rows = []
            for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y_model, groups), start=1):
                pipeline = Pipeline(
                    steps=[
                        ("preprocess", clone(preprocessor)),
                        ("model", clone(model)),
                    ]
                )
                pipeline.fit(X.iloc[train_idx], y_model[train_idx])
                pred_model = pipeline.predict(X.iloc[valid_idx])
                pred = transform.inverse(pred_model)

                fold_metrics = {
                    "model": model_name,
                    "target_transform": transform.name,
                    "fold": fold,
                    "n_train": len(train_idx),
                    "n_valid": len(valid_idx),
                }
                fold_metrics.update(regression_metrics(y[valid_idx], pred))
                fold_metrics.update(
                    regression_metrics(y_model[valid_idx], pred_model, prefix="transformed_")
                )
                fold_rows.append(fold_metrics)

            fold_df = pd.DataFrame(fold_rows)
            summary = {
                "model": model_name,
                "target_transform": transform.name,
                "n_folds": n_splits,
            }
            for metric in ["mae", "rmse", "r2", "transformed_mae", "transformed_rmse", "transformed_r2"]:
                summary[f"cv_{metric}_mean"] = float(fold_df[metric].mean())
                summary[f"cv_{metric}_std"] = float(fold_df[metric].std(ddof=0))
            records.append(summary)

    return pd.DataFrame(records).sort_values(["cv_rmse_mean", "cv_mae_mean"]).reset_index(drop=True)


def train_test_split_groups(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: pd.Series,
    test_size: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create one group-aware train/test split."""

    unique_groups = groups.nunique()
    if unique_groups < 3:
        raise ValueError("At least three electrolyte groups are required for a grouped test split.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    return next(splitter.split(X, y, groups))


def build_pipeline(
    preprocessor: ColumnTransformer,
    models: dict[str, BaseEstimator],
    model_name: str,
) -> Pipeline:
    """Build a preprocessing + estimator pipeline."""

    return Pipeline(
        steps=[
            ("preprocess", clone(preprocessor)),
            ("model", clone(models[model_name])),
        ]
    )


def get_feature_names(pipeline: Pipeline, original_features: list[str]) -> list[str]:
    """Return post-preprocessing feature names where available."""

    preprocessor = pipeline.named_steps["preprocess"]
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        return original_features


def compute_feature_importance(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y_model: np.ndarray,
    transform_name: str,
    random_seed: int,
) -> pd.DataFrame:
    """Compute model feature importance or fall back to permutation importance."""

    estimator = pipeline.named_steps["model"]
    feature_names = get_feature_names(pipeline, X.columns.tolist())

    if hasattr(estimator, "feature_importances_"):
        importance = np.asarray(estimator.feature_importances_, dtype=float)
        return pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importance,
                "importance_type": "model_feature_importance",
                "target_transform": transform_name,
            }
        ).sort_values("importance", ascending=False)

    if hasattr(estimator, "coef_"):
        coefficients = np.ravel(estimator.coef_).astype(float)
        return pd.DataFrame(
            {
                "feature": feature_names,
                "importance": np.abs(coefficients),
                "signed_coefficient": coefficients,
                "importance_type": "absolute_coefficient",
                "target_transform": transform_name,
            }
        ).sort_values("importance", ascending=False)

    permutation = permutation_importance(
        pipeline,
        X,
        y_model,
        n_repeats=30,
        random_state=random_seed,
        scoring="neg_root_mean_squared_error",
    )
    return pd.DataFrame(
        {
            "feature": X.columns,
            "importance": permutation.importances_mean,
            "importance_std": permutation.importances_std,
            "importance_type": "permutation_importance_transformed_target",
            "target_transform": transform_name,
        }
    ).sort_values("importance", ascending=False)


def save_prediction_plots(predictions: pd.DataFrame, feature_importance: pd.DataFrame, output_dir: Path) -> None:
    """Save diagnostic plots for surrogate quality and interpretation."""

    y_true = predictions["true_conductivity_ms_cm"].to_numpy()
    y_pred = predictions["predicted_conductivity_ms_cm"].to_numpy()
    residual = predictions["residual_ms_cm"].to_numpy()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, color="#287c8e", edgecolor="black", linewidth=0.4, alpha=0.85)
    min_value = float(np.nanmin([y_true.min(), y_pred.min()]))
    max_value = float(np.nanmax([y_true.max(), y_pred.max()]))
    ax.plot([min_value, max_value], [min_value, max_value], color="#333333", linestyle="--", linewidth=1)
    ax.set_xlabel("True conductivity (mS/cm)")
    ax.set_ylabel("Predicted conductivity (mS/cm)")
    ax.set_title("Predicted vs true simulated conductivity")
    fig.tight_layout()
    fig.savefig(output_dir / "predicted_vs_true.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(residual, bins=min(12, max(4, len(residual))), color="#7a9e44", edgecolor="black")
    axes[0].axvline(0.0, color="#333333", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Residual (mS/cm)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Residual distribution")

    axes[1].scatter(predictions["q_scale"], residual, color="#b65f2a", edgecolor="black", linewidth=0.4)
    axes[1].axhline(0.0, color="#333333", linestyle="--", linewidth=1)
    axes[1].set_xlabel("q_scale")
    axes[1].set_ylabel("Residual (mS/cm)")
    axes[1].set_title("Residuals vs q_scale")

    if "weighted_permittivity" in predictions.columns:
        x = predictions["weighted_permittivity"]
        xlabel = "Weighted permittivity"
    else:
        x = predictions["true_conductivity_ms_cm"]
        xlabel = "True conductivity (mS/cm)"
    axes[2].scatter(x, residual, color="#5b5ea6", edgecolor="black", linewidth=0.4)
    axes[2].axhline(0.0, color="#333333", linestyle="--", linewidth=1)
    axes[2].set_xlabel(xlabel)
    axes[2].set_ylabel("Residual (mS/cm)")
    axes[2].set_title(f"Residuals vs {xlabel.lower()}")
    fig.tight_layout()
    fig.savefig(output_dir / "residuals.png", dpi=300)
    plt.close(fig)

    top = feature_importance.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, max(4, 0.28 * len(top))))
    ax.barh(top["feature"], top["importance"], color="#4c78a8")
    ax.set_xlabel("Importance")
    ax.set_title("Top surrogate features")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=300)
    plt.close(fig)


def write_summary(
    output_dir: Path,
    best_row: pd.Series,
    test_metrics: dict[str, float],
    feature_importance: pd.DataFrame,
    n_rows: int,
    n_groups: int,
) -> None:
    """Write a compact text summary of the selected surrogate."""

    q_scale_rank = None
    ranked = feature_importance.reset_index(drop=True)
    if "q_scale" in ranked["feature"].astype(str).tolist():
        q_scale_rank = int(ranked.index[ranked["feature"].astype(str) == "q_scale"][0]) + 1
    else:
        matches = ranked[ranked["feature"].astype(str).str.endswith("__q_scale")]
        if not matches.empty:
            q_scale_rank = int(matches.index[0]) + 1

    lines = [
        "Step E surrogate model summary",
        "================================",
        f"training_rows_after_collapse: {n_rows}",
        f"electrolyte_groups: {n_groups}",
        f"best_model: {best_row['model']}",
        f"target_transform: {best_row['target_transform']}",
        f"cv_rmse_mean_mS_cm: {best_row['cv_rmse_mean']:.6g}",
        f"cv_mae_mean_mS_cm: {best_row['cv_mae_mean']:.6g}",
        f"cv_r2_mean: {best_row['cv_r2_mean']:.6g}",
        f"test_rmse_mS_cm: {test_metrics['rmse']:.6g}",
        f"test_mae_mS_cm: {test_metrics['mae']:.6g}",
        f"test_r2: {test_metrics['r2']:.6g}",
        f"q_scale_importance_rank: {q_scale_rank if q_scale_rank is not None else 'not_available'}",
        "",
        "Top features:",
    ]
    for _, row in feature_importance.head(10).iterrows():
        lines.append(f"- {row['feature']}: {row['importance']:.6g}")

    (output_dir / "surrogate_model_summary.txt").write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> None:
    """Run the full Step E training workflow."""

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_campaign_results(args.input_csv, tuple(args.success_statuses.split(",")))
    merged = merge_descriptor_data(results, args.descriptor_csv)
    training_data = collapse_duplicate_simulations(merged)

    numeric_features, categorical_features = feature_columns(training_data, args.include_categorical)
    selected_features = numeric_features + categorical_features
    if not selected_features:
        raise ValueError("No usable feature columns were found.")

    keep_columns = [
        column
        for column in [
            "simulation_key",
            "electrolyte_group",
            "original_row_index",
            "duplicate_identity",
            "cluster_id",
            "q_scale",
            TARGET_COLUMN,
            UNCERTAINTY_COLUMN,
            "weighted_permittivity",
            "n_collapsed_simulations",
        ]
        if column in training_data.columns
    ]
    training_data = training_data.dropna(subset=[TARGET_COLUMN, "q_scale"]).copy()
    training_output_columns = list(
        dict.fromkeys(keep_columns + selected_features + [TARGET_COLUMN, UNCERTAINTY_COLUMN])
    )
    training_output_columns = [
        column
        for column in training_output_columns
        if column in training_data.columns and column not in {"conductivity", "conductivity_descriptor"}
    ]
    training_data[training_output_columns].to_csv(output_dir / "surrogate_training_data.csv", index=False)

    X = training_data[selected_features].copy()
    y = training_data[TARGET_COLUMN].to_numpy(dtype=float)
    groups = training_data["electrolyte_group"].astype(str)

    train_idx, test_idx = train_test_split_groups(X, y, groups, args.test_size, args.random_seed)
    preprocessor = make_preprocessor(numeric_features, categorical_features)
    models = candidate_models(args.random_seed, len(train_idx), args.gpr_threshold)
    transforms = target_transforms(args.target_transform)

    cv_results = evaluate_candidates(
        X.iloc[train_idx].reset_index(drop=True),
        y[train_idx],
        groups.iloc[train_idx].reset_index(drop=True),
        preprocessor,
        models,
        transforms,
        args.cv_splits,
    )
    cv_results.to_csv(output_dir / "surrogate_cv_results.csv", index=False)
    best = cv_results.iloc[0]
    best_transform = next(transform for transform in transforms if transform.name == best["target_transform"])

    holdout_pipeline = build_pipeline(preprocessor, models, str(best["model"]))
    holdout_pipeline.fit(X.iloc[train_idx], best_transform.forward(y[train_idx]))
    test_pred_model = holdout_pipeline.predict(X.iloc[test_idx])
    test_pred = best_transform.inverse(test_pred_model)
    test_metrics = regression_metrics(y[test_idx], test_pred)
    test_metrics.update(
        regression_metrics(
            best_transform.forward(y[test_idx]),
            test_pred_model,
            prefix="transformed_",
        )
    )
    test_metrics_df = pd.DataFrame(
        [
            {
                "model": best["model"],
                "target_transform": best["target_transform"],
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_train_groups": groups.iloc[train_idx].nunique(),
                "n_test_groups": groups.iloc[test_idx].nunique(),
                **test_metrics,
            }
        ]
    )
    test_metrics_df.to_csv(output_dir / "surrogate_test_metrics.csv", index=False)

    prediction_columns = [column for column in keep_columns if column in training_data.columns]
    predictions = training_data.iloc[test_idx][prediction_columns].copy()
    predictions["split"] = "test"
    predictions["true_conductivity_ms_cm"] = y[test_idx]
    predictions["predicted_conductivity_ms_cm"] = test_pred
    predictions["predicted_transformed_target"] = test_pred_model
    predictions["residual_ms_cm"] = test_pred - y[test_idx]
    predictions.to_csv(output_dir / "surrogate_predictions.csv", index=False)

    final_pipeline = build_pipeline(preprocessor, models, str(best["model"]))
    final_y_model = best_transform.forward(y)
    final_pipeline.fit(X, final_y_model)
    model_bundle = {
        "pipeline": final_pipeline,
        "target_transform": str(best["target_transform"]),
        "feature_columns": selected_features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target_column": TARGET_COLUMN,
        "group_column": "electrolyte_group",
        "best_model": str(best["model"]),
        "cv_results": cv_results.to_dict(orient="records"),
    }
    joblib.dump(model_bundle, output_dir / "surrogate_model.joblib")
    joblib.dump(final_pipeline.named_steps["preprocess"], output_dir / "preprocessing_pipeline.joblib")

    importance = compute_feature_importance(
        final_pipeline,
        X,
        final_y_model,
        str(best["target_transform"]),
        args.random_seed,
    )
    importance.to_csv(output_dir / "feature_importance.csv", index=False)
    save_prediction_plots(predictions, importance, output_dir)

    metadata = {
        "input_csv": str(args.input_csv),
        "descriptor_csv": str(args.descriptor_csv) if args.descriptor_csv is not None else None,
        "output_dir": str(output_dir),
        "random_seed": args.random_seed,
        "n_rows": int(len(training_data)),
        "n_groups": int(groups.nunique()),
        "features": selected_features,
        "success_statuses": args.success_statuses.split(","),
    }
    (output_dir / "surrogate_run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    write_summary(output_dir, best, test_metrics, importance, len(training_data), groups.nunique())

    print(f"Wrote Step E surrogate outputs to: {output_dir}")
    print(f"Best model: {best['model']} ({best['target_transform']})")
    print(
        "Grouped CV RMSE/MAE/R2: "
        f"{best['cv_rmse_mean']:.4g} / {best['cv_mae_mean']:.4g} / {best['cv_r2_mean']:.4g}"
    )
    print(
        "Grouped holdout RMSE/MAE/R2: "
        f"{test_metrics['rmse']:.4g} / {test_metrics['mae']:.4g} / {test_metrics['r2']:.4g}"
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("results/sparse_campaign_weighted_permittivity/conductivity_results.csv"),
        help="Sparse campaign conductivity_results.csv.",
    )
    parser.add_argument(
        "--descriptor-csv",
        type=Path,
        default=Path("electrolyte_outputs/descriptors.csv"),
        help="Descriptor CSV to merge by original_row_index. Use a missing path to skip merge.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/surrogate_model_weighted_permittivity"),
        help="Directory for model artifacts, metrics, and plots.",
    )
    parser.add_argument(
        "--success-statuses",
        default=",".join(DEFAULT_SUCCESS_STATUSES),
        help="Comma-separated collection_status values treated as successful.",
    )
    parser.add_argument(
        "--target-transform",
        choices=["auto", "raw", "log1p"],
        default="auto",
        help="Target transform to use. auto compares raw and log1p in grouped CV.",
    )
    parser.add_argument(
        "--include-categorical",
        action="store_true",
        help="Also use selected categorical chemistry columns with one-hot encoding.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Maximum number of grouped CV folds.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of electrolyte groups reserved for the final holdout split.",
    )
    parser.add_argument(
        "--gpr-threshold",
        type=int,
        default=250,
        help="Include Gaussian Process Regressor when the training set has at most this many rows.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=RANDOM_SEED,
        help="Fixed random seed for reproducible splits and models.",
    )
    return parser


def main() -> None:
    """Command-line entry point."""

    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
