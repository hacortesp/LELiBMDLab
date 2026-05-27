#!/usr/bin/env python3
"""Infer q_scale targets from the conductivity surrogate and train a simple model.

Step F uses the Step E surrogate to find the charge scaling factor whose
predicted simulated conductivity best matches the experimental conductivity for
each electrolyte. It then trains a compact, interpretable q_scale model using
the weighted-permittivity cluster structure for leakage-aware validation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "codex-cache"))

# Pandas does not need pyarrow here, and the local optional pyarrow build is
# incompatible with the installed NumPy version.
sys.modules.setdefault("pyarrow", None)

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


RANDOM_SEED = 42
EXPERIMENTAL_CONDUCTIVITY_COLUMN = "conductivity"
TARGET_COLUMN = "q_scale_opt"

CURATED_FEATURES = [
    "weighted_permittivity",
    "weighted_viscosity_mpa_s_25c",
    "salt_conc_x_weighted_permittivity",
    "salt_conc_x_weighted_viscosity",
    "temperature",
    "anion_tfsi",
]

OPTIONAL_INTERPRETABLE_FEATURES = [
    "anion_pf6",
    "anion_clo4",
    "anion_fsi",
    "anion_bf4",
    "solvent_entropy",
    "salt_conc",
]


def standardize_column_name(column: str) -> str:
    """Return a lower-case snake_case column name."""

    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", column.strip()).strip("_").lower()
    return re.sub(r"_+", "_", cleaned)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize all DataFrame column names."""

    out = df.copy()
    out.columns = [standardize_column_name(column) for column in out.columns]
    return out


def read_csv_standardized(path: Path) -> pd.DataFrame:
    """Read a CSV with standardized column names."""

    return standardize_columns(pd.read_csv(path))


def coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Coerce columns to numeric, converting invalid entries to NaN."""

    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def load_q_grid(campaign_results_csv: Path, fallback_grid: str | None) -> np.ndarray:
    """Load the available q_scale grid from the sparse campaign results."""

    if fallback_grid:
        values = [float(value) for value in fallback_grid.split(",")]
        return np.array(sorted(set(values)), dtype=float)

    campaign = read_csv_standardized(campaign_results_csv)
    if "q_scale" not in campaign.columns:
        raise ValueError(f"q_scale not found in {campaign_results_csv}")
    q_grid = pd.to_numeric(campaign["q_scale"], errors="coerce").dropna().unique()
    if len(q_grid) == 0:
        raise ValueError("No q_scale values found in campaign results.")
    return np.array(sorted(q_grid), dtype=float)


def load_experimental_table(
    descriptors_csv: Path,
    cleaned_csv: Path,
    cluster_labels_csv: Path,
) -> pd.DataFrame:
    """Merge descriptors, experimental metadata, and weighted-permittivity clusters."""

    descriptors = read_csv_standardized(descriptors_csv)
    cleaned = read_csv_standardized(cleaned_csv)
    labels = read_csv_standardized(cluster_labels_csv)

    if "original_row_index" not in descriptors.columns:
        raise ValueError("Descriptor table must contain original_row_index.")
    if EXPERIMENTAL_CONDUCTIVITY_COLUMN not in descriptors.columns:
        raise ValueError("Descriptor table must contain experimental conductivity.")

    cleaned_keep = [
        column
        for column in [
            "original_row_index",
            "cation_name",
            "salt_conc",
            "temperature",
            "solvents",
            "solvent_fracs",
            "replicate_count",
            "conductivity_std",
        ]
        if column in cleaned.columns
    ]
    labels_keep = [
        column
        for column in ["original_row_index", "cluster_id", "row_status", "criterion_name"]
        if column in labels.columns
    ]

    table = descriptors.merge(cleaned[cleaned_keep], on="original_row_index", how="left")
    table = table.merge(labels[labels_keep], on="original_row_index", how="left")

    numeric_columns = [
        EXPERIMENTAL_CONDUCTIVITY_COLUMN,
        "salt_conc",
        "temperature",
        "cluster_id",
        "weighted_permittivity",
        "weighted_viscosity_mpa_s_25c",
        "salt_conc_x_weighted_permittivity",
        "salt_conc_x_weighted_viscosity",
    ]
    table = coerce_numeric(table, numeric_columns)
    table = table.replace([np.inf, -np.inf], np.nan)
    table = table.dropna(subset=[EXPERIMENTAL_CONDUCTIVITY_COLUMN, "cluster_id"])
    table = table.loc[table[EXPERIMENTAL_CONDUCTIVITY_COLUMN] > 0].copy()
    table["cluster_id"] = table["cluster_id"].astype(int)
    return table


def inverse_surrogate_target(values: np.ndarray, target_transform: str) -> np.ndarray:
    """Invert the Step E target transform."""

    if target_transform == "log1p":
        return np.expm1(values)
    if target_transform == "raw":
        return values
    raise ValueError(f"Unsupported surrogate target transform: {target_transform}")


def infer_qscale_targets(
    experimental: pd.DataFrame,
    surrogate_bundle: dict,
    q_grid: np.ndarray,
    tie_tolerance: float,
) -> pd.DataFrame:
    """Infer q_scale_opt for each electrolyte by matching experiment."""

    feature_columns = surrogate_bundle["feature_columns"]
    missing = [column for column in feature_columns if column != "q_scale" and column not in experimental.columns]
    if missing:
        raise ValueError(f"Experimental table is missing surrogate feature columns: {missing}")

    repeated = []
    for q_scale in q_grid:
        block = experimental.copy()
        block["q_scale"] = q_scale
        repeated.append(block)
    grid_table = pd.concat(repeated, ignore_index=True)
    model_input = grid_table[feature_columns].copy()
    pred_transformed = surrogate_bundle["pipeline"].predict(model_input)
    grid_table["surrogate_predicted_conductivity_ms_cm"] = inverse_surrogate_target(
        pred_transformed,
        surrogate_bundle["target_transform"],
    )
    grid_table["absolute_error_ms_cm"] = (
        grid_table["surrogate_predicted_conductivity_ms_cm"]
        - grid_table[EXPERIMENTAL_CONDUCTIVITY_COLUMN]
    ).abs()

    records: list[dict[str, object]] = []
    for _, group in grid_table.groupby("original_row_index", sort=False):
        ordered = group.sort_values(["absolute_error_ms_cm", "q_scale"]).copy()
        best_error = float(ordered.iloc[0]["absolute_error_ms_cm"])
        near_best = ordered.loc[ordered["absolute_error_ms_cm"] <= best_error + tie_tolerance]
        # Prefer the smallest q_scale among near ties. It is deterministic and
        # conservative with respect to charge scaling.
        best = near_best.sort_values("q_scale").iloc[0]
        record = best.drop(labels=["surrogate_predicted_conductivity_ms_cm", "absolute_error_ms_cm"]).to_dict()
        record[TARGET_COLUMN] = float(best["q_scale"])
        record["q_scale_tie_rule"] = "lowest_q_within_tolerance"
        record["q_scale_tie_tolerance_ms_cm"] = tie_tolerance
        record["best_abs_error_ms_cm"] = best_error
        record["best_predicted_conductivity_ms_cm"] = float(
            best["surrogate_predicted_conductivity_ms_cm"]
        )
        record["n_q_values_evaluated"] = int(len(group))
        record["q_grid"] = ",".join(f"{value:g}" for value in q_grid)
        records.append(record)

    targets = pd.DataFrame(records)
    return targets.sort_values(["cluster_id", "original_row_index"]).reset_index(drop=True)


def build_training_table(
    targets: pd.DataFrame,
    target_level: str,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Build system-level or cluster-level q_scale training data."""

    if target_level == "system":
        columns = [
            "original_row_index",
            "doi",
            "anion_name",
            "composition_summary",
            "cluster_id",
            EXPERIMENTAL_CONDUCTIVITY_COLUMN,
            "best_predicted_conductivity_ms_cm",
            "best_abs_error_ms_cm",
            TARGET_COLUMN,
        ]
        columns = [column for column in columns if column in targets.columns]
        return targets[columns + feature_columns].copy()

    aggregations: dict[str, str] = {TARGET_COLUMN: "median"}
    for column in feature_columns:
        aggregations[column] = "median"
    aggregations["original_row_index"] = "count"
    if "best_abs_error_ms_cm" in targets.columns:
        aggregations["best_abs_error_ms_cm"] = "median"
    cluster_table = targets.groupby("cluster_id", as_index=False).agg(aggregations)
    cluster_table = cluster_table.rename(columns={"original_row_index": "n_systems"})
    return cluster_table


def choose_feature_columns(
    targets: pd.DataFrame,
    feature_importance_csv: Path | None,
    max_features: int,
    include_optional_anions: bool,
) -> list[str]:
    """Choose a compact, interpretable feature subset."""

    candidates = list(CURATED_FEATURES)
    if include_optional_anions:
        candidates.extend(OPTIONAL_INTERPRETABLE_FEATURES)

    if feature_importance_csv is not None and feature_importance_csv.is_file():
        importance = read_csv_standardized(feature_importance_csv)
        if {"feature", "importance"}.issubset(importance.columns):
            cleaned_features = (
                importance.sort_values("importance", ascending=False)["feature"]
                .astype(str)
                .str.replace(r"^[^_]+__", "", regex=True)
                .tolist()
            )
            for feature in cleaned_features:
                if feature == "q_scale":
                    continue
                if feature in targets.columns and feature not in candidates:
                    candidates.append(feature)

    available = [
        feature
        for feature in candidates
        if feature in targets.columns and pd.api.types.is_numeric_dtype(targets[feature])
    ]
    return list(dict.fromkeys(available))[:max_features]


def make_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    """Build a numeric preprocessing pipeline."""

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_columns,
            )
        ],
        remainder="drop",
    )


def candidate_models(random_seed: int, include_small_rf: bool) -> dict[str, BaseEstimator]:
    """Return simple candidate models for q_scale inference."""

    models: dict[str, BaseEstimator] = {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "lasso": Lasso(alpha=0.002, max_iter=10000, random_state=random_seed),
        "shallow_tree": DecisionTreeRegressor(max_depth=2, min_samples_leaf=20, random_state=random_seed),
    }
    if include_small_rf:
        models["small_random_forest"] = RandomForestRegressor(
            n_estimators=100,
            max_depth=3,
            min_samples_leaf=10,
            random_state=random_seed,
            n_jobs=-1,
        )
    return models


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> dict[str, float]:
    """Compute q_scale regression metrics."""

    residual = y_pred - y_true
    metrics = {
        f"{prefix}mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}rmse": float(np.sqrt(np.mean(residual**2))),
    }
    metrics[f"{prefix}r2"] = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan
    return metrics


def evaluate_models(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: pd.Series,
    preprocessor: ColumnTransformer,
    models: dict[str, BaseEstimator],
    cv_splits: int,
) -> pd.DataFrame:
    """Evaluate simple q_scale models with group-aware cross-validation."""

    n_groups = groups.nunique()
    if n_groups < 2:
        raise ValueError("At least two clusters are required for grouped cross-validation.")
    splitter = GroupKFold(n_splits=min(cv_splits, n_groups))
    records: list[dict[str, object]] = []
    simplicity_rank = {
        "mean_baseline": 0,
        "linear_regression": 1,
        "ridge": 2,
        "lasso": 3,
        "shallow_tree": 4,
        "small_random_forest": 5,
    }

    for model_name, model in models.items():
        fold_records = []
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y, groups), start=1):
            pipeline = Pipeline(
                steps=[
                    ("preprocess", clone(preprocessor)),
                    ("model", clone(model)),
                ]
            )
            pipeline.fit(X.iloc[train_idx], y[train_idx])
            pred = np.clip(pipeline.predict(X.iloc[valid_idx]), y.min(), y.max())
            row = {
                "model": model_name,
                "fold": fold,
                "n_train": len(train_idx),
                "n_valid": len(valid_idx),
            }
            row.update(regression_metrics(y[valid_idx], pred))
            fold_records.append(row)

        fold_df = pd.DataFrame(fold_records)
        summary = {
            "model": model_name,
            "n_folds": int(fold_df["fold"].nunique()),
            "simplicity_rank": simplicity_rank.get(model_name, 99),
        }
        for metric in ["mae", "rmse", "r2"]:
            summary[f"cv_{metric}_mean"] = float(fold_df[metric].mean())
            summary[f"cv_{metric}_std"] = float(fold_df[metric].std(ddof=0))
        records.append(summary)

    cv = pd.DataFrame(records)
    baseline_rmse = float(cv.loc[cv["model"] == "mean_baseline", "cv_rmse_mean"].iloc[0])
    cv["beats_baseline"] = cv["cv_rmse_mean"] < baseline_rmse
    return cv.sort_values(["cv_rmse_mean", "cv_mae_mean", "simplicity_rank"]).reset_index(drop=True)


def choose_best_model(cv_results: pd.DataFrame, simplicity_tolerance: float) -> pd.Series:
    """Select the simplest model within tolerance of the best RMSE."""

    non_baseline = cv_results.loc[cv_results["model"] != "mean_baseline"].copy()
    if non_baseline.empty:
        return cv_results.iloc[0]
    best_rmse = float(non_baseline["cv_rmse_mean"].min())
    eligible = non_baseline.loc[non_baseline["cv_rmse_mean"] <= best_rmse + simplicity_tolerance].copy()
    return eligible.sort_values(["simplicity_rank", "cv_rmse_mean", "cv_mae_mean"]).iloc[0]


def group_train_test_split(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: pd.Series,
    test_size: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a cluster-held-out split."""

    if groups.nunique() < 3:
        raise ValueError("At least three groups are needed for a held-out cluster split.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    return next(splitter.split(X, y, groups))


def build_pipeline(preprocessor: ColumnTransformer, model: BaseEstimator) -> Pipeline:
    """Create preprocessing + model pipeline."""

    return Pipeline(
        steps=[
            ("preprocess", clone(preprocessor)),
            ("model", clone(model)),
        ]
    )


def model_importance_table(pipeline: Pipeline, feature_columns: list[str], model_name: str) -> pd.DataFrame:
    """Return coefficients or feature importances for the selected model."""

    estimator = pipeline.named_steps["model"]
    if hasattr(estimator, "coef_"):
        coef = np.ravel(estimator.coef_).astype(float)
        return pd.DataFrame(
            {
                "feature": feature_columns,
                "coefficient": coef,
                "importance": np.abs(coef),
                "importance_type": "standardized_coefficient",
                "model": model_name,
            }
        ).sort_values("importance", ascending=False)
    if hasattr(estimator, "feature_importances_"):
        return pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": estimator.feature_importances_,
                "importance_type": "model_feature_importance",
                "model": model_name,
            }
        ).sort_values("importance", ascending=False)
    return pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": np.nan,
            "importance_type": "not_available",
            "model": model_name,
        }
    )


def save_plots(
    predictions: pd.DataFrame,
    targets: pd.DataFrame,
    importance: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save q_scale diagnostic and interpretation plots."""

    y_true = predictions["true_q_scale"].to_numpy()
    y_pred = predictions["predicted_q_scale"].to_numpy()
    residual = predictions["residual_q_scale"].to_numpy()

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(y_true, y_pred, c=predictions["cluster_id"], cmap="viridis", edgecolor="black", linewidth=0.4)
    lower = float(np.nanmin([y_true.min(), y_pred.min()]))
    upper = float(np.nanmax([y_true.max(), y_pred.max()]))
    ax.plot([lower, upper], [lower, upper], color="#333333", linestyle="--", linewidth=1)
    ax.set_xlabel("True inferred q_scale")
    ax.set_ylabel("Predicted q_scale")
    ax.set_title("Predicted vs true q_scale")
    fig.tight_layout()
    fig.savefig(output_dir / "predicted_vs_true_qscale.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residual, bins=min(14, max(4, len(residual))), color="#4c78a8", edgecolor="black")
    ax.axvline(0.0, color="#333333", linestyle="--", linewidth=1)
    ax.set_xlabel("Residual q_scale")
    ax.set_ylabel("Count")
    ax.set_title("q_scale residuals")
    fig.tight_layout()
    fig.savefig(output_dir / "qscale_residuals.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    clusters = sorted(targets["cluster_id"].dropna().unique())
    data = [targets.loc[targets["cluster_id"] == cluster, TARGET_COLUMN].to_numpy() for cluster in clusters]
    ax.boxplot(data, labels=[str(cluster) for cluster in clusters], patch_artist=True)
    ax.set_xlabel("Weighted-permittivity cluster")
    ax.set_ylabel("Inferred q_scale")
    ax.set_title("q_scale distribution by cluster")
    fig.tight_layout()
    fig.savefig(output_dir / "qscale_distribution_by_cluster.png", dpi=300)
    plt.close(fig)

    cluster_summary = targets.groupby("cluster_id", as_index=False)[TARGET_COLUMN].median()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(cluster_summary["cluster_id"].astype(str), cluster_summary[TARGET_COLUMN], color="#72b7b2")
    ax.set_xlabel("Weighted-permittivity cluster")
    ax.set_ylabel("Median inferred q_scale")
    ax.set_title("Median inferred q_scale by cluster")
    fig.tight_layout()
    fig.savefig(output_dir / "inferred_qscale_by_cluster.png", dpi=300)
    plt.close(fig)

    top = importance.dropna(subset=["importance"]).head(12).iloc[::-1]
    if not top.empty:
        fig, ax = plt.subplots(figsize=(7, max(3.5, 0.34 * len(top))))
        values = top["coefficient"] if "coefficient" in top.columns else top["importance"]
        colors = ["#d95f02" if value < 0 else "#1b9e77" for value in values]
        ax.barh(top["feature"], values, color=colors)
        ax.axvline(0.0, color="#333333", linewidth=0.8)
        ax.set_xlabel("Coefficient" if "coefficient" in top.columns else "Importance")
        ax.set_title("q_scale model feature effects")
        fig.tight_layout()
        fig.savefig(output_dir / "qscale_feature_importance.png", dpi=300)
        plt.close(fig)


def write_summary(
    output_dir: Path,
    best: pd.Series,
    test_metrics: dict[str, float],
    importance: pd.DataFrame,
    n_rows: int,
    n_clusters: int,
    feature_columns: list[str],
) -> None:
    """Write a short plain-text summary."""

    lines = [
        "Step F q_scale inference model summary",
        "======================================",
        f"training_rows: {n_rows}",
        f"weighted_permittivity_clusters: {n_clusters}",
        f"selected_model: {best['model']}",
        f"cv_rmse_q_scale: {best['cv_rmse_mean']:.6g}",
        f"cv_mae_q_scale: {best['cv_mae_mean']:.6g}",
        f"test_rmse_q_scale: {test_metrics['rmse']:.6g}",
        f"test_mae_q_scale: {test_metrics['mae']:.6g}",
        f"test_r2: {test_metrics['r2']:.6g}",
        "features: " + ", ".join(feature_columns),
        "",
        "Top feature effects:",
    ]
    for _, row in importance.head(10).iterrows():
        detail = row.get("coefficient", row.get("importance", np.nan))
        lines.append(f"- {row['feature']}: {detail:.6g}")
    (output_dir / "qscale_model_summary.txt").write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> None:
    """Run the full Step F workflow."""

    args.output_dir.mkdir(parents=True, exist_ok=True)

    q_grid = load_q_grid(args.campaign_results_csv, args.q_grid)
    surrogate_bundle = joblib.load(args.surrogate_model)
    experimental = load_experimental_table(
        args.descriptors_csv,
        args.cleaned_csv,
        args.cluster_labels_csv,
    )

    targets = infer_qscale_targets(
        experimental,
        surrogate_bundle,
        q_grid,
        args.tie_tolerance,
    )
    targets.to_csv(args.output_dir / "inferred_qscale_targets.csv", index=False)

    feature_columns = choose_feature_columns(
        targets,
        args.surrogate_feature_importance,
        args.max_features,
        args.include_optional_anions,
    )
    if len(feature_columns) < 2:
        raise ValueError("Need at least two usable q_scale model features.")

    training_data = build_training_table(targets, args.target_level, feature_columns)
    training_data = training_data.dropna(subset=[TARGET_COLUMN]).copy()
    training_data.to_csv(args.output_dir / "qscale_training_data.csv", index=False)

    X = training_data[feature_columns].copy()
    y = training_data[TARGET_COLUMN].to_numpy(dtype=float)
    groups = training_data["cluster_id"].astype(str)

    preprocessor = make_preprocessor(feature_columns)
    models = candidate_models(args.random_seed, args.include_small_rf)
    cv_results = evaluate_models(X, y, groups, preprocessor, models, args.cv_splits)
    cv_results.to_csv(args.output_dir / "qscale_model_cv_results.csv", index=False)
    best = choose_best_model(cv_results, args.simplicity_tolerance)

    train_idx, test_idx = group_train_test_split(X, y, groups, args.test_size, args.random_seed)
    selected_pipeline = build_pipeline(preprocessor, models[str(best["model"])])
    selected_pipeline.fit(X.iloc[train_idx], y[train_idx])
    test_pred = np.clip(selected_pipeline.predict(X.iloc[test_idx]), y.min(), y.max())
    test_metrics = regression_metrics(y[test_idx], test_pred)
    test_metrics_row = {
        "model": best["model"],
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "n_train_clusters": groups.iloc[train_idx].nunique(),
        "n_test_clusters": groups.iloc[test_idx].nunique(),
        **test_metrics,
    }
    pd.DataFrame([test_metrics_row]).to_csv(args.output_dir / "qscale_model_test_metrics.csv", index=False)

    predictions = training_data.iloc[test_idx].copy()
    predictions["split"] = "test"
    predictions["true_q_scale"] = y[test_idx]
    predictions["predicted_q_scale"] = test_pred
    predictions["residual_q_scale"] = test_pred - y[test_idx]
    predictions.to_csv(args.output_dir / "predicted_qscale.csv", index=False)

    final_pipeline = build_pipeline(preprocessor, models[str(best["model"])])
    final_pipeline.fit(X, y)
    model_bundle = {
        "pipeline": final_pipeline,
        "feature_columns": feature_columns,
        "target_column": TARGET_COLUMN,
        "group_column": "cluster_id",
        "target_level": args.target_level,
        "selected_model": str(best["model"]),
        "q_grid": q_grid.tolist(),
        "cv_results": cv_results.to_dict(orient="records"),
    }
    joblib.dump(model_bundle, args.output_dir / "qscale_model.joblib")

    importance = model_importance_table(final_pipeline, feature_columns, str(best["model"]))
    importance.to_csv(args.output_dir / "feature_importance_or_coefficients.csv", index=False)

    cluster_targets = targets.groupby("cluster_id", as_index=False).agg(
        n_systems=("original_row_index", "count"),
        median_q_scale_opt=(TARGET_COLUMN, "median"),
        mean_q_scale_opt=(TARGET_COLUMN, "mean"),
        std_q_scale_opt=(TARGET_COLUMN, "std"),
        median_best_abs_error_ms_cm=("best_abs_error_ms_cm", "median"),
        mean_weighted_permittivity=("weighted_permittivity", "mean"),
    )
    cluster_targets.to_csv(args.output_dir / "cluster_qscale_summary.csv", index=False)

    save_plots(predictions, targets, importance, args.output_dir)

    metadata = {
        "surrogate_model": str(args.surrogate_model),
        "descriptors_csv": str(args.descriptors_csv),
        "cleaned_csv": str(args.cleaned_csv),
        "cluster_labels_csv": str(args.cluster_labels_csv),
        "campaign_results_csv": str(args.campaign_results_csv),
        "q_grid": q_grid.tolist(),
        "feature_columns": feature_columns,
        "target_level": args.target_level,
        "random_seed": args.random_seed,
        "tie_tolerance_ms_cm": args.tie_tolerance,
    }
    (args.output_dir / "qscale_run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    write_summary(
        args.output_dir,
        best,
        test_metrics,
        importance,
        len(training_data),
        training_data["cluster_id"].nunique(),
        feature_columns,
    )

    print(f"Wrote Step F q_scale outputs to: {args.output_dir}")
    print(f"Selected model: {best['model']}")
    print(
        "Cluster-aware CV RMSE/MAE/R2: "
        f"{best['cv_rmse_mean']:.4g} / {best['cv_mae_mean']:.4g} / {best['cv_r2_mean']:.4g}"
    )
    print(
        "Held-out cluster RMSE/MAE/R2: "
        f"{test_metrics['rmse']:.4g} / {test_metrics['mae']:.4g} / {test_metrics['r2']:.4g}"
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--surrogate-model",
        type=Path,
        default=Path("results/surrogate_model_weighted_permittivity/surrogate_model.joblib"),
        help="Step E surrogate model bundle.",
    )
    parser.add_argument(
        "--surrogate-feature-importance",
        type=Path,
        default=Path("results/surrogate_model_weighted_permittivity/feature_importance.csv"),
        help="Step E feature importance table used to fill compact feature candidates.",
    )
    parser.add_argument(
        "--descriptors-csv",
        type=Path,
        default=Path("electrolyte_outputs/descriptors.csv"),
        help="Descriptor table containing experimental conductivity.",
    )
    parser.add_argument(
        "--cleaned-csv",
        type=Path,
        default=Path("electrolyte_outputs/cleaned_electrolytes.csv"),
        help="Cleaned electrolyte table with salt concentration and temperature.",
    )
    parser.add_argument(
        "--cluster-labels-csv",
        type=Path,
        default=Path("results/criterion_weighted_permittivity/cluster_labels.csv"),
        help="Weighted-permittivity cluster labels.",
    )
    parser.add_argument(
        "--campaign-results-csv",
        type=Path,
        default=Path("results/sparse_campaign_weighted_permittivity/conductivity_results.csv"),
        help="Sparse campaign results used to recover the q_scale grid.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/qscale_inference_weighted_permittivity"),
        help="Output directory for Step F artifacts.",
    )
    parser.add_argument(
        "--q-grid",
        default=None,
        help="Optional comma-separated q_scale grid. Defaults to values in campaign results.",
    )
    parser.add_argument(
        "--target-level",
        choices=["system", "cluster"],
        default="system",
        help="Train on system-level inferred targets or cluster-level median targets.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=6,
        help="Maximum number of compact descriptor features for q_scale inference.",
    )
    parser.add_argument(
        "--include-optional-anions",
        action="store_true",
        help="Allow extra anion one-hot descriptors beyond the default compact set.",
    )
    parser.add_argument(
        "--include-small-rf",
        action="store_true",
        help="Also compare a small random forest. Disabled by default to favor simple models.",
    )
    parser.add_argument(
        "--tie-tolerance",
        type=float,
        default=0.02,
        help="mS/cm tolerance for near-ties when choosing q_scale_opt.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Maximum number of group-aware CV folds.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.33,
        help="Fraction of clusters reserved for the final holdout split.",
    )
    parser.add_argument(
        "--simplicity-tolerance",
        type=float,
        default=0.005,
        help="Choose the simplest model within this q_scale RMSE of the best model.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=RANDOM_SEED,
        help="Fixed random seed.",
    )
    return parser


def main() -> None:
    """Command-line entry point."""

    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
