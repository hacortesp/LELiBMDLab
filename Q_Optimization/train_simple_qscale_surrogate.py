#!/usr/bin/env python3
"""Train a compact, physically guided q_scale conductivity surrogate.

This Step E surrogate intentionally favors interpretability over descriptor
breadth. It asks whether simulated conductivity is explained by:

1. q_scale and weighted_permittivity,
2. those variables plus weighted-permittivity cluster family, and
3. optionally weighted viscosity.

All validation splits are grouped by electrolyte identity so the same
electrolyte never appears in train and validation/test at different q_scale
values.
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

# Pandas can run without pyarrow. The local optional pyarrow build is noisy with
# NumPy 2.x, so mark it unavailable before importing pandas.
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
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor


RANDOM_SEED = 42
TARGET_COLUMN = "conductivity_ms_cm"
SUCCESS_STATUSES = ("parsed", "completed", "complete", "success", "successful", "ok")


@dataclass(frozen=True)
class TargetTransform:
    """Target transform definition."""

    name: str
    forward: Callable[[np.ndarray], np.ndarray]
    inverse: Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class FeatureSet:
    """Compact physical feature set."""

    name: str
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    description: str
    complexity_rank: int


def standardize_column_name(column: str) -> str:
    """Return a snake_case column name."""

    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", column.strip()).strip("_").lower()
    return re.sub(r"_+", "_", cleaned)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize DataFrame column names."""

    out = df.copy()
    out.columns = [standardize_column_name(column) for column in out.columns]
    return out


def read_csv_standardized(path: Path) -> pd.DataFrame:
    """Read a CSV and standardize its columns."""

    return standardize_columns(pd.read_csv(path))


def coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Coerce selected columns to numeric."""

    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def build_group_key(df: pd.DataFrame) -> pd.Series:
    """Create an electrolyte identity key for leakage-safe validation."""

    if "duplicate_identity" in df.columns and df["duplicate_identity"].notna().any():
        duplicate_identity = df["duplicate_identity"].astype(str).str.strip()
        fallback = pd.Series("row_" + df.index.astype(str), index=df.index)
        return duplicate_identity.where(duplicate_identity.ne(""), other=np.nan).fillna(fallback)

    key_columns = [
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
    available = [column for column in key_columns if column in df.columns]
    if not available:
        return pd.Series(["row_" + str(index) for index in df.index], index=df.index)
    return df[available].astype(str).agg("|".join, axis=1)


def load_training_data(results_csv: Path, cluster_summary_csv: Path) -> pd.DataFrame:
    """Load sparse campaign rows and build the compact surrogate table."""

    df = read_csv_standardized(results_csv)
    if "collection_status" not in df.columns:
        raise ValueError("conductivity results must include collection_status.")
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"conductivity results must include {TARGET_COLUMN}.")

    numeric_columns = [
        TARGET_COLUMN,
        "q_scale",
        "weighted_permittivity",
        "weighted_viscosity_mpa_s_25c",
        "cluster_id",
        "salt_conc",
        "temperature",
    ]
    df = coerce_numeric(df, numeric_columns)
    status = df["collection_status"].astype(str).str.strip().str.lower()
    df = df.loc[status.isin(SUCCESS_STATUSES)].copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[TARGET_COLUMN, "q_scale", "weighted_permittivity", "cluster_id"])
    df = df.loc[df[TARGET_COLUMN] > 0].copy()
    df["cluster_id"] = df["cluster_id"].astype(int)
    df["cluster_id_str"] = df["cluster_id"].astype(str)
    df["electrolyte_group"] = build_group_key(df)

    summary = read_csv_standardized(cluster_summary_csv)
    if "cluster_id" in summary.columns:
        summary = coerce_numeric(
            summary,
            [
                "cluster_id",
                "cluster_size",
                "mean_weighted_permittivity",
                "mean_weighted_viscosity_mpa_s_25c",
            ],
        )
        summary = summary.drop_duplicates(subset=["cluster_id"])
        keep = [
            column
            for column in [
                "cluster_id",
                "cluster_size",
                "mean_weighted_permittivity",
                "mean_weighted_viscosity_mpa_s_25c",
            ]
            if column in summary.columns
        ]
        df = df.merge(summary[keep], on="cluster_id", how="left")

    # Collapse exact duplicate simulations for an electrolyte/q_scale pair by
    # averaging numeric fields and keeping representative metadata.
    df["simulation_key"] = (
        df["electrolyte_group"].astype(str) + "|q=" + df["q_scale"].round(8).astype(str)
    )
    sizes = df.groupby("simulation_key", sort=False).size()
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    aggregations: dict[str, str] = {column: "mean" for column in numeric}
    for column in df.columns:
        if column not in aggregations and column != "simulation_key":
            aggregations[column] = "first"
    collapsed = df.groupby("simulation_key", as_index=False, sort=False).agg(aggregations)
    collapsed["n_collapsed_simulations"] = collapsed["simulation_key"].map(sizes).astype(int)
    collapsed["cluster_id"] = collapsed["cluster_id"].round().astype(int)
    collapsed["cluster_id_str"] = collapsed["cluster_id"].astype(str)
    return collapsed


def feature_sets(include_viscosity: bool) -> list[FeatureSet]:
    """Return progressively richer physical feature sets."""

    sets = [
        FeatureSet(
            name="qscale_permittivity",
            numeric_features=("q_scale", "weighted_permittivity"),
            categorical_features=(),
            description="q_scale + weighted_permittivity",
            complexity_rank=1,
        ),
        FeatureSet(
            name="qscale_permittivity_cluster",
            numeric_features=("q_scale", "weighted_permittivity"),
            categorical_features=("cluster_id_str",),
            description="q_scale + weighted_permittivity + weighted-permittivity cluster",
            complexity_rank=2,
        ),
    ]
    if include_viscosity:
        sets.extend(
            [
                FeatureSet(
                    name="qscale_permittivity_viscosity",
                    numeric_features=(
                        "q_scale",
                        "weighted_permittivity",
                        "weighted_viscosity_mpa_s_25c",
                    ),
                    categorical_features=(),
                    description="q_scale + weighted_permittivity + weighted viscosity",
                    complexity_rank=3,
                ),
                FeatureSet(
                    name="qscale_permittivity_viscosity_cluster",
                    numeric_features=(
                        "q_scale",
                        "weighted_permittivity",
                        "weighted_viscosity_mpa_s_25c",
                    ),
                    categorical_features=("cluster_id_str",),
                    description=(
                        "q_scale + weighted_permittivity + weighted viscosity "
                        "+ weighted-permittivity cluster"
                    ),
                    complexity_rank=4,
                ),
            ]
        )
    return sets


def target_transforms(mode: str) -> list[TargetTransform]:
    """Return target transforms to compare."""

    transforms = {
        "raw": TargetTransform("raw", lambda y: y, lambda y: y),
        "log1p": TargetTransform("log1p", np.log1p, np.expm1),
    }
    if mode == "auto":
        return [transforms["raw"], transforms["log1p"]]
    return [transforms[mode]]


def candidate_models(random_seed: int) -> dict[str, tuple[BaseEstimator, int, bool]]:
    """Return simple candidate models.

    The tuple is (estimator, complexity_rank, use_polynomial_numeric_features).
    """

    return {
        "mean_baseline": (DummyRegressor(strategy="mean"), 0, False),
        "linear": (LinearRegression(), 1, False),
        "ridge": (Ridge(alpha=1.0), 2, False),
        "poly2_ridge": (Ridge(alpha=1.0), 3, True),
        "small_tree": (
            DecisionTreeRegressor(max_depth=2, min_samples_leaf=2, random_state=random_seed),
            4,
            False,
        ),
        "shallow_random_forest": (
            RandomForestRegressor(
                n_estimators=150,
                max_depth=3,
                min_samples_leaf=2,
                random_state=random_seed,
                n_jobs=-1,
            ),
            5,
            False,
        ),
    }


def make_preprocessor(feature_set: FeatureSet, polynomial_numeric: bool) -> ColumnTransformer:
    """Build preprocessing for a compact physical feature set."""

    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if polynomial_numeric:
        # Polynomial terms let simple linear models capture q_scale curvature and
        # q_scale-permittivity interactions while keeping the variables explicit.
        numeric_steps.append(("poly2", PolynomialFeatures(degree=2, include_bias=False)))
    numeric_steps.append(("scaler", StandardScaler()))

    transformers: list[tuple[str, Pipeline, list[str]]] = [
        ("numeric", Pipeline(numeric_steps), list(feature_set.numeric_features))
    ]
    if feature_set.categorical_features:
        transformers.append(
            (
                "cluster",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                list(feature_set.categorical_features),
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> dict[str, float]:
    """Compute MAE, RMSE, and R2."""

    residual = y_pred - y_true
    out = {
        f"{prefix}mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}rmse": float(np.sqrt(np.mean(residual**2))),
    }
    out[f"{prefix}r2"] = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan
    return out


def evaluate_model_grid(
    data: pd.DataFrame,
    feature_set_list: list[FeatureSet],
    transforms: list[TargetTransform],
    models: dict[str, tuple[BaseEstimator, int, bool]],
    cv_splits: int,
) -> pd.DataFrame:
    """Compare feature sets, target transforms, and simple models with GroupKFold."""

    groups = data["electrolyte_group"].astype(str)
    y = data[TARGET_COLUMN].to_numpy(dtype=float)
    n_splits = min(cv_splits, groups.nunique())
    if n_splits < 2:
        raise ValueError("At least two electrolyte groups are needed for grouped validation.")
    splitter = GroupKFold(n_splits=n_splits)

    records: list[dict[str, object]] = []
    for feature_set in feature_set_list:
        columns = list(feature_set.numeric_features + feature_set.categorical_features)
        X = data[columns].copy()
        for transform in transforms:
            y_model = transform.forward(y)
            for model_name, (estimator, model_complexity, polynomial) in models.items():
                fold_rows = []
                for fold, (train_idx, valid_idx) in enumerate(
                    splitter.split(X, y_model, groups),
                    start=1,
                ):
                    pipeline = Pipeline(
                        [
                            ("preprocess", make_preprocessor(feature_set, polynomial)),
                            ("model", clone(estimator)),
                        ]
                    )
                    pipeline.fit(X.iloc[train_idx], y_model[train_idx])
                    pred_model = pipeline.predict(X.iloc[valid_idx])
                    pred = transform.inverse(pred_model)
                    fold = {
                        "fold": fold,
                        "n_train": len(train_idx),
                        "n_valid": len(valid_idx),
                    }
                    fold.update(regression_metrics(y[valid_idx], pred))
                    fold.update(
                        regression_metrics(y_model[valid_idx], pred_model, prefix="transformed_")
                    )
                    fold_rows.append(fold)

                fold_df = pd.DataFrame(fold_rows)
                row = {
                    "feature_set": feature_set.name,
                    "feature_description": feature_set.description,
                    "model": model_name,
                    "target_transform": transform.name,
                    "n_folds": n_splits,
                    "feature_complexity_rank": feature_set.complexity_rank,
                    "model_complexity_rank": model_complexity,
                    "uses_polynomial_terms": polynomial,
                    "numeric_features": ",".join(feature_set.numeric_features),
                    "categorical_features": ",".join(feature_set.categorical_features),
                }
                for metric in [
                    "mae",
                    "rmse",
                    "r2",
                    "transformed_mae",
                    "transformed_rmse",
                    "transformed_r2",
                ]:
                    row[f"cv_{metric}_mean"] = float(fold_df[metric].mean())
                    row[f"cv_{metric}_std"] = float(fold_df[metric].std(ddof=0))
                records.append(row)

    return pd.DataFrame(records).sort_values(
        ["cv_rmse_mean", "cv_mae_mean", "feature_complexity_rank", "model_complexity_rank"]
    )


def choose_model(comparison: pd.DataFrame, relative_tolerance: float) -> pd.Series:
    """Choose the simplest model within tolerance of the best grouped CV RMSE."""

    non_baseline = comparison.loc[comparison["model"] != "mean_baseline"].copy()
    best_rmse = float(non_baseline["cv_rmse_mean"].min())
    eligible = non_baseline.loc[non_baseline["cv_rmse_mean"] <= best_rmse * (1 + relative_tolerance)]
    return eligible.sort_values(
        [
            "feature_complexity_rank",
            "model_complexity_rank",
            "cv_rmse_mean",
            "cv_mae_mean",
        ]
    ).iloc[0]


def find_feature_set(feature_set_list: list[FeatureSet], name: str) -> FeatureSet:
    """Return a feature-set definition by name."""

    for feature_set in feature_set_list:
        if feature_set.name == name:
            return feature_set
    raise ValueError(f"Unknown feature set: {name}")


def train_test_split_groups(
    data: pd.DataFrame,
    test_size: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create one held-out electrolyte split."""

    groups = data["electrolyte_group"].astype(str)
    if groups.nunique() < 3:
        raise ValueError("At least three electrolyte groups are needed for a holdout split.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    return next(splitter.split(data, data[TARGET_COLUMN], groups))


def build_pipeline_from_selection(
    selected: pd.Series,
    feature_set_list: list[FeatureSet],
    models: dict[str, tuple[BaseEstimator, int, bool]],
) -> tuple[Pipeline, FeatureSet, TargetTransform]:
    """Build the selected pipeline and associated metadata."""

    feature_set = find_feature_set(feature_set_list, str(selected["feature_set"]))
    estimator, _, polynomial = models[str(selected["model"])]
    transform = target_transforms(str(selected["target_transform"]))[0]
    pipeline = Pipeline(
        [
            ("preprocess", make_preprocessor(feature_set, polynomial)),
            ("model", clone(estimator)),
        ]
    )
    return pipeline, feature_set, transform


def feature_names(pipeline: Pipeline, fallback: list[str]) -> list[str]:
    """Return transformed feature names when available."""

    try:
        return pipeline.named_steps["preprocess"].get_feature_names_out().tolist()
    except Exception:
        return fallback


def coefficient_or_importance(
    pipeline: Pipeline,
    feature_set: FeatureSet,
    selected: pd.Series,
) -> pd.DataFrame:
    """Save standardized coefficients or tree importances for interpretation."""

    fallback = list(feature_set.numeric_features + feature_set.categorical_features)
    names = feature_names(pipeline, fallback)
    estimator = pipeline.named_steps["model"]
    if hasattr(estimator, "coef_"):
        coef = np.ravel(estimator.coef_).astype(float)
        return pd.DataFrame(
            {
                "feature": names,
                "coefficient": coef,
                "importance": np.abs(coef),
                "importance_type": "standardized_coefficient",
                "selected_model": selected["model"],
                "feature_set": selected["feature_set"],
            }
        ).sort_values("importance", ascending=False)
    if hasattr(estimator, "feature_importances_"):
        return pd.DataFrame(
            {
                "feature": names,
                "importance": estimator.feature_importances_,
                "importance_type": "model_feature_importance",
                "selected_model": selected["model"],
                "feature_set": selected["feature_set"],
            }
        ).sort_values("importance", ascending=False)
    return pd.DataFrame(
        {
            "feature": names,
            "importance": np.nan,
            "importance_type": "not_available",
            "selected_model": selected["model"],
            "feature_set": selected["feature_set"],
        }
    )


def cluster_response_analysis(data: pd.DataFrame) -> pd.DataFrame:
    """Estimate q_scale response and optimum region per cluster."""

    records: list[dict[str, object]] = []
    for cluster_id, group in data.groupby("cluster_id", sort=True):
        by_q = (
            group.groupby("q_scale", as_index=False)
            .agg(
                mean_conductivity_ms_cm=(TARGET_COLUMN, "mean"),
                std_conductivity_ms_cm=(TARGET_COLUMN, "std"),
                n_rows=(TARGET_COLUMN, "size"),
            )
            .sort_values("q_scale")
        )
        if by_q["q_scale"].nunique() >= 2:
            slope = float(np.polyfit(by_q["q_scale"], by_q["mean_conductivity_ms_cm"], deg=1)[0])
        else:
            slope = np.nan
        best_idx = by_q["mean_conductivity_ms_cm"].idxmax()
        best_q = float(by_q.loc[best_idx, "q_scale"])
        records.append(
            {
                "cluster_id": int(cluster_id),
                "n_electrolyte_systems": int(group["electrolyte_group"].nunique()),
                "n_simulation_rows": int(len(group)),
                "mean_weighted_permittivity": float(group["weighted_permittivity"].mean()),
                "mean_weighted_viscosity_mpa_s_25c": float(
                    group["weighted_viscosity_mpa_s_25c"].mean()
                )
                if "weighted_viscosity_mpa_s_25c" in group
                else np.nan,
                "conductivity_sensitivity_to_qscale": slope,
                "best_fit_qscale_region": best_q,
                "conductivity_range": float(group[TARGET_COLUMN].max() - group[TARGET_COLUMN].min()),
                "mean_conductivity_min_qscale": float(by_q.iloc[0]["mean_conductivity_ms_cm"]),
                "mean_conductivity_max_qscale": float(by_q.iloc[-1]["mean_conductivity_ms_cm"]),
            }
        )
    return pd.DataFrame(records)


def predict_response_curves(
    pipeline: Pipeline,
    feature_set: FeatureSet,
    transform: TargetTransform,
    data: pd.DataFrame,
    q_grid: np.ndarray,
) -> pd.DataFrame:
    """Generate cluster-level model response curves over q_scale."""

    rows: list[dict[str, object]] = []
    for cluster_id, group in data.groupby("cluster_id", sort=True):
        template = group.iloc[0].copy()
        for feature in feature_set.numeric_features:
            if feature != "q_scale":
                template[feature] = group[feature].mean()
        for q_scale in q_grid:
            row = template.copy()
            row["q_scale"] = q_scale
            X = pd.DataFrame([row[list(feature_set.numeric_features + feature_set.categorical_features)]])
            pred = float(transform.inverse(pipeline.predict(X))[0])
            rows.append(
                {
                    "cluster_id": int(cluster_id),
                    "q_scale": float(q_scale),
                    "predicted_conductivity_ms_cm": pred,
                    "weighted_permittivity": float(group["weighted_permittivity"].mean()),
                }
            )
    return pd.DataFrame(rows)


def save_plots(
    data: pd.DataFrame,
    predictions: pd.DataFrame,
    response_curves: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save diagnostic plots for the simple surrogate."""

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for cluster_id, group in data.groupby("cluster_id", sort=True):
        by_q = group.groupby("q_scale", as_index=False)[TARGET_COLUMN].mean().sort_values("q_scale")
        ax.plot(
            by_q["q_scale"],
            by_q[TARGET_COLUMN],
            marker="o",
            linewidth=1.8,
            label=f"Cluster {int(cluster_id)}",
        )
    ax.set_xlabel("q_scale")
    ax.set_ylabel("Mean simulated conductivity (mS/cm)")
    ax.set_title("Conductivity response to q_scale by weighted-permittivity cluster")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "conductivity_vs_qscale_by_cluster.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(
        data["q_scale"],
        data[TARGET_COLUMN],
        c=data["cluster_id"],
        cmap="viridis",
        edgecolor="black",
        linewidth=0.35,
        alpha=0.85,
    )
    ax.set_xlabel("q_scale")
    ax.set_ylabel("Simulated conductivity (mS/cm)")
    ax.set_title("Cluster-colored conductivity trends")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Cluster ID")
    fig.tight_layout()
    fig.savefig(output_dir / "cluster_colored_conductivity_trends.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    y_true = predictions["true_conductivity_ms_cm"].to_numpy()
    y_pred = predictions["predicted_conductivity_ms_cm"].to_numpy()
    ax.scatter(y_true, y_pred, c=predictions["cluster_id"], cmap="viridis", edgecolor="black", linewidth=0.4)
    lower = float(np.nanmin([y_true.min(), y_pred.min()]))
    upper = float(np.nanmax([y_true.max(), y_pred.max()]))
    ax.plot([lower, upper], [lower, upper], color="#333333", linestyle="--", linewidth=1)
    ax.set_xlabel("True conductivity (mS/cm)")
    ax.set_ylabel("Predicted conductivity (mS/cm)")
    ax.set_title("Predicted vs true conductivity")
    fig.tight_layout()
    fig.savefig(output_dir / "predicted_vs_true_conductivity.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].scatter(
        predictions["weighted_permittivity"],
        predictions["residual_ms_cm"],
        c=predictions["cluster_id"],
        cmap="viridis",
        edgecolor="black",
        linewidth=0.35,
    )
    axes[0].axhline(0.0, color="#333333", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Weighted permittivity")
    axes[0].set_ylabel("Residual (mS/cm)")
    axes[0].set_title("Residuals vs weighted permittivity")

    axes[1].scatter(
        predictions["q_scale"],
        predictions["residual_ms_cm"],
        c=predictions["cluster_id"],
        cmap="viridis",
        edgecolor="black",
        linewidth=0.35,
    )
    axes[1].axhline(0.0, color="#333333", linestyle="--", linewidth=1)
    axes[1].set_xlabel("q_scale")
    axes[1].set_ylabel("Residual (mS/cm)")
    axes[1].set_title("Residuals vs q_scale")
    fig.tight_layout()
    fig.savefig(output_dir / "residual_diagnostics.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for cluster_id, group in response_curves.groupby("cluster_id", sort=True):
        ax.plot(
            group["q_scale"],
            group["predicted_conductivity_ms_cm"],
            marker="o",
            linewidth=1.7,
            label=f"Cluster {int(cluster_id)}",
        )
    ax.set_xlabel("q_scale")
    ax.set_ylabel("Predicted conductivity (mS/cm)")
    ax.set_title("Selected surrogate response curves")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "selected_surrogate_response_curves.png", dpi=300)
    plt.close(fig)


def summarize_scientific_questions(comparison: pd.DataFrame, cluster_analysis: pd.DataFrame) -> dict[str, object]:
    """Extract compact answers for the model summary."""

    def best_rmse(feature_set: str) -> float | None:
        values = comparison.loc[
            (comparison["feature_set"] == feature_set) & (comparison["model"] != "mean_baseline"),
            "cv_rmse_mean",
        ]
        return float(values.min()) if not values.empty else None

    base = best_rmse("qscale_permittivity")
    cluster = best_rmse("qscale_permittivity_cluster")
    visc = best_rmse("qscale_permittivity_viscosity")
    visc_cluster = best_rmse("qscale_permittivity_viscosity_cluster")
    return {
        "best_base_rmse": base,
        "best_cluster_rmse": cluster,
        "best_viscosity_rmse": visc,
        "best_viscosity_cluster_rmse": visc_cluster,
        "cluster_improvement_fraction": None if base in (None, 0) or cluster is None else (base - cluster) / base,
        "viscosity_improvement_fraction": None
        if cluster in (None, 0) or visc_cluster is None
        else (cluster - visc_cluster) / cluster,
        "clusters_with_negative_qscale_sensitivity": int(
            (cluster_analysis["conductivity_sensitivity_to_qscale"] < 0).sum()
        ),
        "clusters_with_positive_qscale_sensitivity": int(
            (cluster_analysis["conductivity_sensitivity_to_qscale"] > 0).sum()
        ),
    }


def write_summary(
    output_dir: Path,
    selected: pd.Series,
    test_metrics: dict[str, float],
    comparison_summary: dict[str, object],
    coefficients: pd.DataFrame,
) -> None:
    """Write a short recommendation summary."""

    lines = [
        "Simple Step E q_scale surrogate summary",
        "=======================================",
        f"selected_feature_set: {selected['feature_set']}",
        f"selected_model: {selected['model']}",
        f"target_transform: {selected['target_transform']}",
        f"grouped_cv_rmse_mS_cm: {selected['cv_rmse_mean']:.6g}",
        f"grouped_cv_mae_mS_cm: {selected['cv_mae_mean']:.6g}",
        f"grouped_cv_r2: {selected['cv_r2_mean']:.6g}",
        f"heldout_rmse_mS_cm: {test_metrics['rmse']:.6g}",
        f"heldout_mae_mS_cm: {test_metrics['mae']:.6g}",
        f"heldout_r2: {test_metrics['r2']:.6g}",
        "",
        "Scientific comparison:",
    ]
    for key, value in comparison_summary.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "Top selected-model effects:"])
    for _, row in coefficients.head(10).iterrows():
        value = row.get("coefficient", row.get("importance", np.nan))
        lines.append(f"- {row['feature']}: {value:.6g}")
    (output_dir / "simple_surrogate_summary.txt").write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> None:
    """Run the simplified Step E workflow."""

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_training_data(args.results_csv, args.cluster_summary_csv)
    data.to_csv(args.output_dir / "surrogate_training_data.csv", index=False)

    fsets = feature_sets(include_viscosity=args.include_viscosity)
    transforms = target_transforms(args.target_transform)
    models = candidate_models(args.random_seed)

    comparison = evaluate_model_grid(data, fsets, transforms, models, args.cv_splits)
    comparison.to_csv(args.output_dir / "surrogate_model_comparison.csv", index=False)
    selected = choose_model(comparison, args.selection_tolerance)

    train_idx, test_idx = train_test_split_groups(data, args.test_size, args.random_seed)
    selected_pipeline, selected_feature_set, selected_transform = build_pipeline_from_selection(
        selected,
        fsets,
        models,
    )
    selected_columns = list(
        selected_feature_set.numeric_features + selected_feature_set.categorical_features
    )
    y = data[TARGET_COLUMN].to_numpy(dtype=float)
    selected_pipeline.fit(
        data.iloc[train_idx][selected_columns],
        selected_transform.forward(y[train_idx]),
    )
    pred_transformed = selected_pipeline.predict(data.iloc[test_idx][selected_columns])
    pred = selected_transform.inverse(pred_transformed)
    test_metrics = regression_metrics(y[test_idx], pred)
    test_metrics.update(
        regression_metrics(
            selected_transform.forward(y[test_idx]),
            pred_transformed,
            prefix="transformed_",
        )
    )
    pd.DataFrame(
        [
            {
                "feature_set": selected["feature_set"],
                "model": selected["model"],
                "target_transform": selected["target_transform"],
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_train_groups": data.iloc[train_idx]["electrolyte_group"].nunique(),
                "n_test_groups": data.iloc[test_idx]["electrolyte_group"].nunique(),
                **test_metrics,
            }
        ]
    ).to_csv(args.output_dir / "surrogate_test_metrics.csv", index=False)

    predictions = data.iloc[test_idx].copy()
    predictions["split"] = "test"
    predictions["true_conductivity_ms_cm"] = y[test_idx]
    predictions["predicted_conductivity_ms_cm"] = pred
    predictions["residual_ms_cm"] = pred - y[test_idx]
    predictions.to_csv(args.output_dir / "surrogate_predictions.csv", index=False)

    final_pipeline, selected_feature_set, selected_transform = build_pipeline_from_selection(
        selected,
        fsets,
        models,
    )
    final_pipeline.fit(data[selected_columns], selected_transform.forward(y))
    bundle = {
        "pipeline": final_pipeline,
        "feature_columns": selected_columns,
        "numeric_features": list(selected_feature_set.numeric_features),
        "categorical_features": list(selected_feature_set.categorical_features),
        "target_column": TARGET_COLUMN,
        "target_transform": selected_transform.name,
        "selected_feature_set": str(selected["feature_set"]),
        "selected_model": str(selected["model"]),
        "group_column": "electrolyte_group",
        "cluster_column": "cluster_id",
    }
    joblib.dump(bundle, args.output_dir / "selected_surrogate_model.pkl")
    joblib.dump(final_pipeline.named_steps["preprocess"], args.output_dir / "preprocessing_pipeline.pkl")

    coefficients = coefficient_or_importance(final_pipeline, selected_feature_set, selected)
    coefficients.to_csv(args.output_dir / "selected_model_coefficients_or_importance.csv", index=False)

    cluster_analysis = cluster_response_analysis(data)
    cluster_analysis.to_csv(args.output_dir / "cluster_qscale_response_analysis.csv", index=False)
    q_grid = np.linspace(data["q_scale"].min(), data["q_scale"].max(), args.response_grid_points)
    response_curves = predict_response_curves(
        final_pipeline,
        selected_feature_set,
        selected_transform,
        data,
        q_grid,
    )
    response_curves.to_csv(args.output_dir / "selected_surrogate_response_curves.csv", index=False)

    save_plots(data, predictions, response_curves, args.output_dir)

    comparison_summary = summarize_scientific_questions(comparison, cluster_analysis)
    metadata = {
        "results_csv": str(args.results_csv),
        "cluster_summary_csv": str(args.cluster_summary_csv),
        "output_dir": str(args.output_dir),
        "random_seed": args.random_seed,
        "n_rows": int(len(data)),
        "n_electrolyte_groups": int(data["electrolyte_group"].nunique()),
        "n_clusters": int(data["cluster_id"].nunique()),
        "selection_tolerance": args.selection_tolerance,
        "selected": selected.to_dict(),
        "scientific_comparison": comparison_summary,
    }
    (args.output_dir / "simple_surrogate_run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    write_summary(args.output_dir, selected, test_metrics, comparison_summary, coefficients)

    print(f"Wrote simplified Step E surrogate outputs to: {args.output_dir}")
    print(
        f"Selected: {selected['feature_set']} / {selected['model']} "
        f"({selected['target_transform']})"
    )
    print(
        "Grouped CV RMSE/MAE/R2: "
        f"{selected['cv_rmse_mean']:.4g} / {selected['cv_mae_mean']:.4g} / "
        f"{selected['cv_r2_mean']:.4g}"
    )
    print(
        "Held-out electrolyte RMSE/MAE/R2: "
        f"{test_metrics['rmse']:.4g} / {test_metrics['mae']:.4g} / {test_metrics['r2']:.4g}"
    )


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=Path("results/sparse_campaign_weighted_permittivity/conductivity_results.csv"),
        help="Sparse campaign conductivity results.",
    )
    parser.add_argument(
        "--cluster-summary-csv",
        type=Path,
        default=Path("results/criterion_weighted_permittivity/cluster_summary.csv"),
        help="Weighted-permittivity cluster summary.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/simple_qscale_surrogate_weighted_permittivity"),
        help="Output directory for compact surrogate artifacts.",
    )
    parser.add_argument(
        "--target-transform",
        choices=["auto", "raw", "log1p"],
        default="auto",
        help="Target transform. auto compares raw and log1p conductivity.",
    )
    parser.add_argument(
        "--include-viscosity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compare feature sets that include weighted viscosity.",
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
        help="Fraction of electrolyte identities reserved for final holdout.",
    )
    parser.add_argument(
        "--selection-tolerance",
        type=float,
        default=0.10,
        help="Prefer the simplest model within this fractional RMSE of the best CV model.",
    )
    parser.add_argument(
        "--response-grid-points",
        type=int,
        default=41,
        help="Number of q_scale points for selected-model response curves.",
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
