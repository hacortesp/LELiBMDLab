#!/usr/bin/env python3
"""Train a direct q_scale-optimal model from sparse conductivity simulations.

This script does not train a conductivity surrogate. Instead, it derives a
system-level target:

    q_scale_optimal = argmin_q |sigma_sim(q) - sigma_exp|

from the available q_scale simulations for each electrolyte system, then trains
simple interpretable models to predict that optimal q_scale from system-level
metadata.
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


RANDOM_SEED = 42
SIM_CONDUCTIVITY = "conductivity_ms_cm"
EXP_CONDUCTIVITY = "conductivity"
TARGET = "q_scale_optimal"
SUCCESS_STATUSES = {"parsed", "completed", "complete", "success", "successful", "ok"}

NUMERIC_FEATURES = [
    "salt_conc",
    "temperature",
    "experimental_conductivity_ms_cm",
    "cluster_id",
    "mean_conductivity_uncertainty_ms_cm",
]
CATEGORICAL_FEATURES = [
    "cation_name",
    "anion_name",
    "solvents",
    "composition_summary",
]


def standardize_column_name(column: str) -> str:
    """Convert a CSV column name to snake_case."""

    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", column.strip()).strip("_").lower()
    return re.sub(r"_+", "_", cleaned)


def read_campaign_csv(path: Path) -> pd.DataFrame:
    """Read and normalize the campaign CSV."""

    df = pd.read_csv(path)
    df.columns = [standardize_column_name(column) for column in df.columns]
    required = {"collection_status", SIM_CONDUCTIVITY, EXP_CONDUCTIVITY, "q_scale"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Campaign CSV is missing required columns: {sorted(missing)}")

    numeric_columns = [
        SIM_CONDUCTIVITY,
        EXP_CONDUCTIVITY,
        "conductivity_uncertainty_ms_cm",
        "q_scale",
        "salt_conc",
        "temperature",
        "cluster_id",
        "original_row_index",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    status = df["collection_status"].astype(str).str.strip().str.lower()
    df = df.loc[status.isin(SUCCESS_STATUSES)].copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[SIM_CONDUCTIVITY, EXP_CONDUCTIVITY, "q_scale"])
    df = df.loc[(df[SIM_CONDUCTIVITY] > 0) & (df[EXP_CONDUCTIVITY] > 0)].copy()
    if "cluster_id" in df.columns:
        df["cluster_id"] = df["cluster_id"].round().astype("Int64")
    if df.empty:
        raise ValueError("No usable completed simulations were found.")
    return df


def electrolyte_group_key(df: pd.DataFrame) -> pd.Series:
    """Build a stable electrolyte-system key so q_scale rows stay together."""

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
        return pd.Series("row_" + df.index.astype(str), index=df.index)
    return df[available].astype(str).agg("|".join, axis=1)


def derive_qscale_targets(df: pd.DataFrame, tie_tolerance: float) -> pd.DataFrame:
    """Create one q_scale_optimal training row per electrolyte system."""

    work = df.copy()
    work["electrolyte_group"] = electrolyte_group_key(work)
    work["absolute_conductivity_error_ms_cm"] = (
        work[SIM_CONDUCTIVITY] - work[EXP_CONDUCTIVITY]
    ).abs()

    records: list[dict[str, object]] = []
    for group_key, group in work.groupby("electrolyte_group", sort=False):
        ordered = group.sort_values(["absolute_conductivity_error_ms_cm", "q_scale"]).copy()
        best_error = float(ordered.iloc[0]["absolute_conductivity_error_ms_cm"])
        near_best = ordered.loc[
            ordered["absolute_conductivity_error_ms_cm"] <= best_error + tie_tolerance
        ]
        # Deterministic tie rule: choose the smallest q_scale within tolerance.
        best = near_best.sort_values("q_scale").iloc[0]

        sim_min = float(group[SIM_CONDUCTIVITY].min())
        sim_max = float(group[SIM_CONDUCTIVITY].max())
        exp_value = float(best[EXP_CONDUCTIVITY])
        record = {
            "electrolyte_group": group_key,
            "original_row_index": best.get("original_row_index", np.nan),
            "doi": best.get("doi", ""),
            "cation_name": best.get("cation_name", ""),
            "anion_name": best.get("anion_name", ""),
            "solvents": best.get("solvents", ""),
            "solvent_fracs": best.get("solvent_fracs", ""),
            "composition_summary": best.get("composition_summary", ""),
            "cluster_id": best.get("cluster_id", np.nan),
            "salt_conc": best.get("salt_conc", np.nan),
            "temperature": best.get("temperature", np.nan),
            "experimental_conductivity_ms_cm": exp_value,
            "mean_conductivity_uncertainty_ms_cm": group.get(
                "conductivity_uncertainty_ms_cm",
                pd.Series([np.nan]),
            ).mean(),
            TARGET: float(best["q_scale"]),
            "optimal_simulated_conductivity_ms_cm": float(best[SIM_CONDUCTIVITY]),
            "optimal_abs_error_ms_cm": best_error,
            "n_q_values": int(group["q_scale"].nunique()),
            "q_scale_min": float(group["q_scale"].min()),
            "q_scale_max": float(group["q_scale"].max()),
            "simulated_conductivity_min_ms_cm": sim_min,
            "simulated_conductivity_max_ms_cm": sim_max,
            "experiment_inside_simulated_range": bool(sim_min <= exp_value <= sim_max),
            "tie_tolerance_ms_cm": tie_tolerance,
            "tie_rule": "lowest_q_scale_within_tolerance",
        }
        records.append(record)

    targets = pd.DataFrame(records)
    targets["cluster_group"] = targets["cluster_id"].astype(str).fillna("unknown")
    return targets.sort_values(["cluster_id", "original_row_index"]).reset_index(drop=True)


def available_features(targets: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return usable numeric and categorical feature columns."""

    numeric = [
        column
        for column in NUMERIC_FEATURES
        if column in targets.columns and pd.api.types.is_numeric_dtype(targets[column])
    ]
    categorical = [column for column in CATEGORICAL_FEATURES if column in targets.columns]
    return numeric, categorical


def make_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """Build preprocessing for simple system-level metadata."""

    transformers: list[tuple[str, Pipeline, list[str]]] = [
        (
            "numeric",
            Pipeline(
                [
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
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def candidate_models(random_seed: int) -> dict[str, BaseEstimator]:
    """Return simple q_scale regressors."""

    return {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "lasso": Lasso(alpha=0.002, max_iter=10000, random_state=random_seed),
        "small_tree": DecisionTreeRegressor(max_depth=2, min_samples_leaf=2, random_state=random_seed),
        "shallow_random_forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=3,
            min_samples_leaf=2,
            random_state=random_seed,
            n_jobs=-1,
        ),
    }


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression metrics."""

    residual = y_pred - y_true
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan,
    }


def cross_validate_models(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: pd.Series,
    preprocessor: ColumnTransformer,
    models: dict[str, BaseEstimator],
    cv_splits: int,
    q_min: float,
    q_max: float,
) -> pd.DataFrame:
    """Evaluate candidate q_scale models with cluster-aware grouped CV."""

    n_groups = groups.nunique()
    splitter = GroupKFold(n_splits=min(cv_splits, n_groups)) if n_groups >= 2 else None
    records: list[dict[str, object]] = []

    for model_name, model in models.items():
        fold_rows = []
        if splitter is None:
            continue
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y, groups), start=1):
            pipeline = Pipeline(
                [
                    ("preprocess", clone(preprocessor)),
                    ("model", clone(model)),
                ]
            )
            pipeline.fit(X.iloc[train_idx], y[train_idx])
            pred = np.clip(pipeline.predict(X.iloc[valid_idx]), q_min, q_max)
            row = {
                "model": model_name,
                "fold": fold,
                "n_train": len(train_idx),
                "n_valid": len(valid_idx),
            }
            row.update(metrics(y[valid_idx], pred))
            fold_rows.append(row)

        fold_df = pd.DataFrame(fold_rows)
        summary = {
            "model": model_name,
            "n_folds": int(fold_df["fold"].nunique()),
            "cv_mae_mean": float(fold_df["mae"].mean()),
            "cv_mae_std": float(fold_df["mae"].std(ddof=0)),
            "cv_rmse_mean": float(fold_df["rmse"].mean()),
            "cv_rmse_std": float(fold_df["rmse"].std(ddof=0)),
            "cv_r2_mean": float(fold_df["r2"].mean()),
            "cv_r2_std": float(fold_df["r2"].std(ddof=0)),
        }
        records.append(summary)

    return pd.DataFrame(records).sort_values(["cv_rmse_mean", "cv_mae_mean"]).reset_index(drop=True)


def select_model(cv_results: pd.DataFrame, tolerance: float) -> str:
    """Choose the simplest model within an RMSE tolerance of the best."""

    simplicity = {
        "mean_baseline": 0,
        "linear_regression": 1,
        "ridge": 2,
        "lasso": 3,
        "small_tree": 4,
        "shallow_random_forest": 5,
    }
    non_baseline = cv_results.loc[cv_results["model"] != "mean_baseline"].copy()
    best_rmse = float(non_baseline["cv_rmse_mean"].min())
    eligible = non_baseline.loc[non_baseline["cv_rmse_mean"] <= best_rmse + tolerance].copy()
    eligible["simplicity"] = eligible["model"].map(simplicity)
    return str(eligible.sort_values(["simplicity", "cv_rmse_mean"]).iloc[0]["model"])


def save_split(targets: pd.DataFrame, test_size: float, random_seed: int, output_dir: Path) -> pd.DataFrame:
    """Save leakage-safe train/test system split."""

    groups = targets["cluster_group"].astype(str)
    if groups.nunique() >= 2 and len(targets) >= 4:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
        train_idx, test_idx = next(splitter.split(targets, targets[TARGET], groups))
    else:
        rng = np.random.default_rng(random_seed)
        indices = np.arange(len(targets))
        rng.shuffle(indices)
        n_test = max(1, int(round(len(indices) * test_size)))
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

    split = targets[["electrolyte_group", "cluster_id", TARGET]].copy()
    split["split"] = "train"
    split.loc[targets.index[test_idx], "split"] = "test"
    split.to_csv(output_dir / "system_split.csv", index=False)
    return split


def train_final_model(
    targets: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    model_name: str,
    random_seed: int,
) -> Pipeline:
    """Fit the selected model on all target rows."""

    preprocessor = make_preprocessor(numeric_features, categorical_features)
    model = candidate_models(random_seed)[model_name]
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipeline.fit(targets[numeric_features + categorical_features], targets[TARGET].to_numpy(dtype=float))
    return pipeline


def feature_effects(pipeline: Pipeline, feature_columns: list[str], model_name: str) -> pd.DataFrame:
    """Save coefficients or feature importances where the estimator exposes them."""

    try:
        names = pipeline.named_steps["preprocess"].get_feature_names_out().tolist()
    except Exception:
        names = feature_columns
    estimator = pipeline.named_steps["model"]
    if hasattr(estimator, "coef_"):
        coef = np.ravel(estimator.coef_).astype(float)
        return pd.DataFrame(
            {
                "feature": names,
                "coefficient": coef,
                "importance": np.abs(coef),
                "importance_type": "standardized_coefficient",
                "model": model_name,
            }
        ).sort_values("importance", ascending=False)
    if hasattr(estimator, "feature_importances_"):
        return pd.DataFrame(
            {
                "feature": names,
                "importance": estimator.feature_importances_,
                "importance_type": "model_feature_importance",
                "model": model_name,
            }
        ).sort_values("importance", ascending=False)
    return pd.DataFrame({"feature": names, "importance": np.nan, "importance_type": "not_available"})


def run(args: argparse.Namespace) -> None:
    """Run training."""

    args.output_dir.mkdir(parents=True, exist_ok=True)
    campaign = read_campaign_csv(args.input_csv)
    targets = derive_qscale_targets(campaign, args.tie_tolerance)
    targets.to_csv(args.output_dir / "qscale_optimal_targets.csv", index=False)

    numeric_features, categorical_features = available_features(targets)
    feature_columns = numeric_features + categorical_features
    X = targets[feature_columns].copy()
    y = targets[TARGET].to_numpy(dtype=float)
    cv_groups = targets["cluster_group"].astype(str)
    q_min = float(campaign["q_scale"].min())
    q_max = float(campaign["q_scale"].max())

    preprocessor = make_preprocessor(numeric_features, categorical_features)
    models = candidate_models(args.random_seed)
    cv_results = cross_validate_models(
        X,
        y,
        cv_groups,
        preprocessor,
        models,
        args.cv_splits,
        q_min,
        q_max,
    )
    cv_results.to_csv(args.output_dir / "model_cv_results.csv", index=False)
    selected_model = select_model(cv_results, args.selection_tolerance)
    split = save_split(targets, args.test_size, args.random_seed, args.output_dir)

    pipeline = train_final_model(
        targets,
        numeric_features,
        categorical_features,
        selected_model,
        args.random_seed,
    )
    model_bundle = {
        "pipeline": pipeline,
        "selected_model": selected_model,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "feature_columns": feature_columns,
        "target_column": TARGET,
        "q_scale_min": q_min,
        "q_scale_max": q_max,
        "tie_tolerance_ms_cm": args.tie_tolerance,
    }
    joblib.dump(model_bundle, args.output_dir / "qscale_model.joblib")
    joblib.dump(pipeline.named_steps["preprocess"], args.output_dir / "preprocessing.joblib")

    train_groups = set(split.loc[split["split"] == "train", "electrolyte_group"])
    train_targets = targets.loc[targets["electrolyte_group"].isin(train_groups)].copy()
    holdout_pipeline = train_final_model(
        train_targets,
        numeric_features,
        categorical_features,
        selected_model,
        args.random_seed,
    )
    holdout_bundle = dict(model_bundle)
    holdout_bundle["pipeline"] = holdout_pipeline
    holdout_bundle["training_scope"] = "train_split_only"
    holdout_bundle["held_out_systems"] = split.loc[
        split["split"] == "test",
        "electrolyte_group",
    ].tolist()
    joblib.dump(holdout_bundle, args.output_dir / "qscale_model_train_split.joblib")
    effects = feature_effects(pipeline, feature_columns, selected_model)
    effects.to_csv(args.output_dir / "feature_effects.csv", index=False)

    metadata = {
        "input_csv": str(args.input_csv),
        "output_dir": str(args.output_dir),
        "random_seed": args.random_seed,
        "n_systems": int(len(targets)),
        "n_clusters": int(targets["cluster_id"].nunique()),
        "features": feature_columns,
        "selected_model": selected_model,
    }
    (args.output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    summary = [
        "Direct q_scale-optimal model training",
        "====================================",
        f"systems: {len(targets)}",
        f"selected_model: {selected_model}",
        f"features: {', '.join(feature_columns)}",
        "",
        "Best CV rows:",
        cv_results.head(5).to_string(index=False),
    ]
    (args.output_dir / "training_summary.txt").write_text("\n".join(summary) + "\n")

    print(f"Wrote q_scale model outputs to: {args.output_dir}")
    print(f"Selected model: {selected_model}")
    print(cv_results.head(5).to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("results/sparse_campaign_weighted_permittivity/conductivity_results.csv"),
        help="Sparse campaign conductivity_results.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/qscale_direct_model_weighted_permittivity"),
        help="Directory for model artifacts.",
    )
    parser.add_argument("--tie-tolerance", type=float, default=0.0, help="mS/cm q-scale tie tolerance.")
    parser.add_argument("--test-size", type=float, default=0.3, help="Held-out cluster fraction.")
    parser.add_argument("--cv-splits", type=int, default=5, help="Maximum grouped CV folds.")
    parser.add_argument(
        "--selection-tolerance",
        type=float,
        default=0.01,
        help="Choose the simplest model within this q_scale RMSE of the best.",
    )
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED, help="Fixed random seed.")
    return parser


def main() -> None:
    """Command-line entry point."""

    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
