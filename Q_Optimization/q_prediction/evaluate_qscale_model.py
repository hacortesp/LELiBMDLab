#!/usr/bin/env python3
"""Evaluate the simplified Extra Trees q_scale model."""

import argparse
import os
import sys

os.environ.setdefault("PANDAS_NO_IMPORT_PYARROW", "1")
sys.modules.setdefault("pyarrow", None)

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit

try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:
    StratifiedGroupKFold = None

from qscale_common import (
    DEFAULT_INPUT_CSV,
    DEFAULT_OUTPUT_DIR,
    FEATURE_COLUMNS,
    build_training_table,
    load_campaign_csv,
    make_model,
    make_preprocessor,
    q_label,
    qscale_step,
    rmse,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate q_scale classifier with a leakage-safe system split."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_CSV,
        help="Filtered conductivity campaign CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory containing q_prediction outputs.",
    )
    parser.add_argument(
        "--training-table",
        default=None,
        help="Optional qscale_training_table.csv path. Rebuilt from input if omitted.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for validation split and model.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Validation fraction for GroupShuffleSplit fallback.",
    )
    return parser.parse_args()


def load_or_build_training_table(args):
    if args.training_table:
        table = pd.read_csv(args.training_table)
        table["q_scale_class"] = table["q_scale_optimal"].apply(q_label)
        return table
    default_table = os.path.join(args.output_dir, "qscale_training_table.csv")
    if os.path.isfile(default_table):
        table = pd.read_csv(default_table)
        table["q_scale_class"] = table["q_scale_optimal"].apply(q_label)
        return table
    raw = load_campaign_csv(args.input)
    table = build_training_table(raw)
    table["q_scale_class"] = table["q_scale_optimal"].apply(q_label)
    return table


def validation_split(table, random_seed, test_size):
    X = table[FEATURE_COLUMNS].copy()
    y = table["q_scale_class"].astype(str).values
    groups = table["system_id"].astype(str).values
    if StratifiedGroupKFold is not None:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_seed)
        train_index, val_index = list(splitter.split(X, y, groups))[0]
        return train_index, val_index, "StratifiedGroupKFold first fold"
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    train_index, val_index = list(splitter.split(X, y, groups))[0]
    return train_index, val_index, "GroupShuffleSplit"


def main():
    args = parse_args()
    output_dir = os.path.abspath(args.output_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    table = load_or_build_training_table(args)
    X = table[FEATURE_COLUMNS].copy()
    y = table["q_scale_class"].astype(str).values
    train_index, val_index, split_method = validation_split(table, args.random_seed, args.test_size)

    preprocessor = make_preprocessor()
    X_train = preprocessor.fit_transform(X.iloc[train_index])
    X_val = preprocessor.transform(X.iloc[val_index])
    model = make_model(args.random_seed)
    model.fit(X_train, y[train_index])
    predicted = model.predict(X_val)

    y_true_float = np.array([float(v) for v in y[val_index]])
    y_pred_float = np.array([float(v) for v in predicted])
    step = qscale_step(sorted(set(y)))
    exact = accuracy_score(y[val_index], predicted)
    within_one = float(np.mean(np.abs(y_pred_float - y_true_float) <= step + 1e-9))

    metrics = pd.DataFrame(
        [
            {"metric": "validation_split_method", "value": split_method},
            {"metric": "training_systems", "value": int(len(train_index))},
            {"metric": "validation_systems", "value": int(len(val_index))},
            {"metric": "MAE", "value": float(mean_absolute_error(y_true_float, y_pred_float))},
            {"metric": "RMSE", "value": float(rmse(y_true_float, y_pred_float))},
            {"metric": "exact_qscale_accuracy", "value": float(exact)},
            {"metric": "within_one_step_accuracy", "value": within_one},
        ]
    )
    metrics.to_csv(os.path.join(output_dir, "evaluation_metrics.csv"), index=False)

    predictions = table.iloc[val_index][
        [
            "system_id",
            "original_row_index",
            "doi",
            "anion_name",
            "salt_conc",
            "temperature",
            "solvents",
            "solvent_fracs",
            "q_scale_optimal",
            "q_scale_class",
        ]
    ].copy()
    predictions["predicted_q_scale"] = predicted
    predictions["absolute_qscale_error"] = np.abs(y_pred_float - y_true_float)
    predictions["exact_prediction"] = predictions["q_scale_class"].astype(str).values == predicted
    predictions["within_one_step"] = predictions["absolute_qscale_error"] <= step + 1e-9
    predictions.to_csv(os.path.join(output_dir, "validation_predictions.csv"), index=False)

    labels = sorted(set(table["q_scale_class"].astype(str)))
    matrix = confusion_matrix(y[val_index], predicted, labels=labels)
    matrix_df = pd.DataFrame(matrix, index=["true_" + x for x in labels], columns=["pred_" + x for x in labels])
    matrix_df.to_csv(os.path.join(output_dir, "confusion_matrix.csv"))

    print("Evaluation complete")
    print("Split method: {0}".format(split_method))
    print("Training systems: {0}".format(len(train_index)))
    print("Validation systems: {0}".format(len(val_index)))
    print("MAE: {0:.4f}".format(float(metrics.loc[metrics["metric"] == "MAE", "value"].iloc[0])))
    print("RMSE: {0:.4f}".format(float(metrics.loc[metrics["metric"] == "RMSE", "value"].iloc[0])))
    print("Exact q_scale accuracy: {0:.4f}".format(exact))
    print("Within-one-step accuracy: {0:.4f}".format(within_one))
    print("Saved metrics: {0}".format(os.path.join(output_dir, "evaluation_metrics.csv")))
    print("Saved validation predictions: {0}".format(os.path.join(output_dir, "validation_predictions.csv")))
    print("Saved confusion matrix: {0}".format(os.path.join(output_dir, "confusion_matrix.csv")))


if __name__ == "__main__":
    main()
