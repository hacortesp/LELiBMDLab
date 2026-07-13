#!/usr/bin/env python3
"""Train a simple Extra Trees model for q_scale recommendation."""

import argparse
import os
import sys

os.environ.setdefault("PANDAS_NO_IMPORT_PYARROW", "1")
sys.modules.setdefault("pyarrow", None)

import joblib
import pandas as pd

from qscale_common import (
    CATEGORICAL_FEATURES,
    DEFAULT_INPUT_CSV,
    DEFAULT_OUTPUT_DIR,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    build_training_table,
    load_campaign_csv,
    make_model,
    make_preprocessor,
    q_label,
    save_metadata,
    transformed_feature_names,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the simplified Extra Trees q_scale classifier."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_CSV,
        help="Filtered conductivity campaign CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for q_prediction outputs.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for the Extra Trees model.",
    )
    return parser.parse_args()


def write_summary(path, input_csv, training_table, feature_importance, random_seed):
    target_counts = training_table["q_scale_class"].value_counts().sort_index()
    lines = []
    lines.append("Simplified Extra Trees q_scale training summary")
    lines.append("")
    lines.append("Input CSV: {0}".format(input_csv))
    lines.append("Number of electrolyte systems: {0}".format(len(training_table)))
    lines.append("Random seed: {0}".format(random_seed))
    lines.append("")
    lines.append("System grouping:")
    lines.append("- Prefer original_row_index when it uniquely identifies system chemistry and conditions.")
    lines.append("- Otherwise use anion_name, salt_conc, solvents, solvent_fracs, temperature, and doi.")
    lines.append("")
    lines.append("Target definition:")
    lines.append("- q_scale_optimal = argmin_q |log(conductivity_mS_cm) - log(conductivity)|")
    lines.append("- Only finite positive experimental and simulated conductivities are used.")
    lines.append("- Ties are resolved by choosing the smallest tied q_scale.")
    lines.append("")
    lines.append("Target distribution:")
    for q_scale, count in target_counts.items():
        lines.append("- q_scale {0}: {1} systems".format(q_scale, int(count)))
    lines.append("")
    lines.append("Features used:")
    for column in FEATURE_COLUMNS:
        lines.append("- {0}".format(column))
    lines.append("")
    lines.append("Numeric features:")
    for column in NUMERIC_FEATURES:
        lines.append("- {0}".format(column))
    lines.append("")
    lines.append("Categorical features:")
    for column in CATEGORICAL_FEATURES:
        lines.append("- {0}".format(column))
    lines.append("")
    lines.append("Preprocessing:")
    lines.append("- Numeric features: median imputation.")
    lines.append("- Categorical features: most-frequent imputation and one-hot encoding.")
    lines.append("")
    lines.append("Model:")
    lines.append("- ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, class_weight='balanced')")
    lines.append("")
    lines.append("Top feature importances:")
    for _, row in feature_importance.head(15).iterrows():
        lines.append("- {0}: {1:.6f}".format(row["feature"], float(row["importance"])))
    lines.append("")
    with open(path, "w") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")


def main():
    args = parse_args()
    output_dir = os.path.abspath(args.output_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    raw = load_campaign_csv(args.input)
    training_table = build_training_table(raw)
    training_table = training_table.sort_values(["anion_name", "system_id"]).reset_index(drop=True)

    X = training_table[FEATURE_COLUMNS].copy()
    y = training_table["q_scale_class"].astype(str).values

    preprocessor = make_preprocessor()
    X_transformed = preprocessor.fit_transform(X)
    model = make_model(args.random_seed)
    model.fit(X_transformed, y)

    feature_names = transformed_feature_names(preprocessor)
    feature_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    training_table.to_csv(os.path.join(output_dir, "qscale_training_table.csv"), index=False)
    feature_importance.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)
    joblib.dump(model, os.path.join(output_dir, "qscale_extratrees_model.joblib"))
    joblib.dump(preprocessor, os.path.join(output_dir, "qscale_preprocessor.joblib"))
    save_metadata(os.path.join(output_dir, "qscale_model_metadata.json"), args.random_seed)
    write_summary(
        os.path.join(output_dir, "training_summary.txt"),
        os.path.abspath(args.input),
        training_table,
        feature_importance,
        args.random_seed,
    )

    print("Trained Extra Trees q_scale classifier")
    print("Input rows: {0}".format(len(raw)))
    print("System-level training rows: {0}".format(len(training_table)))
    print("Available q_scale targets: {0}".format(", ".join(sorted(training_table["q_scale_class"].unique()))))
    print("Features used: {0}".format(", ".join(FEATURE_COLUMNS)))
    print("Saved training table: {0}".format(os.path.join(output_dir, "qscale_training_table.csv")))
    print("Saved model: {0}".format(os.path.join(output_dir, "qscale_extratrees_model.joblib")))
    print("Saved preprocessor: {0}".format(os.path.join(output_dir, "qscale_preprocessor.joblib")))
    print("Saved feature importance: {0}".format(os.path.join(output_dir, "feature_importance.csv")))
    print("Saved training summary: {0}".format(os.path.join(output_dir, "training_summary.txt")))


if __name__ == "__main__":
    main()
