#!/usr/bin/env python3
"""Write a reproducibility and results report for q_prediction."""

import argparse
import json
import os
import sys

os.environ.setdefault("PANDAS_NO_IMPORT_PYARROW", "1")
sys.modules.setdefault("pyarrow", None)

import pandas as pd


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = os.path.join(THIS_DIR, "qscale_reproducibility_report.md")

REQUIRED_FILES = [
    "qscale_training_table.csv",
    "training_summary.txt",
    "feature_importance.csv",
    "evaluation_metrics.csv",
    "validation_predictions.csv",
    "confusion_matrix.csv",
    "qscale_model_metadata.json",
    "min_log_error_by_system.csv",
]

PLOT_FILES = [
    "predicted_vs_true_qscale.png",
    "confusion_matrix.png",
    "feature_importance.png",
    "qscale_target_distribution.png",
    "min_log_error_vs_qscale.png",
    "min_log_error_by_system.png",
    "conductivity_curves_examples.png",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Write q_prediction reproducibility and results report."
    )
    parser.add_argument(
        "--output-dir",
        default=THIS_DIR,
        help="q_prediction directory containing generated workflow outputs.",
    )
    parser.add_argument(
        "--report",
        default=DEFAULT_OUTPUT,
        help="Output markdown report path.",
    )
    return parser.parse_args()


def read_csv_if_exists(path):
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path)


def read_text_if_exists(path):
    if not os.path.isfile(path):
        return None
    with open(path, "r") as handle:
        return handle.read()


def read_json_if_exists(path):
    if not os.path.isfile(path):
        return None
    with open(path, "r") as handle:
        return json.load(handle)


def metric_value(metrics, name):
    if metrics is None:
        return None
    match = metrics.loc[metrics["metric"] == name, "value"]
    if match.empty:
        return None
    return match.iloc[0]


def number_or_text(value, digits=4):
    if value is None:
        return "missing"
    try:
        return ("{0:." + str(digits) + "f}").format(float(value))
    except Exception:
        return str(value)


def markdown_table(df, columns=None, max_rows=None, float_digits=4):
    if df is None:
        return "Required table is missing."
    table = df.copy()
    if columns is not None:
        table = table[columns]
    if max_rows is not None:
        table = table.head(max_rows)
    headers = list(table.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in table.iterrows():
        values = []
        for col in headers:
            value = row[col]
            if isinstance(value, float):
                values.append(("{0:." + str(float_digits) + "f}").format(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def target_distribution(training):
    if training is None or "q_scale_class" not in training.columns:
        return None
    counts = training["q_scale_class"].astype(str).value_counts().sort_index()
    return pd.DataFrame({"q_scale_optimal": counts.index, "systems": counts.values})


def validation_error_summary(predictions):
    if predictions is None:
        return None
    if "anion_name" not in predictions.columns:
        return None
    grouped = predictions.groupby("anion_name").agg(
        systems=("system_id", "count"),
        mean_absolute_error=("absolute_qscale_error", "mean"),
        exact_accuracy=("exact_prediction", "mean"),
        within_one_step_accuracy=("within_one_step", "mean"),
    )
    return grouped.reset_index().sort_values("mean_absolute_error")


def file_status_table(output_dir):
    rows = []
    for name in REQUIRED_FILES + PLOT_FILES:
        path = os.path.join(output_dir, name)
        rows.append(
            {
                "file": name,
                "status": "present" if os.path.isfile(path) else "missing",
            }
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    output_dir = os.path.abspath(args.output_dir)
    report_path = os.path.abspath(args.report)

    training = read_csv_if_exists(os.path.join(output_dir, "qscale_training_table.csv"))
    feature_importance = read_csv_if_exists(os.path.join(output_dir, "feature_importance.csv"))
    metrics = read_csv_if_exists(os.path.join(output_dir, "evaluation_metrics.csv"))
    predictions = read_csv_if_exists(os.path.join(output_dir, "validation_predictions.csv"))
    confusion = read_csv_if_exists(os.path.join(output_dir, "confusion_matrix.csv"))
    training_summary = read_text_if_exists(os.path.join(output_dir, "training_summary.txt"))
    metadata = read_json_if_exists(os.path.join(output_dir, "qscale_model_metadata.json"))
    statuses = file_status_table(output_dir)
    missing = statuses.loc[statuses["status"] == "missing", "file"].tolist()

    report = []
    report.append("# q_prediction Reproducibility and Results Report")
    report.append("")
    report.append("This report was generated from the files currently present in `{0}`.".format(output_dir))
    report.append("")

    if missing:
        report.append("**Missing required or expected files:** {0}.".format(", ".join(missing)))
        report.append("")

    report.append("## 1. Objective")
    report.append("")
    report.append(
        "The scientific goal is to recommend an electrolyte-specific charge scaling factor, "
        "`q_scale`, before running a new molecular-dynamics q-scale sweep. The simplified "
        "workflow focuses on the best model family identified in the previous comparison: "
        "an Extra Trees Classifier."
    )
    report.append("")

    report.append("## 2. Input Data")
    report.append("")
    if training is not None:
        report.append(
            "The processed system-level table is `qscale_training_table.csv` with {0} electrolyte systems "
            "and {1} columns.".format(training.shape[0], training.shape[1])
        )
        input_rows = int(training["raw_rows_for_system"].sum()) if "raw_rows_for_system" in training.columns else "unavailable"
        valid_rows = int(training["valid_rows_for_system"].sum()) if "valid_rows_for_system" in training.columns else "unavailable"
        report.append("It was derived from {0} raw q-scale rows, of which {1} were valid for target construction.".format(input_rows, valid_rows))
    else:
        report.append("`qscale_training_table.csv` is missing, so input table dimensions are unavailable.")
    report.append("")
    report.append("File status summary:")
    report.append("")
    report.append(markdown_table(statuses))
    report.append("")

    report.append("## 3. Target Definition")
    report.append("")
    report.append("For each electrolyte system, the target is:")
    report.append("")
    report.append("```text")
    report.append("q_scale_optimal = argmin_q |log(conductivity_mS_cm) - log(conductivity)|")
    report.append("```")
    report.append("")
    report.append(
        "`conductivity_mS_cm` is the simulated conductivity and `conductivity` is the experimental "
        "conductivity. The logarithmic error compares relative conductivity agreement and prevents "
        "high-conductivity systems from dominating the target definition."
    )
    if training is not None:
        tie_count = int(training["tie_flag"].sum()) if "tie_flag" in training.columns else "unavailable"
        complete = int((training["number_of_qscale_points"] == 5).sum()) if "number_of_qscale_points" in training.columns else "unavailable"
        report.append("")
        report.append("Ties are resolved by selecting the smallest tied q-scale. The current training table records `{0}` tied systems and `{1}` systems with all five q-scale values.".format(tie_count, complete))
        dist = target_distribution(training)
        report.append("")
        report.append("Target distribution:")
        report.append("")
        report.append(markdown_table(dist))
    report.append("")

    report.append("## 4. Feature Engineering")
    report.append("")
    if metadata is not None:
        numeric_features = metadata.get("numeric_features", [])
        categorical_features = metadata.get("categorical_features", [])
        report.append("Numeric features: `{0}`.".format("`, `".join(numeric_features)))
        report.append("")
        report.append("Categorical features: `{0}`.".format("`, `".join(categorical_features)))
        report.append("")
        report.append("Allowed solvents: `{0}`.".format("`, `".join(metadata.get("allowed_solvents", []))))
    else:
        report.append("`qscale_model_metadata.json` is missing, so the saved feature list is unavailable.")
    report.append("")
    report.append(
        "The workflow uses simple interpretable features: anion identity, salt concentration, "
        "temperature, solvent-presence indicators, and solvent-fraction values. It does not use "
        "`q_scale`, simulated conductivity, or simulation uncertainty as model inputs."
    )
    report.append("")

    report.append("## 5. Model Training")
    report.append("")
    report.append(
        "The model is an `ExtraTreesClassifier` with 500 trees, `min_samples_leaf=2`, "
        "`class_weight='balanced'`, and `random_state=42` in the generated run. Numeric features "
        "are median-imputed. Categorical features are imputed with the most frequent value and one-hot encoded."
    )
    report.append("")
    report.append(
        "Extra Trees is used because the previous comparison ranked Extra Trees Classifier as the best "
        "method by cross-validated MAE, and because it handles nonlinear interactions between salt, "
        "temperature, concentration, and solvent descriptors while still producing feature importances."
    )
    if feature_importance is not None:
        report.append("")
        report.append("Top feature importances:")
        report.append("")
        report.append(markdown_table(feature_importance, max_rows=12))
    if training_summary is not None:
        report.append("")
        report.append("The complete plain-text training summary is saved in `training_summary.txt`.")
    report.append("")

    report.append("## 6. Validation Strategy")
    report.append("")
    split_method = metric_value(metrics, "validation_split_method")
    train_systems = metric_value(metrics, "training_systems")
    val_systems = metric_value(metrics, "validation_systems")
    report.append(
        "Evaluation uses a system-level validation split rather than splitting individual q-scale rows. "
        "This avoids leakage from the same electrolyte appearing in both training and validation. "
        "The generated evaluation used `{0}` with {1} training systems and {2} validation systems.".format(
            number_or_text(split_method),
            number_or_text(train_systems, 0),
            number_or_text(val_systems, 0),
        )
    )
    report.append("")

    report.append("## 7. Evaluation Metrics")
    report.append("")
    report.append(
        "The workflow reports MAE and RMSE in q-scale units, exact q-scale accuracy, and "
        "within-one-step accuracy. Exact accuracy is strict because adjacent q-scale values differ by "
        "only 0.05. Within-one-step accuracy is useful because a prediction one grid step away may still "
        "be a practical starting point for a follow-up simulation."
    )
    report.append("")

    report.append("## 8. Results")
    report.append("")
    if metrics is not None:
        report.append("Evaluation metrics from `evaluation_metrics.csv`:")
        report.append("")
        report.append(markdown_table(metrics))
        report.append("")
        report.append(
            "Main validation result: MAE = {0}, RMSE = {1}, exact accuracy = {2}, "
            "within-one-step accuracy = {3}.".format(
                number_or_text(metric_value(metrics, "MAE")),
                number_or_text(metric_value(metrics, "RMSE")),
                number_or_text(metric_value(metrics, "exact_qscale_accuracy")),
                number_or_text(metric_value(metrics, "within_one_step_accuracy")),
            )
        )
    else:
        report.append("`evaluation_metrics.csv` is missing, so validation metrics are unavailable.")
    if predictions is not None:
        report.append("")
        report.append("Validation predictions are saved in `validation_predictions.csv` with {0} rows.".format(len(predictions)))
        report.append("")
        report.append("Example validation predictions:")
        report.append("")
        report.append(
            markdown_table(
                predictions[
                    [
                        "system_id",
                        "anion_name",
                        "salt_conc",
                        "temperature",
                        "q_scale_class",
                        "predicted_q_scale",
                        "absolute_qscale_error",
                        "exact_prediction",
                        "within_one_step",
                    ]
                ],
                max_rows=8,
            )
        )
        salt_summary = validation_error_summary(predictions)
        if salt_summary is not None:
            report.append("")
            report.append("Validation error by anion:")
            report.append("")
            report.append(markdown_table(salt_summary))
    if confusion is not None:
        report.append("")
        report.append("Confusion matrix from `confusion_matrix.csv`:")
        report.append("")
        report.append(markdown_table(confusion))
    report.append("")
    report.append("Generated diagnostic plots:")
    report.append("")
    for plot_file in PLOT_FILES:
        if os.path.isfile(os.path.join(output_dir, plot_file)):
            report.append("- `{0}`".format(plot_file))
        else:
            report.append("- `{0}`: missing".format(plot_file))
    report.append("")

    report.append("## 9. How to Reproduce")
    report.append("")
    report.append("From the repository root, run:")
    report.append("")
    report.append("```bash")
    report.append("python q_prediction/train_qscale_extratrees.py")
    report.append("python q_prediction/evaluate_qscale_model.py")
    report.append("python q_prediction/plot_qscale_model_diagnostics.py")
    report.append("python q_prediction/write_qscale_report.py")
    report.append("```")
    report.append("")
    report.append("The default input is `results/sparse_campaign_salt_name/conductivity_results_filtered.csv`. Use `--input` on the training, evaluation, and plotting scripts to point to another compatible filtered CSV.")
    report.append("")

    report.append("## 10. How to Predict q_scale for a New System")
    report.append("")
    report.append("Example command:")
    report.append("")
    report.append("```bash")
    report.append("python q_prediction/predict_qscale.py \\")
    report.append("  -anion_name PF6 \\")
    report.append("  -salt-conc 1.0 \\")
    report.append("  -solvents EC DMC \\")
    report.append("  -solvent-fracs 0.5 0.5 \\")
    report.append("  -temperature 298.0")
    report.append("```")
    report.append("")
    report.append("The predictor loads `qscale_extratrees_model.joblib` and `qscale_preprocessor.joblib`, normalizes solvent fractions if needed, builds the same feature representation used during training, and prints the recommended q-scale plus class probabilities when available.")
    report.append("")

    report.append("## 11. Limitations and Next Steps")
    report.append("")
    report.append(
        "The current dataset is small: the generated training table contains only {0} systems. "
        "The validation split contains {1} systems, so metrics should be interpreted as a practical "
        "diagnostic rather than a definitive estimate of generalization.".format(
            training.shape[0] if training is not None else "an unavailable number of",
            number_or_text(val_systems, 0),
        )
    )
    report.append("")
    report.append(
        "The simplified model intentionally excludes experimental conductivity and all simulated outputs, "
        "so it is easier to use prospectively but may be less accurate than models that include more context. "
        "Recommended next steps are to add more electrolyte systems, especially underrepresented salt and "
        "solvent families; simulate additional q-scale values near ambiguous optima; and use this model as "
        "an initial recommendation rather than a replacement for validation simulations."
    )
    report.append("")

    report_dir = os.path.dirname(report_path)
    if report_dir and not os.path.isdir(report_dir):
        os.makedirs(report_dir)
    with open(report_path, "w") as handle:
        handle.write("\n".join(report))
        handle.write("\n")

    print("Saved report: {0}".format(report_path))
    if missing:
        print("Missing files noted in report: {0}".format(", ".join(missing)))


if __name__ == "__main__":
    main()
