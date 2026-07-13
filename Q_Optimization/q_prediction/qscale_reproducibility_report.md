# q_prediction Reproducibility and Results Report

This report was generated from the files currently present in `/home/hacortes/Li-EMDLab/Q_Optimization/q_prediction`.

## 1. Objective

The scientific goal is to recommend an electrolyte-specific charge scaling factor, `q_scale`, before running a new molecular-dynamics q-scale sweep. The simplified workflow focuses on the best model family identified in the previous comparison: an Extra Trees Classifier.

## 2. Input Data

The processed system-level table is `qscale_training_table.csv` with 75 electrolyte systems and 31 columns.
It was derived from 375 raw q-scale rows, of which 373 were valid for target construction.

File status summary:

| file | status |
| --- | --- |
| qscale_training_table.csv | present |
| training_summary.txt | present |
| feature_importance.csv | present |
| evaluation_metrics.csv | present |
| validation_predictions.csv | present |
| confusion_matrix.csv | present |
| qscale_model_metadata.json | present |
| min_log_error_by_system.csv | present |
| predicted_vs_true_qscale.png | present |
| confusion_matrix.png | present |
| feature_importance.png | present |
| qscale_target_distribution.png | present |
| min_log_error_vs_qscale.png | present |
| min_log_error_by_system.png | present |
| conductivity_curves_examples.png | present |

## 3. Target Definition

For each electrolyte system, the target is:

```text
q_scale_optimal = argmin_q |log(conductivity_mS_cm) - log(conductivity)|
```

`conductivity_mS_cm` is the simulated conductivity and `conductivity` is the experimental conductivity. The logarithmic error compares relative conductivity agreement and prevents high-conductivity systems from dominating the target definition.

Ties are resolved by selecting the smallest tied q-scale. The current training table records `0` tied systems and `73` systems with all five q-scale values.

Target distribution:

| q_scale_optimal | systems |
| --- | --- |
| 0.7 | 14 |
| 0.75 | 15 |
| 0.8 | 18 |
| 0.85 | 16 |
| 0.9 | 12 |

## 4. Feature Engineering

Numeric features: `salt_conc_numeric`, `temperature_numeric`, `has_ec`, `frac_ec`, `has_pc`, `frac_pc`, `has_dme`, `frac_dme`, `has_dmc`, `frac_dmc`, `has_dec`, `frac_dec`, `has_emc`, `frac_emc`.

Categorical features: `anion_name`.

Allowed solvents: `EC`, `PC`, `DME`, `DMC`, `DEC`, `EMC`.

The workflow uses simple interpretable features: anion identity, salt concentration, temperature, solvent-presence indicators, and solvent-fraction values. It does not use `q_scale`, simulated conductivity, or simulation uncertainty as model inputs.

## 5. Model Training

The model is an `ExtraTreesClassifier` with 500 trees, `min_samples_leaf=2`, `class_weight='balanced'`, and `random_state=42` in the generated run. Numeric features are median-imputed. Categorical features are imputed with the most frequent value and one-hot encoded.

Extra Trees is used because the previous comparison ranked Extra Trees Classifier as the best method by cross-validated MAE, and because it handles nonlinear interactions between salt, temperature, concentration, and solvent descriptors while still producing feature importances.

Top feature importances:

| feature | importance |
| --- | --- |
| temperature_numeric | 0.1645 |
| salt_conc_numeric | 0.1252 |
| anion_name_PF6 | 0.0877 |
| anion_name_BF4 | 0.0851 |
| anion_name_TFSI | 0.0843 |
| anion_name_ClO4 | 0.0567 |
| frac_pc | 0.0551 |
| frac_ec | 0.0507 |
| anion_name_FSI | 0.0432 |
| has_dmc | 0.0389 |
| has_ec | 0.0358 |
| frac_dme | 0.0352 |

The complete plain-text training summary is saved in `training_summary.txt`.

## 6. Validation Strategy

Evaluation uses a system-level validation split rather than splitting individual q-scale rows. This avoids leakage from the same electrolyte appearing in both training and validation. The generated evaluation used `StratifiedGroupKFold first fold` with 60 training systems and 15 validation systems.

## 7. Evaluation Metrics

The workflow reports MAE and RMSE in q-scale units, exact q-scale accuracy, and within-one-step accuracy. Exact accuracy is strict because adjacent q-scale values differ by only 0.05. Within-one-step accuracy is useful because a prediction one grid step away may still be a practical starting point for a follow-up simulation.

## 8. Results

Evaluation metrics from `evaluation_metrics.csv`:

| metric | value |
| --- | --- |
| validation_split_method | StratifiedGroupKFold first fold |
| training_systems | 60 |
| validation_systems | 15 |
| MAE | 0.043333333333333335 |
| RMSE | 0.061913918736689035 |
| exact_qscale_accuracy | 0.4 |
| within_one_step_accuracy | 0.8 |

Main validation result: MAE = 0.0433, RMSE = 0.0619, exact accuracy = 0.4000, within-one-step accuracy = 0.8000.

Validation predictions are saved in `validation_predictions.csv` with 15 rows.

Example validation predictions:

| system_id | anion_name | salt_conc | temperature | q_scale_class | predicted_q_scale | absolute_qscale_error | exact_prediction | within_one_step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| row_66 | BF4 | 1.0000 | 283.0890 | 0.9000 | 0.9000 | 0.0000 | True | True |
| row_3786 | ClO4 | 0.5000 | 342.1636 | 0.8000 | 0.9000 | 0.1000 | False | False |
| row_37 | FSI | 1.0000 | 273.0531 | 0.7000 | 0.7000 | 0.0000 | True | True |
| row_10874 | PF6 | 0.3914 | 333.0000 | 0.8500 | 0.7500 | 0.1000 | False | False |
| row_10903 | PF6 | 3.3279 | 263.0000 | 0.9000 | 0.7500 | 0.1500 | False | False |
| row_11883 | PF6 | 0.5030 | 263.1500 | 0.7000 | 0.7500 | 0.0500 | False | True |
| row_11905 | PF6 | 1.5050 | 303.1500 | 0.8000 | 0.7500 | 0.0500 | False | True |
| row_11954 | PF6 | 0.0960 | 323.1500 | 0.8500 | 0.8000 | 0.0500 | False | True |

Validation error by anion:

| anion_name | systems | mean_absolute_error | exact_accuracy | within_one_step_accuracy |
| --- | --- | --- | --- | --- |
| BF4 | 1 | 0.0000 | 1.0000 | 1.0000 |
| FSI | 1 | 0.0000 | 1.0000 | 1.0000 |
| TFSI | 5 | 0.0200 | 0.6000 | 1.0000 |
| PF6 | 7 | 0.0643 | 0.1429 | 0.7143 |
| ClO4 | 1 | 0.1000 | 0.0000 | 0.0000 |

Confusion matrix from `confusion_matrix.csv`:

| Unnamed: 0 | pred_0.70 | pred_0.75 | pred_0.80 | pred_0.85 | pred_0.90 |
| --- | --- | --- | --- | --- | --- |
| true_0.70 | 3 | 1 | 0 | 0 | 0 |
| true_0.75 | 1 | 0 | 0 | 0 | 0 |
| true_0.80 | 0 | 1 | 0 | 0 | 1 |
| true_0.85 | 0 | 1 | 2 | 1 | 0 |
| true_0.90 | 0 | 1 | 0 | 1 | 2 |

Generated diagnostic plots:

- `predicted_vs_true_qscale.png`
- `confusion_matrix.png`
- `feature_importance.png`
- `qscale_target_distribution.png`
- `min_log_error_vs_qscale.png`
- `min_log_error_by_system.png`
- `conductivity_curves_examples.png`

## 9. How to Reproduce

From the repository root, run:

```bash
python q_prediction/train_qscale_extratrees.py
python q_prediction/evaluate_qscale_model.py
python q_prediction/plot_qscale_model_diagnostics.py
python q_prediction/write_qscale_report.py
```

The default input is `results/sparse_campaign_salt_name/conductivity_results_filtered.csv`. Use `--input` on the training, evaluation, and plotting scripts to point to another compatible filtered CSV.

## 10. How to Predict q_scale for a New System

Example command:

```bash
python q_prediction/predict_qscale.py \
  -anion_name PF6 \
  -salt-conc 1.0 \
  -solvents EC DMC \
  -solvent-fracs 0.5 0.5 \
  -temperature 298.0
```

The predictor loads `qscale_extratrees_model.joblib` and `qscale_preprocessor.joblib`, normalizes solvent fractions if needed, builds the same feature representation used during training, and prints the recommended q-scale plus class probabilities when available.

## 11. Limitations and Next Steps

The current dataset is small: the generated training table contains only 75 systems. The validation split contains 15 systems, so metrics should be interpreted as a practical diagnostic rather than a definitive estimate of generalization.

The simplified model intentionally excludes experimental conductivity and all simulated outputs, so it is easier to use prospectively but may be less accurate than models that include more context. Recommended next steps are to add more electrolyte systems, especially underrepresented salt and solvent families; simulate additional q-scale values near ambiguous optima; and use this model as an initial recommendation rather than a replacement for validation simulations.

