# Simplified q_scale Prediction Workflow

This folder contains a compact workflow for recommending `q_scale` using the best model from the earlier comparison: an Extra Trees Classifier.

The target is built per electrolyte system:

```text
q_scale_optimal = argmin_q |log(conductivity_mS_cm) - log(conductivity)|
```

The model uses only pre-simulation inputs:

- `anion_name`
- `salt_conc`
- `temperature`
- solvent presence features
- solvent fraction features

It does not use `q_scale`, simulated conductivity, or simulation uncertainty as input features.

## 1. Train

From the repository root:

```bash
python q_prediction/train_qscale_extratrees.py
```

Main outputs:

- `qscale_training_table.csv`
- `qscale_extratrees_model.joblib`
- `qscale_preprocessor.joblib`
- `feature_importance.csv`
- `training_summary.txt`

## 2. Predict One New Electrolyte

```bash
python q_prediction/predict_qscale.py \
  -anion_name PF6 \
  -salt-conc 1.0 \
  -solvents EC DMC \
  -solvent-fracs 0.5 0.5 \
  -temperature 298.0
```

The script prints the recommended `q_scale` and class probabilities.

## 3. Evaluate

```bash
python q_prediction/evaluate_qscale_model.py
```

Main outputs:

- `evaluation_metrics.csv`
- `validation_predictions.csv`
- `confusion_matrix.csv`

The evaluation uses a leakage-safe system-level validation split.

## 4. Plot Diagnostics

```bash
python q_prediction/plot_qscale_model_diagnostics.py
```

Main plots:

- `predicted_vs_true_qscale.png`
- `confusion_matrix.png`
- `feature_importance.png`
- `qscale_target_distribution.png`
- `min_log_error_vs_qscale.png`
- `min_log_error_by_system.png`
- `min_log_error_by_system.csv`
- `conductivity_curves_examples.png`
