# q_scale Optimization Report

## Objective

The goal of this step was to choose an appropriate charge-scaling factor, `q_scale`, for each electrolyte system in the sparse simulation campaign. The parameter is important because it changes the simulated ionic conductivity, and the physically useful value is the one that makes the simulation agree most closely with experiment.

This workflow does not train a model to predict conductivity from `q_scale`. Instead, it first uses the simulations that were already run at several `q_scale` values to infer an optimal `q_scale` for each electrolyte system. A simple model is then trained to predict that optimal `q_scale` from system-level metadata.

This fits into the overall workflow after database cleaning, descriptor generation, weighted-permittivity clustering, and the sparse MD campaign. The output is a first-pass `q_scale` selector that can be used to guide future simulations or to prioritize additional `q_scale` sampling.

## Data

The analysis used:

`results/sparse_campaign_weighted_permittivity/conductivity_results.csv`

After filtering for completed parsed simulations, the data contained 13 electrolyte systems, each sampled at multiple `q_scale` values. The available `q_scale` grid in the campaign was:

`0.70, 0.75, 0.80, 0.85, 0.90`

For each system, the CSV contained both simulated conductivity, `conductivity_mS_cm`, and the experimental conductivity, `conductivity`. The experimental conductivity was used only to define the optimal `q_scale` target and to evaluate reconstructed conductivity.

## Surrogate Approach

The model is a direct `q_scale_optimal` surrogate. For each electrolyte system, all simulated conductivities across the sampled `q_scale` values were compared against the experimental conductivity:

```text
Delta sigma(q) = |sigma_sim(q) - sigma_exp|
```

The optimal charge scaling factor was defined as:

```text
q_scale_optimal = argmin_q Delta sigma(q)
```

If multiple `q_scale` values had the same error, the deterministic rule was to select the lowest `q_scale` among the tied values. In the current run the tie tolerance was `0.0 mS/cm`.

The model inputs were intentionally simple, system-level variables:

- salt concentration
- temperature
- experimental conductivity
- weighted-permittivity cluster ID
- mean conductivity uncertainty from the simulations
- cation identity
- anion identity
- solvent identity
- composition summary

The selected model was a shallow random forest. It remained constrained and interpretable compared with a high-dimensional descriptor model:

- `n_estimators = 100`
- `max_depth = 3`
- `min_samples_leaf = 2`

The model was selected by grouped cross-validation against other simple models: mean baseline, linear regression, ridge, lasso, a shallow decision tree, and the shallow random forest.

## Leakage Control

Rows were grouped by electrolyte identity so that the same electrolyte system at different `q_scale` values could not appear in both training and validation. This matters because the simulation rows for one electrolyte are not independent: they are the same composition sampled at different charge scaling factors.

For final validation, the saved split held out whole cluster/composition groups. The validation model was trained only on the training systems and then evaluated on the held-out systems. This avoided fitting on the same systems used to report validation metrics.

## Reconstructed Conductivity

After predicting `q_scale_optimal` for a held-out system, the prediction was clipped to the sampled campaign range and snapped to the nearest available simulated `q_scale` for that same system. The corresponding simulated conductivity was then used as the reconstructed conductivity.

This reconstruction step answers the practical question: if the model recommends a `q_scale`, how close would the available simulation at that charge scaling be to experiment?

## Metrics

### q_scale MAE

The q-scale mean absolute error is the average absolute difference between predicted and true optimal `q_scale`.

Small values mean the model usually selects a charge scaling close to the simulation-derived optimum. Because the grid spacing is `0.05`, an MAE well below `0.05` means predictions are usually within one grid interval.

### q_scale RMSE

The q-scale root mean squared error penalizes larger `q_scale` mistakes more strongly than MAE. It is useful for identifying whether a few systems have large errors.

### Reconstructed Conductivity MAE

The reconstructed conductivity MAE is the average absolute error between experimental conductivity and the simulated conductivity recovered from the predicted `q_scale`.

This is the most physically direct metric: it measures the conductivity error remaining after using the model to select `q_scale`.

### Reconstructed Conductivity RMSE

The reconstructed conductivity RMSE is the square-root mean squared reconstructed conductivity error. It emphasizes larger conductivity mismatches and is useful for identifying problematic systems.

### R2

R2 was reported for both `q_scale` and reconstructed conductivity. In this small-data setting, R2 should be interpreted cautiously because the validation set contained only 6 systems. Positive R2 indicates improvement over predicting the validation mean, while negative R2 indicates worse-than-mean performance.

### Residual Diagnostics

Residual plots were generated for both `q_scale` and reconstructed conductivity. These show whether prediction errors are centered near zero and whether failures are dominated by a small number of systems.

### Simulated Conductivity Envelope

For each electrolyte, the analysis checked whether the experimental conductivity fell inside the range of simulated conductivities spanned by the sampled `q_scale` grid. This is a coverage diagnostic. If experiment lies outside the simulated range, no model can recover the experimental value exactly using only the existing `q_scale` grid.

## Results

### Target Construction

The optimal `q_scale` labels were derived for 13 electrolyte systems. Across all systems, experiment lay inside the sampled simulated conductivity range for 5 of 13 systems.

The inferred optimal `q_scale` values were distributed across the sampled grid:

- `0.70`: several high-conductivity systems where the lowest sampled charge scaling was closest to experiment
- `0.75`: one system
- `0.80`: several systems
- `0.85`: two systems
- `0.90`: one low-conductivity system

### Model Selection

The best grouped cross-validation result was obtained by the shallow random forest:

| Model | q_scale CV MAE | q_scale CV RMSE | CV R2 |
|---|---:|---:|---:|
| shallow random forest | 0.032565 | 0.037256 | 0.170345 |
| small tree | 0.042917 | 0.051272 | -0.431146 |
| linear regression | 0.044571 | 0.053102 | -0.717192 |
| ridge | 0.044752 | 0.053340 | -0.475120 |
| lasso | 0.048108 | 0.056392 | -0.602445 |
| mean baseline | 0.060601 | 0.069200 | -2.038344 |

The selected model improved substantially over the mean baseline and modestly over the shallow tree and linear models. The improvement justified using the shallow random forest as the first-pass selector, although the dataset is still small.

### Held-Out Validation

The validation set contained 6 held-out systems. Experiment lay inside the simulated conductivity envelope for 3 of those 6 systems.

Held-out q-scale prediction metrics:

- q_scale MAE: `0.027482`
- q_scale RMSE: `0.031814`
- q_scale R2: `0.676123`

Held-out reconstructed conductivity metrics:

- conductivity MAE: `1.08991 mS/cm`
- conductivity RMSE: `1.30958 mS/cm`
- conductivity R2: `0.960232`

Because the q-scale grid spacing is `0.05`, the q-scale MAE of about `0.0275` indicates that most predictions are within roughly one grid interval of the inferred optimum.

### Hardest Validation Systems

The largest reconstructed conductivity errors in the validation set were:

| original_row_index | cluster_id | true q_scale_optimal | predicted q_scale | nearest sampled q_scale | experimental conductivity | reconstructed conductivity | absolute error |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3786 | 2 | 0.80 | 0.780851 | 0.80 | 2.579896 | 0.8061 | 1.773796 |
| 119 | 2 | 0.70 | 0.736410 | 0.75 | 8.373333 | 6.6875 | 1.685833 |
| 192 | 1 | 0.85 | 0.797776 | 0.80 | 22.553356 | 24.2014 | 1.648044 |

Two of these failures were in cluster 2. For row 3786, the predicted `q_scale` snapped to the same sampled value as the true optimum, but the reconstructed conductivity still differed from experiment by `1.773796 mS/cm`. This indicates that the sampled simulation itself did not closely match experiment at its best available `q_scale`.

### Feature Effects

For the selected shallow random forest, the largest feature importances were:

| Feature | Importance |
|---|---:|
| anion_name = TFSI | 0.665787 |
| anion_name = PF6 | 0.128445 |
| experimental conductivity | 0.061722 |
| mean conductivity uncertainty | 0.036668 |
| temperature | 0.034084 |
| cluster_id | 0.028402 |
| salt concentration | 0.021259 |

This suggests that the model mostly separates q-scale behavior by ion family, especially TFSI versus PF6, with smaller contributions from experimental conductivity, temperature, cluster ID, and concentration.

## Interpretation

The results support the idea that a simple direct `q_scale_optimal` model is useful as a first-pass selector. The model predicts the inferred optimum within less than one `q_scale` grid interval on average, and the reconstructed conductivities are reasonably close to experiment for the held-out systems.

Physically, the strongest model signal comes from anion identity. This is plausible because different anions can produce different conductivity responses to charge scaling. Cluster ID and concentration contribute less strongly in the current model, but they still provide chemistry-family context.

The validation results also show a key limitation: even when the predicted `q_scale` is close to the inferred optimum, the best available simulated conductivity may remain far from experiment. This happens when the experimental value is outside, or near the edge of, the conductivity range spanned by the sampled `q_scale` grid.

The coverage diagnostic is therefore important. Across all 13 systems, only 5 had experimental conductivity inside the simulated conductivity envelope. In the held-out validation set, 3 of 6 were inside the envelope. For systems outside the envelope, more q-scale points or additional simulation conditions may be needed before any model can reconstruct experiment accurately.

## Limitations

The main limitation is data size. The model was trained on only 13 electrolyte systems, and the validation set contained 6 systems. Metrics such as R2 are therefore sensitive to which systems are held out.

The q-scale target is also discrete because it is chosen only from the sampled grid. If the true optimum lies between sampled values, the current label can only approximate it.

Some experimental conductivities lie outside the simulated conductivity range. For these systems, the inferred optimum is a boundary value rather than a well-resolved optimum. Boundary optima are less informative because they indicate that the q-scale grid may not fully cover the relevant behavior.

The selected shallow random forest is still simple compared with a high-dimensional black-box model, but it is less directly interpretable than a linear model or a shallow decision tree. Its selection was justified by better grouped cross-validation performance, but this should be revisited as more systems are added.

## Next Steps

1. Add more electrolyte systems to improve model stability, especially in clusters with few representatives.

2. Add additional `q_scale` points near boundary optima, especially when the best available value is `0.70` or `0.90`.

3. For systems where experiment lies outside the simulated conductivity envelope, expand the q-scale range or inspect whether other simulation parameters are limiting agreement.

4. Consider fitting a local interpolation curve for each system once more q-scale points are available, so `q_scale_optimal` is not restricted to the sampled grid.

5. Re-evaluate whether a shallow tree or linear model becomes sufficient after adding more data. A simpler model is preferable if it approaches the shallow random forest performance.

6. Use the current model as a first-pass q-scale selector, not as a final calibration model. The diagnostic plots should be checked before treating any predicted q-scale as definitive.

## Generated Artifacts

Training artifacts:

- `qscale_optimal_targets.csv`
- `model_cv_results.csv`
- `qscale_model.joblib`
- `qscale_model_train_split.joblib`
- `preprocessing.joblib`
- `feature_effects.csv`
- `system_split.csv`

Validation artifacts:

- `validation/validation_predictions.csv`
- `validation/validation_metrics.csv`
- `validation/predicted_vs_true_qscale.png`
- `validation/experimental_vs_reconstructed_conductivity.png`
- `validation/selected_system_conductivity_curves.png`
- `validation/residual_error_distributions.png`
- `validation/qscale_sampling_coverage_diagnostic.png`
