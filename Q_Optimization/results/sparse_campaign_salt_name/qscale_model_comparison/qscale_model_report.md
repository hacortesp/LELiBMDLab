# q_scale model-comparison reproducibility and results report

Generated from the actual outputs in `results/sparse_campaign_salt_name/qscale_model_comparison/`.

## 1. Objective

The objective of this workflow is to build a system-level model that predicts the optimal charge-scaling factor, `q_scale`, for each electrolyte system before running a new q-scale sweep. The target is defined from the simulated and experimental ionic conductivities as:

```text
q_scale_optimal = argmin_q |log(sigma_calc(q)) - log(sigma_exp)|
```

Here, `sigma_calc(q)` is the simulated conductivity at a particular `q_scale`, and `sigma_exp` is the experimental conductivity for the same electrolyte system. Logarithmic conductivity error was used because conductivities span orders of magnitude; on the log scale, relative mismatches are weighted more evenly than absolute mS/cm differences, so a low-conductivity system is not overwhelmed by high-conductivity systems.

## 2. Input data

The input file was:

`/home/hacortes/Li-EMDLab/Q_Optimization/results/sparse_campaign_salt_name/conductivity_results_filtered.csv`

The actual input columns were: `original_row_index`, `doi`, `q_scale`, `temperature`, `anion_name`, `salt_conc`, `solvents`, `solvent_fracs`, `conductivity`, `conductivity_mS_cm`, `conductivity_uncertainty_mS_cm`.

The analysis used these column mappings from `training_metadata.json`:

| role | column |
| --- | --- |
| calculated_conductivity | conductivity_mS_cm |
| calculated_uncertainty | conductivity_uncertainty_mS_cm |
| concentration | salt_conc |
| doi | doi |
| experimental_conductivity | conductivity |
| original_row_index | original_row_index |
| q_scale | q_scale |
| salt | anion_name |
| solvent_fracs | solvent_fracs |
| solvents | solvents |
| temperature | temperature |

Electrolyte systems were identified using `original_row_index` because it uniquely identified the chemistry and condition fields in this filtered file. The chemistry and condition fields checked for system identity were `doi`, `anion_name`, `salt_conc`, `temperature`, `solvents`, and `solvent_fracs`. The final `system_id` values therefore have the form `row_<original_row_index>`.

The model features were built only from information available before choosing a `q_scale`: salt identity, solvent identities, solvent fractions, concentration, temperature, experimental conductivity, and the log of experimental conductivity. The columns `q_scale`, `conductivity_mS_cm`, and `conductivity_uncertainty_mS_cm` were not used as model input features. `q_scale` was used only to define the grid over which the target was selected. `conductivity_mS_cm` was used only for target construction. `conductivity_uncertainty_mS_cm` was excluded from the predictors because it is produced by the simulation workflow, not known before simulation.

## 3. Target construction

Target construction was performed by `train_compare_qscale_models.py` and saved in `optimal_qscale_targets.csv`. The steps were:

1. Rows were grouped into unique electrolyte systems using `system_id`. In this run, `system_id` was derived from `original_row_index`.
2. For each system, all available `q_scale` simulations were collected. The actual grid was `0.70, 0.75, 0.80, 0.85, 0.90`.
3. Rows were valid for target construction only if `q_scale`, `conductivity`, and `conductivity_mS_cm` were finite numeric values and both conductivities were strictly positive.
4. For each valid q-scale row, the error `|log(conductivity_mS_cm) - log(conductivity)|` was calculated.
5. `q_scale_optimal` was assigned to the q-scale value with the smallest log-conductivity error.
6. If ties occurred within numerical tolerance, the smallest tied q-scale value was selected and `tie_flag` was recorded. In this run, the target table contains `0` tied systems.
7. Systems with incomplete q-scale coverage were retained if at least one valid positive simulated conductivity and one valid positive experimental conductivity were available. Partial coverage was recorded in `complete_qscale_coverage_flag`.
8. `coverage_flag` checked whether the experimental conductivity lay between the minimum and maximum valid simulated conductivity across the q-scale values for that system.

Target diagnostics from `qscale_target_diagnostics.csv`:

| quantity | value |
| --- | --- |
| Total raw simulation rows | 375 |
| Valid rows for target construction | 373 |
| Unique electrolyte systems | 75 |
| Systems with complete q_scale coverage | 73 |
| Systems with partial q_scale coverage | 2 |
| Systems excluded because all rows were invalid | 0 |
| Fraction with experiment inside simulated conductivity envelope | 0.72 |

The distribution of optimal targets was:

| q_scale_optimal | systems |
| --- | --- |
| 0.70 | 14 |
| 0.75 | 15 |
| 0.80 | 18 |
| 0.85 | 16 |
| 0.90 | 12 |

The target and diagnostic files to inspect first are `optimal_qscale_targets.csv` and `qscale_target_diagnostics.csv`.

## 4. Features used for q_scale prediction

The final system-level training table was `system_level_training_table.csv` with 75 systems and 40 columns.

Numeric features from `training_metadata.json`: `concentration_numeric`, `temperature_numeric`, `experimental_conductivity_numeric`, `log_experimental_conductivity`, `solvent_count`, `has_solvent_dec`, `frac_solvent_dec`, `has_solvent_dmc`, `frac_solvent_dmc`, `has_solvent_dme`, `frac_solvent_dme`, `has_solvent_ec`, `frac_solvent_ec`, `has_solvent_emc`, `frac_solvent_emc`, `has_solvent_pc`, `frac_solvent_pc`.

Categorical features from `training_metadata.json`: `salt_identity`, `solvent_set`.

Salt identity was represented as `salt_identity`, derived from `anion_name`, and one-hot encoded inside the preprocessing pipeline. Solvent identities were parsed safely from list-like strings such as `["EC","DMC"]`; the script created binary solvent-presence features, solvent-fraction features, `solvent_count`, and a categorical `solvent_set`. Concentration and temperature were converted to numeric columns. Experimental conductivity was included both as `experimental_conductivity_numeric` and as `log_experimental_conductivity`.

All pipelines used numeric median imputation and categorical most-frequent imputation. Models that benefit from scaled numeric inputs, namely Ridge Regression, Elastic Net Regression, and Multinomial Logistic Regression, also used `StandardScaler` for numeric features. Categorical features were one-hot encoded with unknown-category handling enabled. Preprocessing objects were saved in each method directory as `preprocessing_pipeline.joblib`.

## 5. Models compared

The workflow trained the following models. These are the actual methods present in `model_comparison_summary.csv` and the per-method output folders.

| method_name | model_type | target_representation | snapped_to_grid | preprocessing | hyperparameters |
| --- | --- | --- | --- | --- | --- |
| Ridge Regression | regression | continuous q_scale | yes | median numeric imputation, scaling, categorical imputation, one-hot encoding | Ridge(alpha=1.0, random_state=42) |
| Elastic Net Regression | regression | continuous q_scale | yes | median numeric imputation, scaling, categorical imputation, one-hot encoding | ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000, random_state=42) |
| Random Forest Regressor | regression | continuous q_scale | yes | median numeric imputation, categorical imputation, one-hot encoding | RandomForestRegressor(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=42) |
| Extra Trees Regressor | regression | continuous q_scale | yes | median numeric imputation, categorical imputation, one-hot encoding | ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=42) |
| Gradient Boosting Regressor | regression | continuous q_scale | yes | median numeric imputation, categorical imputation, one-hot encoding | GradientBoostingRegressor(random_state=42) |
| Multinomial Logistic Regression | classification | discrete q_scale class | not needed | median numeric imputation, scaling, categorical imputation, one-hot encoding | LogisticRegression(max_iter=2000, solver=lbfgs, multi_class=multinomial, class_weight=balanced, random_state=42) |
| Random Forest Classifier | classification | discrete q_scale class | not needed | median numeric imputation, categorical imputation, one-hot encoding | RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight=balanced, n_jobs=-1, random_state=42) |
| Extra Trees Classifier | classification | discrete q_scale class | not needed | median numeric imputation, categorical imputation, one-hot encoding | ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, class_weight=balanced, n_jobs=-1, random_state=42) |
| Gradient Boosting Classifier | classification | discrete q_scale class | not needed | median numeric imputation, categorical imputation, one-hot encoding | GradientBoostingClassifier(random_state=42) |

Regression models predicted a continuous q-scale value and then snapped predictions to the nearest available grid value for final comparison. Classification models predicted one of the discrete q-scale classes directly.

## 6. Cross-validation and leakage prevention

The workflow used `StratifiedGroupKFold` with 5 folds, as recorded in `training_metadata.json`. The grouping key was `system_id`, so each electrolyte system appeared in exactly one validation fold. Because the model table has one row per system after target construction, this grouping mainly protects against duplicate or repeated system-level entries. It also reflects the more important scientific rule from the raw simulation table: all q-scale simulations for the same electrolyte must be grouped before modeling, because a naive row-wise split would allow different q-scale simulations of the same electrolyte to appear in both training and validation. That would leak the electrolyte identity and would overstate predictive performance.

The requested fold-membership table was saved as `cross_validation_systems_by_fold.csv`. It contains every system for every fold, with `is_validation_set_for_fold` marking whether that system is in that fold validation set. Validation predictions for the best model identify the validation fold for each system in `best_model_validation_predictions.csv`.

Training/validation counts and validation class counts by fold:

| fold | training_systems | validation_systems | validation_qscale_classes_present | validation_class_counts |
| --- | --- | --- | --- | --- |
| 1 | 60 | 15 | 5 | 0.70: 2, 0.75: 4, 0.80: 4, 0.85: 2, 0.90: 3 |
| 2 | 60 | 15 | 5 | 0.70: 2, 0.75: 3, 0.80: 4, 0.85: 4, 0.90: 2 |
| 3 | 60 | 15 | 5 | 0.70: 3, 0.75: 4, 0.80: 2, 0.85: 5, 0.90: 1 |
| 4 | 60 | 15 | 5 | 0.70: 2, 0.75: 2, 0.80: 4, 0.85: 2, 0.90: 5 |
| 5 | 60 | 15 | 5 | 0.70: 5, 0.75: 2, 0.80: 4, 0.85: 3, 0.90: 1 |

Each validation fold contained at least one system from every q-scale class.

## 7. Evaluation metrics

The reported metrics are validation-fold out-of-fold metrics, not training-set metrics. MAE is the mean absolute q-scale error in q-scale units. RMSE is the root mean squared q-scale error and penalizes larger mistakes more strongly. Median absolute error reports the median system-level absolute error. Exact q-scale accuracy is the fraction of systems for which the predicted q-scale exactly equals the target. Within-one-grid-step accuracy counts predictions no farther than one q-scale grid interval from the target; the grid spacing in this run was 0.05. Macro F1 and balanced accuracy were reported for classifiers to account for class-wise performance rather than only overall accuracy.

Exact accuracy is strict on a five-point discrete grid: predicting 0.80 for a 0.85 target is scientifically less severe than predicting 0.70 for a 0.90 target, but exact accuracy treats both as wrong. Therefore, within-one-step accuracy, MAE, and RMSE are important complementary metrics. For regression models, raw continuous predictions were evaluated internally and snapped predictions were used for the final ranking because the actionable output is one of the available q-scale values.

## 8. Results

The ranked model-performance table below is from `model_comparison_summary.csv`, sorted by lowest MAE, then lowest RMSE, then highest exact q-scale accuracy.

| method_name | model_type | number_of_systems | number_of_folds | MAE | RMSE | median_absolute_error | exact_qscale_accuracy | within_one_step_accuracy | macro_F1 | balanced_accuracy | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Extra Trees Classifier | classification | 75 | 5 | 0.0387 | 0.0577 | 0.0500 | 0.4533 | 0.8267 | 0.4554 | 0.4613 | classification predicts discrete q_scale classes; no snapping needed |
| Random Forest Classifier | classification | 75 | 5 | 0.0393 | 0.0614 | 0.0500 | 0.4800 | 0.8267 | 0.4754 | 0.4780 | classification predicts discrete q_scale classes; no snapping needed |
| Gradient Boosting Classifier | classification | 75 | 5 | 0.0400 | 0.0606 | 0.0500 | 0.4667 | 0.8000 | 0.4468 | 0.4500 | classification predicts discrete q_scale classes; no snapping needed |
| Extra Trees Regressor | regression | 75 | 5 | 0.0407 | 0.0569 | 0.0500 | 0.3867 | 0.8400 |  |  | continuous predictions snapped to nearest available q_scale for final comparison |
| Random Forest Regressor | regression | 75 | 5 | 0.0413 | 0.0554 | 0.0500 | 0.3467 | 0.8533 |  |  | continuous predictions snapped to nearest available q_scale for final comparison |
| Gradient Boosting Regressor | regression | 75 | 5 | 0.0427 | 0.0572 | 0.0500 | 0.3333 | 0.8533 |  |  | continuous predictions snapped to nearest available q_scale for final comparison |
| Multinomial Logistic Regression | classification | 75 | 5 | 0.0440 | 0.0658 | 0.0500 | 0.4400 | 0.7867 | 0.4393 | 0.4365 | classification predicts discrete q_scale classes; no snapping needed |
| Ridge Regression | regression | 75 | 5 | 0.0447 | 0.0597 | 0.0500 | 0.3333 | 0.8133 |  |  | continuous predictions snapped to nearest available q_scale for final comparison |
| Elastic Net Regression | regression | 75 | 5 | 0.0527 | 0.0651 | 0.0500 | 0.2400 | 0.7333 |  |  | continuous predictions snapped to nearest available q_scale for final comparison |

The best-performing model by the specified ranking was **Extra Trees Classifier**. It was selected because it had the lowest cross-validated MAE (0.0387) among the compared methods. Its RMSE was 0.0577, exact accuracy was 0.4533, and within-one-step accuracy was 0.8267. The margin over the Random Forest Classifier was small in MAE (0.0387 versus 0.0393), and Random Forest Classifier had slightly higher exact accuracy (0.4800 versus 0.4533). Therefore, the ranking identifies Extra Trees Classifier as the best according to the chosen primary MAE criterion, but the difference versus the next classifiers should be treated as modest rather than decisive.

Generated global plots:

- `model_comparison_mae_rmse.png`: MAE and RMSE comparison by method.
- `model_comparison_accuracy.png`: exact and within-one-step accuracy comparison by method.
- `model_comparison_global_predicted_vs_true.png`: out-of-fold predicted versus true q-scale for all methods.
- `true_optimal_qscale_distribution.png`: target-class distribution.
- `qscale_error_by_salt.png`: q-scale prediction error by salt identity.

## 9. Validation systems and predictions

The validation systems for all folds are listed in `cross_validation_systems_by_fold.csv`. The best model validation predictions were saved as `best_model_validation_predictions.csv`. That file includes the fold number, system metadata, experimental conductivity, true target, predicted q-scale, absolute error, exact-match flag, and within-one-grid-step flag.

Best-model prediction summary by salt identity:

| salt_identity | systems | mean_absolute_error | exact_accuracy | within_one_step_accuracy |
| --- | --- | --- | --- | --- |
| BF4 | 6 | 0.0167 | 0.6667 | 1.0000 |
| ClO4 | 9 | 0.0222 | 0.6667 | 0.8889 |
| FSI | 5 | 0.0300 | 0.4000 | 1.0000 |
| TFSI | 19 | 0.0395 | 0.4737 | 0.8421 |
| PF6 | 36 | 0.0472 | 0.3611 | 0.7500 |

Best-predicted systems, sorted by absolute q-scale error:

| fold_number | system_id | doi | salt_identity | concentration | temperature | solvent_names | solvent_fractions | true_q_scale_optimal | predicted_q_scale | absolute_q_scale_error | exact_prediction | within_one_grid_step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | row_10893 | 10.1149/1.1872737 | PF6 | 0.9555 | 293.0000 | ["EC","PC","DMC"] | [0.27,0.1,0.63] | 0.7500 | 0.7500 | 0.0000 | True | True |
| 3 | row_10895 | 10.1149/1.1872737 | PF6 | 2.2375 | 293.0000 | ["EC","PC","DMC"] | [0.27,0.1,0.63] | 0.7500 | 0.7500 | 0.0000 | True | True |
| 1 | row_10899 | 10.1149/1.1872737 | PF6 | 0.3914 | 263.0000 | ["EC","PC","DMC"] | [0.27,0.1,0.63] | 0.7500 | 0.7500 | 0.0000 | True | True |
| 2 | row_10901 | 10.1149/1.1872737 | PF6 | 1.7382 | 263.0000 | ["EC","PC","DMC"] | [0.27,0.1,0.63] | 0.7500 | 0.7500 | 0.0000 | True | True |
| 3 | row_11886 | 10.1149/2.0571912jes | PF6 | 2.0000 | 263.1500 | ["EC","DMC"] | [0.5,0.5] | 0.7500 | 0.7500 | 0.0000 | True | True |
| 1 | row_11887 | 10.1149/2.0571912jes | PF6 | 3.0000 | 263.1500 | ["EC","DMC"] | [0.5,0.5] | 0.7500 | 0.7500 | 0.0000 | True | True |
| 2 | row_11905 | 10.1149/2.0571912jes | PF6 | 1.5050 | 303.1500 | ["EC","DMC"] | [0.5,0.5] | 0.8000 | 0.8000 | 0.0000 | True | True |
| 1 | row_11923 | 10.1149/2.0571912jes | PF6 | 3.0020 | 263.1500 | ["EC","EMC"] | [0.3,0.7] | 0.8000 | 0.8000 | 0.0000 | True | True |
| 2 | row_11928 | 10.1149/2.0571912jes | PF6 | 1.9990 | 273.1500 | ["EC","EMC"] | [0.3,0.7] | 0.8000 | 0.8000 | 0.0000 | True | True |
| 1 | row_127 | 10.1016/j.jpowsour.2011.07.071 | TFSI | 1.0000 | 298.0000 | ["EC","DMC"] | [0.8,0.2] | 0.7000 | 0.7000 | 0.0000 | True | True |

Worst-predicted systems, sorted by absolute q-scale error:

| fold_number | system_id | doi | salt_identity | concentration | temperature | solvent_names | solvent_fractions | true_q_scale_optimal | predicted_q_scale | absolute_q_scale_error | exact_prediction | within_one_grid_step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | row_10875 | 10.1149/1.1872737 | PF6 | 0.9609 | 333.0000 | ["EC","PC","DMC"] | [0.27,0.1,0.63] | 0.7000 | 0.8500 | 0.1500 | False | False |
| 4 | row_10903 | 10.1149/1.1872737 | PF6 | 3.3279 | 263.0000 | ["EC","PC","DMC"] | [0.27,0.1,0.63] | 0.9000 | 0.7500 | 0.1500 | False | False |
| 5 | row_3614 | 10.1149/1.1344281 | TFSI | 1.0000 | 271.8870 | ["PC"] | [1.0] | 0.7000 | 0.8500 | 0.1500 | False | False |
| 2 | row_3625 | 10.1149/1.1344281 | TFSI | 1.0000 | 217.6929 | ["PC"] | [1.0] | 0.8500 | 0.7000 | 0.1500 | False | False |
| 3 | row_11894 | 10.1149/2.0571912jes | PF6 | 1.0040 | 283.1500 | ["EC","DMC"] | [0.5,0.5] | 0.8000 | 0.7000 | 0.1000 | False | False |
| 5 | row_247 | 10.1016/j.jpowsour.2011.07.071 | TFSI | 1.0000 | 334.9026 | ["EC","DMC"] | [0.5,0.5] | 0.8000 | 0.7000 | 0.1000 | False | False |
| 1 | row_10874 | 10.1149/1.1872737 | PF6 | 0.3914 | 333.0000 | ["EC","PC","DMC"] | [0.27,0.1,0.63] | 0.8500 | 0.7500 | 0.1000 | False | False |
| 2 | row_12013 | 10.1149/2.0641503jes | PF6 | 1.2014 | 313.1500 | ["EC","DEC"] | [0.5,0.5] | 0.8000 | 0.9000 | 0.1000 | False | False |
| 4 | row_12014 | 10.1149/2.0641503jes | PF6 | 1.5000 | 283.1500 | ["EC","DEC"] | [0.5,0.5] | 0.9000 | 0.8000 | 0.1000 | False | False |
| 5 | row_13615 | 10.1016/0378-7753(91)80004-H | PF6 | 1.0000 | 213.1500 | ["EC","PC"] | [0.15,0.85] | 0.9000 | 0.8000 | 0.1000 | False | False |

The largest observed best-model errors were 0.15 q-scale units, corresponding to three grid steps. Inspection of the worst-predicted rows shows errors across multiple salt identities and solvent systems rather than one single failure mode. Mean absolute error by salt was lowest for BF4 and ClO4 and highest for PF6 in this run, but the salt-family counts are small enough that these patterns should be considered diagnostic rather than conclusive. Additional inspection should use `best_model_validation_predictions.csv` together with `qscale_error_by_salt.png`.

## 10. How to reproduce the workflow

Required input file:

`/home/hacortes/Li-EMDLab/Q_Optimization/results/sparse_campaign_salt_name/conductivity_results_filtered.csv`

Main script:

`/home/hacortes/Li-EMDLab/Q_Optimization/train_compare_qscale_models.py`

The run used Python with pandas, NumPy, scikit-learn, joblib, and matplotlib. The local environment used for the recorded outputs reported pandas 2.2.3, NumPy 2.4.3, and scikit-learn 1.6.1 during workflow development. The script avoids `from __future__ import annotations` and uses compatibility fallbacks for older sklearn one-hot encoder and grouped-CV APIs.

Reproduce the model-comparison workflow from the repository root with:

```bash
python train_compare_qscale_models.py \
  --input /home/hacortes/Li-EMDLab/Q_Optimization/results/sparse_campaign_salt_name/conductivity_results_filtered.csv \
  --output-dir /home/hacortes/Li-EMDLab/Q_Optimization/results/sparse_campaign_salt_name/qscale_model_comparison \
  --folds 5 \
  --random-seed 42 \
  --overwrite true
```

Use `--folds` to request a different number of folds, `--random-seed` to change the reproducible seed, and `--overwrite false` to prevent replacing an existing output directory. Note that the supported option is `--folds`, not `--n-folds`.

Inspect these files first after a run:

1. `qscale_target_diagnostics.csv` for target construction health checks.
2. `optimal_qscale_targets.csv` for per-system targets and coverage flags.
3. `model_comparison_summary.csv` for ranked model metrics.
4. `<method>/out_of_fold_predictions.csv` for per-method validation predictions.
5. `best_model_validation_predictions.csv` for the selected model validation details.

This markdown report was generated after the modeling run from the saved outputs; rerunning `train_compare_qscale_models.py` overwrites the output directory by default, so regenerate this report afterward if the model outputs change.

## 11. Limitations

The validation set contains only 75 unique electrolyte systems. Although the target classes are not severely collapsed into one value, each class contains only 12 to 18 systems, which limits the reliability of five-fold class-wise validation. Two systems had partial q-scale coverage, and 28% of systems had experimental conductivity outside the simulated conductivity envelope because the envelope fraction was 0.72. Those systems can still be assigned a closest q-scale, but the target is less satisfying scientifically because no simulated q-scale brackets the experiment.

The chemistry space is sparse. Salt counts in the training table were: BF4: 6; ClO4: 9; FSI: 5; PF6: 36; TFSI: 19. Solvent mixtures were also unevenly distributed, with the most common solvent sets being: DMC+EC: 16; EC+EMC: 16; PC: 11; EC+PC: 10; DMC+EC+PC: 9; DME: 8; DEC+EC: 3; DMC: 1. These imbalances mean that performance by salt or solvent family may reflect sampling density as much as model capability. Conductivity uncertainty estimates were saved in the original filtered data but were not used as model inputs, and the target construction did not weight systems by MD uncertainty. Finally, the best exact accuracy was below 50%, so the current model should not be interpreted as a replacement for q-scale validation simulations.

The saved warning file says: `No major warnings.`.

## 12. Recommended next steps

1. Use the Extra Trees Classifier as an initial q-scale recommendation model, not as a final substitute for validation simulations.
2. Simulate additional q-scale values near ambiguous optima, especially systems where neighboring q-scales have similar log-conductivity errors.
3. Add more systems for underrepresented salt and solvent families so fold-level class and chemistry coverage are more robust.
4. Prioritize new simulations for systems where `coverage_flag` is false, because the current q-scale grid did not bracket the experimental conductivity for those systems.
5. Consider uncertainty-aware target construction or reporting, using `conductivity_uncertainty_mS_cm` as target uncertainty metadata rather than as a predictor.
6. Re-run the workflow after expanding the dataset and compare whether the model ranking remains stable, especially between Extra Trees Classifier and Random Forest Classifier.
