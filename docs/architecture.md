# Software architecture

## Modules

| Module | Content |
|---|---|
| `windows` | trailing moving windows (`rolling`, `rolling_pair`), window checks |
| `metrics_single` | standard deviation, turbulence intensity, normalized range, spectral entropy; `single_series_metrics` |
| `metrics_pair` | moving correlation, MAPE, nRMSE, nMAE; `agreement_metrics` |
| `metrics_ensemble` | spread indices, total spread index, predictability index, ensemble CRPS (plain and fair), observation rank, rank histogram and its summary; `ensemble_metrics` |
| `metrics_spatial` | distances (planar, great circle), Matheron semivariogram, exponential/Gaussian/spherical models, weighted least-squares fit, `variogram_over_time` |
| `probabilistic` | location-scale quantiles and CDF (Gaussian, Laplace), pinball loss, closed-form CRPS, reliability, central-interval coverage and width, default levels |
| `calibration` | Step A: exact pinball-loss-optimal scale per step, per group or per centred window; CRPS-optimal scale; isotonic recalibration |
| `features` | `PlantSeries`, `FeatureSpec`, `SpatialSpec`; Step B: feature tables, targets, training and prediction tables |
| `surrogate` | Step C: `Surrogate` (GP regression, SVR, random forest, ridge, constant baseline), GP kernels and kernel selection |
| `validation` | Step D: plant and time splits, cross-validation, scale and predictive scores, cross-fitted recalibration, oracle reference |
| `workflow` | `WorkflowConfig`, `run_workflow` (Steps A–D and prediction at unobserved plants), `save_result` |
| `synthetic` | invented fleet of plants with ground truth |
| `io` | input-directory reader and writer, observed/unobserved split |
| `cli` | `wind-uq-surrogate synthetic | metrics | run` |

Dependencies: NumPy, SciPy, pandas, scikit-learn. No solver or GPU is needed.

## Data flow

```
plants.csv + series/ (+ turbines/)            read_plants ─► [PlantSeries]
        │                                                       │ split_observed
        ▼                                                       ▼
observed plants ── calibration_targets (Step A) ──┐     unobserved plants
        │                                          │            │
        └── build_features (Step B) ───────────────┴► training_table      prediction_table ◄─ build_features
                                                        │                        │
                         plant_splits ► cross_validate (Step C/D, every surrogate)│
                                                        │                        │
                     score_scale, score_predictive, recalibrated_scores, oracle_scores
                                                        │                        │
                               chosen surrogate fitted on all observed rows ─────┴► scale ► quantiles (MW)
                                                        │
                                                   save_result ► CSV + summary.json
```

## Main types

- `PlantSeries(plant_id, data, capacity, x, y, turbine_coords, turbine_field)`: `data` is a DataFrame indexed by time with `forecast`, optional `actual` and covariates.
- `FeatureSpec(window, single_series, pairs, member_prefix, si_pairs, pi_history, spatial, include_level, ...)`: which metrics become features.
- `WorkflowConfig(features, levels, dist, block, target_window, models, gpr_max_train, n_splits, recalibrate, coverages, final_model, random_state)`.
- `WorkflowResult`: every table below plus the chosen and fitted surrogate.

## Output tables (`save_result`)

| File | One row per | Columns |
|---|---|---|
| `training_table.csv` | observed plant and time step | `plant_id`, `time`, features, `forecast_pu`, `actual_pu`, `target_scale`, `target_loss` |
| `cv_predictions.csv` | held-out row and surrogate | `model`, `fold`, `plant_id`, `time`, `forecast_pu`, `actual_pu`, `target_scale`, `pred_scale`, `pred_log_std` (GP only) |
| `scale_scores.csv` | surrogate | `n`, `scale_mae`, `scale_rmse`, `log_scale_rmse`, `log_scale_r2` |
| `predictive_scores.csv` | surrogate | `n`, `pinball`, `crps`, `reliability_mad`, `coverage_<c>`, `width_<c>` |
| `reliability.csv` | surrogate and level | `level`, `observed`, `deviation` |
| `recalibrated_scores.csv`, `recalibrated_reliability.csv` | as above, after cross-fitted isotonic recalibration | no CRPS (no closed form after recalibration) |
| `oracle_scores.csv` | one row | scores of the targets themselves (in sample) |
| `unobserved_predictions.csv` | unobserved plant and time step | `plant_id`, `time`, `forecast_pu`, `pred_scale_pu`, `q<level>_mw` |
| `summary.json` | — | chosen surrogate, plants, predictive scores, configuration, fit warnings |

All scores are per unit of installed capacity.

## Size and run time

The cost is dominated by the surrogates. Exact GP regression scales as $`O(n^3)`$ in the number of training rows, so the GP surrogate is fitted on at most `gpr_max_train` random rows (600 by default) per fold; the other surrogates use all rows. Leave-one-plant-out validation fits every surrogate once per observed plant plus once for the final model.

Measured on a laptop (Apple silicon, one process), Python 3.12:

| Run | Rows | Time |
|---|---|---|
| `examples/minimal.py`: 6 observed plants × 42 days hourly, 12 features, 5 surrogates, leave-one-plant-out | 5,760 | about 26 s |
| `examples/validation_design.py`: 2 fleets × 2 target types × 4 surrogates | 4 × 5,544 | about 1 min |
| `examples/metrics_tour.py`, `examples/calibration_targets.py` | — | a few seconds |

Step A is exact and fast: single-step targets are vectorized; window and block targets use a bisection over the breakpoints of a convex piecewise-linear function.

## Tests

`pytest` runs the suite in about ten seconds. It covers:

- the window convention and the absence of look-ahead;
- every metric against direct formulas or hand-computed cases (spectral entropy limits, spread indices, predictability index, ensemble CRPS against numerical integration, flat rank histograms for exchangeable members, Matheron estimator, variogram models and fits);
- closed-form CRPS against numerical integration; pinball loss; reliability and interval scores;
- pinball-loss-optimal scales against brute-force grids (per step, per block, per window), their dense-level limit, bounds and edge cases; isotonic recalibration;
- feature construction (observed power never used, no look-ahead, spatial features), targets and tables;
- surrogate fitting and prediction; plant and time splits; a surrogate beating the constant baseline when the synthetic error scale follows the regime, and no material gain in the negative control;
- the end-to-end workflow (non-crossing quantiles, oracle lower bound for single-step targets), input-directory round trip and checks, and the command line.
