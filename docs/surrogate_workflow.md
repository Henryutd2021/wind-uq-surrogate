# Surrogate workflow: from UQ metrics to plant-level power uncertainty

**Problem.** Many wind plants have a point forecast or a modelled power series but no measured power, so their power uncertainty cannot be estimated from their own errors. At plants with measurements it can. The workflow learns, at observed plants, a mapping from UQ metrics of inputs that exist everywhere (meteorological data products, ensembles, the forecast itself, turbine-level fields) to a calibrated measure of power uncertainty, and applies the mapping at unobserved plants.

**Research question.** Can UQ metrics of the inputs of a wind power model serve as the inputs of a surrogate model that predicts calibrated plant-level power uncertainty at plants without power measurements?

The workflow extends the two-step method of Sun et al. (2019), in which a point forecast is the location of a predictive distribution and a surrogate (support vector regression) maps the forecast to a scale chosen by pinball-loss optimization. Here the surrogate's inputs are UQ metrics of several uncertainty sources, several surrogates are compared, and the validation holds out whole plants.

```
observed plants                                   unobserved plants
 forecast, actual,  ──A──► target scale s_t        forecast, covariates
 covariates        ──B──► UQ-metric features x_t   ──B──► features x_t
                           │                                   │
                           C  surrogate  x ↦ ŝ  ───────────────┤
                           │                                   ▼
                           D  validation on held-out plants    ŝ_t ─► predictive quantiles
                              (+ isotonic recalibration)
```

| Step | Function | Module |
|---|---|---|
| A. Calibrated targets at observed plants | `calibration_targets`, `optimal_scale`, `rolling_optimal_scale` | `calibration`, `features` |
| B. Features | `FeatureSpec`, `build_features`, `training_table`, `prediction_table` | `features` (metrics: [metrics.md](metrics.md)) |
| C. Surrogates | `Surrogate`, `select_gpr_kernel` | `surrogate` |
| D. Validation and prediction | `plant_splits`, `cross_validate`, `score_scale`, `score_predictive`, `recalibrated_scores` | `validation` |
| A–D end to end | `WorkflowConfig`, `run_workflow`, `save_result` | `workflow` |

## 1. Notation

| Symbol | Meaning | Code |
|---|---|---|
| $`p`$, $`t`$ | plant, time step | `plant_id`, `time` |
| $`C_p`$ | installed capacity (MW) | `PlantSeries.capacity` |
| $`m_{p,t}`$ | point forecast or modelled power, per unit of $`C_p`$ | `forecast_pu` |
| $`y_{p,t}`$ | measured power, per unit (observed plants only) | `actual_pu` |
| $`e_{p,t} = y_{p,t} - m_{p,t}`$ | forecast error | — |
| $`s_{p,t}`$ | scale of the predictive distribution (target) | `target_scale` |
| $`\mathbf{x}_{p,t}`$ | UQ-metric features | feature columns |
| $`\mathcal{T} = \{\tau_1, \dots, \tau_L\}`$ | quantile levels | `levels` (default 0.05, 0.10, …, 0.95) |
| $`F_0`$, $`z_\tau = F_0^{-1}(\tau)`$ | standardized distribution and its quantiles | `dist`, `standard_quantiles` |

Power is expressed per unit of capacity throughout, so that one surrogate serves plants of different size; predictions are converted back to MW at the end.

## 2. Step A: calibrated uncertainty targets

The predictive distribution around the point forecast is a location-scale distribution,

```math
F_{p,t}(y) = F_0\!\left(\frac{y - m_{p,t}}{s_{p,t}}\right),
\qquad
q_{\tau}(s) = m_{p,t} + s\, z_\tau ,
```

with $`F_0`$ the standard Laplace distribution (`dist="laplace"`, scale = the Laplace scale $`b`$, standard deviation $`\sqrt{2}\,b`$) or the standard normal (`dist="gaussian"`, scale = the standard deviation). The pinball loss of a quantile $`q`$ at level $`\tau`$ for an observation $`y`$ is (Koenker and Bassett, 1978)

```math
\rho_\tau(y - q) = \max\{\tau (y - q),\; (\tau - 1)(y - q)\} .
```

**Pinball-loss optimization** (Sun et al., 2019). The target scale minimizes the mean pinball loss over the levels and over a set $`S`$ of time steps:

```math
s^{*}_S = \arg\min_{s \in [s_{\min}, s_{\max}]} \; \frac{1}{|S|\,L} \sum_{t \in S} \sum_{\tau \in \mathcal{T}} \rho_\tau\!\left(e_t - s\, z_\tau\right) .
```

Three choices of $`S`$ are implemented:

| Option | $`S`$ for the target at step $`t`$ | Use |
|---|---|---|
| default | $`\{t\}`$ | one target per step from that step alone, as in Sun et al. (2019) |
| `block=k` | the block of $`k`$ consecutive steps that contains $`t`$ | one target per block |
| `target_window=k` (odd) | the centred window of $`k`$ steps around $`t`$ (cut at the ends) | one target per step, smoother |

A target from a single step rests on one observation: for dense levels it is proportional to $`|e_t|`$ (see below), a very noisy estimate of the local error scale. Windows and blocks trade temporal resolution for a less noisy target; the choice is part of the validation design (`examples/validation_design.py`). The target at $`t`$ may use observations after $`t`$ (centred window): targets are labels at observed plants, not inputs, so this is not look-ahead in the features.

**Exact solution.** Each term $`\rho_\tau(e_t - s z_\tau)`$ is convex and piecewise linear in $`s`$, so the objective is convex and piecewise linear, with breakpoints at $`s = e_t / z_\tau`$. Its minimum lies at a breakpoint inside $`(s_{\min}, s_{\max})`$ or at a bound. `optimal_scale` evaluates all candidates for single steps and, for blocks and windows, finds the first interval between sorted candidates where the derivative is non-negative by bisection. The result is exact and deterministic (no stochastic search); among equal minima the smallest scale is returned. When $`e_t = 0`$ the optimum is the lower bound $`s_{\min}`$. Default bounds: $`s_{\min} = 10^{-9}\max(1, \max_t|e_t|)`$ and $`s_{\max} = 1.01 \max_t|e_t| / \min_{\tau}|z_\tau|`$ (over the levels with $`z_\tau \ne 0`$), which contains every single-step optimum.

**Link to the CRPS.** The CRPS equals twice the pinball loss integrated over all levels, $`\mathrm{CRPS}(F, y) = 2\int_0^1 \rho_\tau(y - F^{-1}(\tau))\, d\tau`$ (Gneiting and Raftery, 2007). As the levels become dense, the single-step optimum therefore tends to the scale that minimizes the closed-form CRPS for one observation (`crps_optimal_scale`):

```math
s^{*}_t = \frac{|e_t|}{c},
\qquad
c_{\mathrm{gau}} = \sqrt{\ln 2},
\qquad
c_{\mathrm{lap}} \text{ solves } (1 + c)\, e^{-c} = \tfrac{3}{4} .
```

These follow from setting the derivative in $`s`$ of the closed-form CRPS (Section 4) to zero: $`2\varphi(e/s) - 1/\sqrt{\pi} = 0`$ for the Gaussian and $`(1 + |e|/b)\, e^{-|e|/b} - 3/4 = 0`$ for the Laplace distribution. The tests check `optimal_scale` against this limit.

## 3. Steps B and C: features and surrogates

**Features** ([metrics.md](metrics.md)). `FeatureSpec` selects them:

- temporal metrics (standard deviation, turbulence intensity, normalized range, spectral entropy) of the per-unit forecast and of any other column;
- agreement metrics (correlation, MAPE, nRMSE, nMAE) of pairs of data products;
- spread and predictability indices of an ensemble (columns sharing a prefix);
- semivariogram nugget, sill and range of a turbine-level field;
- optionally the per-unit forecast level itself, the only input of the surrogate in Sun et al. (2019).

Two rules keep the features honest. The measured power never enters a feature (`FeatureSpec` rejects it): it does not exist at an unobserved plant. All metrics use trailing windows, so a feature at $`t`$ uses data up to $`t`$ only. Metrics that need observations (ensemble CRPS, rank histogram) are therefore not features.

**Surrogates** (`Surrogate`). Each maps the standardized features to $`\log s`$ (the scale is positive and right-skewed; `log_target=False` regresses $`s`$ itself) and returns $`\hat{s} = \exp(\widehat{\log s})`$, the median of the implied distribution of the scale:

| Name | Model (scikit-learn; Pedregosa et al., 2011) | Note |
|---|---|---|
| `gpr` | Gaussian process regression (Rasmussen and Williams, 2006); kernel $`C \cdot k_{\mathrm{Matérn}\,5/2}(\mathbf{x}, \mathbf{x}') + \sigma_n^2 \delta`$ with one length scale per feature; hyperparameters by maximum marginal likelihood; normalized target | also returns its own predictive standard deviation of $`\log s`$ (`predict_log_std`); exact GP costs $`O(n^3)`$, so it is fitted on at most `max_train_samples` random rows |
| `svr` | support vector regression, RBF kernel | the surrogate type of Sun et al. (2019) |
| `random_forest` | random forest regression | |
| `ridge` | ridge regression | linear reference |
| `constant` | median of the training targets | baseline: no information from the features |

`select_gpr_kernel` compares squared-exponential, Matérn 3/2, Matérn 5/2 and rational quadratic kernels by their log marginal likelihood on the same data. Optimizer convergence warnings are kept in `Surrogate.fit_warnings_` (and in the workflow notes) instead of being printed.

## 4. Step D: validation and scores

**Held-out plants.** `plant_splits` builds leave-one-plant-out folds (or $`K`$ folds of plants). Each held-out plant plays the role of an unobserved plant: its targets and observations are never used to fit the surrogate that predicts it. This measures what the workflow is for, generalization to new plants. `time_split` gives a within-plant split (last part of each plant's record) for comparison; it measures a different, easier task.

**Scores of the scale** (`score_scale`): MAE and RMSE of $`\hat{s}`$ against the target, RMSE of $`\log \hat{s}`$, and the coefficient of determination of $`\log s`$. Because the targets are noisy estimates of the true scale, these numbers understate the surrogate's accuracy; the scores below judge the predictive distributions against the observations directly.

**Scores of the predictive distributions** (`score_predictive`), with quantiles $`\hat{q}_{\tau,t} = m_t + \hat{s}_t z_\tau`$ at the held-out plants:

```math
\mathrm{PL} = \frac{1}{nL} \sum_{t} \sum_{\tau} \rho_\tau\!\left(y_t - \hat{q}_{\tau,t}\right),
\qquad
\mathrm{CRPS} = \frac{1}{n} \sum_t \mathrm{CRPS}\!\left(F_t, y_t\right),
```

```math
\mathrm{Rel}(\tau) = \frac{1}{n} \sum_t \mathbf{1}\{y_t \le \hat{q}_{\tau,t}\},
\qquad
\mathrm{Cov}(c) = \frac{1}{n} \sum_t \mathbf{1}\{\hat{q}_{(1-c)/2,t} \le y_t \le \hat{q}_{(1+c)/2,t}\},
\qquad
\mathrm{Width}(c) = \frac{1}{n} \sum_t \left(\hat{q}_{(1+c)/2,t} - \hat{q}_{(1-c)/2,t}\right).
```

The closed-form CRPS is (Gneiting et al., 2005; Jordan et al., 2019), with $`z = (y - m)/s`$ and $`e = y - m`$:

```math
\mathrm{CRPS}_{\mathrm{gau}} = s\left[z\,(2\Phi(z) - 1) + 2\varphi(z) - \frac{1}{\sqrt{\pi}}\right],
\qquad
\mathrm{CRPS}_{\mathrm{lap}} = |e| + b\, e^{-|e|/b} - \frac{3b}{4} .
```

Reliability $`\mathrm{Rel}(\tau)`$ should be close to $`\tau`$ (calibration); the summary reports the mean absolute deviation over the levels. Interval width measures sharpness, which matters only among calibrated predictions (Gneiting et al., 2007). Pinball loss and CRPS are proper scores and reward both. The constant baseline is the reference: a surrogate is useful only if it beats it on held-out plants.

**Oracle reference** (`oracle_scores`): the scores of the targets themselves, in sample. With single-step targets each target minimizes its own step's pinball loss, so their mean pinball loss is a lower bound that no surrogate can reach; with blocks or windows it is only a reference.

**Recalibration** (`IsotonicRecalibrator`, `recalibrated_scores`; Kuleshov et al., 2018). Systematic miscalibration, for example from regressing $`\log s`$ (which predicts a median, not the loss-optimal scale) or from a distribution family that does not fit, is corrected by an isotonic map of the probability integral transform: with $`u_t = F_t(y_t)`$ on held-out data, $`R(\tau)`$ = share of $`u_t \le \tau`$, made monotone; the recalibrated $`\tau`$-quantile is the original quantile at level $`R^{-1}(\tau)`$. In validation it is cross-fitted (each fold is recalibrated with a map fitted on the other folds); for the unobserved plants the map is fitted on all held-out predictions of the chosen surrogate.

## 5. Prediction at unobserved plants

`run_workflow` chooses the surrogate with the lowest cross-validated pinball loss (or `final_model`), fits it on all observed plants, predicts $`\hat{s}_{p,t}`$ at the unobserved plants, and returns predictive quantiles in MW, $`C_p\,(m_{p,t} + \hat{s}_{p,t} z_\tau)`$ (recalibrated if `recalibrate=True`), together with all validation tables. `save_result` writes them as CSV files and a `summary.json`.

**Scope and limits of the design.**

- The surrogate transfers a relation learned at observed plants. Its predictions at an unobserved plant are as good as the observed plants are representative of it (terrain, climate, turbine type, data quality). Leave-one-plant-out scores estimate this only for plants like the observed ones.
- Plants close to each other share weather, so neighbouring held-out plants are not independent tests; holding out spatial groups of plants (`n_splits` with a grouping by region) gives a stricter estimate.
- The location-scale family fixes the shape of the predictive distribution; the bounds of power (zero and capacity) are not part of it. Recalibration corrects average miscalibration but not shape errors that vary with conditions.
- The features describe uncertainty of the inputs. Whether they carry information about power-forecast uncertainty at a given set of plants is an empirical question that the validation answers; `examples/validation_design.py` includes a negative control in which they do not.

## References

- Gneiting, T., Balabdaoui, F., and Raftery, A. E. (2007). Probabilistic forecasts, calibration and sharpness. *Journal of the Royal Statistical Society: Series B*, 69(2), 243–268. https://doi.org/10.1111/j.1467-9868.2007.00587.x
- Gneiting, T., and Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association*, 102(477), 359–378. https://doi.org/10.1198/016214506000001437
- Gneiting, T., Raftery, A. E., Westveld, A. H., and Goldman, T. (2005). Calibrated probabilistic forecasting using ensemble model output statistics and minimum CRPS estimation. *Monthly Weather Review*, 133(5), 1098–1118. https://doi.org/10.1175/MWR2904.1
- Jordan, A., Krüger, F., and Lerch, S. (2019). Evaluating probabilistic forecasts with scoringRules. *Journal of Statistical Software*, 90(12). https://doi.org/10.18637/jss.v090.i12
- Koenker, R., and Bassett, G. (1978). Regression quantiles. *Econometrica*, 46(1), 33–50. https://doi.org/10.2307/1913643
- Kuleshov, V., Fenner, N., and Ermon, S. (2018). Accurate uncertainties for deep learning using calibrated regression. *Proceedings of the 35th International Conference on Machine Learning*, PMLR 80, 2796–2804. https://arxiv.org/abs/1807.00263
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
- Rasmussen, C. E., and Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
- Sun, M., Feng, C., Chartan, E. K., Hodge, B.-M., and Zhang, J. (2019). A two-step short-term probabilistic wind forecasting methodology based on predictive distribution optimization. *Applied Energy*, 238, 1497–1505. https://doi.org/10.1016/j.apenergy.2019.01.182
