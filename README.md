# Wind UQ surrogate framework: UQ metrics and surrogate models for plant-level wind power uncertainty

Author: [Honglin Li](https://henryutd2021.github.io/)

A documented, runnable framework for two linked tasks:

1. **UQ metrics.** Quantify the uncertainty of the inputs of a wind power model: the variability of a series over time, the disagreement between data sources, the spread of an ensemble, and the spatial variability across the turbines of a plant.
2. **Surrogate models.** Learn, at plants with power measurements, a mapping from these metrics to a calibrated measure of power uncertainty. Then predict probabilistic power (predictive quantiles) at plants without measurements.

The surrogate workflow extends the two-step probabilistic forecasting method of Sun et al. (2019). In that method, a point forecast is the centre of a predictive distribution, and a surrogate predicts the distribution's scale, chosen by pinball-loss optimization. Here the surrogate's inputs are UQ metrics of several uncertainty sources, several surrogates are compared, and the validation holds out whole plants.

## Research question

Can UQ metrics of the inputs of a wind power model serve as the inputs of a surrogate model that predicts calibrated plant-level power uncertainty at plants without power measurements?

## Scope

| Aspect | In this framework |
|---|---|
| Temporal metrics | moving-window standard deviation, turbulence intensity (coefficient of variation), normalized range, spectral entropy |
| Data-source metrics | moving-window correlation, MAPE, nRMSE, nMAE between two series of the same quantity |
| Ensemble metrics | spread indices between percentiles, total spread, predictability index, ensemble CRPS, rank histogram |
| Spatial metrics | semivariogram of turbine-level values; nugget, sill and range from a weighted least-squares fit |
| Uncertainty target | scale of a Laplace or Gaussian predictive distribution around the point forecast, chosen by pinball-loss optimization (exact solution), per step, per block or per centred window |
| Surrogates | Gaussian process regression (with its own predictive uncertainty), support vector regression, random forest, ridge regression, constant baseline |
| Validation | leave-one-plant-out (or plant or regional folds); scores of the scale and of the predictive distributions (pinball loss, CRPS, reliability, interval coverage and width); PIT recalibration validated with nested plant folds |
| Output | per-unit scale and predictive quantiles in MW at unobserved plants |
| Data | any plants in the input format of [docs/data_pipeline.md](docs/data_pipeline.md); synthetic generator for examples and tests |

## What is in this repository

| Path | Content |
|---|---|
| [`docs/metrics.md`](docs/metrics.md) | definitions of every UQ metric, conventions, references |
| [`docs/surrogate_workflow.md`](docs/surrogate_workflow.md) | Steps A–D: calibrated targets, features, surrogates, validation and scores, prediction at unobserved plants; scope and limits |
| [`docs/data_pipeline.md`](docs/data_pipeline.md) | inputs, preparation from typical wind data sources, input-directory format, checks, synthetic data, terms of use |
| [`docs/architecture.md`](docs/architecture.md) | modules, data flow, output tables, run time, tests |
| [`src/wind_uq_surrogate/`](src/wind_uq_surrogate/) | the implementation (NumPy, SciPy, pandas, scikit-learn) |
| [`examples/`](examples/) | `minimal.py` (end-to-end, about half a minute), `metrics_tour.py` (every metric family), `calibration_targets.py` (Step A in detail), `validation_design.py` (target choice and a negative control, about a minute); all on synthetic data |
| [`tests/`](tests/) | 64 tests: metrics against direct formulas, scores against numerical integration, exact optimization against brute force, features without look-ahead, leakage guards, workflow, input files, command line |

## Quick start

```bash
git clone https://github.com/Henryutd2021/wind-uq-surrogate.git
cd wind-uq-surrogate
python3 -m venv .venv && source .venv/bin/activate    # needs Python 3.10+ (check: python3 --version); if older, use e.g. python3.12
pip install -e ".[test]"
python examples/minimal.py                              # invented plants, about half a minute
pytest
```

With conda instead: `conda env create -f environment.yml && conda activate wind-uq-surrogate && pip install -e .`

## Command line

```bash
wind-uq-surrogate synthetic my_inputs/                               # invented plants in the input format
wind-uq-surrogate metrics my_inputs/ --out-dir outputs/metrics \
    --window 18 --pair ws_a:ws_b --members ens_ --pi-history 48      # feature table per plant
wind-uq-surrogate run my_inputs/ --out-dir outputs/run \
    --window 18 --pair ws_a:ws_b --members ens_ --pi-history 48 --target-window 25
```

`run` prints the cross-validated scores of every surrogate and writes the tables of [docs/architecture.md](docs/architecture.md#output-tables), including predictive quantiles at the unobserved plants.

## In Python

```python
import wind_uq_surrogate as w

plants = w.read_plants("my_inputs/")                     # your own data, see docs/data_pipeline.md
observed, unobserved = w.split_observed(plants)

spec = w.FeatureSpec(window=18, pairs=(("ws_a", "ws_b"),), member_prefix="ens_", pi_history=48)
config = w.WorkflowConfig(features=spec, dist="laplace", target_window=25)
result = w.run_workflow(observed, unobserved, config)

print(result.predictive_scores)                          # held-out plants: pinball, CRPS, reliability, coverage
print(result.predictions.head())                         # quantiles in MW at the unobserved plants
```

The metrics can also be used on their own, for example `w.spread_index(members, 75, 25)`, `w.moving_spectral_entropy(series, 18)` or `w.variogram_over_time(field, coords, bin_edges)`.

**About the example data:** `wind_uq_surrogate.synthetic` generates invented plants, data products, ensembles and errors. They show how the methods work and how to read their output. Their numbers say nothing about any real plant, data product or forecast system.

## Using your own data

1. Build the input directory as described in [docs/data_pipeline.md](docs/data_pipeline.md). The same input columns must exist at observed and unobserved plants.
2. Choose the window lengths and the target type, and keep them fixed across plants. Compare against the constant baseline: a surrogate is useful only if it beats the baseline on held-out plants ([docs/surrogate_workflow.md](docs/surrogate_workflow.md#4-step-d-validation-and-scores)).
3. Report the validation scores with the predictions. They show how well the relation learned at the observed plants transfers to plants like them.

## Related

- [mfgpr-wind-power-curve](https://github.com/Henryutd2021/mfgpr-wind-power-curve): multi-fidelity Gaussian process regression for probabilistic wind power curves. It can provide the point forecast and power-curve uncertainty inputs used here.

## What this repository does not contain

This repository presents the metrics and the surrogate workflow as a framework, implemented afresh and tested on synthetic data. It contains no data, code or results from the project in which the methods were developed.

## Acknowledgment

This work was developed based upon funding from the Alliance for Sustainable Energy, LLC, Managing and Operating Contractor for the National Renewable Energy Laboratory for the U.S. Department of Energy.

## Citation

If you use this framework, please cite it with the metadata in [`CITATION.cff`](CITATION.cff), and cite the methods it builds on (references in [docs/](docs/)).

## Licence

- Code: [MIT](LICENSE).
- Documentation (`README.md`, `docs/`): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Reference

Sun, M., Feng, C., Chartan, E. K., Hodge, B.-M., and Zhang, J. (2019). A two-step short-term probabilistic wind forecasting methodology based on predictive distribution optimization. *Applied Energy*, 238, 1497–1505. https://doi.org/10.1016/j.apenergy.2019.01.182
