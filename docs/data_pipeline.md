# Data pipeline

What the framework needs, how to prepare it from typical wind data sources, the input-directory format and the checks applied when it is read. The repository ships no real data; the examples use the synthetic generator (Section 5).

## 1. Inputs

| Input | Unit | Observed plants | Unobserved plants | Role |
|---|---|---|---|---|
| Point forecast or modelled power `forecast` | MW | required | required | location of the predictive distribution; temporal metrics; optional feature `level` |
| Measured power `actual` | MW | required | absent | Step A targets and all validation scores; never a feature |
| Meteorological data products (e.g. hub-height wind speed from two products) | m/s | optional | same columns needed | agreement metrics, temporal metrics |
| Ensemble members (columns sharing a prefix) | m/s or MW | optional | same columns needed | spread and predictability indices |
| Turbine coordinates and turbine-level series | km; m/s | optional | same needed | semivariogram nugget, sill and range |
| Installed capacity | MW | required | required | per-unit normalization |

The surrogate can only use features that exist at both kinds of plants. Every column used by a `FeatureSpec` must therefore be present at the unobserved plants; `run_workflow` stops with an error otherwise.

## 2. Preparing the inputs

Typical sources and what to do with them:

- **Point forecast or modelled power.** The output of an operational forecast system, or a power-curve model applied to a meteorological data product (for a probabilistic power curve from a theoretical curve and measured data, see the sibling framework [mfgpr-wind-power-curve](https://github.com/Henryutd2021/mfgpr-wind-power-curve)). It must be produced the same way at observed and unobserved plants; otherwise the surrogate learns differences between forecast systems, not between conditions.
- **Measured plant power.** Metered plant output or the sum of turbine SCADA power. Remove or flag intervals with curtailment, outages, communication gaps and capacity changes before computing targets: the targets describe forecast-error uncertainty under normal operation, and the flags are not known for unobserved plants. Plant-level measurements often come with restrictive data-use terms; keep them out of shared repositories.
- **Meteorological data products.** Reanalyses, mesoscale model data sets and numerical weather prediction analyses or forecasts at the plant location and hub height. Use the same products at every plant. Interpolate to the plant location; extrapolate to hub height with a documented profile (for example a power law with a stated exponent); record the product version and access date.
- **Ensembles.** Members of a numerical weather prediction ensemble, or several data products treated as an ensemble. Keep the member set fixed over time; a changing number of members changes the spread indices.
- **Turbine-level data.** Nacelle wind speeds (or power) per turbine with turbine coordinates in a planar system in km (or latitude and longitude with `kind="haversine"`). Choose lag bins from the turbine spacing: the first edge above the shortest spacing, the last near the plant diameter, and enough pairs per bin.

Common preparation rules:

1. **One time base.** Same time zone (UTC), same resolution, and the same timestamp convention (period-beginning or period-ending) for every column of every plant. Averaged quantities (power, wind speed) and instantaneous ones must not be mixed without a stated alignment.
2. **Gaps.** The trailing windows propagate missing values; short gaps may be interpolated and flagged, long gaps should stay missing. Rows with a missing feature or target are dropped from the training table.
3. **Per unit.** The workflow divides power by the installed capacity (forecast, actual, targets). If the capacity changed during the record, split the record or scale by the time-varying capacity before writing the inputs.
4. **Window lengths.** Choose the metric window (`FeatureSpec.window`), the predictability-index history (`pi_history`) and the target window (`target_window` or `block`) from the time resolution and the horizon of interest, and keep them fixed across plants. They are part of the study design and should be reported with the results.
5. **Record provenance.** For each column: source, version, access date, processing steps, and the license or data-use terms.

## 3. Input directory format

```
my_inputs/
├── plants.csv                       plant_id, capacity_mw[, x_km, y_km]
├── series/<plant_id>.csv            time, forecast[, actual], other columns
├── turbines/<plant_id>.csv          turbine_id, x, y                     (optional)
└── turbine_series/<plant_id>.csv    time, one column per turbine_id      (optional)
```

| File | Column | Type | Rule |
|---|---|---|---|
| `plants.csv` | `plant_id` | text | unique; names the series files |
| | `capacity_mw` | number | positive |
| | `x_km`, `y_km` | number | optional plant location (informational) |
| `series/<plant_id>.csv` | `time` | ISO 8601 timestamps or integers | unique and increasing |
| | `forecast` | MW | required, finite |
| | `actual` | MW | present with values only at observed plants; an empty column means unobserved |
| | others | any unit | covariates named in the feature specification; ensemble members share a prefix (e.g. `ens_00`, `ens_01`, …) |
| `turbines/<plant_id>.csv` | `turbine_id`, `x`, `y` | text, km, km | one row per turbine |
| `turbine_series/<plant_id>.csv` | `time`, one column per `turbine_id` | as `series` | same time index as the plant's series |

`wind_uq_surrogate.write_plants` writes this layout and `read_plants` reads it back into `PlantSeries` objects.

## 4. Checks

Applied by `read_plants` and `PlantSeries`:

- `plants.csv` has `plant_id` and `capacity_mw`; plant ids are unique; capacities are positive.
- Every plant has a series file with a `time` column; times are unique and increasing; `forecast` exists and is finite.
- Turbine files come in pairs, every listed turbine has a series column, coordinates have one row per turbine, and the turbine series share the plant's time index.

Applied in the workflow:

- Feature specifications never use `actual` (`FeatureSpec` raises an error).
- Targets need finite forecast and actual values (missing rows are dropped first); scale bounds must satisfy $`0 < s_{\min} < s_{\max}`$.
- Window lengths are integers within the series length; the spectral entropy needs at least four steps; the target window is odd.
- At least two observed plants are needed, since validation holds out plants.
- The quantile levels must contain the bounds of every requested central interval.

## 5. Synthetic data

`synthetic_fleet` generates invented plants for the examples and tests:

- plant locations, capacities and mean wind speeds are random draws;
- a slowly varying regime $`r_t \in (0, 1)`$ per plant (partly shared between plants) makes the wind more variable, the second data product noisier and the ensemble wider when it is high;
- the point forecast is an invented smooth power curve applied to one data product; the observed power is the forecast plus Laplace noise with scale $`C_p \cdot \mathrm{base} \cdot \exp(\mathrm{signal} \cdot \mathrm{gain} \cdot (r_t - 0.5))`$, clipped to $`[0, C_p]`$;
- with `n_turbines > 0` each plant gets a turbine-level wind field with exponential spatial correlation whose amplitude follows the regime;
- `signal = 0` keeps the metrics' dependence on the regime but removes it from the error scale: a negative control.

The generator also returns its ground truth (observed power and true error scale at every plant, including the unobserved ones), which only synthetic data can provide. Its numbers say nothing about any real plant, data product or forecast system.

## 6. Terms of use

The framework is licensed under MIT (code) and CC BY 4.0 (documentation). Data used with it keep their own terms: check the license of every meteorological product and the data-use agreement of every measured series before sharing inputs, features or results derived from them.
