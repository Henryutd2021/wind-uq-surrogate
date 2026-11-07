# Data

This repository contains no raw data. The examples and tests use invented plants from the synthetic generator (`wind_uq_surrogate.synthetic`). The case study in [`case_studies/perform_ercot/`](../case_studies/perform_ercot/) uses one public data set, which its scripts download from the publisher's site; only tables and figures derived from it are stored here.

## Public data used

| Data set | Publisher | What the case study uses | Licence | How to obtain |
|---|---|---|---|---|
| ARPA-E PERFORM data sets, ERCOT 2018 (version 1.0.1) | National Renewable Energy Laboratory, for ARPA-E | for 264 wind plants: simulated plant power (5-minute, MW), day-ahead and 2-day-ahead power forecasts of the 51 ECMWF ensemble members (hourly, MW), plant metadata | [Creative Commons Attribution 3.0 United States](https://creativecommons.org/licenses/by/3.0/us/) | `case_studies/perform_ercot/download.py` (anonymous HTTPS from the public bucket `s3://arpa-e-perform`); file list with sizes and SHA-256 in [`case_studies/perform_ercot/data_manifest.csv`](../case_studies/perform_ercot/data_manifest.csv) |

Citation of the data set, as recommended by the publisher:

> Sergi, B., Feng, C., Zhang, F., Hodge, B.-M., Ring-Jarvi, R., Bryce, R., Doubleday, K., Rose, M., Buster, G., and Rossol, M. (2022). ARPA-E PERFORM datasets. United States. https://doi.org/10.25984/1891136. https://data.openei.org/submissions/5772

Method description of the ERCOT data: Bryce, R., Buster, G., Doubleday, K., Feng, C., Ring-Jarvi, R., Rossol, M., Zhang, F., and Hodge, B.-M. (2023). *Solar PV, Wind Generation, and Load Forecasting Dataset for ERCOT 2018: Performance-Based Energy Resource Feedback, Optimization, and Risk Management (P.E.R.F.O.R.M.)*. NREL/TP-5D00-79498. National Renewable Energy Laboratory. Documentation of the files: https://github.com/PERFORM-Forecasts/documentation

Points to keep in mind:

- **The plant power is simulated, not metered.** In PERFORM, wind speed from the WIND Toolkit (a mesoscale model data set) is converted to power with plant configurations; the forecasts convert ECMWF ensemble forecasts with the same configurations. "Existing" plants are at the locations of operating plants, "proposed" plants at sites of the ERCOT interconnection queue. Results on these data describe the method on a consistent public benchmark, not the uncertainty of any real plant's output.
- **Not used:** PERFORM's probabilistic (percentile) forecasts. They are trained on each plant's own history, which an unmeasured plant does not have.

## Derived data stored in this repository

All under `case_studies/perform_ercot/`, derived from the PERFORM data and shared under the same licence (CC BY 3.0 US) with the attribution above. Changes made: hourly averages of the 5-minute power, per-unit scaling by installed capacity, ensemble means and UQ metrics, surrogate predictions and scores.

| File | Content |
|---|---|
| `data_manifest.csv` | the 793 PERFORM files used: key, size, SHA-256, access date |
| `results/plants.csv` | plant id (`w001`–`w264`, in the alphabetical order of the normalized PERFORM site names), PERFORM site name, installed capacity, ERCOT weather zone, status (existing or proposed) |
| `results/forecast_summary.csv`, `results/alignment.csv`, `results/prepare_summary.json` | per plant: mean simulated power and forecast, bias, error and correlation of the ensemble-mean forecast; the time-label check; preparation settings |
| `results/scores_*.csv`, `results/reliability_*.csv`, `results/per_plant_*.csv`, `results/scale_scores_*.csv`, `results/feature_correlation_*.csv`, `results/design_*.json` | scores and settings of the three validation designs, with all features and (suffix `_level`) with the forecast level only (see the case study's README) |
| `results/fig_*.png` | figures made from these tables by `figures.py` |
| `results/README.md` | source, licence and changes, for readers of the results folder alone |

No other data are used or included: no measured plant data, no data under a data-use agreement, and no data from the project in which the methods were developed.

## Acknowledgment

This work was developed based upon funding from the Alliance for Sustainable Energy, LLC, Managing and Operating Contractor for the National Renewable Energy Laboratory for the U.S. Department of Energy.

## Disclaimer

The views expressed in this repository do not necessarily represent the views of the U.S. Department of Energy or the U.S. Government. Neither the data publisher nor the funding organizations have reviewed or endorsed this work. The derived tables are provided as is, without warranty of any kind.
