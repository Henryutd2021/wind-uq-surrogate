# Results of the ERCOT case study

Tables and figures derived from the **ARPA-E PERFORM data sets** (Sergi, B., Feng, C., Zhang, F., Hodge, B.-M., Ring-Jarvi, R., Bryce, R., Doubleday, K., Rose, M., Buster, G., and Rossol, M., 2022, https://doi.org/10.25984/1891136), ERCOT 2018, published by the National Renewable Energy Laboratory for ARPA-E under [CC BY 3.0 US](https://creativecommons.org/licenses/by/3.0/us/). They are shared under the same licence.

- **Changes made to the source data:** hourly averages of the 5-minute simulated power; per-unit scaling by installed capacity; ensemble means and UQ metrics; surrogate predictions, scores and summaries.
- **The plant power is simulated, not metered.** The publisher has not reviewed or endorsed this work.

Produced by `prepare.py` and `run.py` in the folder above, from the files listed in `../data_manifest.csv`; figures by `figures.py`. What each file contains: [../README.md](../README.md) and [../../../data/README.md](../../../data/README.md).

| Files | Content |
|---|---|
| `scores_<design>.csv`, `reliability_<design>.csv`, `per_plant_<design>.csv`, `scale_scores_<design>.csv`, `feature_correlation_<design>.csv`, `design_<design>.json` | the three validation designs `plants10`, `zones`, `proposed` with all features; the same with suffix `_level` for the level-only ablation |
| `plants.csv`, `forecast_summary.csv`, `alignment.csv`, `prepare_summary.json` | plant identifiers and metadata, forecast bias and error per plant, the time-label check, preparation settings |
| `fig_skill.png`, `fig_reliability.png`, `fig_plants.png` | figures |
