# UQ metrics

The metrics describe how uncertain or variable the inputs of a wind power model are. They are computed from series that exist at every plant: point forecasts or modelled power, meteorological data products, ensemble members, and turbine-level values. In the surrogate workflow ([surrogate_workflow.md](surrogate_workflow.md)) they are the inputs of the surrogate model. Every metric here follows a definition from the literature cited in each section; this document fixes the conventions of this implementation.

| Family | What it describes | Metrics | Module |
|---|---|---|---|
| Temporal (one series) | variability of a series within a moving window | standard deviation, turbulence intensity (coefficient of variation), normalized range, spectral entropy | `metrics_single` |
| Data source (two series) | agreement of two series of the same quantity within a moving window | correlation, MAPE, nRMSE, nMAE | `metrics_pair` |
| Ensemble (several series) | spread of members or data sources at each time step | spread indices, total spread index, predictability index, CRPS, rank histogram | `metrics_ensemble` |
| Spatial (turbines in a plant) | spatial structure of turbine-level values at each time step | semivariogram nugget, sill and range | `metrics_spatial` |

## 1. Conventions

- **Trailing windows.** A moving-window metric at time step $`t`$ with window length $`w`$ uses $`x_{t-w+1}, \dots, x_t`$. The first $`w - 1`$ values are missing (`NaN`). A metric never uses values after $`t`$, so features built from it are available in real time and contain no look-ahead.
- **Missing values.** A window that contains a missing value gives a missing metric. Fill or flag gaps before computing the metrics ([data_pipeline.md](data_pipeline.md)).
- **Window length.** The window length is a required argument everywhere. It sets the time scale of the uncertainty that the metric describes and should be chosen with the time resolution of the data and the forecast horizon of interest.
- **Units.** Ratios (turbulence intensity, normalized range, nRMSE, nMAE) are fractions; MAPE and the spread indices are in percent; standard deviation, CRPS and the variogram parameters are in the unit of the data (squared unit for nugget and sill). In the workflow, power is expressed per unit of installed capacity so that plants of different size share one surrogate.

## 2. Temporal metrics of one series

For a window $`W_t = (x_{t-w+1}, \dots, x_t)`$ with mean $`\bar{x}_t`$:

```math
\sigma_t = \sqrt{\frac{1}{w} \sum_{i=t-w+1}^{t} (x_i - \bar{x}_t)^2},
\qquad
\mathrm{TI}_t = \frac{\sigma_t}{\bar{x}_t},
\qquad
\mathrm{NR}_t = \frac{\max W_t - \min W_t}{\bar{x}_t}.
```

- **Standard deviation** $`\sigma_t`$ (`moving_std`): population form by default; `ddof=1` gives the sample form.
- **Turbulence intensity** $`\mathrm{TI}_t`$ (`turbulence_intensity`): the coefficient of variation of the window. For wind speed sampled within the averaging period of IEC 61400-1 (ten minutes) this is the turbulence intensity of that standard; over longer windows it measures relative variability.
- **Normalized range** $`\mathrm{NR}_t`$ (`normalized_range`): the range relative to the mean. Some authors call it a variability index; other quantities share that name.
- Windows whose mean is not above `min_mean` (default 0) give a missing value for the two ratios.

**Spectral entropy** (`spectral_entropy`, `moving_spectral_entropy`; Inouye et al., 1991). The window is detrended (mean removed by default; `detrend="linear"` also removes a straight line). Its periodogram at the Fourier frequencies $`k/w`$, $`k = 1, \dots, K = \lfloor w/2 \rfloor`$, is normalized to a probability mass $`p_k`$, and

```math
H_t = -\frac{1}{\ln K} \sum_{k=1}^{K} p_k \ln p_k \in [0, 1].
```

$`H_t = 0`$ when all variance sits at one frequency (a regular oscillation); $`H_t = 1`$ for a flat spectrum (white-noise-like, least predictable from its own past). The value does not change when the window is shifted or scaled. A window whose variation is at the level of floating-point rounding of its values counts as constant and gives a missing value. The zero frequency is excluded, so a constant offset does not count as information. The window needs at least four steps.

## 3. Agreement of two series (data-source uncertainty)

For a reference series $`r`$ and a second series $`s`$ of the same quantity (two data products at one site, a model and a measurement, two ensemble members), within each trailing window:

```math
\rho_t = \mathrm{corr}(r, s),
\qquad
\mathrm{MAPE}_t = \frac{100}{|I_t|} \sum_{i \in I_t} \frac{|r_i - s_i|}{|r_i|},
```

```math
\mathrm{nRMSE}_t = \frac{1}{N_t} \sqrt{\frac{1}{w} \sum_i (r_i - s_i)^2},
\qquad
\mathrm{nMAE}_t = \frac{1}{N_t} \frac{1}{w} \sum_i |r_i - s_i|.
```

- $`\rho_t`$ is the Pearson correlation (`moving_correlation`); it is missing when either series is constant in the window (variation at the level of floating-point rounding); it does not change when either series is shifted or scaled.
- MAPE (`moving_mape`) averages over the steps $`I_t`$ where $`|r_i|`$ exceeds `min_reference`, because small reference values make relative errors explode (for wind speed a threshold near calm conditions is common, as in Fernández-González et al., 2017). A window without such steps gives a missing value.
- The normalizer $`N_t`$ of nRMSE and nMAE (`moving_nrmse`, `moving_nmae`) is a fixed positive number (for power: the installed capacity), the range of the reference in the window (`"range"`), or its mean (`"mean"`). A zero range or a non-positive mean gives a missing value; there is no small-number fallback. These are among the standard deterministic metrics of forecast evaluation (Zhang et al., 2015).

## 4. Ensemble metrics

At each time step the ensemble is $`x_{t,1}, \dots, x_{t,M}`$ (`members`, shape `(n_times, M)`). The members can be the members of a numerical weather prediction ensemble, or several data sources for the same quantity used as an ensemble.

**Spread indices** (`spread_index`, `total_spread_index`; Fernández-González et al., 2017). With $`q_{t,p}`$ the $`p`$-th percentile of the members and $`\bar{x}_t`$ their mean:

```math
\mathrm{SI}^{u\text{-}l}_t = 100 \, \frac{q_{t,u} - q_{t,l}}{\bar{x}_t},
\qquad
\mathrm{SI}^{\mathrm{total}}_t = 100 \, \frac{\max_m x_{t,m} - \min_m x_{t,m}}{\bar{x}_t}.
```

Percentile-based spreads are used instead of the standard deviation because wind data are skewed. The default pair is the interquartile range (75–25), the index selected in that study; any pair can be given. Percentiles use linear interpolation between order statistics (`numpy.percentile`, method `"linear"`). A non-positive ensemble mean gives a missing value.

**Absolute spread** (`relative=False`; `FeatureSpec(si_relative=False)`). For ensembles of power, the ensemble mean is often zero at low wind, where the relative indices are undefined. The absolute form keeps the numerator in the unit of the members,

```math
\mathrm{SI}^{u\text{-}l}_{t,\mathrm{abs}} = q_{t,u} - q_{t,l},
\qquad
\mathrm{SI}^{\mathrm{total}}_{t,\mathrm{abs}} = \max_m x_{t,m} - \min_m x_{t,m},
```

and is per unit of installed capacity when the members are (`FeatureSpec(members_per_unit=True)`), so that it is comparable across plants. The predictability index below is then the difference of absolute widths.

**Predictability index** (`predictability_index`; after Fernández-González et al., 2017, building on the relative measure of predictability of Toth et al., 2001). The current spread compared with its recent average over the previous $`L`$ steps (`history`):

```math
\mathrm{PI}_t = \frac{1}{L} \sum_{j=1}^{L} \mathrm{SI}_{t-j} - \mathrm{SI}_t .
```

$`\mathrm{PI}_t > 0`$: less spread than usual, a more predictable situation; $`\mathrm{PI}_t < 0`$: more spread than usual. The cited study averages over the previous 30 days for a daily forecast; here the history is a number of time steps chosen by the user.

**Continuous ranked probability score of the ensemble** (`crps_ensemble`), where an observation $`y_t`$ exists (Gneiting and Raftery, 2007; Hersbach, 2000):

```math
\mathrm{CRPS}_t = \frac{1}{M} \sum_{m=1}^{M} |x_{t,m} - y_t| - \frac{1}{2M^2} \sum_{m=1}^{M} \sum_{k=1}^{M} |x_{t,m} - x_{t,k}| .
```

This is the CRPS of the empirical distribution of the members (lower is better; same unit as the data). With `fair=True` the second term uses $`1/(2M(M-1))`$ (Ferro, 2014); the fair version assumes that the members are a random sample of independent draws from the forecast distribution, which does not hold for correlated members or for several data products used as an ensemble. The double sum is computed from the sorted members in $`O(M \log M)`$.

**Rank histogram** (`observation_rank`, `rank_histogram`, `rank_histogram_summary`; Hamill, 2001). The rank of the observation among the members (0 to $`M`$, ties broken at random) is counted over many time steps. A statistically consistent ensemble gives a flat histogram; a U shape points to too little spread or a bias, a dome to too much spread. The summary reports the chi-square statistic against a flat histogram ($`M`$ degrees of freedom) and the share of the two outer bins next to its expected value $`2/(M+1)`$. Serial correlation makes the chi-square test too liberal (Hamill, 2001); read it as a descriptive statistic.

Only the spread and predictability indices can be computed where no observation exists; CRPS and the rank histogram need observations and therefore describe observed sites only.

## 5. Spatial metrics inside a plant

At each time step the values $`z_i`$ at the turbines $`i`$ with positions $`\mathbf{s}_i`$ (for example nacelle wind speeds) describe the spatial variability inside the plant (`metrics_spatial`; Matheron, 1963; Cressie, 1993).

**Empirical semivariogram** (Matheron's estimator, `empirical_variogram`). Pairs are grouped into lag bins $`(e_k, e_{k+1}]`$ of their distance $`d_{ij} = \|\mathbf{s}_i - \mathbf{s}_j\|`$:

```math
\hat{\gamma}(h_k) = \frac{1}{2 N_k} \sum_{(i,j):\, e_k < d_{ij} \le e_{k+1}} (z_i - z_j)^2 ,
```

with $`N_k`$ pairs and $`h_k`$ the mean distance of the pairs in bin $`k`$. Bins with fewer than `min_pairs` pairs are dropped; a common rule of thumb asks for several tens of pairs per bin. Distances are planar (coordinates in km, `kind="euclidean"`) or great-circle distances from latitude and longitude (`kind="haversine"`, mean Earth radius 6371.0088 km). `detrend="linear"` removes a planar trend in the coordinates first, so that a large-scale gradient across the plant is not read as spatial variability.

**Variogram models** (`variogram_model`) with nugget $`c_0`$, partial sill $`c_1`$ and practical range $`a`$; $`\gamma(0) = 0`$ and, for $`h > 0`$:

```math
\gamma_{\exp}(h) = c_0 + c_1 \left(1 - e^{-3h/a}\right),
\qquad
\gamma_{\mathrm{gau}}(h) = c_0 + c_1 \left(1 - e^{-3h^2/a^2}\right),
```

```math
\gamma_{\mathrm{sph}}(h) = c_0 + c_1 \left(\frac{3h}{2a} - \frac{h^3}{2a^3}\right) \text{ for } h < a,
\qquad
\gamma_{\mathrm{sph}}(h) = c_0 + c_1 \text{ for } h \ge a .
```

The exponential and Gaussian models reach 95 % of the partial sill at the practical range; the spherical model reaches the sill exactly there.

**Fit** (`fit_variogram`). By default the weighted least-squares criterion of Cressie (1985),

```math
\min_{c_0, c_1, a} \; \sum_k N_k \left( \frac{\hat{\gamma}(h_k)}{\gamma(h_k)} - 1 \right)^2 ,
```

with $`c_0, c_1 \ge 0`$ and $`0 < a \le a_{\max}`$ ($`a_{\max}`$ defaults to three times the largest lag). `weighting="pairs"` minimizes $`\sum_k N_k (\hat{\gamma}_k - \gamma(h_k))^2`$ and `"none"` is ordinary least squares. The reported metrics are the **nugget** $`c_0`$ (variation at very short distance and measurement noise), the **sill** $`c_0 + c_1`$ (total variance across the plant) and the **range** $`a`$ (distance beyond which turbines are practically uncorrelated). `variogram_over_time` repeats the fit at every time step; steps with fewer than three usable bins or a failed fit give missing values. With a few dozen turbines a single time step identifies the range poorly (a fitted range at the bound $`a_{\max}`$ signals this); the sill is the most stable of the three.

## References

- Cressie, N. (1985). Fitting variogram models by weighted least squares. *Journal of the International Association for Mathematical Geology*, 17(5), 563–586. https://doi.org/10.1007/BF01032109
- Cressie, N. (1993). *Statistics for Spatial Data* (revised ed.). Wiley. https://doi.org/10.1002/9781119115151
- Fernández-González, S., Martín, M. L., Merino, A., Sánchez, J. L., and Valero, F. (2017). Uncertainty quantification and predictability of wind speed over the Iberian Peninsula. *Journal of Geophysical Research: Atmospheres*, 122(7), 3877–3890. https://doi.org/10.1002/2017JD026533
- Ferro, C. A. T. (2014). Fair scores for ensemble forecasts. *Quarterly Journal of the Royal Meteorological Society*, 140(683), 1917–1923. https://doi.org/10.1002/qj.2270
- Gneiting, T., and Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association*, 102(477), 359–378. https://doi.org/10.1198/016214506000001437
- Hamill, T. M. (2001). Interpretation of rank histograms for verifying ensemble forecasts. *Monthly Weather Review*, 129(3), 550–560. https://doi.org/10.1175/1520-0493(2001)129<0550:IORHFV>2.0.CO;2
- Hersbach, H. (2000). Decomposition of the continuous ranked probability score for ensemble prediction systems. *Weather and Forecasting*, 15(5), 559–570. https://doi.org/10.1175/1520-0434(2000)015<0559:DOTCRP>2.0.CO;2
- IEC 61400-1:2019. *Wind energy generation systems – Part 1: Design requirements* (edition 4.0). International Electrotechnical Commission.
- Inouye, T., Shinosaki, K., Sakamoto, H., Toi, S., Ukai, S., Iyama, A., Katsuda, Y., and Hirano, M. (1991). Quantification of EEG irregularity by use of the entropy of the power spectrum. *Electroencephalography and Clinical Neurophysiology*, 79(3), 204–210. https://doi.org/10.1016/0013-4694(91)90138-T
- Matheron, G. (1963). Principles of geostatistics. *Economic Geology*, 58(8), 1246–1266. https://doi.org/10.2113/gsecongeo.58.8.1246
- Toth, Z., Zhu, Y., and Marchok, T. (2001). The use of ensembles to identify forecasts with small and large uncertainty. *Weather and Forecasting*, 16(4), 463–477. https://doi.org/10.1175/1520-0434(2001)016<0463:TUOETI>2.0.CO;2
- Zhang, J., Florita, A., Hodge, B.-M., Lu, S., Hamann, H. F., Banunarayanan, V., and Brockway, A. M. (2015). A suite of metrics for assessing the performance of solar power forecasting. *Solar Energy*, 111, 157–175. https://doi.org/10.1016/j.solener.2014.10.016
