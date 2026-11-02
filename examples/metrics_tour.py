"""Every UQ-metric family on one invented plant, grouped by the generator's regime (a few seconds).

In the synthetic generator a high regime value means gusty, less predictable weather; the table shows how each
metric moves with it. With real data the regime is unknown: this is only a reading aid for the metrics.
"""

import numpy as np
import pandas as pd

import wind_uq_surrogate as w

pd.set_option("display.width", 150)

fleet = w.synthetic_fleet(n_plants=2, n_unobserved=0, n_steps=24 * 30, n_members=12, n_turbines=25, seed=1)
plant = fleet.plants[0]
df = plant.data
window = 18

single = w.single_series_metrics(df["forecast"].to_numpy() / plant.capacity, window, prefix="forecast_")
pair = w.agreement_metrics(df["ws_a"].to_numpy(), df["ws_b"].to_numpy(), window, normalizer="mean", prefix="a_vs_b_")
members = df[[c for c in df.columns if c.startswith("ens_")]].to_numpy()
ens = w.ensemble_metrics(members, obs=df["ws_a"].to_numpy(), si_pairs=((75, 25), (90, 10)), history=48,
                         prefix="ens_")
spatial = w.variogram_over_time(plant.turbine_field.to_numpy(), plant.turbine_coords,
                                bin_edges=np.linspace(0.0, 6.0, 9), model="exponential", prefix="vg_")
table = pd.concat([single, pair, ens, spatial], axis=1)
table.index = df.index

truth = fleet.truth[fleet.truth["plant_id"] == plant.plant_id].set_index("time")
table["regime"] = pd.qcut(truth["regime"], 3, labels=["low", "middle", "high"])
print("Mean of each metric by regime tercile (synthetic data):")
print(table.groupby("regime", observed=True).mean().T.round(3).to_string(), "\n")

counts = w.rank_histogram(members, df["ws_a"].to_numpy(), rng=0)
print("Rank histogram of a data product among the ensemble members:", counts.tolist())
print("Summary:", {k: round(v, 3) if isinstance(v, float) else v for k, v in w.rank_histogram_summary(counts).items()})

sill = spatial["vg_sill"].to_numpy()
t = int(np.nanargmin(np.abs(sill - np.nanmedian(sill))))   # a time step with a typical sill
ev = w.empirical_variogram(plant.turbine_field.iloc[t].to_numpy(), plant.turbine_coords, np.linspace(0.0, 6.0, 9))
print(f"\nEmpirical semivariogram at step {t} and the fitted exponential model:")
fit = w.fit_variogram(ev["lag"], ev["gamma"], ev["n_pairs"], model="exponential")
ev["model"] = w.variogram_model(ev["lag"], fit["nugget"], fit["partial_sill"], fit["range"], "exponential")
print(ev.round(3).to_string(index=False))
print({k: round(v, 3) if isinstance(v, float) else v for k, v in fit.items()})
