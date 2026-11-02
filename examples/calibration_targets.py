"""Step A in detail: pinball-loss-optimal scales, their CRPS limit, and PIT recalibration (a few seconds)."""

import numpy as np
import pandas as pd

import wind_uq_surrogate as w

pd.set_option("display.width", 150)

fleet = w.synthetic_fleet(n_plants=2, n_unobserved=0, n_steps=24 * 30, n_members=0, seed=2)
plant = fleet.plants[0]
truth = fleet.truth[fleet.truth["plant_id"] == plant.plant_id]
true_pu = truth["true_scale"].to_numpy() / plant.capacity
levels = w.default_levels(0.05)

f = plant.data["forecast"].to_numpy() / plant.capacity
a = plant.data["actual"].to_numpy() / plant.capacity
rows = []
for label, kwargs in [("per step", {}), ("blocks of 24", {"block": 24}), ("centred window of 25", {"target_window": 25})]:
    tgt = w.calibration_targets(plant, levels, dist="laplace", **kwargs)
    r = np.corrcoef(np.log(tgt["target_scale"]), np.log(true_pu))[0, 1]
    rows.append({"target": label, "median scale": tgt["target_scale"].median(),
                 "corr(log target, log true scale)": r, "mean pinball": tgt["target_loss"].mean()})
print(pd.DataFrame(rows).round(4).to_string(index=False), "\n")

dense = np.arange(1, 1000) / 1000
s_dense, _ = w.optimal_scale(a[:5], f[:5], dense, dist="laplace")
print("Per-step optimum with 999 levels vs the closed-form CRPS optimum |e| / c:")
print(np.c_[s_dense, w.crps_optimal_scale(a[:5], f[:5], "laplace")].round(5), "\n")

# A deliberately overconfident constant scale, recalibrated on even days and checked on odd days.
fit_days = (np.arange(len(a)) // 24) % 2 == 0
chk = ~fit_days
scale = np.full(len(a), 0.5 * np.median(true_pu))
before = w.reliability(a[chk], w.quantiles(f[chk], scale[chk], levels, "laplace"), levels)
rec = w.PITRecalibrator("laplace").fit(a[fit_days], f[fit_days], scale[fit_days])
after = w.reliability(a[chk], rec.quantiles(f[chk], scale[chk], levels), levels)
print("Reliability (observed share below each quantile) before and after recalibration:")
print(pd.DataFrame({"level": levels, "before": before["observed"], "after": after["observed"]}).round(3)
      .iloc[::3].to_string(index=False))
