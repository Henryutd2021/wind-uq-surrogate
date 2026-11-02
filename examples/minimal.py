"""Minimal end-to-end run on invented plants (about half a minute).

Steps A-D of docs/surrogate_workflow.md: calibrated targets at six observed plants, UQ-metric features,
five surrogates compared by leave-one-plant-out validation, and predictive quantiles at two unobserved plants.
The synthetic generator knows the true error scale everywhere, so the unobserved predictions can be checked here;
with real data they cannot.
"""

import numpy as np
import pandas as pd

import wind_uq_surrogate as w

pd.set_option("display.width", 150)
pd.set_option("display.max_columns", 20)

fleet = w.synthetic_fleet(n_plants=8, n_unobserved=2, n_steps=24 * 42, n_members=10, seed=0)
print(fleet.plant_table.round(1).to_string(index=False), "\n")

spec = w.FeatureSpec(
    window=18,                      # trailing window of the moving-window metrics (steps)
    single_series=("forecast",),    # temporal metrics of the per-unit point forecast
    pairs=(("ws_a", "ws_b"),),      # agreement of two wind-speed data products
    member_prefix="ens_",           # ensemble spread indices
    pi_history=48,                  # history of the predictability index (steps)
)
config = w.WorkflowConfig(features=spec, dist="laplace", target_window=25)
result = w.run_workflow(fleet.observed, fleet.unobserved, config)

print(f"training rows: {len(result.table)}, features: {len(w.feature_columns(result.table))}\n")
print("Accuracy of the predicted scale at held-out plants (per unit of capacity):")
print(result.scale_scores.round(4).to_string(index=False), "\n")
print("Predictive distributions at held-out plants (before recalibration):")
print(result.predictive_scores.round(4).to_string(index=False), "\n")
print(f"{result.chosen_model} after PIT recalibration (nested plant folds):")
print(result.recalibrated_scores.round(4).to_string(index=False), "\n")
print(f"chosen surrogate: {result.chosen_model}")

# Check against the generator's truth (possible only with synthetic data).
pred = result.predictions.merge(fleet.truth, on=["plant_id", "time"])
cap = pred["plant_id"].map({p.plant_id: p.capacity for p in fleet.plants})
inside = (pred["actual"] >= pred["q5_mw"]) & (pred["actual"] <= pred["q95_mw"])
r = np.corrcoef(np.log(pred["pred_scale_pu"]), np.log(pred["true_scale"] / cap))[0, 1]
print(f"unobserved plants: share inside the 90 % interval {inside.mean():.3f}; "
      f"correlation of log predicted and log true scale {r:.2f}")
