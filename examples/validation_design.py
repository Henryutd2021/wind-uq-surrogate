"""Validation design: targets per step vs from a window, and a negative control (about a minute).

Leave-one-plant-out validation of three surrogates and the constant baseline on two invented fleets: one whose
forecast-error scale follows the weather regime (signal = 1) and one where it does not (signal = 0). In the
second fleet the metrics still vary with the regime, so any gain over the constant baseline there would point to
leakage or overfitting.
"""

import pandas as pd

import wind_uq_surrogate as w
from wind_uq_surrogate.features import training_table
from wind_uq_surrogate.validation import cross_validate, plant_splits, score_predictive, score_scale

pd.set_option("display.width", 150)
levels = w.default_levels(0.05)
spec = w.FeatureSpec(window=18, pairs=(("ws_a", "ws_b"),), member_prefix="ens_", pi_history=48)
models = {name: (lambda name=name: w.Surrogate(name, max_train_samples=600))
          for name in ("gpr", "random_forest", "ridge", "constant")}

rows = []
for signal in (1.0, 0.0):
    fleet = w.synthetic_fleet(n_plants=7, n_unobserved=0, n_steps=24 * 35, n_members=10, signal=signal, seed=3)
    for label, target_window in (("per step", None), ("window 25", 25)):
        table = training_table(fleet.observed, spec, levels, target_window=target_window)
        cv = cross_validate(table, models, plant_splits(table["plant_id"]))
        pred, _ = score_predictive(cv, levels)
        scale = score_scale(cv)
        merged = pred[["model", "pinball", "crps"]].merge(scale[["model", "log_scale_r2"]], on="model")
        base = merged.loc[merged["model"] == "constant", "pinball"].iloc[0]
        merged["pinball_vs_constant"] = merged["pinball"] / base - 1.0
        merged.insert(0, "targets", label)
        merged.insert(0, "signal", signal)
        rows.append(merged)
print(pd.concat(rows, ignore_index=True).round(4).to_string(index=False))
