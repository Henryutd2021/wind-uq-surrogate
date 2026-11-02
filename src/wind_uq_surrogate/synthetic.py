"""Synthetic wind plants for examples and tests.

Everything here is invented: plant locations and sizes, wind statistics, data products, ensembles, the power
curve and the error model. The generator exists to show how the framework is used and to test it. Its numbers say
nothing about any real plant, data product or forecast.

Structure of the generator (``docs/data_pipeline.md``, Section 5):

* each plant has a slowly varying "regime" ``r_t`` in (0, 1) (calm and steady near 0, gusty near 1), partly
  shared between plants;
* the wind speed fluctuates more when ``r_t`` is high; two data products observe it with different errors, and
  an ensemble spreads more when ``r_t`` is high;
* the point forecast is the power curve applied to one data product;
* the observed power is the forecast plus Laplace noise with scale
  ``capacity * base * exp(signal * gain * (r_t - 0.5))``, clipped to ``[0, capacity]``.

``signal`` sets how strongly the forecast-error scale follows the regime. With ``signal=0`` the metrics still vary
with the regime, but the error scale does not: a negative control in which no surrogate should beat the constant
baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .features import PlantSeries


def power_curve(v, midpoint: float = 8.7, width: float = 1.3) -> np.ndarray:
    """Invented smooth power curve (per unit of capacity), zero at zero wind and close to one at high wind."""
    v = np.asarray(v, dtype=float)
    lo = 1.0 / (1.0 + np.exp(midpoint / width))
    out = (1.0 / (1.0 + np.exp(-(v - midpoint) / width)) - lo) / (1.0 - lo)
    return np.clip(out, 0.0, 1.0)


def _ar1(rng, n_steps: int, n_series: int, phi: float) -> np.ndarray:
    eps = rng.standard_normal((n_steps, n_series))
    out = np.empty_like(eps)
    out[0] = eps[0]
    c = np.sqrt(1.0 - phi**2)
    for t in range(1, n_steps):
        out[t] = phi * out[t - 1] + c * eps[t]
    return out


@dataclass
class SyntheticFleet:
    """Plants (observed ones carry an ``actual`` column), the plant table, and the ground truth of the generator
    (observed power and true error scale at every plant, including the unobserved ones)."""

    plants: list
    plant_table: pd.DataFrame
    truth: pd.DataFrame
    settings: dict = field(default_factory=dict)

    @property
    def observed(self) -> list:
        return [p for p in self.plants if p.observed]

    @property
    def unobserved(self) -> list:
        return [p for p in self.plants if not p.observed]


def synthetic_fleet(n_plants: int = 8, n_unobserved: int = 2, n_steps: int = 24 * 42, n_members: int = 10,
                    n_turbines: int = 0, signal: float = 1.0, base: float = 0.035, gain: float = 3.0,
                    seed: int = 0) -> SyntheticFleet:
    """Generate an invented fleet of wind plants with hourly series.

    Columns of every plant's ``data``: ``forecast`` (MW), ``ws_a`` and ``ws_b`` (two invented wind-speed data
    products, m/s), ``ens_00`` ... (ensemble members of wind speed, m/s) and, at observed plants, ``actual`` (MW).
    With ``n_turbines > 0`` each plant also gets a turbine-level wind-speed field for the spatial metrics.
    The last ``n_unobserved`` plants have no ``actual`` column.
    """
    if n_plants < 2 or not 0 <= n_unobserved < n_plants:
        raise ValueError("need at least two plants and at least one observed plant")
    if n_members < 2 and n_members != 0:
        raise ValueError("n_members must be 0 or at least 2")
    if signal < 0:
        raise ValueError("signal must be non-negative")
    rng = np.random.default_rng(seed)
    times = pd.date_range("2001-01-01", periods=n_steps, freq="h")
    xy = rng.uniform(0.0, 400.0, size=(n_plants, 2))
    capacity = np.round(rng.uniform(60.0, 320.0, size=n_plants) / 5.0) * 5.0
    mean_speed = rng.uniform(6.0, 9.5, size=n_plants)

    shared = _ar1(rng, n_steps, 1, 0.985)
    own = _ar1(rng, n_steps, n_plants, 0.98)
    regime = 1.0 / (1.0 + np.exp(-1.6 * (0.6 * shared + 0.8 * own)))

    hours = np.arange(n_steps)
    diurnal = 0.8 * np.sin(2 * np.pi * (hours[:, None] - rng.uniform(0, 24, n_plants)) / 24.0)
    innov = rng.standard_normal((n_steps, n_plants)) * (0.25 + 0.9 * regime)
    anom = np.empty_like(innov)
    anom[0] = innov[0]
    for t in range(1, n_steps):
        anom[t] = 0.9 * anom[t - 1] + innov[t]
    speed = np.clip(mean_speed + diurnal + anom, 0.0, None)

    ws_a = np.clip(speed + rng.normal(0.3, 1.0, size=(1, n_plants)) * 0.4
                   + rng.standard_normal(speed.shape) * (0.5 + 0.8 * regime), 0.0, None)
    kernel = np.ones(5) / 5.0
    smooth = np.column_stack([np.convolve(speed[:, p], kernel, mode="same") for p in range(n_plants)])
    ws_b = np.clip(smooth + rng.standard_normal(speed.shape) * 0.6, 0.0, None)

    plants, truth = [], []
    for p in range(n_plants):
        pid = f"P{p + 1:02d}"
        cap = float(capacity[p])
        fc = cap * power_curve(ws_a[:, p])
        true_scale = cap * base * np.exp(signal * gain * (regime[:, p] - 0.5))
        actual = np.clip(fc + rng.laplace(0.0, true_scale), 0.0, cap)
        data = {"forecast": fc, "ws_a": ws_a[:, p], "ws_b": ws_b[:, p]}
        if n_members:
            spread = 0.3 + 1.6 * regime[:, p]
            bias = rng.normal(0.0, 0.2, size=n_members)
            members = np.clip(speed[:, p, None] + bias[None, :] + spread[:, None] * rng.standard_normal(
                (n_steps, n_members)), 0.0, None)
            for k in range(n_members):
                data[f"ens_{k:02d}"] = members[:, k]
        is_observed = p < n_plants - n_unobserved
        if is_observed:
            data["actual"] = actual
        coords = field_df = None
        if n_turbines:
            coords, field_df = _turbine_field(rng, speed[:, p], regime[:, p], n_turbines, times)
        plants.append(PlantSeries(pid, pd.DataFrame(data, index=times), cap, x=float(xy[p, 0]), y=float(xy[p, 1]),
                                  turbine_coords=coords, turbine_field=field_df))
        truth.append(pd.DataFrame({"plant_id": pid, "time": times, "actual": actual, "true_scale": true_scale,
                                   "regime": regime[:, p], "observed": is_observed}))
    plant_table = pd.DataFrame({"plant_id": [p.plant_id for p in plants], "x_km": xy[:, 0], "y_km": xy[:, 1],
                                "capacity_mw": capacity, "observed": [p.observed for p in plants]})
    settings = {"n_plants": n_plants, "n_unobserved": n_unobserved, "n_steps": n_steps, "n_members": n_members,
                "n_turbines": n_turbines, "signal": signal, "base": base, "gain": gain, "seed": seed}
    return SyntheticFleet(plants, plant_table, pd.concat(truth, ignore_index=True), settings)


def _turbine_field(rng, speed, regime, n_turbines, times, size_km: float = 6.0, corr_range_km: float = 2.5,
                   nugget_sd: float = 0.15):
    """Turbine-level wind speeds: plant speed plus a spatially correlated anomaly (exponential correlation with
    practical range ``corr_range_km``) whose amplitude follows the regime, plus small independent noise."""
    coords = rng.uniform(0.0, size_km, size=(n_turbines, 2))
    d = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=-1))
    corr = np.exp(-3.0 * d / corr_range_km) + 1e-9 * np.eye(n_turbines)
    chol = np.linalg.cholesky(corr)
    z = rng.standard_normal((len(speed), n_turbines)) @ chol.T
    amp = 0.2 + 0.9 * regime
    vals = speed[:, None] + amp[:, None] * z + nugget_sd * rng.standard_normal((len(speed), n_turbines))
    cols = [f"T{k + 1:02d}" for k in range(n_turbines)]
    return coords, pd.DataFrame(np.clip(vals, 0.0, None), index=times, columns=cols)
