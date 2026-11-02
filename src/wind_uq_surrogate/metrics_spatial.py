"""Spatial variability inside a plant: empirical semivariogram of turbine-level values and a fitted model.

At each time step the values at the turbines (for example nacelle wind speeds) give an empirical semivariogram;
a model fitted to it summarizes the spatial structure in three numbers: nugget (variation at very short distance
and measurement noise), sill (total variance reached at large distance) and range (distance beyond which values
are practically uncorrelated). Definitions and references: ``docs/metrics.md``, Section 5 (Matheron, 1963;
Cressie, 1985 and 1993).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

EARTH_RADIUS_KM = 6371.0088  # mean Earth radius (IUGG)
MODELS = ("exponential", "gaussian", "spherical")


def pairwise_distances(coords, kind: str = "euclidean") -> np.ndarray:
    """Distance matrix of ``coords`` (shape ``(n, 2)``).

    ``kind="euclidean"``: planar coordinates, distances in the coordinates' unit (for example km).
    ``kind="haversine"``: columns are latitude and longitude in degrees; great-circle distance in km.
    """
    xy = np.asarray(coords, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("coords must have shape (n, 2)")
    if not np.isfinite(xy).all():
        raise ValueError("coords must be finite")
    if kind == "euclidean":
        diff = xy[:, None, :] - xy[None, :, :]
        return np.sqrt((diff**2).sum(axis=-1))
    if kind == "haversine":
        lat = np.radians(xy[:, 0])
        lon = np.radians(xy[:, 1])
        dlat = lat[:, None] - lat[None, :]
        dlon = lon[:, None] - lon[None, :]
        a = np.sin(dlat / 2) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2) ** 2
        return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    raise ValueError("kind must be 'euclidean' or 'haversine'")


def _detrend(values: np.ndarray, coords: np.ndarray, detrend: str) -> np.ndarray:
    if detrend == "none":
        return values
    if detrend == "linear":
        design = np.column_stack([np.ones(len(values)), coords])
        coef, *_ = np.linalg.lstsq(design, values, rcond=None)
        return values - design @ coef
    raise ValueError("detrend must be 'none' or 'linear'")


def empirical_variogram(values, coords, bin_edges, kind: str = "euclidean", detrend: str = "none",
                        min_pairs: int = 1, distances=None) -> pd.DataFrame:
    """Matheron's estimator ``gamma(h_k) = 1 / (2 N_k) sum_{pairs in bin k} (z_i - z_j)^2``.

    ``bin_edges`` are increasing lag-distance edges; a pair with distance ``d`` falls in bin ``k`` when
    ``edges[k] < d <= edges[k + 1]``. Returns one row per bin with at least ``min_pairs`` pairs: ``lag`` (mean
    distance of the pairs in the bin), ``gamma`` and ``n_pairs``. ``detrend="linear"`` first removes a planar trend
    in the coordinates. Turbines with a missing value are left out. Pass ``distances`` (a precomputed matrix) to
    avoid recomputing it at every time step.
    """
    z = np.asarray(values, dtype=float)
    xy = np.asarray(coords, dtype=float)
    if z.ndim != 1 or xy.shape != (len(z), 2):
        raise ValueError("values must be 1-D and coords of shape (len(values), 2)")
    edges = np.asarray(bin_edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0) or edges[0] < 0:
        raise ValueError("bin_edges must be at least two increasing non-negative values")
    dist = pairwise_distances(xy, kind) if distances is None else np.asarray(distances, dtype=float)
    keep = np.isfinite(z)
    z, xy, dist = z[keep], xy[keep], dist[np.ix_(keep, keep)]
    if len(z) < 3:
        return pd.DataFrame({"lag": [], "gamma": [], "n_pairs": []})
    z = _detrend(z, xy, detrend)
    iu = np.triu_indices(len(z), k=1)
    d = dist[iu]
    sq = (z[iu[0]] - z[iu[1]]) ** 2
    idx = np.searchsorted(edges, d, side="left") - 1  # edges[k] < d <= edges[k+1]
    ok = (idx >= 0) & (idx < len(edges) - 1)
    rows = []
    for k in range(len(edges) - 1):
        sel = ok & (idx == k)
        n = int(sel.sum())
        if n >= max(1, min_pairs):
            rows.append((float(d[sel].mean()), float(sq[sel].sum() / (2.0 * n)), n))
    return pd.DataFrame(rows, columns=["lag", "gamma", "n_pairs"])


def variogram_model(h, nugget: float, partial_sill: float, range_: float, model: str = "exponential") -> np.ndarray:
    """Semivariogram model with practical range ``range_`` (the model reaches 95 % of the partial sill there for
    the exponential and Gaussian models and exactly the sill for the spherical model). ``gamma(0) = 0``; for
    ``h > 0`` the model starts at the nugget."""
    h = np.asarray(h, dtype=float)
    if range_ <= 0:
        raise ValueError("range_ must be positive")
    r = h / range_
    if model == "exponential":
        shape = 1.0 - np.exp(-3.0 * r)
    elif model == "gaussian":
        shape = 1.0 - np.exp(-3.0 * r**2)
    elif model == "spherical":
        shape = np.where(r < 1.0, 1.5 * r - 0.5 * r**3, 1.0)
    else:
        raise ValueError(f"model must be one of {MODELS}")
    return np.where(h > 0, nugget + partial_sill * shape, 0.0)


def fit_variogram(lags, gamma, n_pairs=None, model: str = "exponential", weighting: str = "cressie",
                  max_range: float | None = None) -> dict:
    """Fit nugget, partial sill and practical range to an empirical semivariogram by (weighted) least squares.

    ``weighting="cressie"`` minimizes ``sum_k N_k (gamma_k / gamma_model(h_k) - 1)^2`` (Cressie, 1985);
    ``"pairs"`` minimizes ``sum_k N_k (gamma_k - gamma_model)^2``; ``"none"`` is ordinary least squares.
    Bounds: nugget and partial sill non-negative, range in ``(0, max_range]`` (default: 3 x the largest lag).
    Returns ``nugget``, ``partial_sill``, ``sill`` (nugget + partial sill), ``range``, ``rmse`` (unweighted fit
    error) and ``success``.
    """
    h = np.asarray(lags, dtype=float)
    g = np.asarray(gamma, dtype=float)
    if h.shape != g.shape or h.ndim != 1:
        raise ValueError("lags and gamma must be 1-D arrays of the same length")
    if len(h) < 3:
        raise ValueError("at least three lag bins are needed to fit three parameters")
    if not (np.isfinite(h).all() and np.isfinite(g).all()) or np.any(h <= 0) or np.any(g < 0):
        raise ValueError("lags must be positive and gamma non-negative and finite")
    npairs = np.ones_like(h) if n_pairs is None else np.asarray(n_pairs, dtype=float)
    if weighting not in ("cressie", "pairs", "none"):
        raise ValueError("weighting must be 'cressie', 'pairs' or 'none'")
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}")
    max_range = 3.0 * h.max() if max_range is None else float(max_range)
    scale = max(g.max(), 1e-12)

    def resid(p):
        model_g = variogram_model(h, p[0] * scale, p[1] * scale, p[2], model)
        if weighting == "cressie":
            return np.sqrt(npairs) * (g / np.maximum(model_g, 1e-12 * scale) - 1.0)
        w = np.sqrt(npairs) if weighting == "pairs" else 1.0
        return w * (g - model_g) / scale

    x0 = np.array([min(g[0], 0.5 * g.max()) / scale * 0.5, max(g.max() - 0.5 * g[0], 1e-6 * scale) / scale,
                   min(0.5 * h.max(), max_range)])
    lower = [0.0, 0.0, 1e-6 * max_range]
    upper = [np.inf, np.inf, max_range]
    x0 = np.clip(x0, np.array(lower) + 1e-9, np.array([1e6, 1e6, max_range]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sol = least_squares(resid, x0, bounds=(lower, upper), method="trf")
    nug, psill, rng_ = sol.x[0] * scale, sol.x[1] * scale, sol.x[2]
    fitted = variogram_model(h, nug, psill, rng_, model)
    return {"nugget": float(nug), "partial_sill": float(psill), "sill": float(nug + psill), "range": float(rng_),
            "rmse": float(np.sqrt(np.mean((g - fitted) ** 2))), "success": bool(sol.success)}


def variogram_over_time(field, coords, bin_edges, model: str = "exponential", kind: str = "euclidean",
                        detrend: str = "none", min_pairs: int = 1, weighting: str = "cressie",
                        max_range: float | None = None, prefix: str = "", index=None) -> pd.DataFrame:
    """Fit a semivariogram at every time step of ``field`` (shape ``(n_times, n_turbines)``).

    Returns ``<prefix>nugget``, ``<prefix>sill``, ``<prefix>range`` and ``<prefix>vg_rmse`` per time step;
    time steps with fewer than three usable bins, or where the fit fails, give ``NaN``.
    """
    arr = np.asarray(field, dtype=float)
    xy = np.asarray(coords, dtype=float)
    if arr.ndim != 2 or xy.shape != (arr.shape[1], 2):
        raise ValueError("field must be (n_times, n_turbines) and coords (n_turbines, 2)")
    dist = pairwise_distances(xy, kind)
    out = np.full((arr.shape[0], 4), np.nan)
    for t in range(arr.shape[0]):
        ev = empirical_variogram(arr[t], xy, bin_edges, kind=kind, detrend=detrend, min_pairs=min_pairs,
                                 distances=dist)
        if len(ev) < 3 or not (ev["gamma"] > 0).any():
            continue
        try:
            fit = fit_variogram(ev["lag"].to_numpy(), ev["gamma"].to_numpy(), ev["n_pairs"].to_numpy(),
                                model=model, weighting=weighting, max_range=max_range)
        except ValueError:
            continue
        if fit["success"]:
            out[t] = [fit["nugget"], fit["sill"], fit["range"], fit["rmse"]]
    cols = [f"{prefix}{c}" for c in ("nugget", "sill", "range", "vg_rmse")]
    return pd.DataFrame(out, columns=cols, index=index)
