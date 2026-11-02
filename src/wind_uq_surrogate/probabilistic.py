"""Location-scale predictive distributions around a point forecast, and scores for probabilistic predictions.

A predictive distribution here is ``F(y) = F0((y - m) / s)`` with location ``m`` (the point forecast) and scale
``s``. ``dist="gaussian"``: ``s`` is the standard deviation. ``dist="laplace"``: ``s`` is the Laplace scale ``b``
(density ``exp(-|y - m| / b) / (2 b)``, standard deviation ``sqrt(2) b``).
Definitions and references: ``docs/surrogate_workflow.md``, Section 4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

DISTRIBUTIONS = ("gaussian", "laplace")


def _family(dist: str):
    if dist == "gaussian":
        return stats.norm
    if dist == "laplace":
        return stats.laplace
    raise ValueError(f"dist must be one of {DISTRIBUTIONS}, got {dist!r}")


def check_levels(levels) -> np.ndarray:
    """Quantile levels: a 1-D increasing array strictly inside (0, 1)."""
    tau = np.asarray(levels, dtype=float)
    if tau.ndim != 1 or len(tau) == 0:
        raise ValueError("levels must be a non-empty 1-D array")
    if not np.isfinite(tau).all() or np.any(tau <= 0) or np.any(tau >= 1) or np.any(np.diff(tau) <= 0):
        raise ValueError("levels must be increasing and strictly between 0 and 1")
    return tau


def standard_quantiles(levels, dist: str) -> np.ndarray:
    """Quantiles ``z_tau`` of the standardized distribution (location 0, scale 1)."""
    return _family(dist).ppf(check_levels(levels))


def quantiles(loc, scale, levels, dist: str) -> np.ndarray:
    """Predictive quantiles, shape ``(n, len(levels))``: ``q_tau = m + s z_tau``."""
    m = np.atleast_1d(np.asarray(loc, dtype=float))
    s = np.broadcast_to(np.asarray(scale, dtype=float), m.shape)
    if not np.isfinite(m).all():
        raise ValueError("loc must be finite")
    if np.any(~np.isfinite(s)) or np.any(s <= 0):
        raise ValueError("scale must be positive and finite")
    z = standard_quantiles(levels, dist)
    return m[:, None] + s[:, None] * z[None, :]


def cdf(y, loc, scale, dist: str) -> np.ndarray:
    """Predictive CDF at ``y`` (the probability integral transform, PIT, when ``y`` is the observation)."""
    y = np.asarray(y, dtype=float)
    s = np.asarray(scale, dtype=float)
    if np.any(s <= 0):
        raise ValueError("scale must be positive")
    return _family(dist).cdf((y - np.asarray(loc, dtype=float)) / s)


def _check_obs_quantiles(y, qpred, n_levels):
    y = np.asarray(y, dtype=float)
    q = np.asarray(qpred, dtype=float)
    if y.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if q.shape != (len(y), n_levels):
        raise ValueError(f"qpred must have shape ({len(y)}, {n_levels}), got {q.shape}")
    if not (np.isfinite(y).all() and np.isfinite(q).all()):
        raise ValueError("y and qpred must be finite")
    return y, q


def pinball_loss(y, qpred, levels, average: bool = True):
    """Pinball (quantile) loss ``rho_tau(u) = max(tau u, (tau - 1) u)`` with ``u = y - q_tau``.

    ``qpred`` has shape ``(n, len(levels))``. ``average=True`` returns the mean over samples and levels (a float);
    ``average=False`` returns the mean over samples for each level (an array).
    """
    tau = check_levels(levels)
    y, q = _check_obs_quantiles(y, qpred, len(tau))
    u = y[:, None] - q
    loss = np.maximum(tau * u, (tau - 1.0) * u)
    per_level = loss.mean(axis=0)
    return float(per_level.mean()) if average else per_level


def crps_gaussian(y, loc, scale) -> np.ndarray:
    """Closed-form CRPS of a Gaussian: ``s [z (2 Phi(z) - 1) + 2 phi(z) - 1 / sqrt(pi)]``, ``z = (y - m) / s``
    (Gneiting et al., 2005)."""
    s = np.asarray(scale, dtype=float)
    if np.any(s <= 0):
        raise ValueError("scale must be positive")
    z = (np.asarray(y, dtype=float) - np.asarray(loc, dtype=float)) / s
    return s * (z * (2.0 * stats.norm.cdf(z) - 1.0) + 2.0 * stats.norm.pdf(z) - 1.0 / np.sqrt(np.pi))


def crps_laplace(y, loc, scale) -> np.ndarray:
    """Closed-form CRPS of a Laplace distribution: ``|e| + b exp(-|e| / b) - 3 b / 4``, ``e = y - m``
    (Jordan et al., 2019)."""
    b = np.asarray(scale, dtype=float)
    if np.any(b <= 0):
        raise ValueError("scale must be positive")
    e = np.abs(np.asarray(y, dtype=float) - np.asarray(loc, dtype=float))
    return e + b * np.exp(-e / b) - 0.75 * b


def crps(y, loc, scale, dist: str) -> np.ndarray:
    """Closed-form CRPS for ``dist`` (``"gaussian"`` or ``"laplace"``)."""
    if dist == "gaussian":
        return crps_gaussian(y, loc, scale)
    if dist == "laplace":
        return crps_laplace(y, loc, scale)
    raise ValueError(f"dist must be one of {DISTRIBUTIONS}")


def reliability(y, qpred, levels) -> pd.DataFrame:
    """Observed share of observations at or below each predictive quantile, next to the nominal level.

    A calibrated prediction has ``observed`` close to ``level`` for every level (reliability diagram).
    """
    tau = check_levels(levels)
    y, q = _check_obs_quantiles(y, qpred, len(tau))
    observed = (y[:, None] <= q).mean(axis=0)
    return pd.DataFrame({"level": tau, "observed": observed, "deviation": observed - tau})


def central_intervals(levels, coverages) -> list[tuple[float, int, int]]:
    """For each nominal coverage ``c`` find the level indices of ``(1 - c) / 2`` and ``(1 + c) / 2``."""
    tau = check_levels(levels)
    out = []
    for c in coverages:
        c = float(c)
        if not 0 < c < 1:
            raise ValueError("coverages must be strictly between 0 and 1")
        lo = np.flatnonzero(np.isclose(tau, (1 - c) / 2))
        hi = np.flatnonzero(np.isclose(tau, (1 + c) / 2))
        if len(lo) == 0 or len(hi) == 0:
            raise ValueError(f"levels do not contain the bounds of the central {c:.0%} interval")
        out.append((c, int(lo[0]), int(hi[0])))
    return out


def available_coverages(levels, coverages) -> tuple:
    """The nominal coverages whose interval bounds ``(1 -/+ c) / 2`` are among ``levels``."""
    tau = check_levels(levels)
    return tuple(float(c) for c in coverages
                 if np.isclose(tau, (1 - c) / 2).any() and np.isclose(tau, (1 + c) / 2).any())


def interval_scores(y, qpred, levels, coverages) -> pd.DataFrame:
    """Coverage (share of observations inside) and sharpness (mean width) of central prediction intervals."""
    y, q = _check_obs_quantiles(y, qpred, len(check_levels(levels)))
    rows = []
    for c, i, j in central_intervals(levels, coverages):
        inside = (y >= q[:, i]) & (y <= q[:, j])
        rows.append({"nominal": c, "coverage": float(inside.mean()), "mean_width": float((q[:, j] - q[:, i]).mean())})
    return pd.DataFrame(rows)


def default_levels(step: float = 0.05) -> np.ndarray:
    """Evenly spaced levels ``step, 2 step, ..., 1 - step``."""
    n = int(round(1.0 / step))
    if n < 2 or not np.isclose(n * step, 1.0):
        raise ValueError("1 / step must be an integer of at least 2")
    return np.arange(1, n) / n
