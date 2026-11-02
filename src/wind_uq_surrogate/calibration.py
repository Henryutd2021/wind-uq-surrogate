"""Step A of the workflow: calibrated uncertainty targets at observed plants, and recalibration of predictions.

Pinball-loss optimization (after Sun et al., 2019): around each point forecast ``m`` assume a location-scale
predictive distribution and choose its scale ``s`` so that the predictive quantiles minimize the mean pinball loss
against the observation(s). Because ``q_tau = m + s z_tau`` is linear in ``s`` and the pinball loss is convex, the
objective is a convex piecewise-linear function of ``s``; its minimum lies at a breakpoint ``s = e / z_tau``
(``e = y - m``) or at a bound, so it is found exactly (no stochastic search).
Details: ``docs/surrogate_workflow.md``, Section 2.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .probabilistic import _family, cdf, check_levels, standard_quantiles


def _identifiable(z: np.ndarray) -> None:
    if not np.any(np.abs(z) > 1e-12):
        raise ValueError("the levels must include at least one level other than 0.5: with the median alone the "
                         "scale does not affect the loss and is not identifiable")


def default_bounds(errors, levels, dist: str) -> tuple[float, float]:
    """Bounds that contain every per-sample optimum: from a tiny positive value to ``max|e| / min|z_tau|``."""
    e = np.abs(np.asarray(errors, dtype=float))
    z = np.abs(standard_quantiles(levels, dist))
    _identifiable(z)
    z = z[z > 1e-12]
    emax = float(np.nanmax(e)) if e.size else 0.0
    hi = 1.01 * max(emax, 1e-12) / z.min()
    lo = 1e-9 * max(emax, 1.0)
    return lo, max(hi, 2 * lo)


def _objective(e: np.ndarray, s: np.ndarray, z: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Mean pinball loss for errors ``e`` (n,) at candidate scales ``s`` (n, k): returns (n, k)."""
    u = e[:, None, None] - s[:, :, None] * z[None, None, :]
    return np.maximum(tau * u, (tau - 1.0) * u).mean(axis=2)


def _per_sample(e, z, tau, lo, hi, chunk=20000):
    n = len(e)
    out = np.empty(n)
    loss = np.empty(n)
    zz = np.where(np.abs(z) > 1e-12, z, np.nan)
    for start in range(0, n, chunk):
        ee = e[start:start + chunk]
        with np.errstate(invalid="ignore", divide="ignore"):
            bp = ee[:, None] / zz[None, :]
        bp = np.where((bp > lo) & (bp < hi), bp, lo)
        cand = np.sort(np.concatenate([np.full((len(ee), 1), lo), bp, np.full((len(ee), 1), hi)], axis=1), axis=1)
        obj = _objective(ee, cand, z, tau)
        best = obj.min(axis=1, keepdims=True)
        tol = 64.0 * np.finfo(float).eps * (np.abs(ee)[:, None] + hi * np.abs(z).max())
        first = np.argmax(obj <= best + tol, axis=1)  # smallest minimizing scale
        rows = np.arange(len(ee))
        out[start:start + chunk] = cand[rows, first]
        loss[start:start + chunk] = obj[rows, first]
    return out, loss


def _slope(e, s, z, tau):
    """Derivative of the summed pinball loss in ``s`` at a point that is not a breakpoint."""
    u = e[:, None] - s * z[None, :]
    psi = np.where(u > 0, tau, tau - 1.0)
    return float((-z[None, :] * psi).sum())


def _slope_tolerance(e, z) -> float:
    """Rounding tolerance of :func:`_slope`: a slope this close to zero is a flat stretch of the objective."""
    return 64.0 * np.finfo(float).eps * len(e) * float(np.abs(z).sum())


def _group(e, z, tau, lo, hi):
    zz = z[np.abs(z) > 1e-12]
    bp = (e[:, None] / zz[None, :]).ravel()
    cand = np.unique(np.concatenate([[lo, hi], bp[(bp > lo) & (bp < hi)]]))
    if len(cand) == 1:
        s = cand[0]
    else:
        mids = 0.5 * (cand[:-1] + cand[1:])
        tol = _slope_tolerance(e, z)
        a, b = 0, len(mids)  # first interval with non-negative slope; its left end is the smallest minimizer
        while a < b:
            k = (a + b) // 2
            if _slope(e, mids[k], z, tau) >= -tol:
                b = k
            else:
                a = k + 1
        s = cand[a]
    return s, float(_objective(e, np.full((len(e), 1), s), z, tau).mean())


def optimal_scale(actual, forecast, levels, dist: str = "laplace", bounds=None, groups=None):
    """Pinball-loss-optimal scale of a location-scale predictive distribution centred on the point forecast.

    Parameters
    ----------
    actual, forecast : 1-D arrays of observations and point forecasts (same unit).
    levels : increasing quantile levels in (0, 1) used in the loss.
    dist : ``"laplace"`` or ``"gaussian"`` (see :mod:`wind_uq_surrogate.probabilistic` for the scale convention).
    bounds : ``(lower, upper)`` for the scale; default :func:`default_bounds`. The lower bound must be positive.
    groups : ``None`` for one scale per sample (a target per time step), or an array of group labels (one per
        sample, e.g. a block of consecutive hours) to fit one scale per group.

    Returns
    -------
    scale : array with one scale per sample (groups: the group's scale repeated).
    loss : the attained mean pinball loss per sample (groups: the group's mean loss repeated).
    """
    y = np.asarray(actual, dtype=float)
    m = np.asarray(forecast, dtype=float)
    if y.shape != m.shape or y.ndim != 1:
        raise ValueError("actual and forecast must be 1-D arrays of the same length")
    if not (np.isfinite(y).all() and np.isfinite(m).all()):
        raise ValueError("actual and forecast must be finite (drop or fill missing values first)")
    tau = check_levels(levels)
    z = standard_quantiles(tau, dist)
    _identifiable(z)
    e = y - m
    lo, hi = default_bounds(e, tau, dist) if bounds is None else (float(bounds[0]), float(bounds[1]))
    if not (0 < lo < hi and np.isfinite(hi)):
        raise ValueError("bounds must satisfy 0 < lower < upper < inf")
    if groups is None:
        return _per_sample(e, z, tau, lo, hi)
    g = np.asarray(groups)
    if g.shape != e.shape:
        raise ValueError("groups must have one label per sample")
    scale = np.empty_like(e)
    loss = np.empty_like(e)
    for label in np.unique(g):
        sel = g == label
        s, l_ = _group(e[sel], z, tau, lo, hi)
        scale[sel] = s
        loss[sel] = l_
    return scale, loss


def rolling_optimal_scale(actual, forecast, levels, dist: str = "laplace", window: int = 25, bounds=None):
    """One pinball-loss-optimal scale per time step, fitted on the centred window of ``window`` steps around it
    (``window`` odd; the window is cut at the ends of the series). A smoother target than the per-step optimum,
    which rests on a single observation, at the price of temporal resolution. Returns ``(scale, loss)`` as
    :func:`optimal_scale`; ``loss`` is the mean pinball loss over the window."""
    y = np.asarray(actual, dtype=float)
    m = np.asarray(forecast, dtype=float)
    if y.shape != m.shape or y.ndim != 1:
        raise ValueError("actual and forecast must be 1-D arrays of the same length")
    if not (np.isfinite(y).all() and np.isfinite(m).all()):
        raise ValueError("actual and forecast must be finite (drop or fill missing values first)")
    if isinstance(window, (bool, np.bool_)) or not isinstance(window, (int, np.integer)) or window < 1 \
            or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    tau = check_levels(levels)
    z = standard_quantiles(tau, dist)
    _identifiable(z)
    e = y - m
    lo, hi = default_bounds(e, tau, dist) if bounds is None else (float(bounds[0]), float(bounds[1]))
    if not (0 < lo < hi and np.isfinite(hi)):
        raise ValueError("bounds must satisfy 0 < lower < upper < inf")
    half = window // 2
    scale = np.empty_like(e)
    loss = np.empty_like(e)
    for t in range(len(e)):
        scale[t], loss[t] = _group(e[max(0, t - half):t + half + 1], z, tau, lo, hi)
    return scale, loss


def block_labels(n: int, block: int) -> np.ndarray:
    """Labels ``0, 0, ..., 1, 1, ...`` for consecutive blocks of ``block`` samples (last block may be shorter)."""
    if block < 1:
        raise ValueError("block must be a positive integer")
    return np.arange(n) // int(block)


def _laplace_constant() -> float:
    return brentq(lambda u: (1.0 + u) * np.exp(-u) - 0.75, 1e-6, 10.0)


CRPS_SCALE_CONSTANT = {"gaussian": float(np.sqrt(np.log(2.0))), "laplace": _laplace_constant()}


def crps_optimal_scale(actual, forecast, dist: str = "laplace", floor: float = 0.0) -> np.ndarray:
    """Scale that minimizes the CRPS for a single observation: ``|e| / c`` with ``c = sqrt(ln 2)`` (Gaussian) or
    the root of ``(1 + u) e^{-u} = 3/4`` (Laplace). The pinball-loss optimum of :func:`optimal_scale` approaches
    it as the levels become dense, because the CRPS equals twice the pinball loss integrated over all levels.
    Values below ``floor`` are raised to ``floor``."""
    if dist not in CRPS_SCALE_CONSTANT:
        raise ValueError(f"dist must be one of {tuple(CRPS_SCALE_CONSTANT)}")
    e = np.abs(np.asarray(actual, dtype=float) - np.asarray(forecast, dtype=float))
    return np.maximum(e / CRPS_SCALE_CONSTANT[dist], floor)


class PITRecalibrator:
    """Recalibrate predictive distributions with the distribution of their PIT values (Kuleshov et al., 2018).

    Fit on held-out observations ``y`` with predictive CDFs ``F``: the PIT values ``u = F(y)`` have the empirical
    CDF ``R(p)`` = share of ``u`` at or below ``p``. ``R`` is the recalibration map of Kuleshov et al. (2018): at the
    PIT values their isotonic regression of ``R(u)`` on ``u`` returns ``R`` itself, which is already non-decreasing. A
    recalibrated ``tau``-quantile is the original quantile at the level ``R^{-1}(tau)``, the ``tau``-quantile of the
    PIT values. ``interpolation="inverted_cdf"`` uses the generalized inverse ``inf{p : R(p) >= tau}``; the default
    ``"linear"`` interpolates between order statistics (a continuous version). Ties are handled by both: if all PIT
    values are equal, every recalibrated level equals that value. Levels are kept inside ``[eps, 1 - eps]`` so that
    quantiles stay finite.
    """

    def __init__(self, dist: str, interpolation: str = "linear", eps: float = 1e-6):
        _family(dist)
        if interpolation not in ("linear", "inverted_cdf"):
            raise ValueError("interpolation must be 'linear' or 'inverted_cdf'")
        if not 0 < eps < 0.5:
            raise ValueError("eps must be between 0 and 0.5")
        self.dist = dist
        self.interpolation = interpolation
        self.eps = float(eps)
        self.pit_ = None

    def fit(self, y, loc, scale) -> "PITRecalibrator":
        y = np.asarray(y, dtype=float)
        s = np.asarray(scale, dtype=float)
        m = np.asarray(loc, dtype=float)
        if not (np.isfinite(y).all() and np.isfinite(m).all() and np.isfinite(s).all()):
            raise ValueError("y, loc and scale must be finite")
        if np.any(s <= 0):
            raise ValueError("scale must be positive")
        pit = np.asarray(cdf(y, m, s, self.dist), dtype=float).ravel()
        if len(pit) < 2:
            raise ValueError("at least two observations are needed")
        self.pit_ = np.sort(pit)
        return self

    def calibrated_levels(self, levels) -> np.ndarray:
        """Levels ``R^{-1}(tau)`` at which to query the original distribution (non-decreasing)."""
        if self.pit_ is None:
            raise RuntimeError("fit the recalibrator first")
        tau = check_levels(levels)
        new = np.quantile(self.pit_, tau, method=self.interpolation)
        return np.clip(np.maximum.accumulate(new), self.eps, 1.0 - self.eps)

    def quantiles(self, loc, scale, levels) -> np.ndarray:
        """Recalibrated predictive quantiles, shape ``(n, len(levels))``."""
        m = np.atleast_1d(np.asarray(loc, dtype=float))
        s = np.broadcast_to(np.asarray(scale, dtype=float), m.shape)
        if not (np.isfinite(m).all() and np.isfinite(s).all()) or np.any(s <= 0):
            raise ValueError("loc must be finite and scale positive and finite")
        z = _family(self.dist).ppf(self.calibrated_levels(levels))
        return m[:, None] + s[:, None] * z[None, :]
