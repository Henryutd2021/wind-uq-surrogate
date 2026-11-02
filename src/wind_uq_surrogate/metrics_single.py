"""Moving-window metrics of one time series (temporal variability).

Each function returns an array aligned with the input (trailing windows, see :mod:`wind_uq_surrogate.windows`).
Definitions and references: ``docs/metrics.md``, Section 2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .windows import as_1d, check_window, rolling


def moving_std(x, window: int, ddof: int = 0) -> np.ndarray:
    """Standard deviation in each trailing window (``ddof=0``: divide by the window length)."""
    window = check_window(window)
    if ddof not in (0, 1):
        raise ValueError("ddof must be 0 or 1")
    return rolling(x, window, lambda w: np.std(w, axis=1, ddof=ddof))


def _ratio_to_mean(num: np.ndarray, mean: np.ndarray, min_mean: float) -> np.ndarray:
    out = np.full_like(num, np.nan, dtype=float)
    ok = mean > min_mean
    out[ok] = num[ok] / mean[ok]
    return out


def turbulence_intensity(x, window: int, min_mean: float = 0.0, ddof: int = 0) -> np.ndarray:
    """Window standard deviation divided by the window mean (coefficient of variation).

    For wind speed sampled within the averaging period of IEC 61400-1 this is the turbulence intensity; for
    coarser series it measures relative variability. Windows whose mean is not above ``min_mean`` give ``NaN``.
    """
    window = check_window(window)
    if ddof not in (0, 1):
        raise ValueError("ddof must be 0 or 1")
    return rolling(x, window, lambda w: _ratio_to_mean(np.std(w, axis=1, ddof=ddof), w.mean(axis=1), min_mean))


def normalized_range(x, window: int, min_mean: float = 0.0) -> np.ndarray:
    """(maximum - minimum) / mean in each window.

    Some authors call this a variability index; other quantities share that name, so this package uses the
    descriptive name. Windows whose mean is not above ``min_mean`` give ``NaN``.
    """
    window = check_window(window)
    return rolling(x, window, lambda w: _ratio_to_mean(w.max(axis=1) - w.min(axis=1), w.mean(axis=1), min_mean))


def _spectral_entropy_rows(win: np.ndarray, detrend: str) -> np.ndarray:
    n = win.shape[1]
    if detrend == "mean":
        y = win - win.mean(axis=1, keepdims=True)
    elif detrend == "linear":
        t = np.arange(n) - (n - 1) / 2.0
        slope = (win * t).sum(axis=1, keepdims=True) / (t**2).sum()
        y = win - win.mean(axis=1, keepdims=True) - slope * t
    else:
        raise ValueError("detrend must be 'mean' or 'linear'")
    # rescale each window by its largest absolute deviation: the entropy is invariant to shifts and scaling
    dev = np.abs(y).max(axis=1, keepdims=True)
    # a window counts as constant when its variation is at the level of floating-point rounding of its values
    floor = 64.0 * np.finfo(float).eps * np.abs(win).max(axis=1, keepdims=True)
    ok = (dev > floor)[:, 0]
    y = np.divide(y, dev, out=np.zeros_like(y), where=dev > 0)
    power = np.abs(np.fft.rfft(y, axis=1)[:, 1:]) ** 2  # drop the zero frequency
    total = power.sum(axis=1, keepdims=True)
    out = np.full(win.shape[0], np.nan)
    ok &= total[:, 0] > 0
    p = power[ok] / total[ok]
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.where(p > 0, p * np.log(p), 0.0).sum(axis=1)
    out[ok] = h / np.log(power.shape[1])
    return out


def spectral_entropy(x, detrend: str = "mean") -> float:
    """Normalized spectral entropy of one series (0 = one frequency, 1 = flat spectrum).

    The periodogram of the detrended series is computed at the Fourier frequencies ``k / n``,
    ``k = 1, ..., floor(n / 2)`` (the zero frequency is dropped), normalized to a probability mass ``p_k`` and
    ``H = -sum_k p_k ln p_k / ln K`` with ``K`` the number of frequencies. A series without variation gives ``NaN``.
    Needs at least 4 values.
    """
    arr = as_1d(x)
    if len(arr) < 4:
        raise ValueError("spectral entropy needs at least 4 values")
    if np.isnan(arr).any():
        return float("nan")
    return float(_spectral_entropy_rows(arr[None, :], detrend)[0])


def moving_spectral_entropy(x, window: int, detrend: str = "mean") -> np.ndarray:
    """Normalized spectral entropy in each trailing window (window of at least 4 steps)."""
    window = check_window(window, minimum=4)
    return rolling(x, window, lambda w: _spectral_entropy_rows(w, detrend))


SINGLE_METRICS = ("std", "ti", "nrange", "sentropy")


def single_series_metrics(x, window: int, prefix: str = "", min_mean: float = 0.0,
                          metrics=SINGLE_METRICS, index=None) -> pd.DataFrame:
    """All moving-window metrics of one series as a table (columns ``<prefix><metric>``)."""
    unknown = set(metrics) - set(SINGLE_METRICS)
    if unknown:
        raise ValueError(f"unknown single-series metrics: {sorted(unknown)}")
    funcs = {
        "std": lambda: moving_std(x, window),
        "ti": lambda: turbulence_intensity(x, window, min_mean=min_mean),
        "nrange": lambda: normalized_range(x, window, min_mean=min_mean),
        "sentropy": lambda: moving_spectral_entropy(x, window),
    }
    data = {f"{prefix}{m}": funcs[m]() for m in metrics}
    if index is None and isinstance(x, pd.Series):
        index = x.index
    return pd.DataFrame(data, index=index)
